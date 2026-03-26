import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HestonParams:
    """Heston model parameters: mu, kappa, theta, xi, rho."""
    mu:    float = 0.05    # drift of S
    kappa: float = 2.0     # mean-reversion speed of variance
    theta: float = 0.04    # long-run variance (≈ 20% vol)
    xi:    float = 0.3     # vol-of-vol
    rho:   float = -0.7    # correlation W1 <-> W2

    def to_array(self) -> np.ndarray:
        """Pack parameters into a 1-D array for optimisation."""
        return np.array([self.mu, self.kappa, self.theta, self.xi, self.rho])

    @staticmethod
    def from_array(arr: np.ndarray) -> "HestonParams":
        """Unpack a 1-D array produced by `to_array` back into a HestonParams."""
        return HestonParams(mu=arr[0], kappa=arr[1], theta=arr[2], xi=arr[3], rho=arr[4])

    # Feller condition: 2κθ > ξ² ensures v_t stays positive a.s.
    @property
    def feller_satisfied(self) -> bool:
        """Return True when the Feller condition 2κθ > ξ² holds."""
        return 2 * self.kappa * self.theta > self.xi ** 2


@dataclass
class UKFState:
    """Internal mutable state of the UKF at a single time step."""
    v_hat:  float                  # filtered variance estimate  v̂_t
    P:      float                  # scalar error variance of v̂_t
    log_likelihood: float = 0.0   # accumulated log-likelihood contribution


@dataclass
class UKFResult:
    """Output returned by UKF.filter after processing a full price series."""
    v_hat:          np.ndarray   # filtered variance  v̂_t  (length T)
    sigma_hat:      np.ndarray   # estimated realised vol √v̂_t (length T)
    log_likelihood: float        # total log-likelihood of the path
    params:         HestonParams # Heston params used (fitted or provided)
    innovations:    np.ndarray   # one-step prediction errors on log-returns


# ──────────────────────────────────────────────────────────────────────────────
# UKF core
# ──────────────────────────────────────────────────────────────────────────────

class UKF:
    """
    Unscented Kalman Filter on the Heston State Space Model.

    State  : v_t  (instantaneous variance) — scalar, 1-D.
    Observation: log-return r_t = log(S_t / S_{t-1})  — scalar.

    State transition (Euler-Maruyama, Eq. from slide 19):
        v_t = v_{t-1} + κ(θ - v_{t-1})dt + ξ√v_{t-1} W2·√dt

    Observation equation (first-order approximation):
        r_t ~ N( (μ - v_{t-1}/2)dt,  v_{t-1}·dt )

    UKF sigma-point recipe: for a scalar state we use 2n+1 = 3 points
        χ_0  = v̂_{t|t-1}
        χ_±1 = v̂_{t|t-1} ± √((λ+n)·P_{t|t-1})
    with weights W_m = [λ/(n+λ), 1/(2(n+λ)), 1/(2(n+λ))]
    and  W_c = W_m + [(1-α²+β)] for index 0.

    Parameters
    ----------
    dt : float
        Time step in years (default 1/252 for daily data).
    alpha : float
        Spread of sigma points around the mean (1e-3 ≤ α ≤ 1).
    beta : float
        Prior knowledge of distribution (2 is optimal for Gaussian).
    kappa_ukf : float
        Secondary scaling parameter (0 is standard).
    Q_factor : float
        Multiplier on the process noise covariance (tuning knob).
    R_factor : float
        Multiplier on the measurement noise covariance (tuning knob).
    v_min : float
        Hard floor on variance to avoid numerical blow-up.
    """

    # ── constructor ──────────────────────────────────────────────────────────

    def __init__(
        self,
        dt:        float = 1.0 / 252,
        alpha:     float = 0.1,
        beta:      float = 2.0,
        kappa_ukf: float = 0.0,
        Q_factor:  float = 1.0,
        R_factor:  float = 1.0,
        v_min:     float = 1e-8,
    ) -> None:
        # alpha ∈ (0,1]: spread of sigma points. 1e-3 is the textbook default (Wan & Merwe 2000)
        # but leads to very large weights for a scalar state (n=1); 0.1 is a safer practical choice.
        self.dt        = dt
        self.alpha     = alpha
        self.beta      = beta
        self.kappa_ukf = kappa_ukf
        self.Q_factor  = Q_factor
        self.R_factor  = R_factor
        self.v_min     = v_min
        # pre-compute UKF scaling constants (scalar state → n=1)
        self._n   = 1
        self._lam = alpha**2 * (self._n + kappa_ukf) - self._n
        self._Wm, self._Wc = self._compute_weights()

    # ── weight computation (called once at init) ─────────────────────────────

    def _compute_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return mean weights Wm and covariance weights Wc for the 3 sigma points."""
        n, lam = self._n, self._lam
        Wm = np.array([lam / (n + lam), 1.0 / (2 * (n + lam)), 1.0 / (2 * (n + lam))])
        Wc = Wm.copy()
        Wc[0] += (1 - self.alpha**2 + self.beta)   # prior distribution correction
        return Wm, Wc

    # ── sigma points (scalar state) ──────────────────────────────────────────

    def _sigma_points(self, v_hat: float, P: float) -> np.ndarray:
        """Compute the 3 sigma points for a scalar state v_hat with variance P."""
        spread = np.sqrt((self._n + self._lam) * max(P, self.v_min))
        return np.array([v_hat, v_hat + spread, v_hat - spread])

    # ── Heston transition applied to a single sigma point ───────────────────

    def _f_transition(self, v: float, p: HestonParams) -> float:
        """Apply the Euler-Maruyama Heston variance transition (deterministic part)."""
        # deterministic mean only — noise is captured via Q
        v_pos = max(v, self.v_min)
        return v_pos + p.kappa * (p.theta - v_pos) * self.dt

    # ── process noise Q (state-dependent) ───────────────────────────────────

    def _process_noise(self, v_hat: float, p: HestonParams) -> float:
        """Return Q = ξ² · v̂ · dt (variance of the diffusion term of v_t)."""
        return self.Q_factor * p.xi**2 * max(v_hat, self.v_min) * self.dt

    # ── measurement noise R ──────────────────────────────────────────────────

    def _measurement_noise(self, v_hat: float) -> float:
        """Return R: variance of log-return observation given current variance estimate."""
        # Under Heston the log-return variance is v̂·dt; R_factor allows fine-tuning.
        return self.R_factor * max(v_hat, self.v_min) * self.dt

    # ── predict step ─────────────────────────────────────────────────────────

    def _predict(self, state: UKFState, p: HestonParams) -> tuple[float, float]:
        """UKF predict step: propagate sigma points through f, return (v_pred, P_pred)."""
        chi     = self._sigma_points(state.v_hat, state.P)
        chi_f   = np.array([self._f_transition(c, p) for c in chi])
        # predicted mean
        v_pred  = float(np.dot(self._Wm, chi_f))
        v_pred  = max(v_pred, self.v_min)
        # predicted covariance + process noise
        P_pred  = float(np.dot(self._Wc, (chi_f - v_pred)**2)) + self._process_noise(v_pred, p)
        P_pred  = max(P_pred, self.v_min)
        return v_pred, P_pred

    # ── update step ──────────────────────────────────────────────────────────

    def _update(
        self,
        v_pred:  float,
        P_pred:  float,
        r_t:     float,
        p:       HestonParams,
    ) -> tuple[float, float, float, float]:
        """
        UKF update step: assimilate observed log-return r_t.
        Returns (v_hat_new, P_new, innovation, S_innov).
        """
        # sigma points from predicted distribution
        chi      = self._sigma_points(v_pred, P_pred)
        # observation function h(v) = expected log-return = (μ - v/2)·dt
        h_chi    = np.array([(p.mu - c / 2.0) * self.dt for c in chi])
        y_pred   = float(np.dot(self._Wm, h_chi))   # predicted log-return
        # innovation covariance S = Σ Wc·(h_i - y_pred)² + R
        R        = self._measurement_noise(v_pred)
        S        = float(np.dot(self._Wc, (h_chi - y_pred)**2)) + R
        S        = max(S, self.v_min)
        # cross-covariance Pxy = Σ Wc·(chi_i - v_pred)·(h_i - y_pred)
        Pxy      = float(np.dot(self._Wc, (chi - v_pred) * (h_chi - y_pred)))
        # Kalman gain
        K_gain   = Pxy / S
        # innovation (actual minus predicted log-return)
        innov    = r_t - y_pred
        # posterior update
        v_new    = max(v_pred + K_gain * innov, self.v_min)
        P_new    = max(P_pred - K_gain * Pxy, self.v_min)
        return v_new, P_new, innov, S

    # ── single-step log-likelihood contribution ──────────────────────────────

    @staticmethod
    def _log_likelihood_step(innov: float, S: float) -> float:
        """Gaussian log-likelihood contribution: -0.5*(log(2πS) + innov²/S)."""
        return -0.5 * (np.log(2 * np.pi * S) + innov**2 / S)

    # ── full filter pass ─────────────────────────────────────────────────────

    def filter(
        self,
        log_returns: np.ndarray,
        params:      HestonParams,
        v0:          Optional[float] = None,
        P0:          Optional[float] = None,
    ) -> UKFResult:
        """
        Run the UKF forward pass on a log-return series given Heston params.

        Parameters
        ----------
        log_returns : array of shape (T,)
            Daily log-returns log(S_t / S_{t-1}).
        params : HestonParams
            Heston model parameters.
        v0 : float, optional
            Initial variance guess. Defaults to params.theta.
        P0 : float, optional
            Initial error variance. Defaults to params.theta.

        Returns
        -------
        UKFResult
        """
        T            = len(log_returns)
        v0           = v0 if v0 is not None else params.theta
        P0           = P0 if P0 is not None else params.theta
        # storage
        v_hat_arr    = np.empty(T)
        innov_arr    = np.empty(T)
        total_ll     = 0.0
        state        = UKFState(v_hat=v0, P=P0)

        for t in range(T):
            # ── predict ──────────────────────────────────────────────────────
            v_pred, P_pred = self._predict(state, params)
            # ── update ───────────────────────────────────────────────────────
            v_new, P_new, innov, S = self._update(v_pred, P_pred, log_returns[t], params)
            # ── accumulate log-likelihood ─────────────────────────────────────
            total_ll    += self._log_likelihood_step(innov, S)
            # ── store ─────────────────────────────────────────────────────────
            state.v_hat  = v_new
            state.P      = P_new
            v_hat_arr[t] = v_new
            innov_arr[t] = innov

        sigma_hat = np.sqrt(np.maximum(v_hat_arr, 0.0))
        return UKFResult(v_hat=v_hat_arr, sigma_hat=sigma_hat, log_likelihood=total_ll, params=params, innovations=innov_arr)

    # ── parameter calibration ────────────────────────────────────────────────

    def calibrate(
        self,
        log_returns:    np.ndarray,
        initial_params: Optional[HestonParams] = None,
        v0:             Optional[float]        = None,
        n_restarts:     int                    = 3,
        method:         str                    = "L-BFGS-B",
        max_iter:       int                    = 500,
    ) -> HestonParams:
        """
        Maximise the UKF log-likelihood over Heston params via numerical optimisation.

        Runs `n_restarts` independent optimisations from perturbed starting points
        and returns the parameter set achieving the highest log-likelihood.

        Parameters
        ----------
        log_returns : array of shape (T,)
            Daily log-returns used for calibration.
        initial_params : HestonParams, optional
            Starting point. Defaults to HestonParams() (class defaults).
        v0 : float, optional
            Initial variance passed to filter. Defaults to theta.
        n_restarts : int
            Number of random restarts to escape local optima.
        method : str
            SciPy optimiser (L-BFGS-B recommended for bounded problems).
        max_iter : int
            Maximum iterations per restart.

        Returns
        -------
        HestonParams with highest log-likelihood found.
        """
        initial_params = initial_params or HestonParams()
        # bounds: mu, kappa, theta, xi, rho
        bounds = [
            (-0.5,  0.5),    # mu
            (1e-4, 20.0),    # kappa  (fast mean-reversion allowed)
            (1e-5,  1.0),    # theta  (long-run var: [0%, 100%] vol)
            (1e-4,  5.0),    # xi     (vol-of-vol)
            (-0.999, 0.999), # rho
        ]

        def neg_ll(arr: np.ndarray) -> float:
            """Negative log-likelihood (minimised by scipy)."""
            p   = HestonParams.from_array(arr)
            res = self.filter(log_returns, p, v0=v0 or p.theta)
            return -res.log_likelihood

        best_ll     = -np.inf
        best_params = initial_params
        x0_base     = initial_params.to_array()

        for restart in range(n_restarts):
            # perturb starting point (first restart uses the provided guess)
            if restart == 0:
                x0 = x0_base.copy()
            else:
                noise = np.array([0.01, 0.5, 0.005, 0.05, 0.05])
                x0    = x0_base + np.random.randn(5) * noise
                # clip to bounds
                x0    = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

            logging.info("UKF calibration restart %d / %d — x0=%s", restart + 1, n_restarts, np.round(x0, 4))
            try:
                result = minimize(
                    neg_ll,
                    x0,
                    method=method,
                    bounds=bounds,
                    options={"maxiter": max_iter, "ftol": 1e-9},
                )
                if -result.fun > best_ll:
                    best_ll     = -result.fun
                    best_params = HestonParams.from_array(result.x)
                    logging.info("New best LL=%.4f  params=%s", best_ll, np.round(result.x, 4))
            except Exception as e:
                logging.warning("Restart %d failed: %s", restart + 1, e)

        logging.info("Calibration done. Best LL=%.4f  Feller=%s", best_ll, best_params.feller_satisfied)
        return best_params

    # ── rolling-window calibration + filter ──────────────────────────────────

    def rolling_filter(
        self,
        log_returns:         np.ndarray,
        window:              int,
        initial_params:      Optional[HestonParams] = None,
        recalibrate_every:   int                    = 21,
        n_restarts:          int                    = 1,
    ) -> UKFResult:
        """
        Calibrate Heston params on a rolling window, then filter the full series.

        The function slides a window of size `window` over the log-return series.
        Every `recalibrate_every` steps it re-fits the parameters on the latest
        window; in between it reuses the most recently fitted set.  A single
        forward UKF pass with time-varying params collects filtered variances.

        Parameters
        ----------
        log_returns : array of shape (T,)
            Full daily log-return series.
        window : int
            Number of observations in each calibration window.
        initial_params : HestonParams, optional
            Warm-start for the first calibration.
        recalibrate_every : int
            Recalibrate every this many steps (default: monthly ≈ 21 days).
        n_restarts : int
            Restarts per calibration call (keep low for speed).

        Returns
        -------
        UKFResult for the out-of-sample portion (log_returns[window:]).
        """
        T              = len(log_returns)
        params_t       = initial_params or HestonParams()
        # storage for out-of-sample filtered output
        v_hat_list  : list[float] = []
        innov_list  : list[float] = []
        total_ll       = 0.0
        state          = UKFState(v_hat=params_t.theta, P=params_t.theta)

        for t in range(window, T):
            # ── recalibrate if scheduled ──────────────────────────────────────
            if (t - window) % recalibrate_every == 0:
                window_returns = log_returns[t - window : t]
                params_t       = self.calibrate(window_returns, initial_params=params_t, v0=state.v_hat, n_restarts=n_restarts)
                logging.info("t=%d  recalibrated: kappa=%.3f theta=%.4f xi=%.3f rho=%.3f", t, params_t.kappa, params_t.theta, params_t.xi, params_t.rho)
            # ── one-step predict + update ─────────────────────────────────────
            v_pred, P_pred           = self._predict(state, params_t)
            v_new, P_new, innov, S   = self._update(v_pred, P_pred, log_returns[t], params_t)
            total_ll                += self._log_likelihood_step(innov, S)
            state.v_hat              = v_new
            state.P                  = P_new
            v_hat_list.append(v_new)
            innov_list.append(innov)

        v_hat_arr  = np.array(v_hat_list)
        sigma_hat  = np.sqrt(np.maximum(v_hat_arr, 0.0))
        return UKFResult(v_hat=v_hat_arr, sigma_hat=sigma_hat, log_likelihood=total_ll, params=params_t, innovations=np.array(innov_list))


# ──────────────────────────────────────────────────────────────────────────────
# Spread & signal helpers (used in projet.ipynb)
# ──────────────────────────────────────────────────────────────────────────────

def compute_iv_rv_spread(
    sigma_iv:  np.ndarray,
    sigma_hat: np.ndarray,
) -> np.ndarray:
    """
    Compute the daily implied-realised spread s_t = σ_IV,t − σ̂_t.
    Both inputs must be in the same unit (annualised volatility, not variance).
    """
    return sigma_iv - sigma_hat


def spread_to_weight(
    spread:       np.ndarray,
    scale:        float = 1.0,
    clip_min:     float = 0.0,
    clip_max:     float = 2.0,
    transform:    str   = "linear",
) -> np.ndarray:
    """
    Map the IV-RV spread s_t to a position weight w_t ∈ [clip_min, clip_max].

    The idea (slide 19): when s_t is large (IV >> RV) the carry is attractive →
    increase allocation; when s_t ≤ 0 the edge has disappeared → reduce/exit.

    Parameters
    ----------
    spread : array of s_t values.
    scale : multiplier applied before the non-linearity (sensitivity knob).
    clip_min / clip_max : hard bounds on the output weight.
    transform : one of {'linear', 'tanh', 'sigmoid'}.
        linear  → w = scale · s_t
        tanh    → w = tanh(scale · s_t)   (smooth, bounded in (-1,1) then clipped)
        sigmoid → w = σ(scale · s_t)      (always positive, in (0,1) then clipped)

    Returns
    -------
    np.ndarray of weights, clipped to [clip_min, clip_max].
    """
    x = scale * spread
    if transform == "tanh":
        w = np.tanh(x)
    elif transform == "sigmoid":
        w = 1.0 / (1.0 + np.exp(-x))
    else:
        # default: linear
        w = x
    return np.clip(w, clip_min, clip_max)
