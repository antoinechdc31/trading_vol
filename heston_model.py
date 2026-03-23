import numpy as np

class HestonModel:
    def __init__(self, mu, kappa, theta, xi, rho, dt):
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.dt = dt

    def simulate(self, S0, v0, n_steps):
        S = np.zeros(n_steps)
        v = np.zeros(n_steps)
        S[0] = S0
        v[0] = v0

        for t in range(1, n_steps):
            
            z1 = np.random.randn()
            z2 = np.random.randn()
            W1 = z1
            W2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2
            # cette forme de W2 permet d'avoir la correlation + variance = 1

            # Ensure variance stays positive
            v_prev = max(v[t-1], 1e-8)

            # Heston version discrete dvt = vt - v(t-1)
            v[t] = v_prev + self.kappa * (self.theta - v_prev) * self.dt \
                   + self.xi * np.sqrt(v_prev) * np.sqrt(self.dt) * W2

            v[t] = max(v[t], 1e-8)

            # de meme ici on fait la version discrete
            S[t] = S[t-1] + self.mu * S[t-1] * self.dt \
                   + S[t-1] * np.sqrt(v_prev) * np.sqrt(self.dt) * W1

        return S, v