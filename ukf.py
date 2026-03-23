import numpy as np

class UKF:
    def __init__(self, dim_x, dim_z, alpha=1e-3, beta=2, kappa=0):
        self.dim_x = dim_x  # dimension état
        self.dim_z = dim_z  # dimension observation

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.lambda_ = alpha**2 * (dim_x + kappa) - dim_x
        self.gamma = np.sqrt(dim_x + self.lambda_)

        # poids
        self.Wm = np.full(2 * dim_x + 1, 1 / (2 * (dim_x + self.lambda_)))
        self.Wc = self.Wm.copy()

        self.Wm[0] = self.lambda_ / (dim_x + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

    # -----------------------------
    # Sigma points
    # -----------------------------
    def sigma_points(self, x, P):
        A = np.linalg.cholesky(P)
        sigma = np.zeros((2 * self.dim_x + 1, self.dim_x))

        sigma[0] = x
        for i in range(self.dim_x):
            sigma[i+1] = x + self.gamma * A[:, i]
            sigma[self.dim_x + i + 1] = x - self.gamma * A[:, i]

        return sigma

    # -----------------------------
    # Prediction step
    # -----------------------------
    def predict(self, x, P, f, Q):
        sigma = self.sigma_points(x, P)

        # propagation
        sigma_pred = np.array([f(s) for s in sigma])

        # moyenne prédite
        x_pred = np.sum(self.Wm[:, None] * sigma_pred, axis=0)

        # covariance prédite
        P_pred = Q.copy()
        for i in range(len(sigma)):
            diff = sigma_pred[i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)

        return x_pred, P_pred, sigma_pred

    # -----------------------------
    # Update step
    # -----------------------------
    def update(self, x_pred, P_pred, sigma_pred, z, h, R):
        # projection dans espace observation
        Z_sigma = np.array([h(s) for s in sigma_pred])

        # moyenne observation
        z_pred = np.sum(self.Wm[:, None] * Z_sigma, axis=0)

        # covariance innovation
        S = R.copy()
        for i in range(len(Z_sigma)):
            diff = Z_sigma[i] - z_pred
            S += self.Wc[i] * np.outer(diff, diff)

        # covariance croisée
        C = np.zeros((self.dim_x, self.dim_z))
        for i in range(len(sigma_pred)):
            dx = sigma_pred[i] - x_pred
            dz = Z_sigma[i] - z_pred
            C += self.Wc[i] * np.outer(dx, dz)

        # gain de Kalman
        K = C @ np.linalg.inv(S)

        # mise à jour
        x_new = x_pred + K @ (z - z_pred)
        P_new = P_pred - K @ S @ K.T

        return x_new, P_new