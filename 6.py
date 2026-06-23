import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic non-linear data (a sine wave with noise)
np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) + 0.1 * np.random.randn(100)

# Add bias column (x0 = 1) for matrix multiplication
X_mat = np.c_[np.ones(100), X]
x_eval = np.linspace(0, 2 * np.pi, 200)
tau = 0.5  # Bandwidth parameter


def lwr_predict(x_query, X, y, tau):
    # Vectorized calculation of Gaussian weights for the query point
    weights = np.exp(-np.sum((X - x_query) ** 2, axis=1) / (2 * tau**2))
    W = np.diag(weights)

    # Normal equation for weighted linear regression: theta = (X^T * W * X)^-1 * X^T * W * y
    theta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    return x_query @ theta


# Predict over evaluation points
y_pred = [lwr_predict(np.array([1, x]), X_mat, y, tau) for x in x_eval]

# Plotting
plt.scatter(X, y, alpha=0.6, label="Data")
plt.plot(x_eval, y_pred, label=f"LWR (tau={tau})")
plt.title("Locally Weighted Regression")
plt.legend()
plt.show()