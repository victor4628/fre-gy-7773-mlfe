import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import add_dummy_feature


def logistic_loss(theta, X_b, y):
    """Compute the logistic (binary cross-entropy) loss.

    Parameters
    ----------
    theta : np.ndarray, shape (n_features,)
        Model parameters.
    X_b : np.ndarray, shape (m, n_features)
        Design matrix with bias column.
    y : np.ndarray, shape (m,)
        Binary target labels.

    Returns
    -------
    float
        The mean logistic loss.
    """
    z = X_b @ theta
    # np.logaddexp more stable than log(1 + exp(z))
    # logaddexp(0, z) = log(1 + exp(z))
    # (1/m) * sum(log(1 + exp(z)) - y * z)
    return np.mean(np.logaddexp(0, z) - y * z)


def logistic_grad(theta, X_b, y):
    """Compute the gradient of the logistic loss.

    Parameters
    ----------
    theta : np.ndarray, shape (n_features,)
        Model parameters.
    X_b : np.ndarray, shape (m, n_features)
        Design matrix with bias column.
    y : np.ndarray, shape (m,)
        Binary target labels.

    Returns
    -------
    np.ndarray, shape (n_features,)
        The gradient of the logistic loss w.r.t. theta.
    """
    m = len(y)
    z = X_b @ theta
    p = expit(z)
    return (1 / m) * (X_b.T @ (p - y)).ravel()


def learning_schedule(t, t0, t1):
    """Compute the learning rate at step t.

    Parameters
    ----------
    t : int
        Current iteration number.
    t0 : float
        Numerator constant for the schedule.
    t1 : float
        Denominator offset for the schedule.

    Returns
    -------
    float
        The learning rate at step t.
    """
    return t0 / (t + t1)


rng = np.random.default_rng(seed=42)
m = 2000  # number of samples
X = rng.normal(size=(m, 1))
X_b = add_dummy_feature(X)
theta_true = np.array([-0.5, 2.0])
logits = X_b @ theta_true
probs = expit(logits)
y = rng.binomial(1, probs)


# batch gradient descent
eta = 1.0
n_epochs = 200

theta = rng.standard_normal(2)
theta_path_bgd = [theta.copy()]

for _ in range(n_epochs):
    grad = logistic_grad(theta, X_b, y)
    theta = theta - eta * grad
    theta_path_bgd.append(theta.copy())

theta_path_bgd = np.array(theta_path_bgd)

# stochastic gradient descent
theta = rng.standard_normal(2)
theta_path_sgd = [theta.copy()]
n_epochs_sgd = 100

for epoch in range(n_epochs_sgd):
    for iteration in range(m):
        random_index = rng.integers(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        grad = logistic_grad(theta, xi, yi)
        eta = learning_schedule(epoch * m + iteration, t0=50, t1=500)
        theta = theta - eta * grad
        theta_path_sgd.append(theta.copy())

theta_path_sgd = np.array(theta_path_sgd)

# mini-batch gradient descent
n_epochs_mgd = 100
minibatch_size = 20
n_batches_per_epoch = m // minibatch_size

theta = rng.standard_normal(2)
theta_path_mgd = [theta.copy()]

for epoch in range(n_epochs_mgd):
    shuffled_indices = rng.permutation(m)
    X_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    for iteration in range(n_batches_per_epoch):
        idx = iteration * minibatch_size
        xi = X_shuffled[idx : idx + minibatch_size]
        yi = y_shuffled[idx : idx + minibatch_size]
        grad = logistic_grad(theta, xi, yi)
        eta = learning_schedule(
            epoch * n_batches_per_epoch + iteration,
            t0=200,
            t1=1000,
        )
        theta = theta - eta * grad
        theta_path_mgd.append(theta.copy())

theta_path_mgd = np.array(theta_path_mgd)

# for all methods, we need m to be large enough to converge to the true theta
print(f"BGD converged to: {theta_path_bgd[-1]}")
print(f"SGD converged to: {theta_path_sgd[-1]}")
print(f"MGD converged to: {theta_path_mgd[-1]}")
# with sklearn logistic regression
clf = LogisticRegression(fit_intercept=True, solver="lbfgs")
clf.fit(X, y)
theta_sklearn = np.array([clf.intercept_[0], clf.coef_[0, 0]])
print(f"sklearn converged to: {theta_sklearn}")
print(f"True theta: {theta_true}")

# plot the paths of the parameters for each optimization method
fig, ax = plt.subplots()
ax.plot(
    theta_path_sgd[:, 0], theta_path_sgd[:, 1], "*-r", linewidth=1, label="Stochastic"
)
ax.plot(
    theta_path_mgd[:, 0], theta_path_mgd[:, 1], "x-g", linewidth=2, label="Mini-batch"
)
ax.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], ".-b", linewidth=3, label="Batch")
ax.plot([theta_true[0]], [theta_true[1]], ".k", label="Optimal")
ax.legend(loc="upper left")
ax.set_xlabel(r"$\theta_0$")
ax.set_ylabel(r"$\theta_1$")
plt.show()
