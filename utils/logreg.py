from scipy.linalg import toeplitz
from numpy.random import multivariate_normal
import numpy as np
from scipy.special import expit


def simu_logreg(w0, n_samples=1000, corr=0.5):
    """
    Simulate a logistic regression model with correlated Gaussian features.

    Generates synthetic data from a logistic regression model where features
    follow a centered multivariate Gaussian distribution with Toeplitz
    covariance structure.

    Parameters
    ----------
    w0 : numpy.ndarray, shape (n_features,)
        Model weight coefficients.
    n_samples : int, default=1000
        Number of samples to generate.
    corr : float, default=0.5
        Correlation parameter for the Toeplitz covariance matrix.

    Returns
    -------
    X : numpy.ndarray, shape (n_samples, n_features)
        Simulated feature matrix with samples from a centered Gaussian
        distribution with Toeplitz covariance structure.
    y : numpy.ndarray, shape (n_samples,)
        Simulated binary labels in {-1, 1}.

    """
    n_features = w0.shape[0]
    cov = toeplitz(corr ** np.arange(0, n_features))
    X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    p = expit(X @ w0)
    y = np.random.binomial(1, p, size=n_samples)
    # transform the label in (-1, 1) instead of (0, 1)
    y[:] = 2 * y - 1
    return X, y
