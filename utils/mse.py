import numpy as np
from sklearn.metrics import mean_squared_error


def cv_mse(model_class, X, y, alphas, kf):
    """
    Performs k-fold cross-validation to compute mean squared error for
    different alpha regularization parameters.

    Parameters
    ----------
    model_class : sklearn.base.BaseEstimator
        A scikit-learn model class (e.g., Ridge, Lasso) that accepts an
        alpha parameter.
    X : ndarray
        Feature matrix of shape (n_samples, n_features).
    y : ndarray
        Target vector of shape (n_samples,).
    alphas : array-like
        Array of regularization parameters to evaluate.
    kf : sklearn.model_selection.KFold
        A K-Fold cross-validator object.

    Returns
    -------
    mean_mse : list
        List of mean squared errors for each alpha value.
    """
    mean_mse = []

    for a in alphas:
        fold_mse = []

        for train_idx, val_idx in kf.split(X):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            model = model_class(alpha=a)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            fold_mse.append(mean_squared_error(y_val, y_pred))

        mean_mse.append(np.mean(fold_mse))

    return mean_mse
