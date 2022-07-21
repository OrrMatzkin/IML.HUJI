import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    return np.power((y_true-y_pred), 2).mean()


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """

    eq = np.sum(y_true != y_pred).__float__()
    if normalize:
        return eq/y_true.size
    else:
        return eq


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    # total true labels of 1
    P = np.sum(y_true == 1)
    # total true labels of -1
    N = np.sum(y_true == -1)
    # True Positive: we predict a label of 1, and the true label is 1.
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    # True Negative: we predict a label of -1, and the true label is -1.
    TN = np.sum(np.logical_and(y_pred == -1, y_true == -1))

    return (TP+TN)/(P+N)


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    eps = 1e-4
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)   # values outside [0.0001, 0.9999] are set to the interval edges
    y_true_clipped = np.zeros_like(y_pred_clipped)
    y_true_clipped[np.arange(len(y_pred_clipped)), y_true] = 1
    return np.sum(-(y_true_clipped * np.log(y_pred_clipped)), axis=1)[0]


def softmax(X: np.ndarray) -> np.ndarray:
    """
    Compute the Softmax function for each sample in given data

    Parameters:
    -----------
    X: ndarray of shape (n_samples, n_features)

    Returns:
    --------
    output: ndarray of shape (n_samples, n_features)
        Softmax(x) for every sample x in given data X
    """
    return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
