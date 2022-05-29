from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np

import IMLearn.base
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_folds = np.split(X, cv)
    scores = []

    for i in range(cv):
        test_fold = train_folds.pop(i)
        estimator.fit(train_folds, y)
        scores.append(scoring(y, test_fold))
        train_folds.insert(i, test_fold)

    mean = np.mean(scores)[0]
    std = np.std(scores)[0]

    return mean, std
