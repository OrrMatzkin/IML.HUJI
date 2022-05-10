import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples = X.shape[0]
        models, weights = [], []

        # set initial distribution to be uniform
        self.D_ = np.ones(n_samples)
        self.D_ = self.D_ / n_samples

        for t in range(self.iterations_):
            # fit base learner
            curr_model = self.wl_()
            curr_model.fit(X, y * self.D_)
            # compute w_t (curr_weight)
            y_predict = curr_model.predict(X)
            train_err = np.sum(np.where(y_predict != y, self.D_, 0))
            curr_weight = 0.5 * np.log((1 / train_err) - 1)
            # add both to lists
            models.append(curr_model)
            weights.append(curr_weight)
            # update and normalize the sample weights
            self.D_ *= np.exp(-y * curr_weight * y_predict)
            self.D_ /= np.sum(self.D_)

        self.models_ = np.array(models)
        self.weights_ = np.array(weights)

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        n_samples = X.shape[0]
        y_predict = np.zeros(n_samples)

        for i in range(self.iterations_):
            y_predict += self.weights_[i] * self.models_[i].predict(X)

        return np.sign(y_predict)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from IMLearn.metrics import misclassification_error
        return misclassification_error(np.sign(y), self.predict(X))

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if T > self.iterations_:
            return np.zeros(X.shape[0])

        old_iterations = self.iterations_
        self.iterations_ = T
        y_predict = self.predict(X)
        self.iterations_ = old_iterations
        return y_predict


    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from IMLearn.metrics import misclassification_error
        return misclassification_error(np.sign(y), self.partial_predict(X, T))
