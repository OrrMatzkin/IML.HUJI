from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]

        self.mu_ = np.zeros([n_classes, n_features])
        self.cov_ = np.zeros([n_features, n_features])
        self.pi_ = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            # Only select the rows where the label equals the given class
            X_c = X[y == c]
            self.mu_[i] = X_c.mean(axis=0)
            self.pi_[i] = X_c.shape[0] / n_samples

        for i in range(n_samples):
            self.cov_ += np.outer(X[i] - self.mu_[y[i]], X[i] - self.mu_[y[i]])
        self.cov_ = self.cov_ / n_samples
        self._cov_inv = np.linalg.inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
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

        likelihoods = self.likelihood(X)
        y = []
        for likelihood in likelihoods:
            y.append(self.classes_[np.argmax(likelihood)])

        return np.array(y)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)xq
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        n, d = X.shape
        z = 1 / np.sqrt(np.power(2*np.pi, d) * np.linalg.det(self.cov_))

        likelihoods = np.zeros([n, self.classes_.shape[0]])
        for i in range(n):
            curr_likelihood = []
            for c in range(len(self.classes_)):
                x_mu = X[i] - self.mu_[c]
                exp = -0.5 * x_mu.T @ self._cov_inv @ x_mu
                curr_likelihood.append(z * np.exp(exp) * self.pi_[c])

            likelihoods[i] = np.array(curr_likelihood)

        return likelihoods

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
