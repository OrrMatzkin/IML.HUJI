from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        self.vars_ = np.zeros([n_classes, n_features])
        self.pi_ = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            # Only select the rows where the label equals the given class
            X_c = X[y == c]
            self.mu_[i, :] = X_c.mean(axis=0)
            self.pi_[i] = X_c.shape[0] / n_samples
            self.vars_[c] = X_c.var(axis=0)

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
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        n, d = X.shape
        likelihoods = np.zeros([n, self.classes_.shape[0]])
        for i in range(n):
            for c in self.classes_:
                likelihoods[i, c] = np.log(self.pi_[c])
                for j in range(d):
                    z = 1 / np.sqrt(2 * np.pi * self.vars_[c, j])
                    x_mu_var = (X[i, j] - self.mu_[c, j]) / np.sqrt(self.vars_[c, j])
                    likelihoods[i, c] += np.log(z) - 0.5 * np.power(x_mu_var, 2)

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
