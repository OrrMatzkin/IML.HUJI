from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    y = []
    X, y = np.zeros(n_samples), np.zeros(n_samples)
    for i in range(n_samples):
        eps = np.random.normal(0, noise)  # mean = 0, std = 5
        x = np.random.uniform(-1.2, 2)  # x is selected uniformly in the range [-1.2, 2]
        X[i], y[i] = x, f(x) + eps

    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), 2 / 3)
    train_X, train_y, test_X, test_y = train_X.to_numpy().flatten(), train_y.to_numpy(),\
                                       test_X.to_numpy().flatten(), test_y.to_numpy()

    X_true = np.linspace(-1.2, 2, num=n_samples)
    y_true = f(X_true)

    go.Figure([go.Scatter(x=X_true, y=y_true, mode='markers+lines', name="True Data"),
               go.Scatter(x=train_X, y=train_y, mode='markers', name="Train Data"),
               go.Scatter(x=test_X, y=test_y, mode='markers', name="Test Data")],
              layout=go.Layout(
                  title=rf"$\textbf{{(1) True (noiseless) Model and Noisy Model for Function f (noise = {noise}, samples = {n_samples})}}$",
                  xaxis_title=dict(text="X"), yaxis_title=dict(text="y"))).show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    max_k = 10
    train_errs, val_errs = np.zeros(max_k+1), np.zeros(max_k+1)

    for k in range(max_k+1):
        train_errs[k], val_errs[k] = cross_validate(estimator=PolynomialFitting(k), X=train_X,
                                                        y=train_y, scoring=mean_square_error, cv=5)

    go.Figure([go.Scatter(x=[i for i in range(max_k+1)], y=train_errs, mode='markers+lines', name="Training Error"),
               go.Scatter(x=[i for i in range(max_k+1)], y=val_errs, mode='markers+lines', name="Validation Error")],
              layout=go.Layout(
                  title=rf"$\textbf{{(2) Average Training and Validation Error as function of the Polynomial Degree"
                        f" (noise = {noise}, samples = {n_samples})}}$",
                  xaxis_title=dict(text="k"), yaxis_title=dict(text="Average Error Score"))).show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(val_errs)
    poly_es = PolynomialFitting(best_k)
    poly_es.fit(train_X, train_y)
    test_err = mean_square_error(test_y, poly_es.predict(test_X))
    print(f"k* = {best_k}, test error = {test_err}")
    print(val_errs)

def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree()
    # select_polynomial_degree(noise=0)
    # select_polynomial_degree(noise=10, n_samples=1500)
