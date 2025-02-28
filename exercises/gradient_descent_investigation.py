import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate

import plotly.graph_objects as go

from sklearn.metrics import auc


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    weights_ls, values_ls = [], []

    def callback(weights, val, grad, t, eta, delta):
        weights_ls.append(weights)
        values_ls.append(val)

    return callback, values_ls, weights_ls


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for module in [L1, L2]:
        callback, values, weights = get_gd_state_recorder_callback()
        for eta in etas:
            gd = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
            gd.fit(f=module(weights=init), X=None, y=None)
            fg = plot_descent_path(module=module, descent_path=np.array(weights),
                                   title=f"of {module.__name__} module with learning rate of {eta}")
            fg.show()
            go.Figure([go.Scatter(x=np.array(range(len(values))), y=values, mode='lines')],
                      layout=go.Layout(
                          title=rf"$\textbf{{(3) the {module.__name__} norm as a function of the GD iteration"
                                f" with learning rate of {eta}}}$",
                          yaxis_title=dict(text=f"{module.__name__} norm value"),
                          xaxis_title=dict(text="Iteration"))).show()
            print(f"The {module.__name__} norm with learning rate of {eta} achieved minimum loss of"
                  f" {format(min(values), '.10f')}")
            values.clear()
            weights.clear()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    callback, values, weights = get_gd_state_recorder_callback()
    gammas_values, gammas_weights = {}, {}
    for decay_rate in gammas:
        gd = GradientDescent(learning_rate=ExponentialLR(eta, decay_rate=decay_rate), callback=callback)
        gd.fit(f=L1(weights=init, ), X=None, y=None)

        gammas_values[decay_rate] = values.copy()
        gammas_weights[decay_rate] = weights.copy()

        print(f"The L1 norm with learning rate of {eta} and decay rate of {decay_rate} achieved minimum loss of"
              f" {format(min(values), '.10f')}")

        values.clear()
        weights.clear()

    # Plot algorithm's convergence for the different values of gamma
    min_iteration = min([len(i) for i in gammas_values.values()])
    go.Figure([go.Scatter(x=np.array(range(len(gammas_values[decay_rate]))), y=gammas_values[decay_rate], mode='lines',
                          name=decay_rate) for decay_rate in gammas],
              layout=go.Layout(title=rf"$\textbf{{(5) L1 norm value as a function of the GD iteration"
                                     f" with learning rate of 0.1 and different decay rate}}$",
                               yaxis_title=dict(text=f"L1 norm value"),
                               xaxis_title=dict(text="Iteration"),
                               legend_title=dict(text="Decay Rate"))).show()
    # Plot descent path for gamma=0.95
    for module in [L1, L2]:
        gd = GradientDescent(learning_rate=ExponentialLR(eta, decay_rate=0.95), callback=callback)
        gd.fit(f=module(weights=init), X=None, y=None)

        plot_descent_path(module=module, descent_path=np.array(weights),
                          title=f"of {module.__name__} module with learning rate of {eta} and decay rate of 0.95").show()
        values.clear()
        weights.clear()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heart Disease dataset (as numpy)
    X_train, y_train, X_test, y_test = [data.to_numpy() for data in load_data(train_portion=.8)]

    # Plotting convergence rate of logistic regression over SA heart disease data
    callback, descent_path, values = get_gd_state_recorder_callback()
    gd = GradientDescent(callback=callback, learning_rate=FixedLR(1e-4), max_iter=20000)
    lr = LogisticRegression(solver=gd)
    lr.fit(X_train, y_train)
    predicted_proba = lr.predict_proba(X_train)

    FPR, TPR = [], []
    ROC = []
    for alpha in np.arange(0, 1.01, .01):
        y_pred = np.where(predicted_proba >= alpha, 1, 0)
        fp = np.sum((y_pred == 1) & (y_train == 0))
        tp = np.sum((y_pred == 1) & (y_train == 1))
        fn = np.sum((y_pred == 0) & (y_train == 1))
        tn = np.sum((y_pred == 0) & (y_train == 0))
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        TPR.append(tpr)
        FPR.append(fpr)
        ROC.append(tpr - fpr)

    go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                          name="Random Class Assignment"),
               go.Scatter(x=FPR, y=TPR, mode='markers+lines', text=np.arange(0, 1.01, .01), name="", showlegend=False,
                          marker_size=5)],
              layout=go.Layout(title=rf"$\textbf{{(8) ROC Curve Of Fitted Model - AUC}}={auc(FPR, TPR):.6f}$",
                               xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                               yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    optimal_alpha = int(np.argmax(ROC)) / 100
    print(f"The alpha which achieves the optimal ROC is: {optimal_alpha}")
    lr = LogisticRegression(solver=gd, alpha=optimal_alpha)
    lr.fit(X_train, y_train)
    print(f"The test error using the optimal alpha is: {lr.loss(X_test, y_test)}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for penalty in ["l1", "l2"]:
        train_errors, validation_errors = [], []
        for lam in lambdas:
            train_err, validation_err = cross_validate(LogisticRegression(solver=gd, penalty=penalty, lam=lam,
                                                                          alpha=0.5), X=X_train, y=y_train,
                                                       scoring=misclassification_error, cv=5)
            train_errors.append(train_err)
            validation_errors.append(validation_err)

        best_lam = lambdas[int(np.argmin(validation_errors))]
        lr = LogisticRegression(solver=gd, penalty=penalty, alpha=0.5, lam=best_lam)
        lr.fit(X_train, y_train)
        print(f"Results for {penalty} penalty:")
        print(f"Best lambda value: {best_lam}")
        print(f"Error of LogisticRegression on the test set is: {lr.loss(X_test, y_test)}\n")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
