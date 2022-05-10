import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise=0, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(wl=DecisionStump, iterations=n_learners)
    adaboost.fit(train_X, train_y)

    x, train_err, test_err = [], [], []
    for t in range(1, n_learners):
        print(f"iteration No {t}")
        train_err.append(adaboost.partial_loss(train_X, train_y, t))
        test_err.append(adaboost.partial_loss(test_X, test_y, t))
        x.append(t)

    go.Figure(data=[go.Scatter(x=x, y=train_err, name="Train Error"),
                    go.Scatter(x=x, y=test_err, name="Test Error")],
              layout=go.Layout(title=rf"$\textbf{{(1) Training and Test Errors as a Function of the Number of Fitted Learners (noise = {noise})}}$",
                               xaxis_title=dict(text="number of fitted learners"),
                               yaxis_title=dict(text="loss"))).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"{m} Iterations" for m in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)

    for i, m in enumerate(T):
        fig.add_traces([decision_surface(lambda X: adaboost.partial_predict(X, m), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(title=rf"$\textbf{{(2) AdaBoost Decision Boundaries of Different Number of Iterations (noise = {noise})}}$",
                      margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 3: Decision surface of best performing ensemble
    lowest_err = np.argmin(test_err)
    fig2 = go.Figure([decision_surface(lambda X: adaboost.partial_predict(X, lowest_err), lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))],)
    fig2.update_layout(title=rf"$\textbf{{(3) AdaBoost Decision Boundaries for {lowest_err} Iterations with accuracy of {1 - test_err[lowest_err]} (noise = {noise})}}$",
                      margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False).show()


    # Question 4: Decision surface with weighted samples
    D = adaboost.D_ / np.max(adaboost.D_) * 5
    fig3 = go.Figure([decision_surface(lambda X: adaboost.predict(X), lims[0], lims[1], showscale=False),
                      go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                                 mode="markers",
                                 showlegend=False,
                                 marker=dict(color=train_y, size=D,
                                             colorscale=[custom[0],
                                                         custom[-1]],
                                             line=dict(color="black",
                                                       width=1)))], )

    fig3.update_layout(
        title=rf"$\textbf{{(4) AdaBoost Decision Boundaries for {250} Iterations (noise = {noise})}}$",
        margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False).show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost()
    fit_and_evaluate_adaboost(noise=0.4)

