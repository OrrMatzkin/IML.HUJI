import numpy as np
import pandas as pd
from typing import Tuple, List
from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, CrossEntropyLoss, Identity
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, FixedLR
from IMLearn.utils.utils import split_train_test
from utils import *

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "simple_white"


def generate_nonlinear_data(
        samples_per_class: int = 100,
        n_features: int = 2,
        n_classes: int = 2,
        train_proportion: float = 0.8) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a multiclass non linearly-separable dataset. Adopted from Stanford CS231 course code.

    Parameters:
    -----------
    samples_per_class: int, default = 100
        Number of samples per class

    n_features: int, default = 2
        Data dimensionality

    n_classes: int, default = 2
        Number of classes to generate

    train_proportion: float, default=0.8
        Proportion of samples to be used for train set

    Returns:
    --------
    train_X : ndarray of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : ndarray of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : ndarray of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : ndarray of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    X, y = np.zeros((samples_per_class*n_classes, n_features)), np.zeros(samples_per_class*n_classes, dtype='uint8')
    for j in range(n_classes):
        ix = range(samples_per_class * j, samples_per_class * (j + 1))
        r = np.linspace(0.0, 1, samples_per_class)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, samples_per_class) + np.random.randn(samples_per_class) * 0.2  # theta
        X[ix], y[ix] = np.c_[r * np.sin(t), r * np.cos(t)], j

    split = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion)
    return tuple(map(lambda x: x.values, split))


def plot_decision_boundary(nn: NeuralNetwork, lims, X: np.ndarray = None, y: np.ndarray = None, title=""):
    data = [decision_surface(nn.predict, lims[0], lims[1], density=40, showscale=False)]
    if X is not None:
        col = y if y is not None else "black"
        data += [go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                            marker=dict(color=col, colorscale=custom, line=dict(color="black", width=1)))]

    return go.Figure(data,
                     go.Layout(title=rf"$\text{{Network Decision Boundaries {title}}}$",
                               xaxis=dict(title=r"$x_1$"), yaxis=dict(title=r"$x_2$"),
                               width=400, height=400))


def animate_decision_boundary(nn: NeuralNetwork, weights: List[np.ndarray], lims, X: np.ndarray, y: np.ndarray,
                              title="", save_name=None):
    frames = []
    for i, w in enumerate(weights):
        nn.weights = w
        frames.append(go.Frame(data=[decision_surface(nn.predict, lims[0], lims[1], density=40, showscale=False),
                                     go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                                marker=dict(color=y, colorscale=custom, line=dict(color="black", width=1)))
                                     ],
                               layout=go.Layout(title=rf"$\text{{{title} Iteration {i+1}}}$")))

    fig = go.Figure(data=frames[0]["data"], frames=frames[1:],
                    layout=go.Layout(title=frames[0]["layout"]["title"]))
    if save_name:
        animation_to_gif(fig, save_name, 200, width=400, height=400)


def get_callback():
    loss = []
    iters = []
    grads = []
    weights = []

    def callback(**kwargs):
        loss.append(kwargs["val"])
        iters.append(kwargs["t"])
        grads.append(np.linalg.norm(kwargs["grad"]))
        if len(iters) > 0 and len(iters) % 100 == 0:
            weights.append(kwargs["weights"])
    return callback, loss, iters, grads, weights


if __name__ == '__main__':
    np.random.seed(0)

    # Generate and visualize dataset
    n_features, n_classes = 2, 3
    train_X, train_y, test_X, test_y = generate_nonlinear_data(
        samples_per_class=500, n_features=n_features, n_classes=n_classes, train_proportion=0.8)
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    go.Figure(data=[go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                               marker=dict(color=train_y, colorscale=custom, line=dict(color="black", width=1)))],
              layout=go.Layout(title=r"$\text{Train Data}$", xaxis=dict(title=r"$x_1$"), yaxis=dict(title=r"$x_2$"),
                               width=400, height=400)).show()

        # .write_image(f"../figures2/nonlinear_data.png")

    callback, loss, iterations, grads, weights = get_callback()

    # ---------------------------------------------------------------------------------------------#
    # Question 1: Fitting simple network with two hidden layers                                    #
    # ---------------------------------------------------------------------------------------------#

    nn1_layers = [FullyConnectedLayer(input_dim=n_features, output_dim=16, activation=ReLU()),
                  FullyConnectedLayer(input_dim=16, output_dim=16, activation=ReLU()),
                  FullyConnectedLayer(input_dim=16, output_dim=n_classes)]

    nn1 = NeuralNetwork(modules=nn1_layers,
                        loss_fn=CrossEntropyLoss(),
                        solver=GradientDescent(learning_rate=FixedLR(0.1), max_iter=5000, callback=callback))

    nn1.fit(train_X, train_y)
    plot_decision_boundary(nn1, lims, test_X, test_y, title="w/ 2 hidden layers").show()
    print(f"(Q1) Neural Network w/ 2 hidden layers accuracy = {accuracy(test_y, nn1.predict(test_X))}")

    # ---------------------------------------------------------------------------------------------#
    # Question 2: Fitting a network with no hidden layers                                          #
    # ---------------------------------------------------------------------------------------------#
    nn2_layers = [FullyConnectedLayer(input_dim=n_features, output_dim=n_classes)]

    nn2 = NeuralNetwork(modules=nn2_layers,
                         loss_fn=CrossEntropyLoss(),
                         solver=GradientDescent(learning_rate=FixedLR(0.1), max_iter=5000))

    nn2.fit(train_X, train_y)

    plot_decision_boundary(nn2, lims, test_X, test_y, title="w/ no hidden layers").show()
    print(f"(Q2) Neural Network w/ no hidden layers accuracy = {accuracy(test_y, nn2.predict(test_X))}")

    # ---------------------------------------------------------------------------------------------#
    # Question 3+4: Plotting network convergence process                                           #
    # ---------------------------------------------------------------------------------------------#

    # Question 3
    plot_convergence1 = go.Figure()
    plot_convergence1.add_trace(go.Scatter(x=iterations, y=loss, mode="lines",
                                marker=dict(color="blue"), name="Loss",
                                showlegend=True))
    plot_convergence1.add_trace(go.Scatter(x=iterations, y=grads, mode="lines",
                                marker=dict(color="red"), name="Gradient Norm",
                                showlegend=True))

    plot_convergence1.update_layout(
        title=f" (Q3) Convergence Process of the Neural Network Loss and Gradient Norm",
        xaxis_title="Iteration",
        yaxis_title="Loss/Gradient Norm",
        title_x=0.5)

    plot_convergence1.show()

    animate_decision_boundary(nn1, weights, lims, test_X, test_y, save_name="q3.gif")

    # Question 4
    callback2, loss2, iterations2, grads2, weights2 = get_callback()

    nn4_layers = [FullyConnectedLayer(input_dim=n_features, output_dim=6, activation=ReLU()),
                  FullyConnectedLayer(input_dim=6, output_dim=6, activation=ReLU()),
                  FullyConnectedLayer(input_dim=6, output_dim=n_classes)]

    nn4 = NeuralNetwork(modules=nn4_layers,
                        loss_fn=CrossEntropyLoss(),
                        solver=GradientDescent(learning_rate=FixedLR(0.1), max_iter=5000, callback=callback2))

    nn4.fit(train_X, train_y)

    plot_convergence2 = go.Figure()
    plot_convergence2.add_trace(go.Scatter(x=iterations2, y=loss2, mode="lines",
                               marker=dict(color="blue"), name="Loss",
                               showlegend=True))
    plot_convergence2.add_trace(go.Scatter(x=iterations2, y=grads2, mode="lines",
                               marker=dict(color="red"), name="Gradient Norm",
                               showlegend=True))
    plot_convergence2.update_layout(
        title=f" (Q4) Convergence Process of the Neural Network Loss and Gradient Norm",
        xaxis_title="Iteration",
        yaxis_title="Loss/Gradient Norm",
        title_x=0.5)
    plot_convergence2.show()

    animate_decision_boundary(nn4, weights2, lims, test_X, test_y, save_name="q4.gif")