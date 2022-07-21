import time
import numpy as np
import gzip
from typing import Tuple

from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, CrossEntropyLoss, softmax
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, StochasticGradientDescent, FixedLR
from IMLearn.utils.utils import confusion_matrix

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset

    Returns:
    --------
    train_X : ndarray of shape (60,000, 784)
        Design matrix of train set

    train_y : ndarray of shape (60,000,)
        Responses of training samples

    test_X : ndarray of shape (10,000, 784)
        Design matrix of test set

    test_y : ndarray of shape (10,000, )
        Responses of test samples
    """

    def load_images(path):
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            raw_data = np.frombuffer(f.read(), 'B', offset=16)
        # converting raw data to images (flattening 28x28 to 784 vector)
        return raw_data.reshape(-1, 784).astype('float32') / 255

    def load_labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            return np.frombuffer(f.read(), 'B', offset=8)

    return (load_images('../datasets/mnist-train-images.gz'),
            load_labels('../datasets/mnist-train-labels.gz'),
            load_images('../datasets/mnist-test-images.gz'),
            load_labels('../datasets/mnist-test-labels.gz'))


def plot_images_grid(images: np.ndarray, title: str = ""):
    """
    Plot a grid of images

    Parameters
    ----------
    images : ndarray of shape (n_images, 784)
        List of images to print in grid

    title : str, default="
        Title to add to figure

    Returns
    -------
    fig : plotly figure with grid of given images in gray scale
    """
    side = int(len(images) ** 0.5)
    subset_images = images.reshape(-1, 28, 28)

    height, width = subset_images.shape[1:]
    grid = subset_images.reshape(side, side, height, width).swapaxes(1, 2).reshape(height * side, width * side)

    return px.imshow(grid, color_continuous_scale="gray")\
        .update_layout(title=dict(text=title, y=0.97, x=0.5, xanchor="center", yanchor="top"),
                       font=dict(size=16), coloraxis_showscale=False)\
        .update_xaxes(showticklabels=False)\
        .update_yaxes(showticklabels=False)


def get_time_callback():
    time_record = []
    loss = []

    def callback(**kwargs):
        time_record.append(time.time())
        loss.append(np.mean(kwargs["val"]))

    return callback, time_record, loss


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_mnist()
    (n_samples, n_features), n_classes = train_X.shape, 10

    # ---------------------------------------------------------------------------------------------#
    # Question 5+6+7: Network with ReLU activations using SGD + recording convergence              #
    # ---------------------------------------------------------------------------------------------#
    # Initialize, fit and test network
    from exercises.nn_simulated_data import get_callback
    callback1, loss1, iterations1, grads1, weights1 = get_callback()

    nn1_layers = [FullyConnectedLayer(input_dim=train_X.shape[1], output_dim=64, activation=ReLU()),
                  FullyConnectedLayer(input_dim=64, output_dim=64, activation=ReLU()),
                  FullyConnectedLayer(input_dim=64, output_dim=n_classes)]

    nn1 = NeuralNetwork(modules=nn1_layers,
                        loss_fn=CrossEntropyLoss(),
                        solver=StochasticGradientDescent(learning_rate=FixedLR(0.1), max_iter=10000, batch_size=256,
                                                         callback=callback1))
    nn1.fit(train_X, train_y)
    test_y_predict = nn1.predict(test_X)
    print(f"(Q5) Neural Network w/ 2 hidden layers accuracy = {accuracy(test_y, test_y_predict)}")

    # Plotting convergence process
    plot_convergence1 = go.Figure()
    plot_convergence1.add_trace(go.Scatter(x=iterations1, y=loss1, mode="lines",
                                           marker=dict(color="blue"), name="Loss",
                                           showlegend=True))
    plot_convergence1.add_trace(go.Scatter(x=iterations1, y=grads1, mode="lines",
                                           marker=dict(color="red"), name="Gradient Norm",
                                           showlegend=True))

    plot_convergence1.update_layout(
        title=f" (Q6) Convergence Process of the Neural Network Loss and Gradient Norm",
        xaxis_title="Iteration",
        yaxis_title="Loss/Gradient Norm",
        title_x=0.5)

    plot_convergence1.show()

    # Plotting test true- vs predicted confusion matrix
    print(confusion_matrix(test_y, test_y_predict))

    # # ---------------------------------------------------------------------------------------------#
    # # Question 8: Network without hidden layers using SGD                                          #
    # # ---------------------------------------------------------------------------------------------#
    callback2, loss2, iterations2, grads2, weights2 = get_callback()
    nn2_layers = [FullyConnectedLayer(input_dim=train_X.shape[1], output_dim=n_classes)]
    nn2 = NeuralNetwork(modules=nn2_layers,
                        loss_fn=CrossEntropyLoss(),
                        solver=StochasticGradientDescent(learning_rate=FixedLR(0.1), max_iter=10000, batch_size=256,
                                                         callback=callback2))
    nn2.fit(train_X, train_y)
    test_y_predict = nn2.predict(test_X)
    print(f"(Q8) Neural Network w/ no hidden layers accuracy = {accuracy(test_y, test_y_predict)}")

    # Plotting convergence process
    plot_convergence2 = go.Figure()
    plot_convergence2.add_trace(go.Scatter(x=iterations2, y=loss2, mode="lines",
                                           marker=dict(color="blue"), name="Loss",
                                           showlegend=True))
    plot_convergence2.add_trace(go.Scatter(x=iterations2, y=grads2, mode="lines",
                                           marker=dict(color="red"), name="Gradient Norm",
                                           showlegend=True))

    plot_convergence2.update_layout(
        title=f" (Q8) Convergence Process of the Neural Network Loss and Gradient Norm",
        xaxis_title="Iteration",
        yaxis_title="Loss/Gradient Norm",
        title_x=0.5)

    plot_convergence2.show()

    # ---------------------------------------------------------------------------------------------#
    # Question 9: Most/Least confident predictions                                                 #
    # ---------------------------------------------------------------------------------------------#
    train_X_7, train_y_7 = train_X[train_y == 7], train_y[train_y == 7]
    test_X_7, test_y_7 = test_X[test_y == 7],  test_y[test_y == 7]

    nn1.fit(train_X_7, train_y_7)
    nn1.predict(test_X_7)

    # extract from the network our probability vector
    pred_probs = nn1.probability_vector
    pred_probs_7 = np.concatenate((pred_probs, test_X_7), axis=1)
    sorted_pred_prob_7 = pred_probs_7[pred_probs_7[:, 0].argsort()]

    least_confident_7 = sorted_pred_prob_7[:64, 10:]
    most_confident_7 = sorted_pred_prob_7[-64:, 10:]

    most_confident_7_images = plot_images_grid(images=np.array(most_confident_7), title="(Q9) most confident 7")
    least_confident_7_images = plot_images_grid(images=np.array(least_confident_7), title="(Q9) least confident 7")

    most_confident_7_images.show()
    least_confident_7_images.show()

    # ---------------------------------------------------------------------------------------------#
    # Question 10: GD vs GDS Running times                                                         #
    # ---------------------------------------------------------------------------------------------#
    train_X, train_y = train_X[:2500], train_y[:2500]

    callback_SGD, time_SGD, loss_SGD = get_time_callback()
    callback_GD, time_GD, loss_GD = get_time_callback()

    # Stochastic Gradient Descent Neural Network
    nn_layers = [FullyConnectedLayer(input_dim=train_X.shape[1], output_dim=64, activation=ReLU()),
                     FullyConnectedLayer(input_dim=64, output_dim=64, activation=ReLU()),
                     FullyConnectedLayer(input_dim=64, output_dim=n_classes)]

    nn_SGD = NeuralNetwork(modules=nn_layers,
                           loss_fn=CrossEntropyLoss(),
                           solver=StochasticGradientDescent(learning_rate=FixedLR(0.1), max_iter=10000, batch_size=256,
                                                            tol=10e-10, callback=callback_SGD))

    SGD_start_time = time.time()
    nn_SGD.fit(train_X, train_y)
    SGD_time_from_start = [round(t - SGD_start_time, 4) for t in time_SGD]

    SGD_loss_vs_time = go.Figure()
    SGD_loss_vs_time.add_trace(go.Scatter(x=SGD_time_from_start, y=loss_SGD, mode="lines", marker=dict(color="blue"),
                                          name="loss_SGD", showlegend=True))
    SGD_loss_vs_time.update_layout(
        title=f"(Q10) SGD Neural Network - Loss vs Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Loss",
        title_x=0.5)
    SGD_loss_vs_time.show()

    # Gradient Descent Neural Network
    nn_GD = NeuralNetwork(modules=nn_layers,
                          loss_fn=CrossEntropyLoss(),
                          solver=GradientDescent(learning_rate=FixedLR(0.1), max_iter=10000, tol=10e-10,
                                                 callback=callback_GD))


    GD_start_time = time.time()
    nn_GD.fit(train_X, train_y)

    GD_time_from_start = [round(t - GD_start_time, 4) for t in time_GD]

    GD_loss_vs_time = go.Figure()
    GD_loss_vs_time.add_trace(go.Scatter(x=GD_time_from_start, y=loss_GD, mode="lines", marker=dict(color="blue"), name="loss_GD",
                                         showlegend=True))
    GD_loss_vs_time.update_layout(
        title=f"(Q10) GD Neural Network - Loss vs Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Loss",
        title_x=0.5)
    GD_loss_vs_time.show()

    # added SGD plot to GD graph
    GD_loss_vs_time.add_trace(go.Scatter(x=SGD_time_from_start, y=loss_SGD, mode="lines", marker=dict(color="red"),
                                         name="loss_SGD", showlegend=True))
    GD_loss_vs_time.show()
