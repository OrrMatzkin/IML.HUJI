import IMLearn.learners.regressors.linear_regression
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import os

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    house_data = pd.read_csv(filename)

    house_data = house_data.sort_values(by=['price'])
    house_data = house_data.dropna()  # removes missing values
    house_data = house_data[house_data['price'] > 0]  # removes negative or zero prices
    prices = house_data['price']

    # removes unnecessary columns
    house_data = house_data.drop(columns=['price', 'id', 'date', 'lat', 'long', 'sqft_living15', 'sqft_lot15'])

    # added a new feature bedrooms + bathrooms number
    house_data = pd.concat([house_data, (house_data['bedrooms'] + house_data['bathrooms']).
                           rename("bedrooms & bathrooms")], axis=1)

    # added a new feature total sqft of house
    house_data = pd.concat([house_data, (house_data['sqft_above'] + house_data['sqft_basement'] +
                                         house_data['sqft_living'] + house_data['sqft_lot']).
                           rename("sqft_garden")], axis=1)

    # added a new feature total year built + renovated
    house_data = pd.concat([house_data, (house_data['yr_built'] + house_data['yr_renovated']).
                           rename("year built + renovated")], axis=1)

    # added the dummies features for all 200 zipcodes
    house_data = pd.concat([house_data, pd.get_dummies(house_data['zipcode'])], axis=1)

    return house_data, prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # creates the directory if not already exist
    if output_path != "." and not os.path.exists(output_path):
        os.makedirs(output_path)

    for feature in X:
        # if the current feature is one of the dummies features of the zipcodes don't evaluate it
        if str(feature).startswith('98'):
            continue
        cov = np.cov(X[feature], y)
        dev_X, dev_y = np.std(X[feature]), np.std(y)
        p_corr = (cov / (dev_X * dev_y))[0, 1]  # we take the (0,1) index of the correlation matrix [[XX, XY], [YX,YY]]
        fig_title = f"price vs {feature} - Pearson Correlation: {p_corr}"
        fig = go.Figure(go.Scatter(x=X[feature], y=y, mode='markers'),
                        layout=go.Layout(title=fig_title, xaxis_title=feature, yaxis_title=dict(text="price")))
        fig.write_image(f"{output_path}/{fig_title}.png")
        print(f"created {fig_title}")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "figures")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, train_proportion=0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    es = LinearRegression()
    percentage_loss_data = pd.DataFrame()
    p_percentage = []
    print("\nStarted training the LinerRegression Estimator...\n(takes approx 1 minute)")
    for p in range(10, 101):
        loss = []
        for _ in range(10):
            sample_X = train_X.sample(frac=p/100)
            sample_y = train_y.reindex_like(sample_X)
            es.fit(sample_X.to_numpy(), sample_y.to_numpy())
            loss.append(es.loss(test_X.to_numpy(), test_y.to_numpy()))
        p_loss = pd.DataFrame({p: loss})
        percentage_loss_data = pd.concat([percentage_loss_data, p_loss], axis=1)
        p_percentage.append(p)

    go.Figure([go.Scatter(x=p_percentage, y=percentage_loss_data.mean(), mode="markers+lines", name="Mean Loss",
                          marker=dict(color="green", opacity=.7)),
               go.Scatter(x=p_percentage, y=percentage_loss_data.mean() - 2 * percentage_loss_data.std(),
                          fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
               go.Scatter(x=p_percentage, y=percentage_loss_data.mean() + 2 * percentage_loss_data.std(),
                          fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False)],
              layout=go.Layout(title="Mean loss as a function of percentage of train set used",
                               xaxis_title=dict(text="p% of the train set used"),
                               yaxis_title=dict(text="MSE loss"))).show()
