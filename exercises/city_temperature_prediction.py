import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"

All_Months = ["January",
              "February",
              "March",
              "April",
              "May",
              "June",
              "July",
              "August",
              "September",
              "October",
              "November",
              "December"]

All_Countries = ['South Africa', 'The Netherlands', 'Jordan']


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # removes NaN values and duplicates
    temp_city_data = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    # removes temperatures with invalid data of sample (after april the 1'st)
    temp_city_data = temp_city_data[temp_city_data['Date'] < '2022-04-01']
    # removes temperatures lower than -25 degrees celsius - google says it's never happened in those areas
    temp_city_data = temp_city_data[temp_city_data['Temp'] > -25]
    # adds Day of Year feature to the data
    temp_city_data['DayOfYear'] = temp_city_data['Date'].apply(lambda x: x.day_of_year)
    return temp_city_data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    data_Israel = X[X['Country'] == "Israel"]

    # GRAPH - Average daily temperature in Israel as function of the day in the year
    px.scatter(data_Israel, x='DayOfYear', y="Temp", color=data_Israel['Year'].astype(str),
               title="Average daily temperature in Israel as function of the day in the year",
               labels={
                   "DayOfYear": "Day of Year",
                   "Temp": "Average Temperature (Celsius)",
                   "color": "Year"}).show()

    data_Israel_std = data_Israel.groupby('Month').agg(np.std)
    data_Israel_std["TempStandardDeviation"] = data_Israel_std['Temp']

    # GRAPH - Standard Deviation of the daily Temperatures per Month
    px.bar(data_Israel_std, x=All_Months, y="TempStandardDeviation", text_auto=True,
           title="Standard Deviation of the daily Temperatures per Month",
           labels={"TempStandardDeviation": "Temperature Standard Deviation (Celsius)"}).show()

    # Question 3 - Exploring differences between countries
    data_std = X.groupby(["Country", "Month"]).agg(np.std)
    data_avg = X.groupby(["Country", "Month"]).agg(np.average) # currently, works, in future versions of panda might not

    # GRAPH - Average monthly temperature per Country with error bars
    go.Figure(data=[go.Scatter(x=All_Months,
                               y=data_avg.loc[x]['Temp'],
                               error_y=go.scatter.
                               ErrorY(array=data_std.loc[x]['Temp']),
                               name=x) for x in X['Country'].unique()],
              layout=go.Layout(title="Average monthly temperature per Country with error bars",
                               xaxis_title=dict(text="Month"),
                               yaxis_title=dict(text="Average Temperature (Celsius)"),
                               legend_title=dict(text='Country'))).show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(data_Israel.drop(columns=["Temp"]),
                                                        data_Israel['Temp'],
                                                        train_proportion=0.75)

    all_losses = list()
    for k in range(1, 11):
        es = PolynomialFitting(k)
        es.fit(train_X['DayOfYear'].to_numpy(), train_y.to_numpy())
        loss = es.loss(test_X['DayOfYear'].to_numpy(), test_y.to_numpy()).__round__(2)
        print(f"for k = {k} we get a MSE loss value of {loss}")
        all_losses.append(loss)

    # GRAPH - MSE loss for each Polynom degree between 1 and 10
    px.bar(x=range(1, 11), y=all_losses, text_auto=True,
           title="MSE loss for each Polynom degree between 1 and 10",
           labels={"x": "k - Polynom degree", "y": "MSE loss"}). \
        update_xaxes(type="category").show()

    # Question 5 - Evaluating fitted model on different countries
    es = PolynomialFitting(k=5)
    es.fit(data_Israel['DayOfYear'].to_numpy(), data_Israel['Temp'].to_numpy())

    countries_losses = list()

    for country in All_Countries:
        country_data = X[X['Country'] == country]
        countries_losses.append(es.loss(country_data['DayOfYear'], country_data['Temp']))

    # GRAPH - MSE loss of a Polynomial fitting Estimator of degree 5 for each Country
    px.bar(x=All_Countries, y=countries_losses, text_auto=True, color=All_Countries,
           title="MSE loss of a Polynomial fitting Estimator of degree 5 for each Country ",
           labels={"x": "Country", "y": "MSE loss", "color": "Countries"}).update_xaxes(type="category").show()

    # Extra stuff !!!
    # In this section I proof my claim in question 3, that by adding a constant  of -10 to the Israel training
    # response vector we can get a better Estimator for The Netherlands.

    countries_losses.clear()
    Israel_Netherlands_diff_constant = 10

    for country in All_Countries:
        country_data = X[X['Country'] == country]
        if country == "The Netherlands":
            es.fit(data_Israel['DayOfYear'].to_numpy(),
                   (data_Israel['Temp'] - Israel_Netherlands_diff_constant).to_numpy())
        else:
            es.fit(data_Israel['DayOfYear'].to_numpy(), data_Israel['Temp'].to_numpy())
        countries_losses.append(es.loss(country_data['DayOfYear'], country_data['Temp']))

    # GRAPH - EXTRA - MSE loss of The Netherlands (with a constant) and other Countries
    px.bar(x=All_Countries, y=countries_losses, text_auto=True, color=All_Countries,
           title=" EXTRA - MSE loss of The Netherlands (with a constant) and other Countries",
           labels={"x": "Country", "y": "MSE loss", "color": "Countries"}).update_xaxes(type="category").show()
