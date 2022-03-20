from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import math
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, var, m = 10, 1, 1000
    data = np.random.normal(mu, var, m)
    estimator = UnivariateGaussian().fit(data)
    print(f"({estimator.mu_}, {estimator.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    x = np.arange(10, 1010, 10)  # x = [10, 20,..., 100, 110, ..., 1000]
    y = []
    for i in x:
        estimator.fit(data[:i])
        y.append(math.fabs(mu - estimator.mu_))

    go.Figure(go.Scatter(x=x, y=y, mode='markers+lines'),
              layout=go.Layout(title="absolute distance between the estimated and true value of the "
                                     r"expectation, as a function of the sample size",
                               xaxis_title="number of samples",
                               yaxis_title="r$|\mu - \hat\mu|$")).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    estimator.fit(data)
    pdf_data = estimator.pdf(data)

    go.Figure(go.Scatter(x=data, y=pdf_data, mode='markers'),
              layout=go.Layout(title="Empirical PDF of fitted model",
                               xaxis_title="samples",
                               yaxis_title="PDF values")).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
