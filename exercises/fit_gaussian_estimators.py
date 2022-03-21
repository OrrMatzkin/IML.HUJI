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
    m = 1000
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])

    data = np.random.multivariate_normal(mu, cov, m)
    mult_estimator = MultivariateGaussian().fit(data)

    print(mult_estimator.mu_)
    print(mult_estimator.cov_)

    # Question 5 - Likelihood evaluation
    f = np.linspace(-10, 10, 200)
    log_likelihood_data = np.empty([f.size, f.size])

    for i in range(f.size):
        for j in range(f.size):
            mu = np.array([f[i], 0, f[j], 0])
            log_likelihood_data[i][j] = mult_estimator.log_likelihood(mu, cov, data)

    go.Figure(go.Heatmap(x=f, y=f, z=log_likelihood_data),
              layout=go.Layout(
                  title="Log Likelihood Heatmap",
                  xaxis_title="$f3\\ values$",
                  yaxis_title="$f1\\ values$")).show()

    # Question 6 - Maximum likelihood
    f1_index, f3_index = np.where(log_likelihood_data == np.amax(log_likelihood_data))
    f1, f3 = f[f1_index].round(3),  f[f3_index].round(3)
    # The Exercise 1 description didn't ask to print... to print uncomment next line
    # print(f1, f3)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
