import numpy as np
import matplotlib.pyplot as plt


def create_data(a, b, error=2):
    """This function creates and returns two lists of two variables that have a linear relationship. y = a*x + b + error
    a is the slope of the line, b is the y intercept and error is the noise that we add to the measurments.
    The independent variable is created with the function linespace of numpy and has values from 0-10 with 0.1 increment
    """

    x = np.linspace(0, 10, 50)
    y = a * x + b + np.random.uniform(-1, 1, 50) * error

    return x, y


def linear_reg(x, y):
    """This linear regression model uses the least squared loss error function. The error function has two unknown
    variables, the slope and the y intercept we want to estimate. We want this variables to take such values so that the
    error function will be as low as possible. Using calculus, since we have only two variables we can solve the problem
    analytically, by taking the derivative of the cost function with respect to each variable and se it equal to zero.
    Then we conclude with a system of two linear equations with two parameters.
    The code below uses values of the analytical solution.
    """
    denominator = x.dot(x) - x.mean()*x.sum()

    a_ols = (x.dot(y) - y.mean()*x.sum())/denominator
    b_ols = (y.mean()*x.dot(x) - x.mean()*x.dot(y))/denominator
    y_est = a_ols * x + b_ols

    return y_est


def r_squared(y, yest):
    """R squared is a metric of good fit which measures the percentage of the variability that our model captures.
    It ranges from 0 (captures no variance) to 1 (captures all possible variance). The closer to 1 we get the better is
    our model is but we should be careful for overfitting"""

    y_mean = y.mean()
    r = 1 - (y - yest).dot(y - yest)/(y - y_mean).dot(y - y_mean)

    return r


def create_model(x, y, plot=True):
    y_est = linear_reg(x, y)
    r = r_squared(y, y_est).round(4)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.style.use("ggplot")
        plt.title("Linear Regression model")
        plt.scatter(x, y, color="blue")
        plt.plot(x, y_est, label=f"R^2 : {r}")
        plt.legend(loc="best")
        plt.show()


if __name__ == "__main__":
    x, y = create_data(2, 5)
    create_model(x, y)