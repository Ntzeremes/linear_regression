import numpy as np
import matplotlib.pyplot as plt
import random


def create_data(a, b, error=2, plot=False):
    """This function creates and returns two lists of two variables that have a linear relationship. y = a*x + b + error
    a is the slope of the line, b is the y intercept and error is the noise that we add to the measurments.
    The independent variable is created with the function linespace of numpy and has values from 0-10 with 0.1 increment
    """

    x = np.linspace(0, 10, 50)
    y = a * x + np.full(50, b) + np.random.uniform(-1, 1, 50) * error

    if plot:
        plt.scatter(x, y)
        plt.show()

    return x, y

print(create_data(2,2))

