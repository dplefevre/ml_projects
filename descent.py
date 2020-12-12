"""
descent.py

Author: Daniel Lefevre

12/11/2020

Implementation of 1-dimensional gradient descent for linear regression

"""

from collections import deque

import numpy as np
import matplotlib.pyplot as plt


def hypothesis(feature_matrix, theta_matrix):
    """Compute the linear hypothesis function"""
    return np.matmul(feature_matrix, theta_matrix)


def cost_function(m, feature_matrix, theta_matrix, labels):
    """Compute the error from the hypothesis and y values"""
    return (1 / (2 * m)) * np.sum(np.square(hypothesis(feature_matrix, theta_matrix) - labels))


def update_theta(alpha, m, theta_matrix, feature_matrix, labels):
    """Simultaneously update theta_0 and theta_1 according to the algorithm"""
    new_theta = np.zeros((2, 1))
    new_theta[0] = theta_matrix[0] - (alpha/m)*np.sum(hypothesis(feature_matrix, theta_matrix) - labels)
    new_theta[1] = theta_matrix[1] \
                  - (alpha/m)*np.matmul(np.array([feature_matrix[:, 1]]),
                                        (hypothesis(feature_matrix, theta_matrix) - labels))
    return new_theta


def gradient_descent(alpha, m, theta_matrix, feature_matrix, labels):
    """Compute and store the cost function for current theta guess,
       update theta, and repeat until the cost function no longer decreases.

       Return the last computed value of theta"""
    track_cost = deque(maxlen=2)
    J = cost_function(m, feature_matrix, theta_matrix, labels)
    track_cost.append(J)
    while True:
        theta_matrix = update_theta(alpha, m, theta_matrix, feature_matrix, labels)
        new_cost = cost_function(m, feature_matrix, theta_matrix, labels)
        if new_cost < track_cost[0]:
            track_cost.append(new_cost)
        else:
            return theta_matrix


def add_line(axis, feature_matrix, theta_matrix):
    """Plot a line given the slope and intercept (theta_matrix)"""
    intercept = theta_matrix[0]
    slope = theta_matrix[1]
    x_values = np.arange(min(feature_matrix[:, 1]), max(feature_matrix[:, 1]) + 1)
    y_values = intercept + slope * x_values
    axis.plot(x_values, y_values)


# Create some initial data and add noise to y values
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
noise = np.random.random(5) - 0.5
y = y + noise

# Plot the initial data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x, y)


# Gradient Descent
# learning rate
learning_rate = 0.01
# m = number of examples
num_examples = x.shape[0]
# Reshape X, y into column vectors
# Append a column of 1's in front of X for the theta_nought coefficients
features = np.transpose([x])
X = np.hstack((np.ones_like(features), features))
y = np.transpose([y])
# initialize theta to [[0], [0]]
theta = np.zeros((2, 1))

problem_data = {"m": num_examples,
                "alpha": learning_rate,
                "feature_matrix": X,
                "theta_matrix": theta,
                "labels": y}

# perform gradient descent to find optimal theta
optimized_theta = gradient_descent(**problem_data)

# plot the result along with the initial data
result_plot_data = {"axis": ax,
                    "feature_matrix": X,
                    "theta_matrix": optimized_theta}
add_line(**result_plot_data)

plt.show()
