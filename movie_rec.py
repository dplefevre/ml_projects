import numpy as np


def hypothesis(X, theta):
    return X.transpose() @ theta


def cost_function(X, theta, labels, r_matrix):
    cost = (1/2) * np.sum((r_matrix * (hypothesis(X, theta) - labels))**2)
    return cost


def fit_model(X, labels, r_matrix):
    theta = np.zeros((3, 4)) # Initial theta
    alpha = 0.01
    iterations = 50
    pass
    for i in range(iterations):
        error = X @ (R*((X.transpose() @ theta) - y)) # (5x3) @ (5x4)
        theta = theta - alpha * error
    return theta

movies = ["Batman Begins", "Mission Impossible", "Titanic", "Love Actually", "The Matrix"]
viewers = ["Bob", "Alice", "Sally", "Bill"]
# y = ratings
# -1 => movie not rated
y = np.array([[5, 5, 1, 0, 4], [4, 5, 0, 2, 5], [0, 1, 4, 5, 1], [-1, 5, 1, -1, 4]]).transpose() # 5x4
# Try to predict Bill's rating for Batman Begins and Love Actually

# Movie feature vectors:
# X = [<action>, <romance>, <comedy>]
X = np.array([[5, 0, 0], [5, 0, 0], [1, 5, 0], [0, 5, 1], [5, 1, 0]]).transpose() # 3x5
# R = 1 if user j has rated movie i; 0 otherwise
R = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 0, 1]]).transpose() # Dim(y)
# Gradient descent
theta = fit_model(X, y, R)
print(hypothesis(X, theta))
# We predict Bill would rate "Batman Begins" with 4.5, and "Love Actually" with 0