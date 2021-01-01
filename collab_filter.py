import numpy as np


def hypothesis(X_matrix, Theta_matrix):
    return X_matrix.transpose() @ Theta_matrix


def cost_function(X_matrix, Theta_matrix, labels, r_matrix):
    return (1/2) * np.sum((r_matrix *
                           (hypothesis(X_matrix, Theta_matrix) - labels)) ** 2)


def fit_model(num_movies, num_features, num_users, labels, r_matrix):
    # Initialize X
    x = np.random.rand(num_features, num_movies)
    # Initialize theta
    theta_matrix = np.random.rand(num_features, num_users)
    alpha = 0.01
    iterations = 500
    for i in range(iterations):
        grad_theta = x @ (r_matrix*((x.transpose() @ theta_matrix) - labels))
        grad_x = theta_matrix @ \
                 (r_matrix*((x.transpose() @ theta_matrix) -
                            labels)).transpose()
        theta_matrix = theta_matrix - alpha * grad_theta
        x = x - alpha * grad_x
    return x, theta_matrix


movies = ["Batman Begins", "Mission Impossible",
          "Titanic", "Love Actually", "The Matrix"]
viewers = ["Bob", "Alice", "Sally", "Bill"]
# -1 => movie not yet rated
y = np.array([[5, 5, 1, 0, 4],
              [4, 5, 0, 2, 5],
              [0, 1, 4, 5, 1],
              [-1, 5, 1, -1, 4]]).transpose()
# Try to predict Bill's rating for Batman Begins and Love Actually

# Movie feature vectors:
genres = ["action", "romance", "comedy"]
# R = 1 if user j has rated movie i; 0 otherwise
# User "Bill" has not rated "Batman Begins" or "Love Actually"
R = np.array([[1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [0, 1, 1, 0, 1]]).transpose()
# Gradient descent
problem_data = {"num_movies": len(movies),
                "num_features": len(genres),
                "num_users": len(viewers),
                "labels": y,
                "r_matrix": R}
X, theta = fit_model(**problem_data)
np.set_printoptions(precision=3, suppress=True)
predicted_labels = hypothesis(X, theta)
rounded_labels = np.around(predicted_labels, decimals=0)
print(rounded_labels)
print(cost_function(X, theta, labels=y, r_matrix=R))
# Starting from X and Theta initialized to small random values, we see that the
# initial label set is recovered, with reasonable predictions for Bill's
# ratings on the movies he has not yet seen
