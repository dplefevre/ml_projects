"""
grad_desc.py

Daniel Lefevre, 12/07/2020

A rough attempt at implementing gradient descent in Python

The script currently has a hardcoded function and domain. It selects a random point from the domain,
and evaluates the function at that point. It then selects a neighboring point, and evaluates the
function there. This process continues for a set number of iterations, with the goal of finding
a local minimum from the starting point.

At the end, it produces a plot of the function as well as the starting and final points.
"""

# TODO: make function, domain, alpha into user arguments
# TODO: handle evaluation of a point outside the domain
# TODO: generalize to higher dimensions
# TODO: break loop when progress is not being made
# TODO: could also try decreasing alpha
# TODO: make plot more detailed, plot path as series of vectors


import numpy as np
import matplotlib.pyplot as plt


def get_starting_point(x_domain, y_domain):
    return np.array([np.random.choice(x_domain),
                     np.random.choice(y_domain)])


def get_new_point(x0, y0, alph):
    new_points = np.array([(x0 + alph, y0),
                          (x0 - alph, y0),
                          (x0, y0 + alph),
                          (x0, y0 - alph),
                          (x0 + alph, y0 + alph),
                          (x0 - alph, y0 - alph),
                          (x0 + alph, y0 - alph),
                          (x0 - alph, y0 + alph)])
    choice = np.random.randint(0, len(new_points))
    return new_points[choice]


def func(a, b):
    return (a+1) ** 2 + (b+1) ** 2


# Plot the 3d function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(-6, 4)
y = np.linspace(-6, 4)
X, Y = np.meshgrid(x, y)

func3d = np.vectorize(func)
Z = func3d(X, Y)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='terrain')
ax.set(xlabel="x", ylabel="y", zlabel="f(x, y)", title="Cool function")


# Gradient descent
alpha = 0.005
sp = get_starting_point(x, y)
original_point = sp.copy()
starting_elevation = func(sp[0], sp[1])
original_elevation = starting_elevation.copy()
new_elevation = starting_elevation.copy()
new_point = get_new_point(sp[0], sp[1], alpha)
print(new_point)

for i in range(10000):
    new_point = get_new_point(sp[0], sp[1], alpha)
    new_elevation = func(new_point[0], new_point[1])
    if new_elevation < starting_elevation:
        sp = new_point
        starting_elevation = new_elevation
print(f"The function is found to be minimized at point {sp}")
print(f"The value at the computed minimum at that point is {starting_elevation}")

ax.scatter(original_point[0], original_point[1], original_elevation, color='black')
ax.scatter(sp[0], sp[1], starting_elevation, color='black')
#
plt.show()
