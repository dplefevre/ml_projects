import itertools
import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import sympy as sp

from paraboloid import parabola_2d
from plane import plane

parboloid = parabola_2d()
pln = plane()

point = [1, 2]
z1 = parboloid.compute_at_point(point)

grad = nd.Gradient(parboloid.compute_at_point)(point)
print(grad)

points = np.linspace(-5, 5, 90)
x, y = np.meshgrid(points, points)
X = list(itertools.product(x, y))
plane_func = np.vectorize(plane.compute_at_point)

Z = plane_func(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.plot_surface(x, y, Z, rstride=1, cstride=1, cmap='terrain')
ax.set(xlabel="x", ylabel="y", zlabel="f(x, y)", title="Cool function")

plt.show()
