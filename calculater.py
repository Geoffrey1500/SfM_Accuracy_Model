import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import pickle
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

mean = (0, 0, 0)
cov = [[9, 0, 0], [0, 9, 0], [0, 0, 9]]

x = np.random.multivariate_normal(mean, cov, 10000)
print(x)
print(x.shape)

# a = np.random.random_sample((5000, 2)) - 0.5
# print(a)
#
b = np.sum(x**2, axis=0)
print(b)
print(np.sum(x**2, axis=1))

c = np.sqrt(np.sqrt(np.average(np.sum(x**2, axis=1)))**2/3)
d = np.average(np.sum(x**2, axis=1))/3
print(c, d)
print(np.std(x[:, 0]), np.std(x[:, 1]), np.std(x[:, 2]))

plt.scatter(x[:, 0], x[:, 1], s=2, c='r')

plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:, 0], x[:, 1], x[:, 2])
plt.show()
