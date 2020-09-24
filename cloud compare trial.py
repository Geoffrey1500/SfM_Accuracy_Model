import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import pickle
from scipy.optimize import curve_fit


def add_noise(x_, y_, mu_=0, sigma_=0.001):
    x_internal = x_.copy()
    y_internal = y_.copy()
    for i_ in range(x_internal.size):
        x_internal[i_] += np.random.normal(mu_, sigma_)
        y_internal[i_] += np.random.normal(mu_, sigma_)

    return x_internal, y_internal


def M3C2():
    pass


def merge_and_fit():
    pass


def func(x_, a_, b_):
    return a_*np.sin(x_) + b_


def tree_build(x_, y_):
    data_sets_ = np.vstack((x_, y_)).T
    # print(data_sets_.shape)
    tree_ = KDTree(data_sets_, leaf_size=10)
    tree_save_ = pickle.dumps(tree_)
    return data_sets_, tree_save_


def neighbor_search_(data_, tree_, threshold_value):
    pass


x = np.linspace(0, 2*np.pi, 500)
y = np.sin(x)
data1, tree_1 = tree_build(x, y)

x2, y2 = add_noise(x, y, 0, 0.005)
data2, tree_2 = tree_build(x2, y2)
x3, y3 = add_noise(x, y, 0, 0.05)
data3, tree_3 = tree_build(x3, y3)

dist, ind = pickle.loads(tree_2).query(data3[:1], k=1)
print(ind)
print(dist)

#
# popt_2, pcov_2 = curve_fit(func, x2, y2)
# print(popt_2)
# error_2 = func(x2, *popt_2)-y2
# perr_2 = np.sqrt(np.diag(pcov_2))
# print(np.average(error_2), np.std(error_2), perr_2)
# print(pcov_2)
# popt_3, pcov_3 = curve_fit(func, x3, y3)
# print(popt_3)
# error_3 = func(x3, *popt_2)-y3
# perr_3 = np.sqrt(np.diag(pcov_3))
# print(np.average(error_3), np.std(error_3), perr_3)
# print(pcov_3)
#
#
# plt.scatter(x, y, s=2, c='r')
# plt.show()
# plt.scatter(x2, y2, s=2, c='r')
# plt.plot(x2, func(x2, *popt_2), c="black")
#
# plt.show()
# plt.scatter(x3, y3, s=2, c='b')
# plt.plot(x3, func(x3, *popt_3), c="black")
# plt.show()

