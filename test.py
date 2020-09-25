import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import pickle
from scipy.optimize import curve_fit
from scipy.optimize import minimize


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


def neighbor_search_(data_, data_x, tree_, threshold_value=2, i_=1):
    tree_copy_ = pickle.loads(tree_)
    dist_, ind_ = tree_copy_.query(data_[i_].reshape(1, -1), k=1)
    dist_enlonged = dist_[0][0]*threshold_value
    print(dist_enlonged)
    ind_updated_ = tree_copy_.query_radius(data_[i_].reshape(1, -1), r=dist_enlonged)[0]
    print(ind_updated_)
    print(ind_updated_.astype(np.int32))
    print(type(ind_updated_.astype(np.int32)))
    print(ind_updated_.astype(np.int32).tolist())
    neighbor_count = tree_copy_.query_radius(data_[i_].reshape(1, -1), r=dist_enlonged, count_only=True)

    neighbor_set_ = data_x[ind_updated_.astype(np.int32).tolist()]
    print(type(neighbor_count))
    print(neighbor_count)

    print("The total number of included neighbor is: {}".format(str(neighbor_count[0])))
    return neighbor_set_


def dis_to_surface_():
    pass


x = np.linspace(0, 2*np.pi, 500)
y = np.sin(x)
data1, tree_1 = tree_build(x, y)

x2, y2 = add_noise(x, y, 0, 0.005)
data2, tree_2 = tree_build(x2, y2)
x3, y3 = add_noise(x, y, 0, 0.05)
data3, tree_3 = tree_build(x3, y3)

index_num = 300

dist, ind = pickle.loads(tree_2).query(data3[index_num].reshape(1, -1), k=1)
print(ind)
print(dist)


xxx = neighbor_search_(data3, data2, tree_2, i_=index_num)
print(len(xxx))
print(xxx)
print(data3[index_num])

ax = plt.gca()
ax.set_aspect(1)

plt.scatter(xxx[:, 0], xxx[:, -1], c="b")
plt.scatter(data3[index_num][0], data3[index_num][-1], c='r')

popt_2, pcov_2 = curve_fit(func, xxx[:, 0], xxx[:, -1])
print(popt_2, type(popt_2))
print(popt_2)
error_2 = func(xxx[:, 0], *popt_2)-xxx[:, -1]
print(np.average(error_2), np.std(error_2), "xxxxxxx")
plt.plot(xxx[:, 0], func(xxx[:, 0], *popt_2), c="black")
plt.axis('equal')
plt.show()

xxx_min, xxx_max = np.min(xxx[:, 0]), np.max(xxx[:, 0])
yyy_min, yyy_max = np.min(xxx[:, 1]), np.max(xxx[:, 1])

print(xxx_min, xxx_max, yyy_min, yyy_max)

fun_X = lambda x_x: np.sqrt((x_x[0] - data3[index_num][0])**2 + (x_x[1] - data3[index_num][1])**2)

cons = ({'type': 'eq', 'fun': lambda x_x: -x_x[1] + popt_2[0]*np.sin(x_x[0]) + popt_2[1]}, # xyz=1
        {'type': 'ineq', 'fun': lambda x_x: x_x[0] - xxx_min},
        {'type': 'ineq', 'fun': lambda x_x: xxx_max - x_x[0]},
        {'type': 'ineq', 'fun': lambda x_x: x_x[1] - yyy_min},
        {'type': 'ineq', 'fun': lambda x_x: yyy_max - x_x[1]}
       )
x0 = np.array((np.average(xxx[:, 0]), np.average(xxx[:, 1])))
res = minimize(fun_X, x0, method='SLSQP', constraints=cons)
print('最小值：', res.fun)
print('最优解：',res.x)
print('迭代终止是否成功：', res.success)
print('迭代终止原因：', res.message)
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

