import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import pickle
from scipy.optimize import curve_fit
from scipy.optimize import minimize


def add_noise(x_, y_, mu_=0, sigma_=0.001):
    x_internal = x_.copy()
    y_internal = y_.copy()

    mean = (mu_, mu_)
    cov = [[sigma_, 0], [0, sigma_]]

    L_ = np.random.multivariate_normal(mean, cov, len(x_internal))

    x_internal += L_[:, 0]
    y_internal += L_[:, 1]

    return x_internal, y_internal


def M3C2():
    pass


def merge_and_fit():
    pass


def func(x_, a_, b_):
    return a_*x_ + b_


def tree_build(x_, y_):
    data_sets_ = np.vstack((x_, y_)).T
    # print(data_sets_.shape)
    tree_ = KDTree(data_sets_, leaf_size=10)
    tree_save_ = pickle.dumps(tree_)
    return data_sets_, tree_save_


def neighbor_search_(data_a, data_b, tree_b, threshold_value=2, i_=1):
    tree_copy_ = pickle.loads(tree_b)
    dist_, ind_ = tree_copy_.query(data_a[i_].reshape(1, -1), k=1)
    dist_enlonged = dist_[0][0]*threshold_value

    ind_updated_ = tree_copy_.query_radius(data_a[i_].reshape(1, -1), r=dist_enlonged)[0]
    neighbor_set_ = data_b[ind_updated_.astype(np.int32).tolist()]

    # print("The total number of included neighbor is: {}".format(str(len(neighbor_set_))))

    return neighbor_set_


def dis_to_surface_(neighbor_set_, target_set_, para_set_, i_=1):
    if len(neighbor_set_) <= 2:
        # print("Too less")
        pass
    else:
        neighbor_x_min, neighbor_x_max = np.min(neighbor_set_[:, 0]), np.max(neighbor_set_[:, 0])
        neighbor_y_min, neighbor_y_max = np.min(neighbor_set_[:, 1]), np.max(neighbor_set_[:, 1])

        dis_fun_ = lambda cor_xy_: np.sqrt((cor_xy_[0] - target_set_[i_][0]) ** 2 + (cor_xy_[1] - target_set_[i_][1]) ** 2)

        cons_ = ({'type': 'eq', 'fun': lambda cor_xy_: func(cor_xy_[0], *para_set_) - cor_xy_[1]},
                {'type': 'ineq', 'fun': lambda cor_xy_: cor_xy_[0] - neighbor_x_min},
                {'type': 'ineq', 'fun': lambda cor_xy_: neighbor_x_max - cor_xy_[0]},
                {'type': 'ineq', 'fun': lambda cor_xy_: cor_xy_[1] - neighbor_y_min},
                {'type': 'ineq', 'fun': lambda cor_xy_: neighbor_y_max - cor_xy_[1]}
                )
        cor_innitial_ = np.array((np.average(neighbor_set_[:, 0]), np.average(neighbor_set_[:, 1])))
        dist_calculted = minimize(dis_fun_, cor_innitial_, method='SLSQP', constraints=cons_)
        # print('最小值：', dist_calculted.fun)
        # print('最优解：', dist_calculted.x)
        # print('迭代终止是否成功：', dist_calculted.success)
        # print('迭代终止原因：', dist_calculted.message)

        return dist_calculted.fun


x = np.linspace(0, 2*np.pi, 5000)
y = np.sin(x)
data1, tree_1 = tree_build(x, y)

x2, y2 = add_noise(x, y, 0, 0.0001)
data2, tree_2 = tree_build(x2, y2)

x_c = np.linspace(0, 2*np.pi, 500)
y_c = np.sin(x_c)

x3, y3 = add_noise(x_c, y_c, 0, 0.05)
data3, tree_3 = tree_build(x3, y3)

# index_num = 300

dis_var_set = []

for i_x_i in range(len(data3)):
    xxx = neighbor_search_(data3, data2, tree_2, i_=i_x_i)
    if len(xxx) <= 2:
        # print("The total number of neighbors is less than 2")
        pass
    else:
        # ax = plt.gca()
        # ax.set_aspect(1)
        # plt.scatter(xxx[:, 0], xxx[:, -1], c="b")
        # plt.scatter(data3[i_x_i][0], data3[i_x_i][-1], c='r')

        popt_2, pcov_2 = curve_fit(func, xxx[:, 0], xxx[:, -1])
        # print(popt_2)
        #
        # error_2 = func(xxx[:, 0], *popt_2)-xxx[:, -1]
        # print(np.average(error_2), np.std(error_2), "error")
        # plt.plot(xxx[:, 0], func(xxx[:, 0], *popt_2), c="black")
        # plt.show()

        dis = dis_to_surface_(xxx, data3, popt_2, i_=i_x_i)
        # print(dis, type(dis))
        dis_var_set.append(dis)

dis_var_set = np.array(dis_var_set)
print(np.sqrt(np.sum(dis_var_set**2)/len(dis_var_set))/np.sqrt(2), "predict noise")
print(np.std(dis_var_set), np.average(dis_var_set), len(dis_var_set))
# print(dis_var_set)

popt_2, pcov_2 = curve_fit(func, x2, y2)
# print(popt_2)
# error_2 = func(x2, *popt_2)-y2
# perr_2 = np.sqrt(np.diag(pcov_2))
# print(np.average(error_2), np.std(error_2), perr_2)
# print(pcov_2)
popt_8, pcov_8 = curve_fit(func, x3, y3)
# print(dis_var_set)
print(np.average(dis_var_set**2), "参数")
# print(popt_3)
# error_3 = func(x3, *popt_2)-y3
# perr_3 = np.sqrt(np.diag(pcov_3))
# print(np.average(error_3), np.std(error_3), perr_3)
# print(pcov_3)
#
#
# plt.scatter(x, y, s=2, c='r')
# plt.show()
plt.scatter(x2, y2, s=2, c='r')
plt.plot(x, func(x, *popt_2), c="black")

plt.show()
plt.scatter(x3, y3, s=2, c='b')
plt.plot(x, func(x, *popt_8), c="black")
plt.show()

