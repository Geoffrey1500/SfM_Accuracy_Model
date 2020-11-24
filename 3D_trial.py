import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import pickle
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

X1 = np.arange(-5, 5, 0.5)
Y1 = np.arange(-5, 5, 0.5)
X, Y = np.meshgrid(X1, Y1)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

X_reshaped = np.reshape(X, (-1, 1))
Y_reshaped = np.reshape(Y, (-1, 1))
Z_reshaped = np.reshape(Z, (-1, 1))

data_original_1 = np.hstack((X_reshaped, Y_reshaped, Z_reshaped))
print(data_original_1.shape, "1_size")

X2 = np.arange(-5, 5, 0.05)
Y2 = np.arange(-5, 5, 0.05)
X_2, Y_2 = np.meshgrid(X2, Y2)
R_2 = np.sqrt(X_2**2 + Y_2**2)
Z_2 = np.sin(R_2)

X_reshaped_2 = np.reshape(X_2, (-1, 1))
Y_reshaped_2 = np.reshape(Y_2, (-1, 1))
Z_reshaped_2 = np.reshape(Z_2, (-1, 1))

data_original_2 = np.hstack((X_reshaped_2, Y_reshaped_2, Z_reshaped_2))
print(data_original_2.shape, "2 size")


def add_noise(data_ori, mu_=0, sigma_=0.001):
    mean = (mu_, mu_, mu_)
    cov = [[sigma_, 0, 0], [0, sigma_, 0], [0, 0, sigma_]]

    L_ = np.random.multivariate_normal(mean, cov, data_ori.shape[0])
    data_modified_ = data_ori + L_

    return data_modified_


def func(data_, a_, b_, c_):
    return a_*data_[0]**2 + b_*data_[1]**2 + c_


def tree_build(data_):
    tree_ = KDTree(data_, leaf_size=10)
    tree_save_ = pickle.dumps(tree_)
    return tree_save_


def neighbor_search_(data_a, data_b, threshold_value=2, i_=1, leaf_=10):
    tree_b = KDTree(data_b, leaf_size=leaf_)
    dist_, ind_ = tree_b.query(data_a[i_].reshape(1, -1), k=1)
    dist_enlonged = dist_[0][0]*threshold_value

    ind_updated_ = tree_b.query_radius(data_a[i_].reshape(1, -1), r=dist_enlonged)[0]
    neighbor_set_ = data_b[ind_updated_.astype(np.int32).tolist()]

    return neighbor_set_


def dis_to_surface_(neighbor_set_, target_set_, para_set_, i_=1):
    if len(neighbor_set_) <= 2:
        # print("Too less")
        pass
    else:
        neighbor_x_min, neighbor_x_max = np.min(neighbor_set_[:, 0]), np.max(neighbor_set_[:, 0])
        neighbor_y_min, neighbor_y_max = np.min(neighbor_set_[:, 1]), np.max(neighbor_set_[:, 1])
        neighbor_z_min, neighbor_z_max = np.min(neighbor_set_[:, 2]), np.max(neighbor_set_[:, 2])

        dis_fun_ = lambda cor_xy_: np.sqrt((cor_xy_[0] - target_set_[i_][0]) ** 2 + (cor_xy_[1] - target_set_[i_][1]) ** 2
                                           + (cor_xy_[2] - target_set_[i_][2]) ** 2)

        cons_ = ({'type': 'eq', 'fun': lambda cor_xy_: func([cor_xy_[0], cor_xy_[1]], *para_set_) - cor_xy_[2]},
                 {'type': 'ineq', 'fun': lambda cor_xy_: cor_xy_[0] - neighbor_x_min},
                 {'type': 'ineq', 'fun': lambda cor_xy_: neighbor_x_max - cor_xy_[0]},
                 {'type': 'ineq', 'fun': lambda cor_xy_: cor_xy_[1] - neighbor_y_min},
                 {'type': 'ineq', 'fun': lambda cor_xy_: neighbor_y_max - cor_xy_[1]},
                 {'type': 'ineq', 'fun': lambda cor_xy_: cor_xy_[2] - neighbor_z_min},
                 {'type': 'ineq', 'fun': lambda cor_xy_: neighbor_z_max - cor_xy_[2]}
                 )

        cor_innitial_ = np.array((np.average(neighbor_set_[:, 0]), np.average(neighbor_set_[:, 1]),
                                  np.average(neighbor_set_[:, 2])))
        dist_calculted = minimize(dis_fun_, cor_innitial_, method='SLSQP', constraints=cons_)
        # print('最小值：', dist_calculted.fun)
        # print('最优解：', dist_calculted.x)
        # print('迭代终止是否成功：', dist_calculted.success)
        # print('迭代终止原因：', dist_calculted.message)

        return dist_calculted.fun


data_2 = add_noise(data_original_1, 0, 0.04)
data_3 = add_noise(data_original_2, 0, 0.00001)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data_2[:, 0], data_2[:, 1], data_2[:, 2], c="blue")

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data_3[:, 0], data_3[:, 1], data_3[:, 2], c="blue")

plt.show()

dis_var_set = []

for i_x_i in range(len(data_2)):
    xxx = neighbor_search_(data_2, data_3, i_=i_x_i)
    if len(xxx) <= 2:
        # print("The total number of neighbors is less than 2")
        pass
    else:
        # ax = plt.gca()
        # ax.set_aspect(1)
        # plt.scatter(xxx[:, 0], xxx[:, -1], c="b")
        # plt.scatter(data3[i_x_i][0], data3[i_x_i][-1], c='r')

        popt_2, pcov_2 = curve_fit(func, [xxx[:, 0], xxx[:, 1]], xxx[:, 2])
        # print(popt_2)
        #
        error_2 = func([xxx[:, 0], xxx[:, 1]], *popt_2) - xxx[:, -1]
        # print(np.average(error_2), np.std(error_2), "error")
        # plt.plot(xxx[:, 0], func(xxx[:, 0], *popt_2), c="black")
        # plt.show()

        dis = dis_to_surface_(xxx, data_2, popt_2, i_=i_x_i)
        # print(dis)
        # print(dis, type(dis))
        dis_var_set.append(dis)


dis_var_set = np.array(dis_var_set)
print(dis_var_set)
# print(np.sqrt(np.sum(dis_var_set**2)/len(dis_var_set))/np.sqrt(2), "predict noise")
# print(np.std(dis_var_set), np.average(dis_var_set), len(dis_var_set))
print(len(dis_var_set)/len(data_2))
print(np.average(dis_var_set**2), "参数")
