import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import pickle
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

X1 = np.arange(-5, 5, 0.1)
Y1 = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X1, Y1)
R = X + Y
Z = R

X_reshaped = np.reshape(X, (-1, 1))
Y_reshaped = np.reshape(Y, (-1, 1))
Z_reshaped = np.reshape(Z, (-1, 1))

data_original_1 = np.hstack((X_reshaped, Y_reshaped, Z_reshaped))
print(data_original_1.shape, "1_size")

X2 = np.arange(-5, 5, 0.05)
Y2 = np.arange(-5, 5, 0.05)
X_2, Y_2 = np.meshgrid(X2, Y2)
R_2 = X_2 + Y_2
Z_2 = R_2

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
    return a_*data_[0] + b_*data_[1] + c_


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


data_2 = add_noise(data_original_1, 0, 0.1**2)
data_3 = add_noise(data_original_2, 0, 0.05**2)
data_4 = add_noise(data_original_2, 0, 0**2)

np.savetxt('0_1.txt', data_2)
np.savetxt('0_05.txt', data_3)
np.savetxt('0.txt', data_4)

# x_b, y_b, z_b = add_noise(X, Y, Z, 0, 0.001)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data_2[:, 0], data_2[:, 1], data_2[:, 2], c="blue")

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data_3[:, 0], data_3[:, 1], data_3[:, 2], c="blue")

plt.show()

