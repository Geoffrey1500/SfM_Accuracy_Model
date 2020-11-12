import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import pickle
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

X1 = np.arange(-5, 5, 0.05)
Y1 = np.arange(-5, 5, 0.05)
X, Y = np.meshgrid(X1, Y1)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

print(Z)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)

plt.show()


def add_noise(x_, y_, z_, mu_=0, sigma_=0.001):
    x_internal = x_.copy()
    y_internal = y_.copy()
    z_internal = z_.copy()

    mean = (mu_, mu_, mu_)
    cov = [[sigma_, 0, 0], [0, sigma_, 0], [0, 0, sigma_]]

    L_ = np.random.multivariate_normal(mean, cov, len(x_internal))

    x_internal += L_[:, 0]
    y_internal += L_[:, 1]
    z_internal += L_[:, 2]

    return x_internal, y_internal, z_internal


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


x_a, y_a, z_a = add_noise(X, Y, Z, 0, 0.05)

x_b, y_b, z_b = add_noise(X, Y, Z, 0, 0.001)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x_a, y_a, z_a, rstride=1, cstride=1, cmap=cm.viridis)

plt.show()
