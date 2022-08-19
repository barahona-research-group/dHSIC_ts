from tqdm import tqdm
import numpy as np
import pandas as pd
# from statsmodels.tsa.arima_process import arma_generate_sample
from numpy import sign, sin, cos, pi
from numpy.random import normal, randn


def make_iid_example(mode, s=0.99, n_sample=100):
    """
    Returns kernels of iid data that has higher-order interactions (from Bjorn Bottcher's notes: add details)
    """
    if mode == 'multi-normal':
        # Multivariate normal
        mean = [0, 0, 0]
        cov = [[1, s, s], [s, 1, s], [s, s, 1]]
        d1, d2, d3 = np.random.multivariate_normal(mean, cov, n_sample).T

    if mode == 'interpolated':
        # Interpolated complete dependence
        mean = [0, 0, 0, 0]
        cov = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        x, x1, x2, x3 = np.random.multivariate_normal(mean, cov, n_sample).T
        d1, d2, d3 = s * x + (1 - s) * x1, s * x + (1 - s) * x2, s * x + (1 - s) * x3

    if mode == 'higher-order':
        # perturbed higher-order dependence
        mean = [0, 0, 0]
        cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        x1, x2, x3 = np.random.multivariate_normal(mean, cov, n_sample).T

        y1 = np.random.binomial(n=1, p=0.5, size=n_sample)
        y2 = np.random.binomial(n=1, p=0.5, size=n_sample)
        y3 = np.asarray([int(y1[i] == y2[i]) for i in range(len(y1))])

        d1, d2, d3 = y1 + (1 - s) * x1, y2 + (1 - s) * x2, y3 + (1 - s) * x3

    df = pd.DataFrame(list(zip(d1, d2, d3)), columns=['d1', 'd2', 'd3'])

    return df


def make_iid_example_4way(mode, s=0.99, n_sample=100):
    """
    Returns kernels of iid data that has higher-order interactions (from Bjorn Bottcher's notes: add details)
    """
    if mode == 'multi-normal':
        # Multivariate normal
        mean = [0, 0, 0, 0]
        cov = [[1, s, s, s], [s, 1, s, s], [s, s, 1, s], [s, s, s, 1]]
        d1, d2, d3, d4 = np.random.multivariate_normal(mean, cov, n_sample).T

    if mode == 'interpolated':
        # Interpolated complete dependence
        mean = [0, 0, 0, 0, 0]
        cov = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]

        x, x1, x2, x3, x4 = np.random.multivariate_normal(mean, cov, n_sample).T
        d1, d2, d3, d4 = s * x + (1 - s) * x1, s * x + (1 - s) * x2, s * x + (1 - s) * x3, s * x + (1 - s) * x4

    if mode == 'higher-order':
        # perturbed higher-order dependence
        mean = [0, 0, 0, 0]
        cov = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        x1, x2, x3, x4 = np.random.multivariate_normal(mean, cov, n_sample).T

        y1 = np.random.binomial(n=1, p=0.5, size=n_sample)
        y2 = np.random.binomial(n=1, p=0.5, size=n_sample)
        y3 = np.random.binomial(n=1, p=0.5, size=n_sample)
        y4 = np.asarray([int(y1[i] == y2[i] == y3[i]) for i in range(len(y1))])

        d1, d2, d3, d4 = y1 + (1 - s) * x1, y2 + (1 - s) * x2, y3 + (1 - s) * x3, y4 + (1 - s) * x4

    df = pd.DataFrame(list(zip(d1, d2, d3, d4)), columns=['d1', 'd2', 'd3', 'd4'])

    return df


def stationary_pb_ts(t_time, d, mode, a=0.5, order=3):
    # variables * time * 1
    # np.random.seed(seed)
    x = np.zeros(t_time)
    y = np.zeros(t_time)
    w = np.zeros(t_time)
    z = np.zeros(t_time)

    x[0] = randn()
    y[0] = randn()
    w[0] = randn()
    z[0] = randn()

    if order == 3:
        for i in range(1, t_time):
            x[i] = a * x[i - 1] + randn()
            y[i] = a * y[i - 1] + randn()
            if mode == 'case1':
                # pairwise independent but jointly dependent
                z[i] = a * z[i - 1] + d * abs(randn()) * sign(x[i] * y[i]) + randn()
            if mode == 'case2':
                # 2 pairwise dependent and 3-way jointly dependent
                z[i] = a * z[i - 1] + d * (x[i] + y[i]) + randn()
            if mode == 'case3':
                # all independence
                z[i] = a * z[i - 1] + randn()
        df = pd.DataFrame(list(zip(x, y, z)), columns=['d1', 'd2', 'd3'])

    if order == 4:
        for i in range(1, t_time):
            x[i] = a * x[i - 1] + randn()
            y[i] = a * y[i - 1] + randn()
            w[i] = a * w[i - 1] + randn()

            if mode == 'case1':
                # pairwise independent but 4-way jointly dependent
                z[i] = a * z[i - 1] + d * abs(randn()) * sign(x[i] * y[i] * w[i]) + randn()
            if mode == 'case2':
                # 3 pairwise dependent and 4-way jointly dependent
                z[i] = a * z[i - 1] + d * (x[i] + y[i] + w[i]) + randn()
            if mode == 'case3':
                # all independence
                z[i] = a * z[i - 1] + randn()
        df = pd.DataFrame(list(zip(x, y, w, z)), columns=['d1', 'd2', 'd3', 'd4'])
    return df


def stationary_pb_ts_n(n_sample, t_time, d, mode, a=0.5):
    # variables * time * n_sample
    x_mat = []
    y_mat = []
    z_mat = []
    for j in range(n_sample):
        # np.random.seed(seed)
        x = np.zeros(t_time)
        y = np.zeros(t_time)
        z = np.zeros(t_time)

        x[0] = randn()
        y[0] = randn()
        z[0] = randn()
        for i in range(1, t_time):
            x[i] = a * x[i - 1] + randn()
            y[i] = a * y[i - 1] + randn()
            if mode == 'case1':
                # pairwise independent but jointly dependent
                z[i] = a * z[i - 1] + d * max(x[i], y[i]) + randn()
            if mode == 'case2':
                # 2 pairwise dependent and jointly dependent
                z[i] = a * z[i - 1] + d * (x[i] + y[i]) + randn()
            if mode == 'case3':
                # all independence
                z[i] = a * z[i - 1] + randn()

        x_mat.append(x)
        y_mat.append(y)
        z_mat.append(z)
    # np.swapaxes(x_mat, 0, 1), np.swapaxes(y_mat, 0, 1), np.swapaxes(z_mat, 0, 1)
    return np.array(x_mat), np.array(y_mat), np.array(z_mat)


def nonstationary_ts_n(n_sample, t_time, d, mode, a=0.5, order=3):
    """
    Returns nonstationary time series data that has higher-order interactions
    """
    # variables * time * n_sample
    x_mat = []
    y_mat = []
    w_mat = []
    z_mat = []
    for j in range(n_sample):
        # np.random.seed(seed)
        x = np.zeros(t_time)
        y = np.zeros(t_time)
        w = np.zeros(t_time)
        z = np.zeros(t_time)

        x[0] = randn()
        y[0] = randn()
        w[0] = randn()
        z[0] = randn()
        for i in range(1, t_time):
            x[i] = a * x[i - 1] + i * randn()
            y[i] = a * y[i - 1] + i * randn()
            w[i] = a * w[i - 1] + i * randn()
            if order == 3:
                if mode == 'case1':
                    # pairwise independent but jointly dependent
                    z[i] = a * z[i - 1] + d * i * sign(x[i] * y[i]) + normal(0, 0.25)
                if mode == 'case2':
                    # 2 pairwise dependent and jointly dependent
                    z[i] = a * z[i - 1] + d * (x[i] + y[i]) + i * randn()
                if mode == 'case3':
                    # all independence
                    z[i] = a * z[i - 1] + i * randn()
            if order == 4:
                if mode == 'case1':
                    # pairwise independent but jointly dependent
                    z[i] = a * z[i - 1] + d * i * sign(x[i] * y[i] * w[i]) + normal(0, 0.125)
                if mode == 'case2':
                    # 2 pairwise dependent and jointly dependent
                    z[i] = a * z[i - 1] + d * (x[i] + y[i] + w[i]) + i * randn()
                if mode == 'case3':
                    # all independence
                    z[i] = a * z[i - 1] + i * randn()
        x_mat.append(x)
        y_mat.append(y)
        w_mat.append(w)
        z_mat.append(z)
    return np.array(x_mat), np.array(y_mat), np.array(w_mat), np.array(z_mat)


# def ARIMA(phi=np.array([0]), theta=np.array([0]), d=0, t=0, mu=0, sigma=1, n=20, burn=100):
#     """ Simulate data from ARMA model (eq. 1.2.4):
#
#     z_t = phi_1*z_{t-1} + ... + phi_p*z_{t-p} + a_t + theta_1*a_{t-1} + ... + theta_q*a_{t-q}
#
#     with d unit roots for ARIMA model.
#
#     Arguments:
#     phi -- array of shape (p,) or (p, 1) containing phi_1, phi2, ... for AR model
#     theta -- array of shape (q) or (q, 1) containing theta_1, theta_2, ... for MA model
#     d -- number of unit roots for non-stationary time series
#     t -- value deterministic linear trend
#     mu -- mean value for normal distribution error term
#     sigma -- standard deviation for normal distribution error term
#     n -- length time series
#     burn -- number of discarded values because series begins without lagged terms
#
#     Return:
#     x -- simulated ARMA process of shape (n, 1)
#
#     Reference:
#     Time Series Analysis by Box et al.
#     """
#
#     # add "theta_0" = 1 to theta
#     theta = np.append(1, theta)
#
#     # set max lag length AR model
#     p = phi.shape[0]
#
#     # set max lag length MA model
#     q = theta.shape[0]
#
#     # simulate n + q error terms
#     a = np.random.normal(mu, sigma, (n + max(p, q) + burn, 1))
#
#     # create array for returned values
#     x = np.zeros((n + max(p, q) + burn, 1))
#
#     # initialize first time series value
#     x[0] = a[0]
#
#     for i in range(1, x.shape[0]):
#         AR = np.dot(phi[0: min(i, p)], np.flip(x[i - min(i, p): i], 0))
#         MA = np.dot(theta[0: min(i + 1, q)], np.flip(a[i - min(i, q - 1): i + 1], 0))
#         x[i] = AR + MA + t
#
#     # add unit roots
#     if d != 0:
#         ARMA = x[-n:]
#         m = ARMA.shape[0]
#         z = np.zeros((m + 1, 1))  # create temp array
#
#         for i in range(d):
#             for j in range(m):
#                 z[j + 1] = ARMA[j] + z[j]
#             ARMA = z[1:]
#         x[-n:] = z[1:]
#
#     return x[-n:]


def main():
    x, y, z = stationary_pb_ts_n(100, 10, 0.5, 'case1')
    print(x.shape)


if __name__ == "__main__":
    main()
