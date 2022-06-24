from tqdm import tqdm
import numpy as np
import pandas as pd
# from statsmodels.tsa.arima_process import arma_generate_sample
from numpy import append, array, sign
from numpy.random import normal, randn


def make_iid_example(mode='multi-normal', s=0.99):
    """
    Returns kernels of iid data that has higher-order interactions (from Bjorn Bottcher's notes: add details)
    """
    # dHSIC_cor = []
    if mode == 'multi-normal':
        # Multivariate normal
        mean = [0, 0, 0]
        cov = [[1, s, s], [s, 1, s], [s, s, 1]]
        d1, d2, d3 = np.random.multivariate_normal(mean, cov, 100).T

    if mode == 'interpolated':
        # Interpolated complete dependence
        mean = [0, 0, 0, 0]
        cov = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        x, x1, x2, x3 = np.random.multivariate_normal(mean, cov, 100).T
        d1, d2, d3 = s * x + (1 - s) * x1, s * x + (1 - s) * x2, s * x + (1 - s) * x3

    if mode == 'higher-order':
        # perturbed higher-order dependence
        mean = [0, 0, 0]
        cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        x1, x2, x3 = np.random.multivariate_normal(mean, cov, 100).T

        y1 = np.random.binomial(n=1, p=0.5, size=100)
        y2 = np.random.binomial(n=1, p=0.5, size=100)
        y3 = np.asarray([int(y1[i] == y2[i]) for i in range(len(y1))])

        d1, d2, d3 = y1 + (1 - s) * x1, y2 + (1 - s) * x2, y3 + (1 - s) * x3

    else:
        raise ValueError("Invalid example")
    df = pd.DataFrame(list(zip(d1, d2, d3)), columns=['d1', 'd2', 'd3'])

    return df


def stationary_pb_ts(n_sample, seed, d, mode, a=0.5):
    np.random.seed(seed)
    x = np.zeros(n_sample)
    y = np.zeros(n_sample)
    z = np.zeros(n_sample)

    x[0] = randn()
    y[0] = randn()
    z[0] = randn()
    for i in range(1, n_sample):
        x[i] = a * x[i - 1] + randn()
        y[i] = a * y[i - 1] + randn()
        if mode == 'case1':
            # pairwise independent but jointly dependent
            z[i] = a * z[i - 1] + d * abs(randn()) * sign(x[i] * y[i]) + randn()
        if mode == 'case2':
            # pairwise dependent and jointly dependent
            z[i] = a * z[i - 1] + d * (x[i] + y[i]) + randn()
        if mode == 'case3':
            # all independence
            z[i] = a * z[i - 1] + randn()
            
    df = pd.DataFrame(list(zip(x, y, z)), columns=['d1', 'd2', 'd3'])
    return df


def make_nonstat():
    """
    Returns nonstationary time series data that has higher-order interactions
    """
    return


def ARIMA(phi=np.array([0]), theta=np.array([0]), d=0, t=0, mu=0, sigma=1, n=20, burn=100):
    """ Simulate data from ARMA model (eq. 1.2.4):

    z_t = phi_1*z_{t-1} + ... + phi_p*z_{t-p} + a_t + theta_1*a_{t-1} + ... + theta_q*a_{t-q}

    with d unit roots for ARIMA model.

    Arguments:
    phi -- array of shape (p,) or (p, 1) containing phi_1, phi2, ... for AR model
    theta -- array of shape (q) or (q, 1) containing theta_1, theta_2, ... for MA model
    d -- number of unit roots for non-stationary time series
    t -- value deterministic linear trend
    mu -- mean value for normal distribution error term
    sigma -- standard deviation for normal distribution error term
    n -- length time series
    burn -- number of discarded values because series begins without lagged terms

    Return:
    x -- simulated ARMA process of shape (n, 1)

    Reference:
    Time Series Analysis by Box et al.
    """

    # add "theta_0" = 1 to theta
    theta = np.append(1, theta)

    # set max lag length AR model
    p = phi.shape[0]

    # set max lag length MA model
    q = theta.shape[0]

    # simulate n + q error terms
    a = np.random.normal(mu, sigma, (n + max(p, q) + burn, 1))

    # create array for returned values
    x = np.zeros((n + max(p, q) + burn, 1))

    # initialize first time series value
    x[0] = a[0]

    for i in range(1, x.shape[0]):
        AR = np.dot(phi[0: min(i, p)], np.flip(x[i - min(i, p): i], 0))
        MA = np.dot(theta[0: min(i + 1, q)], np.flip(a[i - min(i, q - 1): i + 1], 0))
        x[i] = AR + MA + t

    # add unit roots
    if d != 0:
        ARMA = x[-n:]
        m = ARMA.shape[0]
        z = np.zeros((m + 1, 1))  # create temp array

        for i in range(d):
            for j in range(m):
                z[j + 1] = ARMA[j] + z[j]
            ARMA = z[1:]
        x[-n:] = z[1:]

    return x[-n:]
