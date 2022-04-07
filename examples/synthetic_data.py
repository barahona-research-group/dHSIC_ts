from tqdm import tqdm
import numpy as np
from sklearn.metrics import pairwise_kernels
from HOI.statistics import compute_dHSIC_statistics


def make_iid_example(mode='multi-normal'):
    """
    Returns kernels of iid data that has higher-order interactions (from Bjorn Bottcher's notes)
    """
    # dHSIC_cor = []
    for s in tqdm(np.linspace(0, 1, 201)):
        if mode == 'multi-normal':
            # Multivariate normal
            mean = [0, 0, 0]
            cov = [[1, s, s], [s, 1, s], [s, s, 1]]
            d1, d2, d3 = np.random.multivariate_normal(mean, cov, 4000).T

        if mode == 'interpolated':
            # Interpolated complete dependence
            mean = [0, 0, 0, 0]
            cov = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

            x, x1, x2, x3 = np.random.multivariate_normal(mean, cov, 4000).T
            d1, d2, d3 = s * x + (1 - s) * x1, s * x + (1 - s) * x2, s * x + (1 - s) * x3

        if mode == 'higher-order':
            # perturbed higher-order dependence
            mean = [0, 0, 0]
            cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

            x1, x2, x3 = np.random.multivariate_normal(mean, cov, 4000).T

            y1 = np.random.binomial(n=1, p=0.5, size=4000)
            y2 = np.random.binomial(n=1, p=0.5, size=4000)
            y3 = np.asarray([int(y1[i] == y2[i]) for i in range(len(y1))])

            d1, d2, d3 = y1 + (1 - s) * x1, y2 + (1 - s) * x2, y3 + (1 - s) * x3

        K1 = pairwise_kernels(d1.reshape(-1, 1), metric='rbf', gamma=0.5 / (width(d1.reshape(-1, 1)) ** 2))
        K2 = pairwise_kernels(d2.reshape(-1, 1), metric='rbf', gamma=0.5 / (width(d2.reshape(-1, 1)) ** 2))
        K3 = pairwise_kernels(d3.reshape(-1, 1), metric='rbf', gamma=0.5 / (width(d3.reshape(-1, 1)) ** 2))

        # K_list_all = [K1, K2, K3]
        # K_list_1 = [K1, K1, K1]
        # K_list_2 = [K2, K2, K2]
        # K_list_3 = [K3, K3, K3]
        #
        # dHSIC_all = compute_dHSIC_statistics(K_list_all)
        # dHSIC_1 = compute_dHSIC_statistics(K_list_1)
        # dHSIC_2 = compute_dHSIC_statistics(K_list_2)
        # dHSIC_3 = compute_dHSIC_statistics(K_list_3)
        #
        # dHSIC_cor.append(dHSIC_all / (dHSIC_1 * dHSIC_2 * dHSIC_3) ** (1 / 3))

    return K1, K2, K3


def make_stat():
    """
    Returns stationary time series data that has higher-order interactions
    """
    return


def make_nonstat():
    """
    Returns nonstationary time series data that has higher-order interactions
    """
    return
