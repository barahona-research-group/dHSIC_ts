# this is the file for making data object
import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_kernels
import networkx as nx
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
import pandas as pd
import copy
from sklearn.metrics import pairwise_distances, pairwise_kernels


def width(Z):
    """
    Computes the median heuristic for the kernel bandwidth
    """
    dist_mat = pairwise_distances(Z, metric='euclidean')
    width_Z = np.median(dist_mat[dist_mat > 0])
    return width_Z


def compute_kernel(data, nodes):
    # data preparation

    num_nodes = nodes
    prep_g = np.empty(num_nodes, dtype=object)
    prep_g_K = np.empty(num_nodes, dtype=object)

    for g in range(num_nodes):
        g_list = []
        for i in nodes:
            g_list.append(np.asarray(data[i]))

        g_array = np.asarray(g_list)
        prep_g[g] = g_array

        K_matrix = pairwise_kernels(g_array, metric='rbf', gamma=0.5 / (width(g_array) ** 2))
        prep_g_K[g] = K_matrix

    return prep_g, prep_g_K


def make_iid(mode='multi-normal'):
    """
    Returns kernels of iid data that has higher-order interactions (from Bjorn Bottcher's notes)
    """
    for s in np.linspace(0, 1, 201):
        if mode == 'multi-normal':
            # Multivariate normal
            mean = [0, 0, 0]
            cov = [[1, s, s], [s, 1, s], [s, s, 1]]
            d1, d2, d3 = np.random.multivariate_normal(mean, cov, 4000).T

        if mode == 'interpolated':
            # Interpolated complete dependence
            mean = [0, 0, 0, 0]
            cov = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            for s in np.linspace(0, 1, 201):
                x, x1, x2, x3 = np.random.multivariate_normal(mean, cov, 4000).T
                d1, d2, d3 = s * x + (1 - s) * x1, s * x + (1 - s) * x2, s * x + (1 - s) * x3

        if mode == 'higher-order':
            # perturbed higher-order dependence
            mean = [0, 0, 0]
            cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            for s in np.linspace(0, 1, 201):
                x1, x2, x3 = np.random.multivariate_normal(mean, cov, 4000).T

                y1 = np.random.binomial(n=1, p=0.5, size=4000)
                y2 = np.random.binomial(n=1, p=0.5, size=4000)
                y3 = np.asarray([int(y1[i] == y2[i]) for i in range(len(y1))])

                d1, d2, d3 = y1 + (1 - s) * x1, y2 + (1 - s) * x2, y3 + (1 - s) * x3

    K1 = pairwise_kernels(d1.reshape(-1, 1), metric='rbf', gamma=0.5 / (width(d1.reshape(-1, 1)) ** 2))
    K2 = pairwise_kernels(d2.reshape(-1, 1), metric='rbf', gamma=0.5 / (width(d2.reshape(-1, 1)) ** 2))
    K3 = pairwise_kernels(d3.reshape(-1, 1), metric='rbf', gamma=0.5 / (width(d3.reshape(-1, 1)) ** 2))

    K_list_all = [K1, K2, K3]
    K_list_1 = [K1, K1, K1]
    K_list_2 = [K2, K2, K2]
    K_list_3 = [K3, K3, K3]

    return K_list_all, K_list_1, K_list_2, K_list_3


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
