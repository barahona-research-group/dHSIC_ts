# this is the file for making data
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_kernels, pairwise_distances

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def width(Z):
    """
    Computes the median heuristic for the kernel bandwidth
    """
    dist_mat = pairwise_distances(Z, metric='euclidean')
    width_Z = np.median(dist_mat[dist_mat > 0])
    return width_Z


def compute_kernel(df):
    # data preparation
    """
    To do:
    """

    data_dict = df.to_dict('list')
    kernel_dict = {}

    for i in list(data_dict.keys()):
        data = np.array(data_dict[i]).reshape(-1, 1)
        kernel_matrix = pairwise_kernels(data, metric='rbf', gamma=0.5 / (width(data) ** 2))
        kernel_dict[i] = kernel_matrix

    return data_dict, kernel_dict


def compute_kernel_n(data_mat):
    # data preparation for individual variable matrices
    """
    To do:
    """
    kernel_matrix = pairwise_kernels(data, metric='rbf', gamma=0.5 / (width(data_mat) ** 2))

    return kernel_matrix


def unpack_dict_info(kernel_dict):
    k_list = np.array(list(kernel_dict.values()))
    n_variables = len(k_list)
    n_samples = k_list[0].shape[0]
    return k_list, n_variables, n_samples
