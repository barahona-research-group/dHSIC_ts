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


def make_K_list(X_list, n_samples, n_nodes):
    """
    Computes the kernel matrices of the variables in X_array, where each column represents one variable.
    Returns a list of the kernel matrices of each variable.
    """
    k_list = list(
        pairwise_kernels(X_list[i], metric='rbf', gamma=0.5 / (width(X_list[i]) ** 2)) for i in range(n_nodes))
    return k_list


def compute_kernel():
    # data preparation
    groups_prep_g = {}
    groups_prep_g_K = {}

    for group in groups:
        print(group)

        groups_prep_g[group] = np.empty(17, dtype=object)
        groups_prep_g_K[group] = np.empty(17, dtype=object)

        for g, goal in enumerate(goals):
            g_list = []
            for country in groups[group].dropna():
                g_list.append(np.asarray(goals_values_i[country][g]))

            g_array = np.asarray(g_list)
            groups_prep_g[group][g] = g_array

            K_matrix = pairwise_kernels(g_array, metric='rbf', gamma=0.5 / (width(g_array) ** 2))
            groups_prep_g_K[group][g] = K_matrix

    return groups_prep_g, groups_prep_g_K


def make_iid():
    """
    Returns iid data that has higher-order interactions (from Bjorn Bottcher's paper)
    """
    return


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
