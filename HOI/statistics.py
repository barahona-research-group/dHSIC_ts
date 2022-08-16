import numpy as np


def compute_dHSIC_statistics(k_list):
    """
    Computes the dHSIC statistic

    Parameters
    ----------
    k_list: kernel lists. each entry is kernel matrices for each variable

    Returns
    -------
    single value for dHSIC statistic
    """

    n_variables = len(k_list)
    n_samples = k_list[0].shape[0]

    term1 = k_list[0]
    term2 = np.sum(k_list[0]) / (n_samples ** 2)
    term3 = 2 * np.sum(k_list[0], axis=0) / (n_samples ** 2)
    for j in range(1, n_variables):
        term1 = term1 * k_list[j]
        term2 = term2 * np.sum(k_list[j]) / (n_samples ** 2)
        term3 = term3 * np.sum(k_list[j], axis=0) / n_samples

    term1_sum = np.sum(term1)
    term3_sum = np.sum(term3)
    dHSIC = term1_sum / (n_samples ** 2) + term2 - term3_sum
    return dHSIC


def compute_lancaster_statistics(k_list):
    K = k_list[0]
    L = k_list[1]
    M = k_list[2]
    m = np.shape(K)[0]
    H = np.eye(m) - 1 / m * np.ones(m)
    Kc = H @ K @ H
    Lc = H @ L @ H
    Mc = H @ M @ H
    statMatrix = Kc * Lc * Mc
    lancaster = 1/(m**2) * np.sum(statMatrix)
    return lancaster

# def sq_distance(a, b):
#     aa = np.multiply(a, a)
#     bb = np.multiply(b, b)
#     ab = np.outer(a, b)
#     d = abs(np.tile(aa.reshape((-1, 1)), (1, np.shape(bb)[0])) + np.tile(bb, (np.shape(aa)[0], 1)) - 2 * ab)
#     return d
#
#
# def Gauss_kernel(x, y, sig):
#     H = sq_distance(x.reshape((-1, 1)), y.reshape((-1, 1)))
#     H = np.exp(-H / 2 / sig ^ 2)
#     return H
#
#
# def empirically_centre(gram_matrix):
#     n = np.shape(gram_matrix)[0]
#     centred_matrix = gram_matrix
#     centred_matrix = centred_matrix - (1 / n) * np.tile(np.sum(gram_matrix, 1).reshape((-1, 1)), (1, n))
#     centred_matrix = centred_matrix - (1 / n) * np.tile(np.sum(gram_matrix, 0), (n, 1))
#     centred_matrix = centred_matrix + (1 / n ** 2) * np.tile(np.sum(np.sum(gram_matrix)), (n, n))
#     return centred_matrix
#
#
# def compute_HSIC_statistics(K, L):
#     # K = k_list[0]
#     # L = k_list[1]
#     m = np.shape(K)[0]
#     H = np.eye(m) - 1 / m * np.ones(m, m)
#     Kc = H @ K @ H
#     Lc = H @ L @ H
#     statMatrix = Kc * Lc
#     return statMatrix






















