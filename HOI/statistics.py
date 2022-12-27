import numpy as np


def compute_dHSIC_statistics(K):
    """
    Computes the dHSIC statistic

    Parameters
    ----------
    K: kernel lists stacked along 0-axis.
        Each K[i, :, :] entry is a kernel matrix for i-th variable.

    Returns
    -------
    single value for dHSIC statistic
    """

    n_samples = K.shape[1]

    term1 = np.mean(np.prod(K, axis=0))
    # Normalizing const: Prod(1/n^2) = 1/n^(2d)
    term2 = np.prod(np.mean(K, axis=(1, 2)))
    # Normalizing const: (1/n) Prod(1/n)=1/n^(d+1)
    term3 = (2 / n_samples) * np.sum(np.prod(np.mean(K, axis=2), axis=0))
    return term1 + term2 - term3


def compute_lancaster_statistics(k_list):
    K = k_list[0]
    L = k_list[1]
    M = k_list[2]
    m = np.shape(K)[0]
    H = np.eye(m) - 1 / m * np.ones(m)
    Kc = H @ K @ H
    Lc = H @ L @ H
    Mc = H @ M @ H
    return np.mean(Kc * Lc * Mc)


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
