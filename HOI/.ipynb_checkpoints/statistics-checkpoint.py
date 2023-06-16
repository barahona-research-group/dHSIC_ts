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
