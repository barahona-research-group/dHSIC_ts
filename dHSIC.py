# Perform the independence test
import numpy as np


def compute_dHSIC_statistics(k_list):
    """
    Computes the dHSIC statistic

    Parameters
    ----------
    k_list: kernel list

    Returns
    -------

    """
    n_nodes = len(k_list)
    n_samples = k_list[0].shape[0]

    term1, term2, term3 = 1, 1, 2 / n_samples
    for j in range(n_nodes):
        term1 = term1 * k_list[j]
        term2 = term2 * np.sum(k_list[j]) / (n_samples ** 2)
        term3 = term3 * np.sum(k_list[j], axis=0) / n_samples

    term1_sum = np.sum(term1)
    term3_sum = np.sum(term3)
    dHSIC = term1_sum / (n_samples ** 2) + term2 - term3_sum
    return dHSIC


def compute_all_HSIC(data, num_var):
    """Compute HSIC for all levels of interactions"""
    HSIC_dict = {}
    # compute HSIC(X,X)....HSIC(X,X,X,X,X) in one fn
    for i in range(2, 5+1):
        HSIC_dict[str(i)] = []
        HSICs = []
        for j in np.arange(num_var):
            k_list = [list([data[j] for _ in range(i)])]
            HSIC = compute_dHSIC_statistics(k_list)
            HSICs.append(HSIC)
        HSIC_dict[str(i)] = HSICs

    return HSIC_dict
