import numpy as np


def permutation_test(k_list, n_samples, n_nodes, stat_found, n_perms=5000, alpha=0.05):
    """
    Approximates the null distribution by permuting all variables. Using Monte Carlo approximation.
    """

    "to do:"
    "1: include calc_stat as an input function"

    # initiating statistics
    statistics = np.zeros(n_perms)

    for i in range(n_perms):
        term1 = k_list[0]
        term2 = np.sum(k_list[0]) / (n_samples ** 2)
        term3 = 2 * np.sum(k_list[0], axis=0) / (n_samples ** 2)

        for j in range(1, n_nodes):
            index_perm = np.random.permutation(k_list[j].shape[0])
            k_perm = k_list[j][index_perm, index_perm[:, None]]

            term1 = term1 * k_perm
            term2 = term2 * np.sum(k_perm) / (n_samples ** 2)
            term3 = term3 * np.sum(k_perm, axis=0) / n_samples

        term1_sum = np.sum(term1)
        term3_sum = np.sum(term3)

        statistics[i] = term1_sum / (n_samples ** 2) + term2 - term3_sum

    statistics_sort = np.sort(statistics)
    # computing 1-alpha critical value
    Bind = np.sum(stat_found == statistics_sort) + int(np.ceil((1 - alpha) * (n_perms + 1)))
    critical_value = statistics_sort[Bind]

    return critical_value


def joint_independence_test(k_list, n_perms=5000, alpha=0.05):
    """
    Performs the independence test with HSIC and returns an accept or reject statement

    Inputs:
    k_list: list of Kernel matrices for each variable, each having dimensions (n_samples, n_samples)
    n_perms: number of permutations performed when bootstrapping the null
    alpha: rejection threshold of the test

    Returns:
    reject: 1 if null rejected, 0 if null not rejected
    """

    """
    To do:
    1. let stat == fn that computes
    """


    n_nodes = len(k_list)
    n_samples = k_list[0].shape[0]

    # statistic and threshold
    stat = compute_dHSIC_statistics(k_list)
    critical_value = permutation_test(k_list, n_samples, n_nodes, stat, n_perms, alpha)

    reject = int(stat > critical_value)

    return stat, reject
