import numpy as np

from HOI.preprocessings import compute_kernel
from HOI.statistics import compute_dHSIC_statistics, compute_lancaster_statistics
from scipy.signal import lfilter
import statsmodels.api as sm
# import numba
import cProfile

from examples.synthetic_data import make_iid_example


def permute_K(K, n_perms, seed):

    K_perms = []
    for seed in np.random.SeedSequence(seed).spawn(n_perms):
        K_perm = np.zeros_like(K)
        for d, sub_seed in enumerate(seed.spawn(K.shape[0])):
            rng = np.random.default_rng(sub_seed)
            var_perm = rng.permutation(K.shape[1])
            K_perm[d, :, :] = K[d, var_perm, var_perm[:, None]]

        K_perms.append(K_perm)

    return K_perms


def estimate_tail_head(data_list):
    m = len(data_list[0])
    # sum up all variable vectors
    acf = sm.tsa.acf(np.sum(data_list, axis=0), nlags=50)
    smallestACF = np.where(acf < 0.2)  # returns indices
    head = smallestACF[0][0]  # returns smallest index
    if head > min(75, m):
        raise ValueError(
            'possibly long memory process, the output of test might be FALSE.')
    head = min(head, 50)
    tail = m
    return head, tail


def shift_K(data_list, K, n_perms, seed):

    K_perms = []
    head, tail = estimate_tail_head(data_list)
    index = np.arange(K.shape[1])
    for seed in np.random.SeedSequence(seed).spawn(n_perms):
        K_perm = np.zeros_like(K)
        for d, sub_seed in enumerate(seed.spawn(K.shape[0])):
            rng = np.random.default_rng(sub_seed)

            cut = rng.integers(low=head, high=tail)
            # get permutations after shift
            # index_perm = np.array([cut:tail, 1:cut-1])
            index_perm = np.roll(index, -cut)
            # shift the matrix corresponds to that of the time series
            K_perm[d, :, :] = K[d, index_perm, index_perm[:, None]]

        K_perms.append(K_perm)

    return K_perms


def reject_H0(K0, K_perms, stat_fun, alpha=0.05):
    """
    Approximates the null distribution by permuting all variables. Using Monte Carlo approximation.
    """
    "to do:"
    "1: include calc_stat as an input function"
    s0 = stat_fun(K0)
    stats = list(map(stat_fun, K_perms))
    return int((1 + sum(s >= s0 for s in stats)) / (1 + len(stats)) < alpha)


def dhsic_permutation(k_list, n_samples, n_variables, stat_found, n_perms=5000, alpha=0.05):
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

        for j in range(1, n_variables):
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
    ind = np.sum(stat_found == statistics_sort) + int(np.ceil((1 - alpha) * (n_perms + 1)))
    critical_value = statistics_sort[ind]
    pval = (np.sum(stat_found <= statistics_sort) + 1) / (n_perms + 1)
    return critical_value, pval


def shifting(data_list, k_list, n_samples, n_variables, stat_found, n_perms=5000, alpha=0.05):
    """
    Approximates the null distribution by permuting all variables. Using Monte Carlo approximation.
    """

    "to do:"

    head, tail = estimate_tail_head(data_list)
    statistics = np.zeros(n_perms)
    index = range(0, n_samples)
    for i in range(n_perms):
        term1 = k_list[0]
        term2 = np.sum(k_list[0]) / (n_samples ** 2)
        term3 = 2 * np.sum(k_list[0], axis=0) / (n_samples ** 2)

        for j in range(1, n_variables):
            # choose a random number between head and tail
            cut = np.random.randint(low=head, high=tail)
            # get permutations after shift
            # index_perm = np.array([cut:tail, 1:cut-1])
            index_perm = np.roll(index, cut)
            # shift the matrix corresponds to that of the time series
            k_perm = k_list[j][index_perm, index_perm[:, None]]

            term1 = term1 * k_perm
            term2 = term2 * np.sum(k_perm) / (n_samples ** 2)
            term3 = term3 * np.sum(k_perm, axis=0) / n_samples

        term1_sum = np.sum(term1)
        term3_sum = np.sum(term3)

        statistics[i] = term1_sum / (n_samples ** 2) + term2 - term3_sum

    statistics_sort = np.sort(statistics)
    # computing 1-alpha critical value
    ind = np.sum(stat_found == statistics_sort) + int(np.ceil((1 - alpha) * (n_perms + 1)))
    critical_value = statistics_sort[ind]
    pval = (np.sum(stat_found <= statistics_sort) + 1) / (n_perms + 1)
    return critical_value, pval


def test_independence(k_list, data_list, n_perms=5000, alpha=0.05, mode='permutation'):
    """
    Performs the independence test with dHSIC and returns an accept or reject statement

    Inputs:
    k_list: list of Kernel matrices for each variable, each having dimensions (n_samples, n_samples)
    mode: choose test type
    n_perms: number of permutations performed when bootstrapping the null
    alpha: rejection threshold of the test

    Returns:
    reject: 1 if null rejected, 0 if null not rejected
    """

    """
    To do:
    1. add if conditions to import the right stats/test
    """

    n_variables = len(k_list)
    n_samples = k_list[0].shape[0]
    # statistic and threshold
    if mode == 'permutation':
        statistic = compute_dHSIC_statistics(k_list)
        critical_value, pval = dhsic_permutation(k_list, n_samples, n_variables, statistic, n_perms, alpha)
        reject = int(statistic > critical_value)
        return statistic, critical_value, pval, reject
    if mode == 'shifting':
        statistic = compute_dHSIC_statistics(k_list)
        critical_value, pval = shifting(data_list, k_list, n_samples, n_variables, statistic, n_perms, alpha)
        reject = int(statistic > critical_value)
        return statistic, critical_value, pval, reject
 