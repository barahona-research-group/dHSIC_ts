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
    # if (tail - head) < 100:
    #     raise ValueError('using less than 100 points for a bootstrap approximation, stability of the test might be '
    #                      'affected')
    return head, tail


def shift_K(data_list, K, n_perms, seed):

    K_perms = []
    head, tail = estimate_tail_head(data_list)
    index = np.arange(K.shape[1])
    for seed in np.random.SeedSequence(seed).spawn(n_perms):
        K_perm = np.zeros_like(K)
        for d, sub_seed in enumerate(seed.spawn(K.shape[0])):
            rng = np.random.default_rng(sub_seed)

            cut = rng.randint(low=head, high=tail)
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
    return (1 + sum(s >= s0 for s in stats)) / (1 + len(stats)) < alpha


def lancaster_permutation(k_list, stat_found, n_perms=5000, alpha=0.05):
    K = k_list[0]
    L = k_list[1]
    M = k_list[2]
    m = np.shape(K)[0]
    H = np.eye(m) - 1 / m * np.ones(m)
    Kc = H @ K @ H
    Lc = H @ L @ H
    Mc = H @ M @ H

    statistics = np.zeros(n_perms)
    for i in range(n_perms):
        y_perm = np.random.permutation(Kc.shape[0])
        z_perm = np.random.permutation(Kc.shape[0])
        Lc_perm = Lc[y_perm, y_perm[:, None]]
        Mc_perm = Mc[z_perm, z_perm[:, None]]
        statistics[i] = 1 / (m**2) * np.sum(Kc * Lc_perm * Mc_perm)

    statistics_sort = np.sort(statistics)
    critical_value = np.quantile(statistics_sort, 1 - alpha)
    pval = (np.sum(stat_found <= statistics_sort) + 1) / (n_perms + 1)
    return critical_value, pval


# def lancaster_shifting(data_list,
#                        k_list,
#                        stat_found,
#                        n_samples,
#                        n_perms=5000,
#                        alpha=0.05):
#     head, tail = estimate_tail_head(data_list)
#     index = range(0, n_samples)

#     K = k_list[0]
#     L = k_list[1]
#     M = k_list[2]
#     m = np.shape(K)[0]
#     H = np.eye(m) - 1 / m * np.ones(m)
#     Kc = H @ K @ H
#     Lc = H @ L @ H
#     Mc = H @ M @ H

#     statistics = np.zeros(n_perms)
#     for i in range(n_perms):
#         cut_y = np.random.randint(low=head, high=tail)
#         y_perm = np.roll(index, cut_y)
#         cut_z = np.random.randint(low=head, high=tail)
#         z_perm = np.roll(index, cut_z)

#         Lc_perm = Lc[y_perm, y_perm[:, None]]
#         Mc_perm = Mc[z_perm, z_perm[:, None]]
#         statistics[i] = 1 / (m**2) * np.sum(Kc * Lc_perm * Mc_perm)

#     statistics_sort = np.sort(statistics)
#     critical_value = np.quantile(statistics_sort, 1 - alpha)
#     pval = (np.sum(stat_found <= statistics_sort) + 1) / (n_perms + 1)
#     return critical_value, pval

# def test_lancaster(k_list,
#                    data_list,
#                    n_perms=5000,
#                    alpha=0.05,
#                    mode='permutation'):
#     n_samples = k_list[0].shape[0]
#     if mode == 'permutation':
#         statistic = compute_lancaster_statistics(k_list)
#         critical_value, pval = lancaster_permutation(k_list, statistic,
#                                                      n_perms, alpha)
#         reject = int(statistic > critical_value)
#         return statistic, critical_value, pval, reject
#     if mode == 'shifting':
#         statistic = compute_lancaster_statistics(k_list)
#         critical_value, pval = lancaster_shifting(data_list, k_list, statistic,
#                                                   n_samples, n_perms, alpha)
#         reject = int(statistic > critical_value)
#         return statistic, critical_value, pval, reject

#
# def shifting_kernel(head, tail, k_list):
#     num_var = len(k_list)
#     shifted_kernel_lists = {}
#     for cut in range(head, tail):
#         ind = list(range(cut, tail)) + list(range(0, cut))
#         shifted_kernel_list = {}
#         for i in range(num_var):
#             shifted_kernel_list[str(i)] = k_list[i][:, ind]
#         shifted_kernel_lists[str(cut-head)] = shifted_kernel_list
#     return shifted_kernel_lists
#
#
# def rand_ind(num_cut, num_var):
#     perms = [np.array(range(num_cut))]
#     for i in range(num_var-1):
#         perm = np.random.permutation(num_cut)
#         perms.append(perm)
#     perms = np.transpose(perms)
#     return perms
#
#
# def permutation_stationary_ts(data_list, k_list, alpha):
#     num_var = len(k_list)
#     head, tail = estimate_tail_head(data_list)
#     shifted_kernel_lists = shifting_kernel(head, tail, k_list)
#     num_cut = len(shifted_kernel_lists)
#     perms = rand_ind(num_cut, num_var)
#
#     shifted_dHSICs = []
#     for ind in perms:
#         shifted_k_lists = [shifted_kernel_lists[str(ind[i])][str(i)] for i in range(num_var)]
#         shifted_dHSIC = compute_dHSIC_statistics(shifted_k_lists)
#         shifted_dHSICs.append(shifted_dHSIC)
#
#     critical_value = np.quantile(shifted_dHSICs, 1 - alpha)
#     return critical_value

#                       n_perms=5000,
#                       alpha=0.05,
#                       mode='permutation'):
#     """
#     Performs the independence test with dHSIC and returns an accept or reject statement

#     Inputs:
#     k_list: list of Kernel matrices for each variable, each having dimensions (n_samples, n_samples)
#     mode: choose test type
#     n_perms: number of permutations performed when bootstrapping the null
#     alpha: rejection threshold of the test

#     Returns:
#     reject: 1 if null rejected, 0 if null not rejected
#     """
#     """
#     To do:
#     1. add if conditions to import the right stats/test
#     """

#     n_variables = len(k_list)
#     n_samples = k_list[0].shape[0]
#     # statistic and threshold
#     if mode == 'permutation':
#         statistic = compute_dHSIC_statistics(k_list)
#         critical_value, pval = dhsic_permutation(k_list, n_samples,
#                                                  n_variables, statistic,
#                                                  n_perms, alpha)
#         reject = int(statistic > critical_value)
#         return statistic, critical_value, pval, reject
#     if mode == 'shifting':
#         statistic = compute_dHSIC_statistics(k_list)
#         critical_value, pval = shifting(data_list, k_list, n_samples,
#                                         n_variables, statistic, n_perms, alpha)
#         reject = int(statistic > critical_value)
#         return statistic, critical_value, pval, reject
# if mode == 'stat_ts_n':
#     statistic = compute_dHSIC_statistics(k_list)
#     critical_value = permutation_stationary_ts(np.sum(data_list, axis=0), k_list, alpha=0.05)
#     reject = int(statistic > critical_value)
#     return statistic, critical_value, reject

#
# def bootstrap_series(length, n_bootstrap):
#     # generates the wild bootstrap process
#     ln = 20
#     ar = np.array([np.exp(-1 / ln)])
#     variance = 1 - np.exp(-2 / ln)
#     w = np.sqrt(variance) * np.random.randn(length, n_bootstrap)
#     a = [1, -ar]
#     processes = lfilter([1], a, w)
#     processes = processes.astype(float)
#     return processes
#
#
# def wildbootstrap_test(m, stat_matrix, alpha=0.05, n_bootstrap=300, test_type=2):
#     processes = bootstrap_series(m, n_bootstrap)
#     testStats = np.zeros(n_bootstrap)
#
#     for process in range(n_bootstrap):
#         mn = np.mean(processes[process])
#         if test_type == 1:
#             matFix = np.outer([processes[process] - mn], [processes[process] - mn])
#         else:
#             matFix = np.outer(processes[process], processes[process])
#         testStats[process] = m * np.mean(np.dot(stat_matrix, matFix))
#
#     critical_value = np.quantile(testStats, 1 - alpha)
#     return critical_value
#
#
# def HSIC_3way_test(X, Y, Z):
#     m = len(X)
#
#     K = Gauss_kernel(X.reshape((-1, 1)), X.reshape((-1, 1)), 5)
#     L = Gauss_kernel(Y.reshape((-1, 1)), Y.reshape((-1, 1)), 5)
#     M = Gauss_kernel(Z.reshape((-1, 1)), Z.reshape((-1, 1)), 5)
#
#     # test 2-way independence
#     resultsHSIC_XY = compute_HSIC_statistics(K, L)
#     critical_value_XY = wildbootstrap_test(m, resultsHSIC_XY, alpha=0.05, n_bootstrap=300, test_type=2)
#     reject_XY = int(resultsHSIC_XY > critical_value_XY)
#
#     resultsHSIC_XZ = compute_HSIC_statistics(K, M)
#     critical_value_XZ = wildbootstrap_test(m, resultsHSIC_XZ, alpha=0.05, n_bootstrap=300, test_type=2)
#     reject_XZ = int(resultsHSIC_XZ > critical_value_XZ)
#
#     resultsHSIC_YZ = compute_HSIC_statistics(L, M)
#     critical_value_YZ = wildbootstrap_test(m, resultsHSIC_YZ, alpha=0.05, n_bootstrap=300, test_type=2)
#     reject_YZ = int(resultsHSIC_YZ > critical_value_YZ)
#
#     Kc = empirically_centre(K)
#     Lc = empirically_centre(L)
#     Mc = empirically_centre(M)
#
#     # test 3-way independence
#     resultsHSIC_XY_Z = compute_HSIC_statistics(np.dot(Kc, Lc), Mc)
#     critical_value_XY_Z = wildbootstrap_test(k_list, resultsHSIC_XY_Z, alpha=0.05, n_bootstrap=300, test_type=2)
#     reject_XY_Z = int(resultsHSIC_XY_Z > critical_value_XY_Z)
#
#     resultsHSIC_XZ_Y = compute_HSIC_statistics(np.dot(Kc, Mc), Lc)
#     critical_value_XZ_Y = wildbootstrap_test(k_list, resultsHSIC_XZ_Y, alpha=0.05, n_bootstrap=300, test_type=2)
#     reject_XZ_Y = int(resultsHSIC_XZ_Y > critical_value_XZ_Y)
#
#     resultsHSIC_YZ_X = compute_HSIC_statistics(np.dot(Lc, Mc), Kc)
#     critical_value_YZ_X = wildbootstrap_test(k_list, resultsHSIC_YZ_X, alpha=0.05, n_bootstrap=300, test_type=2)
#     reject_YZ_X = int(resultsHSIC_YZ_X > critical_value_YZ_X)
#
#     return [reject_XY, reject_XZ, reject_YZ], [reject_XY_Z, reject_XZ_Y, reject_YZ_X]
