# Perform the independence test
import numpy as np
import copy


def combinations_tuple(iterable, r):
    """
    e.g.: combinations('ABCD', 2) --> AB AC AD BC BD CD
          combinations(range(4), 3) --> 012 013 023 123
    """
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


def compute_dHSIC(k_list):
    """
    Computes the dHSIC statistic
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


def dHSIC_permutation_MC(k_list, n_samples, n_nodes, stat_found, n_perms=5000, alpha=0.05):
    """
    Approximates the null distribution by permutating all variables. Using Monte Carlo approximation.
    """
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


def joint_independence_test_MC(k_list, n_perms=5000, alpha=0.05):
    """
    Performs the independence test with HSIC and returns an accept or reject statement

    Inputs:
    k_list: list of Kernel matrices for each variable, each having dimensions (n_samples, n_samples)
    n_perms: number of permutations performed when bootstrapping the null
    alpha: rejection threshold of the test

    Returns:
    reject: 1 if null rejected, 0 if null not rejected

    """
    n_nodes = len(k_list)
    n_samples = k_list[0].shape[0]

    # statistic and threshold
    stat = compute_dHSIC(k_list)
    critical_value = dHSIC_permutation_MC(k_list, n_samples, n_nodes, stat)

    reject = int(stat > critical_value)

    return stat, reject


def compute_all_HSIC(data, num_var):
    """Compute """
    HSIC_dict = {}
    # compute HSIC(X,X)....HSIC(X,X,X,X,X) in one fn
    for i in range(2, 5+1):
        HSIC_dict[str(i)] = []
        HSICs = []
        for j in np.arange(num_var):
            k_list = [list([data[j] for _ in range(i)])]
            HSIC = compute_dHSIC(k_list)
            HSICs.append(HSIC)
        HSIC_dict[str(i)] = HSICs

    return HSIC_dict


def dHSIC_links_MC_norm(group_arr, iterable, stop_after_2=False, n_perms=5000, alpha=0.05):
    K = len(iterable)  # number of total variables (17 goals, 76 targets)
    edges = {}  # initialize dictionary with edges according to dependencies found
    Adj2 = np.zeros((K, K))  # initialize KxK adjacency matrix for d=2
    d = 2  # initial number of variables for dHSIC
    e = 0
    HSICs_all = compute_all_HSIC(group_arr, len(iterable))

    indexes = np.arange(K)  # create vector corresponding to indexes of iterable
    # find all possible d-combinations of indexes without order
    g_combinations = list(combinations_tuple(indexes, d))

    weights = {}
    # iterate until no possible combinations of independent variables are left
    while len(g_combinations) > 0:
        print("combinations: ", d)
        print("number of combinations available: ", len(g_combinations))

        f = 0
        hsic_found = {}  # initialize dictionary with decision rule for each d-combination considered
        # iterate over all combinations considered
        for comb in g_combinations:
            # create k_list[i] = Kernel from observed data for variable comb[i]
            k_list = []
            for i in range(d):
                k_list.append(group_arr[comb[i]])

            # test joint independence: if reject H0, reject=1 (dependency found)
            dHSIC_val, reject = joint_independence_test_MC(k_list, n_perms, alpha)

            hsic_found[comb] = reject
            if reject == 1:
                e += 1
                f += 1
                edges[e] = tuple(iterable[i] for i in comb)  # add edge to graph according to dependency found

                # COMPUTE NORMALISATION
                HSIC_XX = HSICs_all[str(d)]
                for n in range(d):
                    mult = 1
                    mult = mult * HSIC_XX[comb[n]]
                HSIC_norm = dHSIC_val / mult ** (1 / d)

                if d == 2:
                    Adj2[comb[0], comb[1]] = HSIC_norm
                    Adj2[comb[1], comb[0]] = HSIC_norm

                else:
                    weights[str(d)][str(comb)] = HSIC_norm

        print("Edges found with ", d, "nodes: ", f)

        if stop_after_2 == True:
            break

        d += 1  # update d
        if d == K + 1:
            break  # stop iteration if d is greater than available variables

        # Find possible d-combinations of iterable. Note that if a dependency has already been found
        # among elements of a combination <d, then we should not consider the combinations involving
        # these elements
        g_combinations_all = list(combinations_tuple(indexes, d))
        g_combinations = copy.deepcopy(g_combinations_all)

        for comb_n in g_combinations_all:
            # consider all possible sub-combinations of d-1 elements in each comb of g_combinations_all
            gg = list(combinations_tuple(comb_n, d - 1))
            for l in range(len(gg)):
                # for each sub_combination a dependency among its elements has already been found if
                # that combination is not in hsic_found (so was already not considered in the previous
                # step), or if it is but has value = 1 (there was a dependency only for the joint dist
                # of all d-1 elements)
                if (gg[l] in hsic_found and hsic_found[gg[l]] == 1) or (gg[l] not in hsic_found):
                    g_combinations.remove(comb_n)  # do not consider such combination
                    break

    return edges, Adj2, weights