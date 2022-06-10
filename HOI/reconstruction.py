import numpy as np
from util import combinations_tuple
from tests import joint_independence_test
import copy


def reconstruct_HOI(group_arr, iterable, stop_after_2=False, n_perms=5000, alpha=0.05):
    """
    To do:
    1. return (var1, var2), reject/accept, test_statistic, critical_value, normalised_weights
    2. preprocess, combinations, load stats, load tests
    3. return rejects, dhsics, critical values,
    """

    K = len(iterable)  # number of total variables (17 goals, 76 targets)
    edges = {}  # initialize dictionary with edges according to dependencies found
    Adj2 = np.zeros((K, K))  # initialize KxK adjacency matrix for d=2
    d = 2  # initial number of variables for dHSIC
    e = 0
    # individual dHSICs for normalisation later
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
            dHSIC_val, reject = joint_independence_test(k_list, n_perms, alpha)

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

        if stop_after_2:
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