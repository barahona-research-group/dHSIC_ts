from HOI.statistics import compute_dHSIC_statistics, compute_lancaster_statistics


def dHcor2_norm(k_list):
    d = len(k_list)
    dHSIC_all = compute_dHSIC_statistics(k_list)
    denominator = 1
    for i in range(d):
        denominator *= compute_dHSIC_statistics([k_list[i] for _ in range(d)])

    dHcor2_norm = dHSIC_all / denominator ** (1 / d)
    return dHcor2_norm


def dHcor2_std(k_list):
    d = len(k_list)
    dHSIC_all = compute_dHSIC_statistics(k_list)
    denominator = 1
    for i in range(d):
        denominator *= compute_dHSIC_statistics([k_list[i] for _ in range(2)])

    dHcor2_std = dHSIC_all / denominator ** (1 / 2)
    return dHcor2_std


def lcor2_norm(k_list):
    d = len(k_list)
    lancaster_all = compute_lancaster_statistics(k_list)
    denominator = 1
    for i in range(d):
        denominator *= compute_lancaster_statistics([k_list[i] for _ in range(d)])

    lcor2_norm = lancaster_all / denominator ** (1 / d)
    return lcor2_norm


def lcor2_std(k_list):
    d = len(k_list)
    lancaster_all = compute_lancaster_statistics(k_list)
    denominator = 1
    for i in range(d):
        denominator *= compute_dHSIC_statistics([k_list[i] for _ in range(2)])

    lcor2_std = lancaster_all / denominator ** (1 / 2)
    return lcor2_std