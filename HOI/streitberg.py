import numpy as np


def centering(K):
    n = length = np.shape(K)[0]
    H = np.eye(n) - 1/n * np.ones(n)
    Kc = H @ K @ H
    return Kc


def first_centering_streitberg_4(K, L, M, N):
    length = np.shape(K)[0]
    Kc = centering(K)
    Lc = centering(L)
    Mc = centering(M)
    Nc = centering(N)
    si = 0
    for i in range(length):
        for j in range(length):
            for k in range(length):
                for l in range(length):
                    p1234_p1234 = Kc[i, k] * Lc[i, k] * Mc[i, k] * Nc[i, k]
                    p1234_p12p34 = Kc[i, k] * Lc[i, k] * Mc[i, j] * Nc[i, j]
                    p1234_p13p24 = Kc[i, k] * Lc[i, j] * Mc[i, k] * Nc[i, j]
                    p1234_p14p23 = Kc[i, k] * Lc[i, j] * Mc[i, j] * Nc[i, k]
                    p12p34_p12p34 = Kc[i, k] * Lc[i, k] * Mc[j, l] * Nc[j, l]
                    p12p34_p13p24 = Kc[i, k] * Lc[i, l] * Mc[j, k] * Nc[j, l]
                    p12p34_p14p23 = Kc[i, k] * Lc[i, l] * Mc[j, l] * Nc[j, k]
                    p13p24_p13p24 = Kc[i, k] * Lc[j, l] * Mc[i, k] * Nc[j, l]
                    p13p24_p14p23 = Kc[i, k] * Lc[j, l] * Mc[i, l] * Nc[j, k]
                    p14p23_p14p23 = Kc[i, k] * Lc[j, l] * Mc[j, l] * Nc[i, k]
                    si += length**2 * p1234_p1234 - 2 * length * p1234_p12p34 - 2 * length * p1234_p13p24 - 2 * length * p1234_p14p23 + p12p34_p12p34 + 2 * p12p34_p13p24 + 2 * p12p34_p14p23+ p13p24_p13p24 + 2 * p13p24_p14p23 + p14p23_p14p23
    return 1/(length ** 4) * si


def t1_t1(K, L, M, N):
    length = np.shape(K)[0]
    sums = 0
    for i in range(length):
        for j in range(length):
            for k in range(length):
                for l in range(length):
                    for m in range(length):
                        for n in range(length):
                            for p in range(length):
                                for q in range(length):
                                    sums = sums + K[i, j] * L[i, j] * M[i, j] * N[i, j]
    sums = 1 / (length ** 8) * sums
    return sums
