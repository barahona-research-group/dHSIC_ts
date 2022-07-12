from tqdm import tqdm
import numpy as np
import pandas as pd

from HOI.preprocessings import compute_kernel
from HOI.tests import test_independence
from examples.synthetic_data import make_iid_example


def test_power():
    power = {}
    for d in tqdm(np.arange(0.1, 1, 0.1)):
        rejects = 0
        for i in np.arange(100):
            df = make_iid_example('higher-order', s=d, n_sample=100)
            data_dict, kernel_dict = compute_kernel(df)
            _, _, reject = test_independence([kernel_dict['d1'], kernel_dict['d2'], kernel_dict['d3']],
                                             [data_dict['d1'], data_dict['d2'], data_dict['d3']],
                                             mode='iid', n_perms=5000, alpha=0.05)
            rejects = rejects + reject
        power[str(d)] = rejects / 100
    return power
