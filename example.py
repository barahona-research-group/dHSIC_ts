from data import iid_example
import matplotlib.pyplot as plt
import numpy as np

mn_dHcor = iid_example(mode='multi-normal')

plt.plot(np.linspace(0, 1, 201), np.sqrt(mn_dHcor))
# plt.title("Multivariate normal", fontdict={'fontsize': 18})
# plt.xlabel('s', fontdict={'fontsize': 18})
# plt.ylabel(r'$\widehat{dHcor}_n$', fontdict={'fontsize': 18})
plt.show()
