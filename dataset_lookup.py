"""
Utility diagnostic plotting distribution characteristics over graph occurrences dynamically.
"""

from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np

dataset = TUDataset(root='./data/', name='MUTAG')
n_tab = [data.num_nodes for data in dataset]
e_tab = [data.num_edges/2 for data in dataset]

unique_n, counts_n = np.unique(n_tab, return_counts=True)
unique_e, counts_e = np.unique(e_tab, return_counts=True)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))


axs[0].bar(unique_n, counts_n, width=0.8, align='center')

axs[0].set_xlabel('Node number (N)')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Distribution of N (number of nodes) in the MUTAG Dataset')

axs[0].set_xticks(unique_n)


axs[1].bar(unique_e, counts_e, width=0.8, align='center')

axs[1].set_xlabel('Node edges (E)')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Distribution of E (number of edges) in the MUTAG Dataset')

axs[1].set_xticks(unique_e)
plt.show()
