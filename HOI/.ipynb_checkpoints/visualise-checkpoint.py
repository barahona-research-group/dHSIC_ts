# %load_ext autoreload
# %autoreload 2
import hypernetx as hnx
from hypernetx.algorithms.s_centrality_measures import *
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np


#
#
# def draw_HG(edges):
#     HG = hnx.Hypergraph(edges)
#     hnx.draw(HG, with_edge_labels=False)
#
#
# def draw_restrictedHG(edges, num):
#     restricted_HG = HG.restrict_to_edges(list(e for e in edges if len(edges[e]) == num))
#     hnx.draw(restricted_HG, pos=nx.spring_layout(restricted_HG), with_edge_labels=False)
#
#
# def draw_G(adj):
#     # graph with only 2-way dependencies
#     G = nx.Graph(adj)  # define graph
#     pos2_li = nx.circular_layout(G)  # compute graph layout
#     # plot
#     plot_network(G, pos2_li, n_labels, 'Low Income', weights=False, eigenvector_centrality=True)


def plot_whole_hx(H, cmaps_list, name, shortname):
    """Plot hypergraphs with edge colors from different colormaps according to their sizes"""
    alpha = 0.5
    sizes = np.array([len(e) for e in H.edges()])

    max_s = sizes.max()

    counts = [np.count_nonzero(sizes == i + 2) for i in range(max_s - 1)]

    norms = []
    cmaps = []
    mids = []
    for i in range(max_s - 1):
        norms.append(plt.Normalize(0, 3 * counts[i]))
        cmaps.append(cmaps_list[i])
        mids.append(round(3 * counts[i] / 2))

    colour_edges = []
    js = [0 for i in range(max_s - 1)]
    for i in range(len(sizes)):
        edge_len = sizes[i]
        norm = norms[edge_len - 2]
        colmap = cmaps[edge_len - 2]
        mid = mids[edge_len - 2]
        js[edge_len - 2] += 1

        if edge_len == 2:
            col_list = list(colmap(norm(mid + js[edge_len - 2])))
            col_list[-1] = alpha
            col_alpha = tuple(col_list)
            colour_edges.append(col_alpha)
        else:
            colour_edges.append(colmap(norm(mid + js[edge_len - 2])))

    plt.figure(figsize=(10, 10))
    plt.tight_layout()
    plt.title(name, fontdict={'fontsize': 30})

    hnx.draw(H, with_edge_labels=False, edges_kwargs={'edgecolors': colour_edges},
             node_labels_kwargs={'fontsize': 16}, pos=nx.circular_layout(H))
    plt.savefig('/Users/saravallejomengod/MathsYear4/M4R/utils/hx/{}_ALL.png'.format(shortname), format='png',
                bbox_inches='tight')


# H_af = hnx.Hypergraph(edges_af)
# plot_whole_hx(H_af, cmaps_list, "Africa", "AF")

# def get_adjacency(H):
#     """ Construct weighted adjacency matrix for HyperGraph H
#
#     Arguments
#     H : Hypernetx hypergraph object
#
#     """
#
#     incidence = H.incidence_matrix().toarray()
#
#     # hyperedge adjacency matrix
#     C = np.matmul(incidence.T, incidence)
#     A = np.matmul(incidence, incidence.T)
#
#     R = np.matmul(incidence, np.matmul(np.diag(np.diag(C)), incidence.T))
#
#     # defining transition matrix
#     adj = R - A
#     np.fill_diagonal(adj, 0)
#
#     return adj


def compute_and_plot_eigenvector_hypergraph(g, pos, n_labels, weight=True):
    plt.figure(figsize=(10, 6))
    g.edges(data=True)
    if weight == True:
        ec = nx.eigenvector_centrality(g, weight='weight')
    else:
        ec = nx.eigenvector_centrality(g)
    colors = list(ec.values())

    cmap = plt.cm.Blues
    vmin = min(colors)
    vmax = max(colors)
    nx.draw(g, pos=pos, node_size=800, node_color=colors, cmap=cmap, vmin=vmin, vmax=vmax)
    labels = nx.draw_networkx_labels(g, pos, labels=n_labels)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm)
    plt.show()

    return ec


colors_SDG = {1:(229/255, 36/255, 59/255), 2:(221/255, 166/255, 58/255), 3:(76/255, 159/255, 56/255),
              4: (197/255, 25/255, 45/255), 5:(1, 58/255, 33/255), 6: (38/255, 189/255, 226/255),
              7: (252/255,195/255, 11/255), 8: (162/255,25/255, 66/255), 9: (253/255, 105/255, 37/255),
              10: (221/255, 19/255, 103/255), 11: (253/255, 157/255, 36/255), 12:(191/255, 139/255, 46/255),
              13: (63/255, 126/255, 68/255), 14: (10/255, 151/255, 217/255), 15: (86/255, 192/255, 43/255),
              16: (0, 104/255, 157/255), 17:(25/255, 72/255, 106/255) }


def plot_eig_centralities_reorg(G, n_labels, name, shortname, colors=None):
    if colors is None:
        colors = colors_SDG
    degree = nx.eigenvector_centrality(G, weight='weight')

    num = len(n_labels)

    values = list(degree.values())
    values_reorg = np.zeros(17)
    for j in range(num):
        sdg_ind = int(n_labels[j])
        values_reorg[sdg_ind - 1] = values[j]

    labels = []
    for i in range(17):
        labels.append('{}'.format(i + 1))

    colors_list = []
    for i in range(17):
        colors_list.append(colors[i + 1])

    width = 0.94
    plt.figure(figsize=(18, 10))
    plt.tight_layout()

    plt.bar(x=labels, height=values_reorg, width=width, color=colors_list, align='center')

    for i, (label, value) in enumerate(zip(labels, values_reorg)):
        img = mpimg.imread('/Users/saravallejomengod/MathsYear4/M4R/utils/SDG_icons/SDG-{}.png'.format(i + 1))
        plt.imshow(img, extent=[i - width / 2, i + width / 2 - 0.01, value - 0.055, value - 0.002], aspect='auto',
                   zorder=2)

    plt.ylim(0, 0.64)
    plt.xlim(-0.5, len(values_reorg) - 0.5)

    plt.xticks(fontsize=18)

    plt.ylabel('Eigenvector centrality', x=-0.06, fontdict={'fontsize': 26})
    plt.xlabel('SDG', fontdict={'fontsize': 26})
    plt.title(name, y=1.05, fontdict={'fontsize': 46})

    plt.savefig('/Users/saravallejomengod/MathsYear4/M4R/utils/hx/{}_EC.png'.format(shortname), format='png',
                bbox_inches='tight')
    plt.show()


def plot_restricted_hypergraph(H, name, shortname):
    plt.figure(figsize=(10, 10))
    plt.tight_layout()
    plt.title(name, fontdict={'fontsize': 30})

    hnx.draw(H, with_edge_labels=False,
             node_labels_kwargs={'fontsize': 16}, pos=nx.circular_layout(H))
    plt.savefig('/Users/saravallejomengod/MathsYear4/M4R/utils/hx/{}_3.png'.format(shortname), format='png',
                bbox_inches='tight')
