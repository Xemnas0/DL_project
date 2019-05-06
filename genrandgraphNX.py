import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def gen_ER_graph(p, n=None):
    if n is None:
        n = 32
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(gen_rand_pairs(nodes, p))
    return G


def gen_BA_graph(m, n=None):
    if n is None:
        n = 32
    G = nx.empty_graph(m)
    targets = list(range(m))
    repeated_nodes = []
    for source in range(m, n):
        G.add_edges_from(zip([source]*m, targets))
        repeated_nodes.extend(targets)
        repeated_nodes.extend([source] * m)
        targets = gen_rand_subset(repeated_nodes, m)
    return G


def gen_WS_graph(k, p, n=None):
    if n is None:
        n = 32
    G = make_ring_lattice(k, n)
    G = rewire_ws(G, p)
    return G


def make_ring_lattice(k, n):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(gen_adj_edges(nodes, k // 2))
    return G


def gen_rand_pairs(nodes, p):
    for edge in check_pairs(nodes):
        if check_prob(p):
            yield edge


def check_prob(p):
    return np.random.random() < p


def check_pairs(nodes):
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i > j:
                yield u, v


def gen_adj_edges(nodes, halfk):
    n = len(nodes)
    for i, u in enumerate(nodes):
        for j in range(i+1, i+halfk+1):
            v = nodes[j % n]
            yield u, v


def rewire_ws(G, p):
    nodes = set(G)
    for u, v in G.edges():
        if check_prob(p):
            choices = nodes - {u} - set(G[u])
            new_v = np.random.choice(list(choices))
            G.remove_edge(u, v)
            G.add_edge(u, new_v)
    return G


def gen_rand_subset(repeated_nodes, k):
    targets = set()
    while len(targets) < k:
        x = random.choice(repeated_nodes)
        targets.add(x)
        return targets


def main():
    G = gen_ER_graph(0.12)
    nx.draw_circular(G)
    plt.show()

    G = gen_BA_graph(12)
    nx.draw_circular(G, with_labels=True)
    plt.show()

    G = gen_WS_graph(3, 0.6)
    nx.draw_circular(G, with_labels=True)
    plt.show()

main()