import networkx as nx
import matplotlib.pyplot as plt
import collections
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from IPython.display import Image
import numpy as np
import sys
# import pygraphviz as pgv
from genrandgraphNX import gen_BA_graph, gen_ER_graph, gen_WS_graph

Node = collections.namedtuple('Node', ['id', 'inputs', 'type'])

MAXINT32 = 4294967295 // 10


def get_graph(args, n):
    """
    Generates a networkx directed acyclic graph.

    Arguments:
        args: dict with hyperparameters for the generation
            of the graph.
        n: number of nodes of the graph.

    Returns:
        Networkx directed graph, nodes sorted with topological sort,
        list of input nodes, list of output nodes.
    """
    k = args['k']
    p = args['p']
    m = args['m']

    if args['graph_mode'] == 'WS':
        graph = gen_WS_graph(k=k, p=p, n=n, seed=np.random.randint(0, MAXINT32))
        # nx.generators.connected_watts_strogatz_graph(n=n, k=k, p=p, seed=np.random.randint(0, MAXINT32))
    elif args['graph_mode'] == 'ER':
        graph = gen_ER_graph(p=p, n=n, seed=np.random.randint(0, MAXINT32))
        # nx.generators.erdos_renyi_graph(n=n, p=p, seed=np.random.randint(0, MAXINT32))
    elif args['graph_mode'] == 'BA':
        graph = gen_BA_graph(m=m, n=n, seed=np.random.randint(0, MAXINT32))
        # nx.barabasi_albert_graph(n=n, m=m, seed=np.random.randint(0, MAXINT32))

    dgraph = nx.DiGraph()
    dgraph.add_nodes_from(graph.nodes)
    dgraph.add_edges_from(graph.edges)

    in_node = []
    out_node = []
    for indeg, outdeg in zip(dgraph.in_degree, dgraph.out_degree):
        if indeg[1] == 0:
            in_node.append(indeg[0])
        if outdeg[1] == 0:
            out_node.append(outdeg[0])

    sorted = list(nx.topological_sort(dgraph))

    # pygraphviz_graph = nx.drawing.nx_agraph.to_agraph(dgraph)
    # pygraphviz_graph.add_subgraph(in_node, rank='same')
    # pygraphviz_graph.add_subgraph(out_node, rank='same')
    # pygraphviz_graph.draw(path='graph_image.png', prog='dot')

    return dgraph, sorted, in_node, out_node


def plot(train, val, metric, title, save_name=None):
    # fig config
    fig = plt.figure()
    plt.grid(True)
    epochs = np.arange(0, len(train), 1)

    plt.title(title)
    plt.plot(epochs, train, color='r')
    plt.plot(epochs, val, color='b')
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.legend(["train {}".format(metric), "validation {}".format(metric)])

    if save_name is None:
        plt.show()
    else:
        fig.savefig(save_name)
        plt.close(fig)


if __name__ == '__main__':
    n = 16
    K = 4
    p = 0.75
    graph = nx.generators.connected_watts_strogatz_graph(n=n, k=K, p=p)

    nx.draw(graph)
    plt.show()

    dgraph = nx.DiGraph()
    dgraph.add_nodes_from(graph.nodes)
    dgraph.add_edges_from(graph.edges)

    # dgraph = graph.to_directed()
    nx.draw(dgraph)
    plt.show()
    in_node = []
    out_node = []
    for indeg, outdeg in zip(dgraph.in_degree, dgraph.out_degree):
        if indeg[1] == 0:
            in_node.append(indeg[0])
        elif outdeg[1] == 0:
            out_node.append(outdeg[0])

    sorted = list(nx.topological_sort(dgraph))
