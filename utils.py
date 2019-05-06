import networkx as nx
import matplotlib.pyplot as plt
import collections
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from IPython.display import Image
import numpy as np
import sys

Node = collections.namedtuple('Node', ['id', 'inputs', 'type'])

MAXINT32 = 4294967295 // 10

def get_graph(args):
    n = args['n']
    k = args['k']
    p = args['p']
    m = args['m']

    if args['graph_mode'] == 'WS':
        graph = nx.generators.connected_watts_strogatz_graph(n=n, k=k, p=p, seed=np.random.randint(0, MAXINT32))
    elif args['graph_mode'] == 'ER':
        graph = nx.generators.erdos_renyi_graph(n=n, p=p, seed=np.random.randint(0, MAXINT32))
    elif args['graph_mode'] == 'BA':
        graph = nx.barabasi_albert_graph(n=n, m=m, seed=np.random.randint(0, MAXINT32))

    dgraph = nx.DiGraph()
    dgraph.add_nodes_from(graph.nodes)
    dgraph.add_edges_from(graph.edges)

    in_node = []
    out_node = []
    for indeg, outdeg in zip(dgraph.in_degree, dgraph.out_degree):
        if indeg[1] == 0:
            in_node.append(indeg[0])
        elif outdeg[1] == 0:
            out_node.append(outdeg[0])

    sorted = list(nx.topological_sort(dgraph))

    # write_dot(dgraph, 'test.dot')

    # pos = nx.drawing.nx_pydot.pydot_layout(dgraph)
    # png_str = d.create_png()
    # Image(data=png_str)
    # pos = graphviz_layout(dgraph, prog='dot')
    # nx.draw(dgraph, pos, with_labels=True, arrows=True, prog='dot')
    # plt.savefig('nx_test.png')

    # plt.show()

    nx.drawing.nx_agraph.view_pygraphviz(dgraph)

    return dgraph, sorted, in_node, out_node


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
