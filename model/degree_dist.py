from genrandgraphNX import gen_ER_graph, gen_BA_graph, gen_WS_graph
import matplotlib.pyplot as plt
import networkx as nx


def deg_dist(G):
    '''
    Calculates the degree distribution of one example of each type of random graph
    :param G: Graph
    :return: unique degrees, degree counts
    '''

    all_degrees = dict(G.degree())
    all_degree_values = all_degrees.values()
    unique_degrees = list(set(all_degree_values))
    unique_degrees.sort()
    count_of_degrees = []

    for i in unique_degrees:
        c = list(all_degree_values).count(i)
        count_of_degrees.append(c)

    print(unique_degrees)
    print(count_of_degrees)

    return unique_degrees, count_of_degrees


def main():
    '''
    Calls the generating and degree distribution funcs and does the plots
    :return: None
    '''

    G1 = gen_ER_graph(0.2, n=10000)
    G2 = gen_BA_graph(10, n=10000)
    G3 = gen_WS_graph(40, 0.75, 10000)

    u1, c1 = deg_dist(G1)
    u2, c2 = deg_dist(G2)
    u3, c3 = deg_dist(G3)

    fig1 = plt.figure(figsize=(5, 3))
    plt.plot(u1, c1, 'ro-', label='p=0.2, n=10000')
    plt.xlabel('Degrees')
    plt.ylabel('Number of nodes')
    plt.title('Degree Distribution')
    plt.legend()
    plt.show()

    fig2 = plt.figure(figsize=(5, 3))
    plt.plot(u2, c2, 'bo-', label='m=10, n=10000')
    plt.xlabel('Degrees')
    plt.ylabel('Number of nodes')
    plt.title('Degree Distribution')
    plt.legend()
    plt.show()

    fig3 = plt.figure(figsize=(5, 3))
    plt.plot(u3, c3, 'go-', label='k=40, p=0.75, n=10000')
    plt.xlabel('Degrees')
    plt.ylabel('Number of nodes')
    plt.title('Degree Distribution')
    plt.legend()
    plt.show()

main()


