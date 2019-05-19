from tensorflow.python import keras
import tensorflow as tf
from tensorflow.python.keras.layers import SeparableConv2D, Dense, Softmax, BatchNormalization, \
    GlobalAveragePooling2D, ReLU, Activation, Dropout

from tensorflow.python.keras.regularizers import l2

import networkx as nx
from utils import get_graph
import numpy as np
import os

WEIGHT_DECAY = 5e-5


class Aggregation(keras.layers.Layer):
    def __init__(self, input_dim):
        """
        Keras Layer representing the node operation
        that aggregates the input of two or more edges
        by computing a weighted sum with positive weights.
        The weights themselves are not positive, but they go
        through a sigmoid before the sum.

        Arguments:
            input_dim: number of incoming edges in the node (in degree).
        """
        super(Aggregation, self).__init__()
        self.w = self.add_weight(shape=(input_dim, 1, 1, 1, 1),
                                 initializer='lecun_normal',
                                 regularizer=l2(WEIGHT_DECAY),
                                 trainable=True)
        self.sigmoid = keras.activations.sigmoid

    def call(self, inputs, **kwargs):
        pos_w = self.sigmoid(self.w)
        x = pos_w * inputs
        x = tf.reduce_sum(x, axis=0)
        return x


class RandLayer(keras.layers.Layer):

    def __init__(self, channels, random_args, activation, N):
        """
        Keras Layer representing the randomly wired layer.
        Generates a graph and the operations for each node.

        Arguments:
            channels: number of filters per node.
            random_args: dict with parameters for the
                generation of the and the stride of the convolution.
            activation: activation function. Usually ReLU.
            N: number of nodes in the layer.
        """

        super(RandLayer, self).__init__()

        self.graph, self.graph_order, self.start_node, self.end_node = get_graph(random_args, N)
        self.triplets = {}
        self.aggregations = {}

        for node in self.graph_order:
            if node in self.start_node:
                self.triplets[node] = Triplet(channels=channels, activation=None, strides=random_args['stride'])
            else:
                in_degree = self.graph.in_degree[node]
                if in_degree > 1:
                    self.aggregations[node] = Aggregation(input_dim=in_degree)
                self.triplets[node] = Triplet(channels=channels, activation=activation)

        self.unweighted_average = tf.reduce_mean

    def call(self, inputs, **kwargs):
        node_results = {}

        for node in self.graph_order:
            if node in self.start_node:
                node_results[node] = self.triplets[node](inputs)
            else:
                parents = list(self.graph.predecessors(node))
                if len(parents) > 1:
                    parents_output = []
                    for parent in parents:
                        parents_output.append(node_results[parent])
                    parents_output = tf.convert_to_tensor(parents_output)
                    output_aggregation = self.aggregations[node](parents_output)
                else:
                    output_aggregation = node_results[parents[0]]
                node_results[node] = self.triplets[node](output_aggregation)

        output_last_nodes = []
        for node in self.end_node:
            output_last_nodes.append(node_results[node])
        output_last_nodes = tf.convert_to_tensor(output_last_nodes)
        final_output = self.unweighted_average(output_last_nodes, axis=0)

        return final_output


class Triplet(keras.layers.Layer):
    def __init__(self, channels, name=None, activation=None, random=False, input_shape=None, strides=(1, 1),
                 kernel_size=(3, 3), N=None, rand_args=None):
        """
        Keras layer that encapsule the triplet ReLU-Conv-BN.
        The Conv component can either be a single separable
        convolution or a RandLayer, depending on the 'random'
        parameter.

        Arguments:
            channels: number of filters.
            name: layer's name. Useful for the summary.
            activation: activation function. Usually ReLU.
            random: bool defining what type of layer this is.
            input_shape: shape of the input, e.g. (None, height, width, channels).
                Necessary only for the first layer of the network.
            strides: stride of the convolution. Only used if random=False.
            kernel_size: size of the filter.
            N: number of nodes in the random layer.
            rand_args: hyperparameters of the random layer.
        """

        super(Triplet, self).__init__(name=name)

        if activation is None or activation == 'linear':
            self.activation = Activation('linear')
        elif activation == 'relu':
            self.activation = Activation('relu')

        if random:
            assert rand_args is not None
            self.conv = RandLayer(channels, rand_args, activation, N)
        else:
            if input_shape is None:
                self.conv = SeparableConv2D(filters=channels, kernel_size=kernel_size,
                                            kernel_regularizer=l2(WEIGHT_DECAY),
                                            strides=strides, padding='same')
            else:  # Only in the first layer
                self.conv = SeparableConv2D(filters=channels, kernel_size=kernel_size,
                                            kernel_regularizer=l2(WEIGHT_DECAY), strides=strides, padding='same',
                                            input_shape=input_shape)

        self.bn = BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.activation(inputs)
        x = self.conv(x)
        x = self.bn(x)
        return x


class Classifier(keras.layers.Layer):
    def __init__(self, n_classes):
        """
        Last layer of the model outputting the probabilities
        for each class.
        Performs a SeparableConv2D 1x1, BN, GlobalAveragePooling2D,
        FC, Dropout, Softmax.

        Arguments:
            n_classes: output size.
        """
        super(Classifier, self).__init__(name='classifier')

        self.conv = SeparableConv2D(filters=1280, kernel_size=(1, 1), kernel_regularizer=l2(WEIGHT_DECAY),
                                    activation='relu')
        self.bn = BatchNormalization()
        self.avg_pool = GlobalAveragePooling2D()
        self.fc = Dense(units=n_classes)
        self.droput = Dropout(0.2)
        self.softmax = Softmax()

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.droput(x)
        x = self.softmax(x)
        return x


class RandWireNN(keras.Model):
    def __init__(self, args, input_shape, n_classes):
        """
        Randomly wired Keras model.

        Arguments:
            args: namespace with parameters for instantiating the model.
            input_shape: shape of the input, e.g. (None, height, width, channels).
            n_classes: size of the output.
        """
        super(RandWireNN, self).__init__(name='randomly_wired_network')

        self.n_classes = n_classes
        self.random_args = {'n': args.N, 'p': args.P, 'k': args.K, 'm': args.M, 'seed': args.seed,
                            'regime': args.regime,
                            'graph_mode': args.graph_mode,
                            'stride': args.stride}

        self.stages = []
        self.regime = args.regime

        self.conv1 = Triplet(channels=args.C // 2, name='conv1', activation=None, random=False, strides=2,
                             input_shape=input_shape)

        if args.regime == 'small':
            self.conv2 = Triplet(channels=args.C, name='conv2', activation='relu', random=False, strides=args.stride)
        elif args.regime == 'regular':
            self.random_args['stride'] = 2
            self.conv2 = Triplet(channels=args.C, name='conv2', activation='relu', random=True, N=args.N // 2,
                                 rand_args=self.random_args)
            self.random_args['stride'] = args.stride

        channels = args.C if args.regime == 'small' else 2 * args.C
        for stage in range(args.stages):
            self.stages.append(
                Triplet(channels=channels, name='conv{}'.format(3 + stage), activation='relu', random=True, N=args.N,
                        rand_args=self.random_args))
            channels *= 2

        self.classifier = Classifier(n_classes=n_classes)

    def call(self, inputs, **kwargs):

        x = self.conv1(inputs)
        x = self.conv2(x)

        for stage in self.stages:
            x = stage(x)

        x = self.classifier(x)

        return x

    def save_graph_image(self, path=''):
        """
        Saves a very beautiful visualization of
        the graph in a .pdf file in the provided path.
        This function requires networkx and pygraphviz.

        Arguments:
            path: destination of the graph image.
                If the path does not exist, it is
                automagically created.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        dgraph = nx.DiGraph()

        dgraph.add_node('conv1', shape='rectangle')

        random_layers = list(self.stages)
        if self.regime == 'regular':
            random_layers.insert(0, self.conv2)

        tot_nodes = 0
        n_stages = len(random_layers)
        for i, stage in enumerate(random_layers):
            stage_graph = stage.conv.graph
            start_node = stage.conv.start_node
            end_node = stage.conv.end_node

            n_nodes = len(stage_graph.nodes)
            tot_nodes += n_nodes
            c = tot_nodes - n_nodes

            for node in stage_graph.nodes:
                dgraph.add_node(node + c, label='', shape='circle')

            for u, v in stage_graph.edges:
                dgraph.add_edge(u + c, v + c)

            if i == 0:
                for node in start_node:
                    dgraph.add_edge('conv1', node + c)
            elif i > 0:
                for node in start_node:
                    dgraph.add_edge('output{}'.format(i - 1), node + c)

            dgraph.add_node('output{}'.format(i), color='orange', style='filled', label='')

            for node in end_node:
                dgraph.add_edge(node + c, 'output{}'.format(i))

            if i == n_stages - 1:
                dgraph.add_node('classifier', shape='rectangle')
                dgraph.add_edge('output{}'.format(i), 'classifier')

        pygraphviz_graph = nx.drawing.nx_agraph.to_agraph(dgraph)
        tot_nodes = 0
        for i, stage in enumerate(random_layers):
            stage_graph = stage.conv.graph
            start_node = stage.conv.start_node
            end_node = stage.conv.end_node

            n_nodes = len(stage_graph.nodes)
            tot_nodes += n_nodes
            c = tot_nodes - n_nodes

            pygraphviz_graph.add_subgraph(list(np.array(start_node) + c), rank='same')
            pygraphviz_graph.add_subgraph(list(np.array(end_node) + c), rank='same')

        filename = self.get_filename() + '.pdf'

        pygraphviz_graph.draw(path=path + filename, prog='dot')
        print('Model printed in file.')

    def get_filename(self):
        """
        Returns:
            a string that summarizes the specifics of the model.
        """
        filename = None
        n_stages = len(self.stages)
        if self.random_args['graph_mode'] == 'WS':
            filename = 'WS_{4}_stages{5}_N{0}_K{1}_P{2}_seed{3}'.format(self.random_args['n'],
                                                                        self.random_args['k'],
                                                                        int(self.random_args['p'] * 100),
                                                                        self.random_args['seed'],
                                                                        self.random_args['regime'],
                                                                        n_stages
                                                                        )
        elif self.random_args['graph_mode'] == 'ER':
            filename = 'ER_{3}_stages{4}_N{0}_P{1}_seed{2}'.format(
                self.random_args['n'],
                int(self.random_args['p'] * 100),
                self.random_args['seed'],
                self.random_args['regime'],
                n_stages
            )
        elif self.random_args['graph_mode'] == 'BA':
            filename = 'BA_{3}_stages{4}_N{0}_M{1}_seed{2}'.format(
                self.random_args['n'],
                self.random_args['m'],
                self.random_args['seed'],
                self.random_args['regime'],
                n_stages
            )
        return filename
