from tensorflow.python import keras
import tensorflow as tf
from tensorflow.python.keras.layers import SeparableConv2D, Dense, Softmax, BatchNormalization, \
    GlobalAveragePooling2D, ReLU, Activation, Dropout

import networkx as nx
from utils import get_graph


class Aggregation(keras.layers.Layer):

    def __init__(self, input_dim):
        super(Aggregation, self).__init__()
        self.w = self.add_weight(shape=(input_dim, 1, 1, 1, 1),
                                 initializer='lecun_normal',
                                 trainable=True)
        self.sigmoid = keras.activations.sigmoid

    def call(self, inputs, **kwargs):
        # TODO: debug this plis
        pos_w = self.sigmoid(self.w)
        x = pos_w * inputs
        x = tf.reduce_sum(x, axis=0)
        return x


class RandLayer(keras.layers.Layer):

    def __init__(self, channels, random_args, activation):
        super(RandLayer, self).__init__()

        self.graph, self.graph_order, self.start_node, self.end_node = get_graph(random_args)

        self.triplets = {}
        self.aggregations = {}

        for node in self.graph_order:
            if node in self.start_node:
                self.triplets[node] = Triplet(channels=channels, activation=None, strides=2)
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
                 kernel_size=(3, 3), rand_args=None):
        super(Triplet, self).__init__(name=name)

        if activation is None or activation == 'linear':
            self.activation = Activation('linear')
        elif activation == 'relu':
            self.activation = Activation('relu')

        if random:
            assert rand_args is not None
            self.conv = RandLayer(channels, rand_args, activation)
        else:
            if input_shape is None:
                self.conv = SeparableConv2D(filters=channels, kernel_size=kernel_size, strides=strides, padding='same')
            else:  # Only in the first layer
                self.conv = SeparableConv2D(filters=channels, kernel_size=kernel_size, strides=strides, padding='same',
                                            input_shape=input_shape)

        self.bn = BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.activation(inputs)
        x = self.conv(x)
        x = self.bn(x)
        return x


class Classifier(keras.layers.Layer):
    def __init__(self, n_classes):
        super(Classifier, self).__init__(name='classifier')

        self.conv = SeparableConv2D(filters=1280, kernel_size=(1, 1), activation='relu')
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
        super(RandWireNN, self).__init__(name='randomly_wired_network')

        self.n_classes = n_classes
        self.random_args = {'n': args.N, 'p': args.P, 'k': args.K, 'm': args.M, 'seed': args.seed,
                            'graph_mode': args.graph_mode}
        if args.regime == 'small':
            self.conv1 = Triplet(channels=args.C // 2, name='conv1', activation=None, random=False,
                                 input_shape=input_shape)
            self.conv2 = Triplet(channels=args.C, name='conv2', activation='relu', random=False)
            self.conv3 = Triplet(channels=args.C, name='conv3', activation='relu', random=True,
                                 rand_args=self.random_args)
            self.conv4 = Triplet(channels=2 * args.C, name='conv4', activation='relu', random=True,
                                 rand_args=self.random_args)
            self.conv5 = Triplet(channels=4 * args.C, name='conv5', activation='relu', random=True,
                                 rand_args=self.random_args)
        elif args.regime == 'regular':
            self.conv1 = Triplet(channels=args.C // 2, name='conv1', activation=None, random=False, strides=2,
                                 input_shape=input_shape)
            self.random_args['n'] = self.random_args['n'] // 2
            self.conv2 = Triplet(channels=args.C, name='conv2', activation='relu', random=True,
                                 rand_args=self.random_args)
            self.random_args['n'] = self.random_args['n'] * 2
            self.conv3 = Triplet(channels=2 * args.C, name='conv3', activation='relu', random=True,
                                 rand_args=self.random_args)
            self.conv4 = Triplet(channels=4 * args.C, name='conv4', activation='relu', random=True,
                                 rand_args=self.random_args)
            self.conv5 = Triplet(channels=8 * args.C, name='conv5', activation='relu', random=True,
                                 rand_args=self.random_args)

        self.classifier = Classifier(n_classes=n_classes)

    # noinspection PyCallingNonCallable
    def call(self, inputs, **kwargs):

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.classifier(x)

        return x
