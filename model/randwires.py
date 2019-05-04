from tensorflow.python import keras
import tensorflow as tf
from tensorflow.python.keras.layers import SeparableConv2D, Dense, Softmax, BatchNormalization, \
    GlobalAveragePooling2D, ReLU, Activation


class RandLayer(keras.layers.Layer):
    pass


class Triplet(keras.layers.Layer):
    def __init__(self, channels, name, activation=None, random=False, input_shape=None, strides=(1, 1),
                 kernel_size=(3, 3), N=32):
        super(Triplet, self).__init__(name=name)

        if activation is None or activation == 'linear':
            self.activation = Activation('linear')
        elif activation == 'relu':
            self.activation = Activation('relu')

        if random:
            self.conv = RandLayer(channels, N, activation)
        else:
            if input_shape is None:
                self.conv = SeparableConv2D(filters=channels, kernel_size=kernel_size, strides=strides)
            else:  # Only in the first layer
                self.conv = SeparableConv2D(filters=channels, kernel_size=kernel_size, strides=strides,
                                            input_shape=input_shape)

        self.bn = BatchNormalization()

    def call(self, inputs):
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
        self.softmax = Softmax()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class RandWireNN(keras.Model):
    def __init__(self, args, input_shape, n_classes):
        super(RandWireNN, self).__init__(name='randomly_wired_network')

        self.n_classes = n_classes

        if args.regime == 'small':
            self.conv1 = Triplet(channels=args.C // 2, name='conv1', activation=None, random=False,
                                 input_shape=input_shape)
            self.conv2 = Triplet(channels=args.C, name='conv2', activation='relu', random=False)
            self.conv3 = Triplet(channels=args.C, name='conv3', activation='relu', random=False, N=args.N)
            self.conv4 = Triplet(channels=2 * args.C, name='conv4', activation='relu', random=False, N=args.N)
            self.conv5 = Triplet(channels=4 * args.C, name='conv5', activation='relu', random=False, N=args.N)
        elif args.regime == 'regular':
            self.conv1 = Triplet(channels=args.C // 2, name='conv1', activation=None, random=False, strides=(2,2),
                                 input_shape=input_shape)
            self.conv2 = Triplet(channels=args.C, name='conv2', activation='relu', random=False, N=args.N // 2)
            self.conv3 = Triplet(channels=2 * args.C, name='conv3', activation='relu', random=False, N=args.N)
            self.conv4 = Triplet(channels=4 * args.C, name='conv4', activation='relu', random=False, N=args.N)
            self.conv5 = Triplet(channels=8 * args.C, name='conv5', activation='relu', random=False, N=args.N)

        self.classifier = Classifier(n_classes=n_classes)

    # noinspection PyCallingNonCallable
    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.classifier(x)

        return x
