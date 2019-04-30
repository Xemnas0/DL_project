import tensorflow as tf

from tensorflow.python.keras.layers import SeparableConv2D, Dense, Softmax, BatchNormalization, \
    GlobalAveragePooling2D


class RandWire(tf.keras.Model):

    def __init__(self, C, input_shape, n_classes):
        super(RandWire, self).__init__(name='randomly_wired_network')

        self.n_classes = n_classes

        self.conv1 = SeparableConv2D(filters=int(C / 2), kernel_size=(3, 3), activation=None,
                                     input_shape=input_shape)
        self.bn1 = BatchNormalization()

        self.conv2 = SeparableConv2D(filters=int(C), kernel_size=(3, 3), activation='relu')
        self.bn2 = BatchNormalization()

        # TODO: change with randomly wired layer
        self.conv3 = SeparableConv2D(filters=int(C), kernel_size=(3, 3), activation='relu')
        self.bn3 = BatchNormalization()
        self.conv4 = SeparableConv2D(filters=int(2 * C), kernel_size=(3, 3), activation='relu')
        self.bn4 = BatchNormalization()
        self.conv5 = SeparableConv2D(filters=int(4 * C), kernel_size=(3, 3), activation='relu')
        self.bn5 = BatchNormalization()
        ##

        self.last_conv = SeparableConv2D(filters=1280, kernel_size=(1, 1), activation='relu')
        self.last_bn = BatchNormalization()
        self.avg_pool = GlobalAveragePooling2D()
        self.fc = Dense(units=n_classes)

        self.softmax = Softmax()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = self.last_conv(x)
        x = self.last_bn(x)
        x = self.avg_pool(x)
        x = self.fc(x)

        x = self.softmax(x)

        return x
