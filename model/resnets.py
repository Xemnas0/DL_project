from tensorflow.python import keras
import tensorflow as tf
from tensorflow.python.keras import applications
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, \
    AveragePooling2D, Flatten, Dense, Input, ZeroPadding2D


class ConvBlock(keras.layers.Layer):

  def __init__(self, kernel_size, filters, stage, block, strides=(2, 2)):

    super(ConvBlock, self).__init__(name='')

    [filters1, filters2, filters3] = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    self.conv2a = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')

    self.bn2a = BatchNormalization(name=bn_name_base + '2a')

    self.conv2b = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')
    self.bn2b = BatchNormalization(name=bn_name_base + '2b')

    self.conv2c = Conv2D( filters3, (1, 1), name=conv_name_base + '2c')
    self.bn2c = BatchNormalization(name=bn_name_base + '2c')

  def call(self, inputs, **kwargs):
    x = self.conv2a(inputs)
    x = self.bn2a(x)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x)

    x += inputs
    return tf.nn.relu(x)



class ResNet(keras.Model):
    def __init__(self, args, input_shape, n_classes):
        super(ResNet, self).__init__(name='ResNet')

        self.n_classes = n_classes
        self.stages = []
        self.input_shape_ = input_shape


        self.zeropadding = ZeroPadding2D(padding=(3, 3), name='conv1_pad')

        # resnet50
        if args.regime == 'small':
            self.conv1 = Conv2D(filters=256, kernel_size=(7, 7), strides=(2, 2),
                                padding='same', name='conv1', kernel_initializer='he_normal')
            self.bn = BatchNormalization()
            self.max_pool = MaxPooling2D((3, 3), strides=(2, 2))

            self.layer_1_a = ConvBlock(kernel_size=3, filters=[64, 64, 256], stage=2, block='a', strides=(1, 1))
            self.layer_1_b = ConvBlock(kernel_size=3, filters=[64, 64, 256], stage=2, block='b', strides=(2, 2))
            self.layer_1_c = ConvBlock(kernel_size=3, filters=[64, 64, 256], stage=2, block='c', strides=(2, 2))

            self.l3a = ConvBlock(kernel_size=3, filters=[128, 128, 512], stage=3, block='a', strides=(2, 2))
            self.l3b = ConvBlock(kernel_size=3, filters=[128, 128, 512], stage=3, block='b', strides=(2, 2))
            self.l3c = ConvBlock(kernel_size=3, filters=[128, 128, 512], stage=3, block='c', strides=(2, 2))
            self.l3d = ConvBlock(kernel_size=3, filters=[128, 128, 512], stage=3, block='d', strides=(2, 2))

            self.l4a = ConvBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block='a', strides=(2, 2))
            self.l4b = ConvBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block='b', strides=(2, 2))
            self.l4c = ConvBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block='c', strides=(2, 2))
            self.l4d = ConvBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block='d', strides=(2, 2))
            self.l4e = ConvBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block='e', strides=(2, 2))
            self.l4f = ConvBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block='f', strides=(2, 2))

            self.l5a = ConvBlock(kernel_size=3, filters=[256, 256, 1024], stage=5, block='a', strides=(2, 2))
            self.l5b = ConvBlock(kernel_size=3, filters=[256, 256, 1024], stage=5, block='b', strides=(2, 2))
            self.l5c = ConvBlock(kernel_size=3, filters=[256, 256, 1024], stage=5, block='c', strides=(2, 2))

            self.stages.append(self.layer_1_a)
            self.stages.append(self.layer_1_b)
            self.stages.append(self.layer_1_c)

            self.stages.append(self.l3a)
            self.stages.append(self.l3b)
            self.stages.append(self.l3c)
            self.stages.append(self.l3d)

            self.stages.append(self.l4a)
            self.stages.append(self.l4b)
            self.stages.append(self.l4c)
            self.stages.append(self.l4d)
            self.stages.append(self.l4e)
            self.stages.append(self.l4f)

            self.stages.append(self.l5a)
            self.stages.append(self.l5b)
            self.stages.append(self.l5c)

        # resnet101
        elif args.regime == 'regular':
            pass

        self.classifier = Classifier(n_classes=n_classes)

    # noinspection PyCallingNonCallable
    def call(self, inputs, **kwargs):

        x = self.zeropadding(inputs)

        x = self.conv1(x)
        x = self.bn(x)
        x = self.max_pool(x)

        for stage in self.stages:
            x = stage(x)

        x = self.classifier(x)

        return x

    def get_ready_model(self):
        return applications.resnet50.ResNet50(weights=None, include_top=True,
                                              input_shape=self.input_shape_)


class Classifier(keras.layers.Layer):
    def __init__(self, n_classes):
        super(Classifier, self).__init__(name='')

        self.avg_pool = AveragePooling2D((7, 7), strides=(7, 7))

        self.flatten = Flatten()
        self.fc = Dense(n_classes)

    def call(self, inputs, **kwargs):
        x = self.avg_pool(inputs)
        x = self.flatten(x)

        return self.fc(x)