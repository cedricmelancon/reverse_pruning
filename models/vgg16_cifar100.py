import tensorflow as tf
from .vgg16 import VGG16
from .layers.bin_conv_2d import BinConv2d
from .layers.bin_conv_2d_2 import BinConv2d2
from .layers.bin_dense import BinDense

__all__ = ['VGG_cifar100']


class VGG_cifar100(VGG16):

    def __init__(self, num_classes=100, depth=18):
        super(VGG_cifar100, self).__init__()
        self.inflate = 1
        self.conv1 = tf.keras.layers.Conv2D(64*self.inflate, kernel_size=3, strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = BinConv2d(64*self.inflate, kernel_size=3, strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        #######################################################

        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        #######################################################

        #########Layer################
        self.conv3 = BinConv2d2(128* self.inflate, kernel_size=3, strides=1, padding='same')
        self.relu3 = tf.keras.layers.ReLU()
        self.conv4 = BinConv2d(128 * self.inflate, kernel_size=3, strides=1, padding='same')
        self.relu4 = tf.keras.layers.ReLU()
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        #######################################################

        #########Layer################
        self.conv5 = BinConv2d2(256*self.inflate, kernel_size=3, strides=1, padding='same')
        self.relu5 = tf.keras.layers.ReLU()
        self.conv6 = BinConv2d(256 * self.inflate, kernel_size=3, strides=1, padding='same')
        self.relu6 = tf.keras.layers.ReLU()
        self.conv7 = BinConv2d(256*self.inflate, kernel_size=3, strides=1, padding='same')
        self.relu7 = tf.keras.layers.ReLU()
        self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        #######################################################

        #########Layer################
        self.conv8 = BinConv2d2(512*self.inflate, kernel_size=3, strides=1, padding='same')
        self.relu8 = tf.keras.layers.ReLU()
        self.conv9 = BinConv2d(512 * self.inflate, kernel_size=3, strides=1, padding='same')
        self.relu9 = tf.keras.layers.ReLU()
        self.conv10 = BinConv2d(512*self.inflate, kernel_size=3, strides=1, padding='same')
        self.relu10 = tf.keras.layers.ReLU()
        self.maxpool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        #######################################################

        #########Layer################
        self.conv11 = BinConv2d2(512*self.inflate, kernel_size=3, strides=1, padding='same')
        self.relu11 = tf.keras.layers.ReLU()
        self.conv12 = BinConv2d2(512*self.inflate, kernel_size=3, strides=1, padding='same')
        self.relu12 = tf.keras.layers.ReLU()
        self.conv13 = BinConv2d(512*self.inflate, kernel_size=3, strides=1, padding='same')
        self.relu13 = tf.keras.layers.ReLU()
        self.maxpool5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        #######################################################

        #########Layer################
        self.fc1 = BinDense(1024)
        self.relu14 = tf.keras.layers.ReLU()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.fc3 = tf.keras.layers.Dense(num_classes)
        self.logsoftmax = tf.keras.layers.Softmax()

        # self.regime = {
        #     0: {'optimizer': 'SGD', 'lr': 5e-3},
        #     101: {'lr': 1e-3},
        #     142: {'lr': 5e-4},
        #     184: {'lr': 1e-4},
        #     220: {'lr': 1e-5}
        # }
