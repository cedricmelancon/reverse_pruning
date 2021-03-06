import tensorflow as tf
from .vgg16 import VGG16
from .layers.conv_2d import Conv2d

from .layers.dense import Dense

__all__ = ['VGG_cifar100']


class VGG_cifar100(VGG16):

    def __init__(self, num_classes=100, depth=18):
        super(VGG_cifar100, self).__init__()
        self.inflate = 1
        self.conv1 = tf.keras.layers.Conv2D(64*self.inflate, kernel_size=3, strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = Conv2d(64*self.inflate, kernel_size=3, strides=1, padding='same', dropout=0.1)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        #######################################################

        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        #######################################################

        #########Layer################
        self.conv3 = Conv2d(128* self.inflate, kernel_size=3, strides=1, padding='same', dropout=0.2)
        self.relu3 = tf.keras.layers.ReLU()
        self.conv4 = Conv2d(128 * self.inflate, kernel_size=3, strides=1, padding='same', dropout=0.2)
        self.relu4 = tf.keras.layers.ReLU()
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        #######################################################

        #########Layer################
        self.conv5 = Conv2d(256*self.inflate, kernel_size=3, strides=1, padding='same', dropout=0.3)
        self.relu5 = tf.keras.layers.ReLU()
        self.conv6 = Conv2d(256 * self.inflate, kernel_size=3, strides=1, padding='same', dropout=0.3)
        self.relu6 = tf.keras.layers.ReLU()
        self.conv7 = Conv2d(256*self.inflate, kernel_size=3, strides=1, padding='same', dropout=0.3)
        self.relu7 = tf.keras.layers.ReLU()
        self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        #######################################################

        #########Layer################
        self.conv8 = Conv2d(512*self.inflate, kernel_size=3, strides=1, padding='same', dropout=0.4)
        self.relu8 = tf.keras.layers.ReLU()
        self.conv9 = Conv2d(512 * self.inflate, kernel_size=3, strides=1, padding='same', dropout=0.4)
        self.relu9 = tf.keras.layers.ReLU()
        self.conv10 = Conv2d(512*self.inflate, kernel_size=3, strides=1, padding='same', dropout=0.4)
        self.relu10 = tf.keras.layers.ReLU()
        self.maxpool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        #######################################################

        #########Layer################
        self.conv11 = Conv2d(512*self.inflate, kernel_size=3, strides=1, padding='same', dropout=0.4)
        self.relu11 = tf.keras.layers.ReLU()
        self.conv12 = Conv2d(512*self.inflate, kernel_size=3, strides=1, padding='same', dropout=0.4)
        self.relu12 = tf.keras.layers.ReLU()
        self.conv13 = Conv2d(512*self.inflate, kernel_size=3, strides=1, padding='same', dropout=0.4)
        self.relu13 = tf.keras.layers.ReLU()
        self.maxpool5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        #######################################################

        #########Layer################
        self.fc1 = Dense(1024)
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
