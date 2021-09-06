import tensorflow as tf
from .bin_conv_2d import BinConv2d


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=3, strides=(1,1), padding='valid', downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = BinConv2d(filters, kernel_size=3, strides=strides, padding='valid', dropout=0)
        self.bn1 = tf.keras.layers.BatchNomalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = BinConv2d(filters, kernel_size=3, strides=(1,1), padding='valid', dropout=0)
        self.bn2 = tf.keras.layers.BatchNomalization()
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = self.relu(out)

        out += residual
        residual2 = out
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual2
        out = self.relu(out)
        return out
