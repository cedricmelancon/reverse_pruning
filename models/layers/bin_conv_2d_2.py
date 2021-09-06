import tensorflow as tf
from .bin_active2 import BinActive2


class BinConv2d2(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=-1, strides=(1, 1), padding='valid', dropout=0):
        super(BinConv2d2, self).__init__()
        self.layer_type = 'BinConv2d'
        self.dropout_ratio = dropout

        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.1)
        if dropout != 0:
            self.dropout = tf.keras.layers.Dropout(dropout)
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)

        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x, mean = BinActive2(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)

        x = self.conv(x)

        # x = self.relu(x)
        return x
