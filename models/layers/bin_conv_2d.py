import tensorflow as tf
import numpy as np


def forward(x):
    input = tf.sign(x)

    @tf.custom_gradient
    def grad(grad_output):
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

    return input, grad


class BinConv2d(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=-1, strides=(1, 1), padding='valid', dropout=0):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.dropout_ratio = dropout

        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.1)
        if dropout != 0:
            self.dropout = tf.keras.layers.Dropout(dropout)
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)

        # self.relu = nn.ReLU(inplace=True)

    def call(self, x):
        x = self.bn(x)
        x, grad = forward(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)

        x = self.conv(x)

        # x = self.relu(x)
        return x

    def get_weights(self):
        weights = self.conv.weights[0]
        weights = tf.transpose(weights, (3, 2, 1, 0))
        return weights

    def set_gradients(self, grad):
        self.conv.grad = grad
