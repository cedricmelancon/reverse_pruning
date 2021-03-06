import tensorflow as tf
import numpy as np

def forward(x):
    mean = tf.experimental.numpy.mean(tf.abs(x))
    input = tf.sign(x)

    @tf.custom_gradient
    def grad(grad_output):
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

    return input, mean


class BinDense(tf.keras.layers.Layer):
    def __init__(self, output_channels, dropout=0):
        super(BinDense, self).__init__()
        self.layer_type = 'BinDense'
        self.dropout_ratio = dropout

        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.1)
        if dropout != 0:
            self.dropout = tf.keras.layers.Dropout(dropout)

        self.dense = tf.keras.layers.Dense(output_channels)

    def call(self, x):
        x = self.bn(x)
        x, mean = forward(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)

        x = self.dense(x)

        return x

    def get_weights(self):
        weights = self.dense.weights[0]
        weights = tf.transpose(weights, (1, 0))
        return weights

    def set_gradients(self, grad):
        self.dense.grad = grad
