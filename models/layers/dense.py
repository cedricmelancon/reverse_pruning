import tensorflow as tf


class Dense(tf.keras.layers.Layer):
    def __init__(self, output_channels, dropout=0):
        super(Dense, self).__init__()
        self.layer_type = 'BinDense'
        self.dropout_ratio = dropout

        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.1)
        if dropout != 0:
            self.dropout = tf.keras.layers.Dropout(dropout)

        self.dense = tf.keras.layers.Dense(output_channels)

    def call(self, x):
        x = self.bn(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)

        x = self.dense(x)

        return x
