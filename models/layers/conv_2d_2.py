import tensorflow as tf


class Conv2d2(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=-1, strides=(1, 1), padding='valid', dropout=0):
        super(Conv2d2, self).__init__()
        self.layer_type = 'BinConv2d'
        self.dropout_ratio = dropout

        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-4, momentum=0.1)
        if dropout != 0:
            self.dropout = tf.keras.layers.Dropout(dropout)
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)

        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)

        if self.dropout_ratio != 0:
            x = self.dropout(x)

        x = self.conv(x)

        # x = self.relu(x)
        return x
