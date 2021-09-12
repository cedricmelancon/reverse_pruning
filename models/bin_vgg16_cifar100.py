import tensorflow as tf
from utils.bin_op import BinOp
from .layers.bin_conv_2d import BinConv2d
from .layers.bin_conv_2d_2 import BinConv2d2
from .layers.bin_dense import BinDense

__all__ = ['VGG_cifar100']


class VGG_cifar100(tf.keras.Sequential):

    def __init__(self, num_classes=100, depth=18):
        super(VGG_cifar100, self).__init__()
        self.inflate = 1

        self.add(tf.keras.layers.Conv2D(64*self.inflate, input_shape=(32, 32, 3), kernel_size=3, strides=1, padding='same'))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.ReLU())
        self.add(BinConv2d(64*self.inflate, kernel_size=3, strides=1, padding='same'))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.ReLU())
        #######################################################

        self.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        #######################################################

        #########Layer################
        self.add(BinConv2d2(128* self.inflate, kernel_size=3, strides=1, padding='same'))
        self.add(tf.keras.layers.ReLU())
        self.add(BinConv2d(128 * self.inflate, kernel_size=3, strides=1, padding='same'))
        self.add(tf.keras.layers.ReLU())
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        #######################################################

        #########Layer################
        self.add(BinConv2d2(256*self.inflate, kernel_size=3, strides=1, padding='same'))
        self.add(tf.keras.layers.ReLU())
        self.add(BinConv2d(256 * self.inflate, kernel_size=3, strides=1, padding='same'))
        self.add(tf.keras.layers.ReLU())
        self.add(BinConv2d(256*self.inflate, kernel_size=3, strides=1, padding='same'))
        self.add(tf.keras.layers.ReLU())
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        #######################################################

        #########Layer################
        self.add(BinConv2d2(512*self.inflate, kernel_size=3, strides=1, padding='same'))
        self.add(tf.keras.layers.ReLU())
        self.add(BinConv2d(512 * self.inflate, kernel_size=3, strides=1, padding='same'))
        self.add(tf.keras.layers.ReLU())
        self.add(BinConv2d(512*self.inflate, kernel_size=3, strides=1, padding='same'))
        self.add(tf.keras.layers.ReLU())
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        #######################################################

        #########Layer################
        self.add(BinConv2d2(512*self.inflate, kernel_size=3, strides=1, padding='same'))
        self.add(tf.keras.layers.ReLU())
        self.add(BinConv2d2(512*self.inflate, kernel_size=3, strides=1, padding='same'))
        self.add(tf.keras.layers.ReLU())
        self.add(BinConv2d(512*self.inflate, kernel_size=3, strides=1, padding='same'))
        self.add(tf.keras.layers.ReLU())
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        #######################################################

        #########Layer################
        self.add(tf.keras.layers.Flatten())
        self.add(BinDense(1024))
        self.add(tf.keras.layers.ReLU())
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Dense(num_classes))
        self.add(tf.keras.layers.Softmax())

        # self.regime = {
        #     0: {'optimizer': 'SGD', 'lr': 5e-3},
        #     101: {'lr': 1e-3},
        #     142: {'lr': 5e-4},
        #     184: {'lr': 1e-4},
        #     220: {'lr': 1e-5}
        # }
        self.bin_op = BinOp(self)

    def initialize(self):
        self.bin_op.initialize()

    def train_step(self, data):
        self.bin_op.binarization()

        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        # restore weights
        self.bin_op.restore()
        self.bin_op.updateBinaryGradWeight()

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
