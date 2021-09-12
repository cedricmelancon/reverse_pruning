import tensorflow as tf

__all__ = ['vgg16']


class VGG16(tf.keras.Model):

    def __init__(self):
        super(VGG16, self).__init__()
