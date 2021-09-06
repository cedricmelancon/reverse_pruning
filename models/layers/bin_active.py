import tensorflow as tf


class BinActive(tf.keras.layers.Layer):
    """
    Binarize the input activations and calculate the mean across channel dimension.
    """

    def __init__(self):
        super(BinActive, self).__init__()

    def call(self, x):
        self.save_for_backward(x)

        mean = tf.mean(x.abs(), 1, keepdim=True)
        input = x.sign()

        @tf.custom_gradient
        def grad(grad_output):
            grad_input = grad_output.clone()
            grad_input[input.ge(1)] = 0
            grad_input[input.le(-1)] = 0
            return grad_input

        return input, mean


