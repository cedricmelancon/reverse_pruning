import tensorflow as tf


class BinActive2(tf.keras.layers.Layer):
    """
    Make the input activations k-bit
    """

    def __init__(self):
        super(BinActive2, self).__init__()

    def forward(self, x):
        mean = tf.mean(x.abs(), 1, keepdim=True)

        num_bits = 2  # Bit-precision
        v0 = 1
        v1 = 2
        v2 = -0.5

        y = 2. ** num_bits - 1.
        x = x.add(v0).div(v1)
        x = x.mul(y).round_()
        x = x.div(y)
        x = x.add(v2)
        x = x.mul(v1)
        input = x

        @tf.custom_gradient
        def grad(grad_output):
            grad_input = grad_output.clone()
            return grad_input

        return input, mean
