import numpy
import tensorflow as tf
from models.layers.bin_dense import BinDense
from models.layers.bin_conv_2d import BinConv2d
from models.layers.bin_conv_2d_2 import BinConv2d2
from sklearn.preprocessing import normalize

class BinOp():
    def __init__(self, model):
        # count the number of Conv2d or linear
        count_targets = 0
        for m in model.layers:
            if isinstance(m, BinConv2d) or isinstance(m, BinConv2d2) or isinstance(m, BinDense):
                count_targets = count_targets + 1

        start_range = 1
        end_range = count_targets-2
        if start_range == end_range:
            self.bin_range = numpy.linspace(start_range,
                                            end_range, end_range-start_range)\
                .astype('int').tolist()
        else:
            self.bin_range = numpy.linspace(start_range,
                                            end_range, end_range-start_range+1)\
                .astype('int').tolist()

        res_conn = numpy.array([])

        # Layers with k-bit weights
        #kbit_conn = numpy.array([22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
        #kbit_conn = numpy.array([])
        self.kbit_conn = numpy.array([3, 5, 8, 11, 12])
        res_conn = res_conn.astype('int').tolist()
        self.kbit_conn = self.kbit_conn.astype('int').tolist()

        self.bin_range = (list(set(self.bin_range) - set(res_conn)))

        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        self.onlybin_range = (list(set(self.bin_range) - set(self.kbit_conn)))

        self.model = model

    @tf.function
    def initialize(self):
        self.target_modules = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
        index = -1
        for m in self.model.layers:
            if isinstance(m, BinConv2d) or isinstance(m, BinConv2d2) or isinstance(m, BinDense):
                index = index + 1
                if index in self.onlybin_range:
                    print('Binarizing')
                    tmp = m.get_weights()
                    self.saved_params.append(tmp)
                    self.target_modules.write(index, m.get_weights())
                elif index in self.kbit_conn:
                    print('Making k-bit')  # Know which layers are made k-bit
                    tmp = m.get_weights()
                    self.saved_params.append(tmp)
                    self.target_modules.write(index, m.get_weights())

    @tf.function
    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    @tf.function
    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            mean = tf.reduce_mean(self.target_modules.read(index))
            neg_mean = tf.multiply(-1.0, mean)
            #neg_mean = tf.broadcast_to(, self.target_modules[index])
            self.target_modules.write(index, tf.add(self.target_modules.read(index), neg_mean))

    @tf.function
    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules.write(index, tf.clip_by_value(self.target_modules.read(index), -1.0, 1.0))

    @tf.function
    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index] = self.target_modules.read(index)

    @tf.function
    def binarizeConvParams(self):
        # k-bit conv layer list (starts from 0, hence -1 from previous list)
        #kbit_conn = numpy.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
        #kbit_conn = numpy.array([])
        kbit_conn = numpy.array([3, 5, 8, 11, 12])
        kbit_conn = kbit_conn.astype('int').tolist()

        for index in range(self.num_of_params):
            n = self.target_modules.read(index)[0].shape[0]
            s = self.target_modules.read(index).shape
            print(self.target_modules.read(index)
            #norm = numpy.linalg.norm(self.target_modules[index], axis=-1, keepdims=True)
            #print(numpy.shape(norm))

            #print(numpy.shape(normal_array))
            if s[0] == 4:
                m = tf.norm(self.target_modules.read(index), 1, 3, keepdims=True)
                m = tf.math.reduce_sum(m, 2, keepdims=True)
                m = tf.math.reduce_sum(m, 1, keepdims=True)
                m = tf.divide(m, n)
            elif s[0] == 2:
                m = tf.linalg.normalize(self.target_modules.read(index), 1, 1)

            if index in kbit_conn:  # Make the k-bit assigned layer weights k-bit
                # print(index)
                #print('Binarizing kbit')
                # print(self.target_modules[index].data.size())
                # input()
                x = self.target_modules.read(index)
                num_bits = 2
                v0 = 1
                v1 = 2
                v2 = -0.5
                y = 2.**num_bits - 1.
                x = tf.divide(tf.add(x, v0), v1)
                x = tf.round(tf.multiply(x, y))
                x = tf.divide(x, y)
                x = tf.add(x, v2)
                x = tf.multiply(x, v1)
                self.target_modules.write(index, tf.multiply(x, tf.broadcast_to(m, s)))
            else:
                # print(index)
                # print(self.target_modules[index].size())
                #print('Binarizing 1bit')
                # input()
                temp = tf.sign(self.target_modules.read(index))
                self.target_modules.write(index, tf.multiply(temp, tf.broadcast_to(m, s)))

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules.write(index, self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules.read(index)
            n = self.target_modules.read(index)[0].shape[0]
            s = self.target_modules.read(index).shape
            print(s)
            if s[0] == 4:
                m = tf.norm(self.target_modules.read(index), 1, 3, keepdims=True)
                m = tf.math.reduce_sum(m, 2, keepdims=True)
                m = tf.math.reduce_sum(m, 1, keepdims=True)
                m = tf.divide(m, n)
            elif s[0] == 2:
                m = tf.linalg.normalize(self.target_modules.read(index), 1, 1, keepdims=True)
                m = tf.divide(m, n)

            m = tf.where(tf.less_equal(weight, -1.0), tf.zeros_like(m), m)
            m = tf.where(tf.greater(m, 1.0), tf.zeros_like(m), m)

            # m = m.add(1.0/n).mul(1.0-1.0/s[1]).mul(n)
            # self.target_modules[index].grad = \
            #         self.target_modules[index].grad.mul(m)
            m = tf.multiply(m, tf.gradients(self.target_modules.read(index), self.target_modules.read(index)))
            m_add = tf.multiply(tf.sign(weight), tf.gradients(self.target_modules.read(index), self.target_modules.read(index)))
            if s[0] == 4:
                m_add = tf.math.reduce_sum(m_add, 3, keepdims=True)
                m_add = tf.math.reduce_sum(m_add, 2, keepdims=True)
                m_add = tf.math.reduce_sum(m_add, 1, keepdims=True)
                m_add = tf.divide(m_add, n)
            elif s[0] == 2:
                m_add = tf.math.reduce_sum(m_add, 1)
                m_add = tf.divide(m_add, n)

            m_add = tf.multiply(m_add, tf.sign(weight))
            grad = tf.multiply(tf.multiply(tf.add(m, m_add), 1.0-1.0/s[1]), n)
            self.target_modules.read(index).grad = tf.transpose(grad[0], perm=(3, 2, 1, 0))
