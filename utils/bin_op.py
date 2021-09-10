import numpy
import tensorflow as tf
from models.layers.bin_dense import BinDense
from models.layers.bin_conv_2d import BinConv2d
from models.layers.bin_conv_2d_2 import BinConv2d2

class BinOp():
    def __init__(self, model):
        # count the number of Conv2d or linear
        count_targets = 0
        for m in model.layers:
            if isinstance(m, BinConv2d) or isinstance(m, BinConv2d2) or isinstance(m, BinDense):
                count_targets = count_targets + 1
        print(count_targets)

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

        print(self.bin_range)

        res_conn = numpy.array([])

        # Layers with k-bit weights
        #kbit_conn = numpy.array([22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
        #kbit_conn = numpy.array([])
        kbit_conn = numpy.array([3, 4, 5, 6, 11, 12, 13, 22, 23, 24, 25])
        res_conn = res_conn.astype('int').tolist()
        kbit_conn = kbit_conn.astype('int').tolist()

        print(kbit_conn)

        self.bin_range = (list(set(self.bin_range) - set(res_conn)))
        print(self.bin_range)

        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        self.onlybin_range = (list(set(self.bin_range) - set(kbit_conn)))
        index = -1
        print(self.num_of_params)

        for m in model.layers:
            if isinstance(m, BinConv2d) or isinstance(m, BinConv2d2) or isinstance(m, BinDense):
                index = index + 1
                if index in self.onlybin_range:
                    print('Binarizing')
                    tmp = m.get_weights()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.get_weights())
                elif index in kbit_conn:
                    print('Making k-bit')  # Know which layers are made k-bit
                    tmp = m.get_weights()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.get_weights())

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            neg_mean = self.target_modules[index].data.mean(1, keepdim=True).\
                mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(neg_mean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        # k-bit conv layer list (starts from 0, hence -1 from previous list)
        kbit_conn = numpy.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
        #kbit_conn = numpy.array([])
        #kbit_conn = numpy.array([2,3,4,5,10,11,12,21,22,23,24])
        kbit_conn = kbit_conn.astype('int').tolist()

        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            if len(s) == 4:
                m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            elif len(s) == 2:
                m = self.target_modules[index].data.norm(
                    1, 1, keepdim=True).div(n)

            if index in kbit_conn:  # Make the k-bit assigned layer weights k-bit
                # print(index)
                #print('Binarizing kbit')
                # print(self.target_modules[index].data.size())
                # input()
                x = self.target_modules[index].data
                xmax = x.abs().max()
                num_bits = 2
                v0 = 1
                v1 = 2
                v2 = -0.5
                y = 2.**num_bits - 1.
                x = x.add(v0).div(v1)
                x = x.mul(y).round_()
                x = x.div(y)
                x = x.add(v2)
                x = x.mul(v1)
                self.target_modules[index].data = x.mul(m.expand(s))
            else:
                # print(index)
                # print(self.target_modules[index].data.size())
                #print('Binarizing 1bit')
                # input()
                self.target_modules[index].data = \
                    self.target_modules[index].data.sign().mul(m.expand(s))

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            if len(s) == 4:
                m = weight.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0
            m[weight.gt(1.0)] = 0
            # m = m.add(1.0/n).mul(1.0-1.0/s[1]).mul(n)
            # self.target_modules[index].grad.data = \
            #         self.target_modules[index].grad.data.mul(m)
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)

            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(
                m_add).mul(1.0-1.0/s[1]).mul(n)
