import tensorflow as tf
from .layers.bin_conv_2d import BinConv2d


class ResNet(tf.keras.Model):

    def __init__(self):
        super(ResNet, self).__init__()

    # def _make_layer(self, block, planes, blocks, strides=(1, 1), do_binary=True):
    #     downsample = None
    #     downsample1 = None
    #     if strides != (1, 1) or self.inplanes != planes * block.expansion:
    #         downsample = tf.keras.Sequential()
    #         downsample.add(BinConv2d(self.inplanes, kernel_size=1, strides=strides, padding='same', dropout=0))
    #         downsample.add(tf.keras.layers.BatchNormalization())
    #
    #         downsample1 = tf.keras.Sequential()
    #         downsample1.add(tf.keras.layers.Conv2D(self.inplanes, kernel_size=1, strides=strides, padding='same'))
    #         downsample1.add(tf.keras.layers.BatchNormalization())
    #
    #     layers = []
    #
    #     if do_binary:
    #         layers.append(block(self.inplanes, planes,
    #                             1, strides, 0, downsample))
    #     else:
    #         layers.append(block(self.inplanes, planes,
    #                             1, strides, 0, downsample1))
    #
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks - 1):
    #         layers.append(block(self.inplanes, planes))
    #     layers.append(block(self.inplanes, planes))
    #     return tf.keras.Sequential(*layers)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        residual1 = x

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu2(out)

        print(out.shape)
        print(residual1.shape)
        out = tf.add(out, residual1)
        residual2 = out
        ###################################
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out += residual2
        residual3 = out
        ###################################
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out += residual3
        residual4 = out
        ###################################
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        out += residual4
        residual5 = out
        ###################################
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu6(out)
        out += residual5
        residual6 = out
        ###################################
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu7(out)
        out += residual6
        residual7 = out
        ###################################
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu8(out)
        out += residual7
        residual8 = out
        ###################################
        out = self.conv9(out)
        out = self.bn9(out)
        out = self.relu9(out)
        out += residual8
        residual9 = out
        ###################################
        out = self.conv10(out)
        out = self.bn10(out)
        out = self.relu10(out)
        out += residual9
        residual10 = out
        ###################################
        out = self.conv11(out)
        out = self.bn11(out)
        out = self.relu11(out)
        out += residual10
        residual11 = out
        ###################################
        #########Layer################
        out = self.conv12(out)
        out = self.bn12(out)
        residual1 = self.resconv1(residual1)
        out = self.relu12(out)
        out += residual11
        residual12 = out
        ###################################
        out = self.conv13(out)
        out = self.bn13(out)
        out = self.relu13(out)
        out += residual12
        residual13 = out
        ###################################
        out = self.conv14(out)
        out = self.bn14(out)
        out = self.relu14(out)
        out += residual13
        residual14 = out
        ###################################
        out = self.conv15(out)
        out = self.bn15(out)
        out = self.relu15(out)
        out += residual14
        residual15 = out
        ###################################
        out = self.conv16(out)
        out = self.bn16(out)
        out = self.relu16(out)
        out += residual15
        residual16 = out
        ###################################
        out = self.conv17(out)
        out = self.bn17(out)
        out = self.relu17(out)
        out += residual16
        residual17 = out
        ###################################
        out = self.conv18(out)
        out = self.bn18(out)
        out = self.relu18(out)
        out += residual17
        residual18 = out
        ###################################
        out = self.conv19(out)
        out = self.bn19(out)
        out = self.relu19(out)
        out += residual18
        residual19 = out
        ###################################
        out = self.conv20(out)
        out = self.bn20(out)
        out = self.relu20(out)
        out += residual19
        residual20 = out
        ###################################
        out = self.conv21(out)
        out = self.bn21(out)
        out = self.relu21(out)
        out += residual20
        residual21 = out
        ###################################
        #########Layer################
        out = self.conv22(out)
        out = self.bn22(out)
        residual22 = self.resconv2(residual21)
        out = self.relu22(out)
        out += residual22
        residual22 = out
        ###################################
        out = self.conv23(out)
        out = self.bn23(out)
        out = self.relu23(out)
        out += residual22
        residual23 = out
        ###################################
        out = self.conv24(out)
        out = self.bn24(out)
        out = self.relu24(out)
        out += residual23
        residual24 = out
        ###################################
        out = self.conv25(out)
        out = self.bn25(out)
        out = self.relu25(out)
        out += residual24
        residual25 = out
        ##################################
        out = self.conv26(out)
        out = self.bn26(out)
        out = self.relu26(out)
        out += residual25
        residual26 = out
        ###################################
        out = self.conv27(out)
        out = self.bn27(out)
        out = self.relu27(out)
        out += residual26
        residual27 = out
        ###################################
        out = self.conv28(out)
        out = self.bn28(out)
        out = self.relu28(out)
        out += residual27
        residual28 = out
        ###################################
        out = self.conv29(out)
        out = self.bn29(out)
        out = self.relu29(out)
        out += residual28
        residual29 = out
        ###################################
        out = self.conv30(out)
        out = self.bn30(out)
        out = self.relu30(out)
        out += residual29
        residual30 = out
        ###################################
        out = self.conv31(out)
        out = self.bn31(out)
        out = self.relu31(out)
        out += residual30
        residual31 = out
        ###################################
        #########Layer################
        x = out

        x = self.avgpool(x)
        #x = x.view(x.size(0), -1)

        x = self.bn32(x)

        #x = tf.keras.layers.Flatten()(x)

        x = self.fc(x)

        x = self.bn33(x)

        x = self.logsoftmax(x)

        return x
