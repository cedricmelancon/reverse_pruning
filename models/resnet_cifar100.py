import tensorflow as tf
from .layers.basic_block import BasicBlock
from .layers.bin_conv_2d import BinConv2d
from .layers.bin_conv_2d_2 import BinConv2d2
from .resnet import ResNet


class ResNet_Cifar100(ResNet):

    def __init__(self, num_classes=100, block=BasicBlock, depth=18):
        super(ResNet_Cifar100, self).__init__()
        self.inflate = 1
        self.inplanes = 16 * self.inflate

        # The layers with binary activations are defined as BinConv2d whereas layers with multi-bit activations are defined as BinConv2d2
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='valid')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.conv2 = BinConv2d(48, kernel_size=3, strides=1, padding='valid')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        #######################################################

        self.conv3 = BinConv2d(64, kernel_size=3, strides=1, padding='valid')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()
        #######################################################

        self.conv4 = BinConv2d(64, kernel_size=3, strides=1, padding='valid')
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.relu4 = tf.keras.layers.ReLU()
        #######################################################

        self.conv5 = BinConv2d(48, kernel_size=3, strides=1, padding='valid')
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.relu5 = tf.keras.layers.ReLU()
        #######################################################

        self.conv6 = BinConv2d(64, kernel_size=3, strides=1, padding='valid')
        self.bn6 = tf.keras.layers.BatchNormalization()
        self.relu6 = tf.keras.layers.ReLU()
        #######################################################

        self.conv7 = BinConv2d(64, kernel_size=3, strides=1, padding='valid')
        self.bn7 = tf.keras.layers.BatchNormalization()
        self.relu7 = tf.keras.layers.ReLU()
        #######################################################

        self.conv8 = BinConv2d(48, kernel_size=3, strides=1, padding='valid')
        self.bn8 = tf.keras.layers.BatchNormalization()
        self.relu8 = tf.keras.layers.ReLU()
        #######################################################

        self.conv9 = BinConv2d(64, kernel_size=3, strides=1, padding='valid')
        self.bn9 = tf.keras.layers.BatchNormalization()
        self.relu9 = tf.keras.layers.ReLU()
        #######################################################

        self.conv10 = BinConv2d(64, kernel_size=3, strides=1, padding='valid')
        self.bn10 = tf.keras.layers.BatchNormalization()
        self.relu10 = tf.keras.layers.ReLU()
        #######################################################

        self.conv11 = BinConv2d(96, kernel_size=3, strides=1, padding='valid')
        self.bn11 = tf.keras.layers.BatchNormalization()
        self.relu11 = tf.keras.layers.ReLU()
        #######################################################

        #########Layer################
        self.conv12 = BinConv2d(128, kernel_size=3, strides=2, padding='valid')
        self.bn12 = tf.keras.layers.BatchNormalization()
        self.resconv1 = tf.keras.Sequential()
        self.resconv1.add(BinConv2d(int(32*self.inflate), kernel_size=1, strides=2, padding='same'))
        self.resconv1.add(tf.keras.layers.BatchNormalization())
        self.resconv1.add(tf.keras.layers.ReLU())
        self.relu12 = tf.keras.layers.ReLU()
        #######################################################

        self.conv13 = BinConv2d(128, kernel_size=3, strides=1, padding='valid')
        self.bn13 = tf.keras.layers.BatchNormalization()
        self.relu13 = tf.keras.layers.ReLU()
        #######################################################

        self.conv14 = BinConv2d(96, kernel_size=3, strides=1, padding='valid')
        self.bn14 = tf.keras.layers.BatchNormalization()
        self.relu14 = tf.keras.layers.ReLU()
        #######################################################

        self.conv15 = BinConv2d(128, kernel_size=3, strides=1, padding='valid')
        self.bn15 = tf.keras.layers.BatchNormalization()
        self.relu15 = tf.keras.layers.ReLU()
        #######################################################

        self.conv16 = BinConv2d(128, kernel_size=3, strides=1, padding='valid')
        self.bn16 = tf.keras.layers.BatchNormalization()
        self.relu16 = tf.keras.layers.ReLU()
        #######################################################

        self.conv17 = BinConv2d(96, kernel_size=3, strides=1, padding='valid')
        self.bn17 = tf.keras.layers.BatchNormalization()
        self.relu17 = tf.keras.layers.ReLU()
        #######################################################

        self.conv18 = BinConv2d(128, kernel_size=3, strides=1, padding='valid')
        self.bn18 = tf.keras.layers.BatchNormalization()
        self.relu18 = tf.keras.layers.ReLU()
        #######################################################

        self.conv19 = BinConv2d(128, kernel_size=3, strides=1, padding='valid')
        self.bn19 = tf.keras.layers.BatchNormalization()
        self.relu19 = tf.keras.layers.ReLU()
        #######################################################

        self.conv20 = BinConv2d(96, kernel_size=3, strides=1, padding='valid')
        self.bn20 = tf.keras.layers.BatchNormalization()
        self.relu20 = tf.keras.layers.ReLU()
        #######################################################

        self.conv21 = BinConv2d(128, kernel_size=3, strides=1, padding='valid')
        self.bn21 = tf.keras.layers.BatchNormalization()
        self.relu21 = tf.keras.layers.ReLU()
        #######################################################

        #########Layer################
        self.conv22 = BinConv2d2(128, kernel_size=3, strides=2, padding='valid')
        self.bn22 = tf.keras.layers.BatchNormalization()
        self.resconv2 = tf.keras.Sequential()
        self.resconv2.add(BinConv2d2(int(64*self.inflate), kernel_size=1, strides=2, padding='same'))
        self.resconv2.add(tf.keras.layers.BatchNormalization())
        self.resconv2.add(tf.keras.layers.ReLU())
        self.relu22 = tf.keras.layers.ReLU()
        #######################################################

        self.conv23 = BinConv2d2(128, kernel_size=3, strides=1, padding='valid')
        self.bn23 = tf.keras.layers.BatchNormalization()
        self.relu23 = tf.keras.layers.ReLU()
        #######################################################

        self.conv24 = BinConv2d2(256, kernel_size=3, strides=1, padding='valid')
        self.bn24 = tf.keras.layers.BatchNormalization()
        self.relu24 = tf.keras.layers.ReLU()
        #######################################################

        self.conv25 = BinConv2d2(256, kernel_size=3, strides=1, padding='valid')
        self.bn25 = tf.keras.layers.BatchNormalization()
        self.relu25 = tf.keras.layers.ReLU()
        #######################################################

        self.conv26 = BinConv2d2(128, kernel_size=3, strides=1, padding='valid')
        self.bn26 = tf.keras.layers.BatchNormalization()
        self.relu26 = tf.keras.layers.ReLU()
        #######################################################

        self.conv27 = BinConv2d2(int(64*self.inflate), kernel_size=3, strides=1, padding='valid')
        self.bn27 = tf.keras.layers.BatchNormalization()
        self.relu27 = tf.keras.layers.ReLU()
        #######################################################

        self.conv28 = BinConv2d2(int(64*self.inflate), kernel_size=3, strides=1, padding='valid')
        self.bn28 = tf.keras.layers.BatchNormalization()
        self.relu28 = tf.keras.layers.ReLU()
        #######################################################

        self.conv29 = BinConv2d2(int(64*self.inflate), kernel_size=3, strides=1, padding='valid')
        self.bn29 = tf.keras.layers.BatchNormalization()
        self.relu29 = tf.keras.layers.ReLU()
        #######################################################

        self.conv30 = BinConv2d2(int(64*self.inflate), kernel_size=3, strides=1, padding='valid')
        self.bn30 = tf.keras.layers.BatchNormalization()
        self.relu30 = tf.keras.layers.ReLU()
        #######################################################

        self.conv31 = BinConv2d2(int(64*self.inflate), kernel_size=3, strides=1, padding='valid')
        self.bn31 = tf.keras.layers.BatchNormalization()
        self.relu31 = tf.keras.layers.ReLU()
        #######################################################

        #########Layer################
        self.avgpool = tf.keras.layers.AveragePooling2D(8)
        self.bn32 = tf.keras.layers.BatchNormalization()
        self.fc = tf.keras.layers.Dense(num_classes)
        self.bn33 = tf.keras.layers.BatchNormalization()
        self.logsoftmax = tf.keras.layers.Softmax()
