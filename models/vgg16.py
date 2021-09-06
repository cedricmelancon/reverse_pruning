import tensorflow as tf

__all__ = ['vgg16']


class VGG16(tf.keras.Model):

    def __init__(self):
        super(VGG16, self).__init__()

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)

        #########Layer################
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)

        #########Layer################
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.maxpool3(x)

        #########Layer################
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.conv9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.relu10(x)
        x = self.maxpool4(x)

        #########Layer################
        x = self.conv11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.relu12(x)
        x = self.conv13(x)
        x = self.relu13(x)
        x = self.maxpool5(x)

        #########Layer################
        x = tf.keras.layers.Flatten()(x)

        x = self.fc1(x)

        x = self.relu14(x)

        x = self.bn3(x)

        x = self.fc3(x)

        x = self.logsoftmax(x)

        return x
