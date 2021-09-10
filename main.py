import tensorflow as tf
from models.bin_vgg16_cifar100 import VGG_cifar100
import utils.preprocess as preprocess
from utils.bin_op import BinOp

import numpy
import random
import os

model = VGG_cifar100()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

preprocessor = preprocess.scale_random_crop(x_train.shape, 32)
x_train = preprocessor(x_train)
x_test = preprocessor(x_test)

#x_train = x_train / 255.0
#x_test = x_test / 255.0

batch_size = 128
max_epochs = 250
learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20

y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)


def lr_scheduler(epoch):
    return learning_rate * (0.1 ** (epoch // lr_drop))


reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

sgd = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', 'top_k_categorical_accuracy'])

bin_op = BinOp(model)

model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=max_epochs, validation_data=(x_test, y_test), callbacks=[reduce_lr])
model.summary()
