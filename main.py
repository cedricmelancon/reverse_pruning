import tensorflow as tf
from models.vgg16_cifar100 import VGG_cifar100

model = VGG_cifar100()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

learning_rate = 0.1
lr_decay = 1e-6

y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)

sgd = tf.keras.optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=10)
model.summary()