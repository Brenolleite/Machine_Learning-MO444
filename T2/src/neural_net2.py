#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import keras.utils

batch_size = 256
epochs = 20

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = keras.utils.to_categorical(y_train, 10) # converte classes em matrizes binárias
y_test = keras.utils.to_categorical(y_test, 10)

print (x_train.shape)

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print (np.prod(x_train.shape[1:]))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = Sequential()
model.add(Dense(786, activation='relu', input_shape=(784,)))
model.add(Dense(786, activation='relu'))
model.add(Dense(10, activation='sigmoid'))

model.summary()

model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


