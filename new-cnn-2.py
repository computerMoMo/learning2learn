# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Reshape, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad

import sys
sysoutfile = open(sys.argv[3], 'w')
old = sys.stdout
sys.stdout = sysoutfile
import cPickle
import numpy as np
np.random.seed(1337)

import tensorflow as tf

if __name__ == '__main__':
    data_path = "/opt/exp_data/learning2learn/text-data/train-data/"
    # lr_in = float(sys.argv[2])
    lr_list = [0.7, 0.8]
    nb_epoch = int(sys.argv[2])
    nb_classes = 2
    max_len = 308
    word_dim = 300
    nb_filters = 250
    batch_size = 128

    X_train = np.load(data_path + "x-data-" + sys.argv[1])
    y_train = np.load(data_path + "y-data-" + sys.argv[1])

    X_test = np.load(data_path + "x-data-test.npy")
    y_test = np.load(data_path + "y-data-test.npy")

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], word_dim, 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], word_dim, 1))
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    with tf.device('/cpu:0'):
        model = Sequential()

        model.add(Convolution2D(nb_filters, 3, word_dim, activation='relu', input_shape=(max_len, word_dim, 1)))
        model.add(MaxPooling2D(pool_size=(max_len - 3 + 1, 1)))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(nb_classes, activation='softmax'))

    with tf.device('/gpu:0'):
        for lr_in in lr_list:
            print("learning rate: ", lr_in)
            sgd = SGD(lr=lr_in, clipnorm=1.0)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(X_test, y_test))

            score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)

            print('Test score:', score)
            print('Test accuracy:', acc)
    print("program finished!")