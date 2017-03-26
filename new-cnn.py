# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Reshape, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad

import sys
import os
sysoutfile = open(sys.argv[1], 'w')
old = sys.stdout
sys.stdout = sysoutfile
import cPickle
import numpy as np
np.random.seed(1337)

if __name__ == '__main__':
    data_path = "/opt/exp_data/learning2learn/text-data/train-data/"

    lr_list = [0.5,0.55,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,
        0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.89,
               0.90,0.95,]

    nb_epoch = 40
    nb_classes = 2
    max_len = 308
    word_dim = 300
    nb_filters = 250
    batch_size = 128

    for train_file_id in sys.argv[2:]:
        print("\n\n\n")
        print("train file id "+train_file_id)
        print("\n\n")
        x_train = np.load(data_path + "x-data-train-" + train_file_id +".npy")
        y_train = np.load(data_path + "y-data-train-" + train_file_id +".npy")

        x_test = np.load(data_path + "x-data-test.npy")
        y_test = np.load(data_path + "y-data-test.npy")

        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], word_dim, 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], word_dim, 1))
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)

        for lr_in in lr_list:
            model = Sequential()
            model.add(Convolution2D(nb_filters, 3, word_dim, activation='relu', input_shape=(max_len, word_dim, 1)))
            model.add(MaxPooling2D(pool_size=(max_len - 3 + 1, 1)))
            model.add(Flatten())
            model.add(Dropout(0.2))
            model.add(Dense(nb_classes, activation='softmax'))
            print("\n\n")
            print("learning rate: ", lr_in)
            sgd = SGD(lr=lr_in, clipnorm=1.0)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(x_test, y_test))
            score, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)
            print('Test score:', score)
            print('Test accuracy:', acc)

        os.remove(data_path + "x-data-train-" + train_file_id +".npy")
        os.remove(data_path + "y-data-train-" + train_file_id + ".npy")

    print("program finished!")