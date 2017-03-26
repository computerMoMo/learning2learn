# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Reshape, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.models import load_model

if __name__ == '__main__':
    # load data
    x_train = np.load('x_train_l2l.npy')
    y_train = np.load('y_train_l2l.npy')
    x_test = np.load('x_test_l2l.npy')
    y_test = np.load('y_test_l2l.npy')
    print("x-train shape:", x_train.shape)
    print("y-train shape:", y_train.shape)
    print("x-test shape:", x_test.shape)
    print("y-test shape:", y_test.shape)
    data_dim = 100
    data_max_len = 2500
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], data_dim, 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], data_dim, 1))

    print("reshape x-train:", x_train.shape)
    print("reshape x-test:", x_test.shape)
    # cnn model
    print("build model......")
    sum_epoch = int(sys.argv[1])
    sum_filters = 300
    batch_size = 32
    nb_rows = 2
    nb_cols = 2

    model = Sequential()
    model.add(Convolution2D(nb_filter=sum_filters, nb_row=nb_rows, nb_col=nb_cols, activation='relu',
              input_shape=(data_max_len, data_dim, 1), border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1))
    sgd = SGD(lr=0.7, clipnorm=1.0)

    # train model
    model.compile(loss='mse', optimizer=sgd, metrics=['mean_absolute_error'])
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=sum_epoch, verbose=1, validation_data=(x_test, y_test))
    acc = model.evaluate(x_test, y_test, batch_size=16, verbose=1)
    print('errors:')
    print(acc)

    print('predicting......')
    result = model.predict(x_test, batch_size=16, verbose=1)
    # np.save('result.npy', result)
    # model.save('l2l_cnn.h5')
    print('program finished!')
