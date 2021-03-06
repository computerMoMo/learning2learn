# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPool2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
np.random.seed(1234)

if __name__ == '__main__':
    # load data
    x_train = np.load('exp-data/x_cnn_feature_train.npy').astype(np.float32)
    y_train = np.load('exp-data/y_cnn_feature_train.npy').astype(np.float32)
    x_test = np.load('exp-data/x_cnn_feature_test.npy').astype(np.float32)
    y_test = np.load('exp-data/y_cnn_feature_test.npy').astype(np.float32)
    # y_new_train = np.asarray(y_train*10.0)
    # y_new_test = np.asarray(y_test*10.0)

    data_dim = 100
    data_max_len = 2500
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], data_dim, 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], data_dim, 1))
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)

    # cnn model
    print("build model......")
    sum_epoch = int(sys.argv[1])
    sum_filters = 128
    batch_size = 64
    nb_rows = 3
    nb_cols = data_dim

    model = Sequential()
    model.add(Conv2D(nb_filter=sum_filters, nb_row=nb_rows, nb_col=nb_cols, activation='relu',
              input_shape=(data_max_len, data_dim, 1)))
    model.add(MaxPool2D(pool_size=(data_max_len-nb_rows+1, 1)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    sgd = SGD(lr=0.7, clipnorm=1.0)

    # train model
    model.compile(loss='mse', optimizer=sgd, metrics=['mean_absolute_error'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=1)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=sum_epoch, verbose=1,
              validation_data=(x_test, y_test), callbacks=[early_stopping])
    acc = model.evaluate(x_test, y_test, batch_size=16, verbose=1)
    print('errors:')
    print(acc)

    print('predicting......')
    result = model.predict(x_test, batch_size=16, verbose=1)
    np.savetxt(sys.argv[2], result, fmt='%.6f', delimiter=',')

    print('program finished!')
