# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPool2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
np.random.seed(1234)

if __name__ == '__main__':
    # load data
    x_train = np.load('exp-data/x_cnn_feature_train.npy').astype(np.float32)
    y_train = np.load('exp-data/y_train_l2l_label.npy').astype(np.int16)
    x_test = np.load('exp-data/x_cnn_feature_test.npy').astype(np.float32)
    y_test = np.load('exp-data/y_test_l2l_label.npy').astype(np.int16)

    data_dim = 100
    data_max_len = 2500
    nb_classes = 11
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], data_dim, 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], data_dim, 1))
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    y_train = np_utils.to_categorical(y_train, num_classes=nb_classes)
    y_test = np_utils.to_categorical(y_test, num_classes=nb_classes)

    # cnn model
    print("build model......")
    sum_epoch = int(sys.argv[1])
    sum_filters = 128
    batch_size = 32
    nb_rows = 3
    nb_cols = data_dim

    model = Sequential()
    model.add(Conv2D(nb_filter=sum_filters, nb_row=nb_rows, nb_col=nb_cols, activation='relu',
              input_shape=(data_max_len, data_dim, 1)))
    model.add(MaxPool2D(pool_size=(data_max_len-nb_rows+1, 1)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(nb_classes, activation='softmax'))
    sgd = SGD(lr=0.7, clipnorm=1.0)

    # train model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=1)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=sum_epoch, verbose=1,
              validation_data=(x_test, y_test), callbacks=[early_stopping])
    acc = model.evaluate(x_test, y_test, batch_size=16, verbose=1)
    print('errors:')
    print(acc)

    print('predicting......')
    result = model.predict_classes(x_test, batch_size=16, verbose=1)
    result = np.asarray(result, dtype=np.int8)
    np.savetxt(sys.argv[2], result, fmt='%1.1d', delimiter=',')

    print('program finished!')
