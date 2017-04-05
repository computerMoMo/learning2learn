# -*- coding: utf-8 -*-
from __future__ import print_function

from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout,Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import load_model
import sys
import numpy as np
np.random.seed(1337)


if __name__ == "__main__":
    data_path = "exp-data/"
    lr_in = 0.7
    nb_epoch = int(sys.argv[1])
    nb_classes = 2
    max_len = 100
    word_dim = 300
    nb_filters = 250
    batch_size = 128

    X_train = np.load(data_path+"x-train-0.npy")
    y_train = np.load(data_path+"y-train-0.npy")

    X_test = np.load(data_path+"x-test.npy")
    y_test = np.load(data_path+"y-test.npy")

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], word_dim, 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], word_dim, 1))
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    # model = Sequential()

    # model.add(Convolution2D(nb_filters, 3, word_dim, activation='relu', input_shape=(max_len, word_dim, 1),
    #                         name='cnn1'))
    # model.add(MaxPooling2D(pool_size=(4, 1), name='maxpooling1'))
    # model.add(Flatten())
    # model.add(Dropout(0.2, name='dropout1'))
    # model.add(Dense(32, activation='relu', name='Dense1'))
    # model.add(Dense(nb_classes, activation='softmax'))
    #
    # sgd = SGD(lr=lr_in, clipnorm=1.0)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #
    # model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,  verbose=1, validation_data=(X_test, y_test))
    #
    # score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    #
    # print('Test score:', score)
    # print('Test accuracy:', acc)
    model = load_model('exp-data/cnn-model.h5')

    #get middle layer output
    layer_name = 'Dense1'
    middle_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    middle_layer_output = middle_layer_model.predict(X_train, batch_size=128, verbose=1)
    middle_layer_output_array = np.asarray(middle_layer_output)
    print(middle_layer_output_array.shape)
    print(middle_layer_output_array[0])
