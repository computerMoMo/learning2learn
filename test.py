# -*- coding: utf-8 -*-
import numpy as np
import sys

if __name__ == '__main__':
    # load data
    data_path = "/opt/exp_data/learning2learn/text-data/train-data/"
    x_test = np.load(data_path + "x-data-test.npy")
    y_test = np.load(data_path + "y-data-test.npy")
    
    # x_train = np.load('x_train_l2l.npy')
    # y_train = np.load('y_train_l2l.npy')
    # x_test = np.load('x_test_l2l.npy')
    # y_test = np.load('y_test_l2l.npy')

    # print(x_train.shape)
    # print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)