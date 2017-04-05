# -*- coding: utf-8 -*-
import numpy as np
import sys

if __name__ == '__main__':
    data_path = "exp-data/"
    y_test = np.load(data_path + "y_test_l2l.npy")
    y_test = y_test.astype(np.float32)
    print y_test.shape
    y_new_test = np.multiply(y_test, 10.0)
    print y_new_test
    print y_new_test.shape
    print y_test*10.0
