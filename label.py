#-*- coding:utf-8 -*-
import numpy as np
import sys

if __name__ == '__main__':
    data_path = 'exp-data/'
    y_array = np.load(data_path+sys.argv[1]).astype(np.float32)
    LabelList = []
    for element in y_array:
        if element < 0.5:
            LabelList.append(0)
        elif element < 0.55:
            LabelList.append(1)
        elif element < 0.6:
            LabelList.append(2)
        elif element < 0.65:
            LabelList.append(3)
        elif element < 0.7:
            LabelList.append(4)
        elif element < 0.75:
            LabelList.append(5)
        elif element < 0.8:
            LabelList.append(6)
        elif element < 0.85:
            LabelList.append(7)
        elif element < 0.9:
            LabelList.append(8)
        elif element < 0.95:
            LabelList.append(9)
        else:
            LabelList.append(10)
    LabelArray = np.asarray(LabelList, dtype=np.int16)
    print(LabelArray.shape)
    np.save(data_path+sys.argv[2], LabelArray)
