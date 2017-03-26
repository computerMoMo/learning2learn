# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import cPickle
import numpy as np
np.random.seed(1337)


def get_idx_from_sent(sent, word_idx_map, word_vector, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(word_vector[0])
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_vector[word_idx_map[word]])
            if len(x) == max_l:
                break
    while len(x) < max_l+2*pad:
        x.append(word_vector[0])
    return x


def make_idx_data_cv(revs, word_idx_map, word_vector, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    x_target = []
    y_target = []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, word_vector, max_l, k, filter_h)
        y_target.append(rev["y"])
        x_target.append(sent)
    x_target = np.asarray(x_target, dtype="float32")
    y_target = np.asarray(y_target, dtype='uint8')

    return x_target, y_target


if __name__ == "__main__":
    data_path = "/opt/exp_data/learning2learn/text-data/train-data/"
    source_file = data_path+sys.argv[1]
    data_name = sys.argv[2]

    print("loading data...")
    dataset_origin = cPickle.load(open(source_file, "rb"))
    revs, W, word_idx_map = dataset_origin[0], dataset_origin[1], dataset_origin[3]
    print("data loaded!")

    print("Saving data to .npy files...")
    text, tag = make_idx_data_cv(revs, word_idx_map, word_vector=W, max_l=100, k=300, filter_h=5)
    print("Text shape: ", text.shape)
    print("Tag shape: ", tag.shape)
    np.save(data_path+'x-data-' + data_name, text)
    np.save(data_path+'y-data-' + data_name, tag)
    print("Saved!")
