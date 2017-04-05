# -*- coding:utf-8 -*-
import sys
import gensim
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


class Mysentences(object):
    def __init__(self, datafile):
        self.datafile = datafile

    def __iter__(self):
        with open(self.datafile, 'r') as f_r:
            for line in f_r.readlines():
                yield line.split()

if __name__ == '__main__':
    # word_vectors = Word2Vec
    word_vectors = Word2Vec.load('exp-data/exp_w2v.w2v')
    # word_array = np.asarray(word_vectors.wv['hello'], dtype=np.float32)
    word_vec_test = []
    word_vec_test.append(word_vectors.wv['hello'])
    word_vec_test.append(np.zeros(shape=(50,)))

    word_vec_test = np.asarray(word_vec_test, dtype=np.float32)
    # w2v_vocab = word_vectors.vocab
    #
    # print w2v_vocab.has_key('中文')

    print word_vec_test.shape
    print word_vec_test
