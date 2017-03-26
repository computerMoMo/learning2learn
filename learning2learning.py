# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cPickle
import sys
import re
import random
import os
from collections import defaultdict
import pandas as pd
import jieba
import jieba.analyse
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Reshape, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.models import load_model

np.random.seed(1337)
sysoutfile = open(sys.argv[1], 'w')
old = sys.stdout
sys.stdout = sysoutfile


def build_data_cv(file_pos, file_neg, cv=10, clean_string=True, ):
    """
    Loads data and split into 10 folds.
    """
    revs = []

    vocab = defaultdict(float)

    with open(file_pos, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 0,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)

    with open(file_neg, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 1,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)

    return revs, vocab

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k))
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

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

def clean(str):
    str = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", str)
    str = re.sub(r"\'s", " \'s", str)
    str = re.sub(r"\'ve", " \'ve", str)
    str = re.sub(r"n\'t", " n\'t", str)
    str = re.sub(r"\'re", " \'re", str)
    str = re.sub(r"\'d", " \'d", str)
    str = re.sub(r"\'ll", " \'ll", str)
    str = re.sub(r",", " , ", str)
    str = re.sub(r"!", " ! ", str)
    str = re.sub(r"\(", " \( ", str)
    str = re.sub(r"\)", " \) ", str)
    str = re.sub(r"\?", " \? ", str)
    str = re.sub(r"\s{2,}", " ", str)

    return str

if __name__ == '__main__':

    lr_list = [0.5, 0.55, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7,
               0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
               0.89,
               0.90, 0.95, ]
    # lr_list = [0.6, 0.7]
    nb_epoch = 40
    nb_classes = 2
    max_len = 108
    word_dim = 300
    nb_filters = 250
    batch_size = 128

    for id_str in sys.argv[2:]:
        # generate train txt
        if (int(id_str) % 2 == 0):
            # print("original data path:/opt/exp_data/learning2learn/text-data/aclimdb/train")
            read_filepath_pos = "/opt/exp_data/learning2learn/text-data/aclimdb/train/pos/"
            read_filepath_neg = "/opt/exp_data/learning2learn/text-data/aclimdb/train/neg/"
        else:
            # print("original data path:/opt/exp_data/learning2learn/text-data/aclimdb/test")
            read_filepath_pos = "/opt/exp_data/learning2learn/text-data/aclimdb/test/pos/"
            read_filepath_neg = "/opt/exp_data/learning2learn/text-data/aclimdb/test/neg/"

        write_filepath = "/opt/exp_data/learning2learn/text-data/train-data/"
        w_pos = open(write_filepath + 'pos-' + id_str + '.txt', 'w')
        w_neg = open(write_filepath + 'neg-' + id_str + '.txt', 'w')

        # read pos
        i = 0
        for filename in os.listdir(read_filepath_pos):
            with open(read_filepath_pos + filename, 'r') as f_r:
                for each_line in f_r.readlines():
                    each_line = clean(each_line) + '\n'
                    r = random.randint(1, 10)
                    if r == 5:
                        w_pos.write(each_line)
                        i += 1
        # read neg
        i = 0
        for filename in os.listdir(read_filepath_neg):
            with open(read_filepath_neg + filename, 'r') as f_r:
                for each_line in f_r.readlines():
                    each_line = clean(each_line) + '\n'
                    r = random.randint(1, 10)
                    if r == 5:
                        w_neg.write(each_line)
                        i += 1
        # read word2vec files
        data_path = "/opt/exp_data/learning2learn/text-data/train-data/"
        w2v_file = "/opt/exp_data/google_news_vector/GoogleNews-vectors-negative300.bin"
        revs, vocab = build_data_cv(data_path + "pos-" + id_str + ".txt", data_path + "neg-" + id_str + ".txt", cv=10, clean_string=False)
        max_l = np.max(pd.DataFrame(revs)["num_words"])
        w2v = load_bin_vec(w2v_file, vocab)
        add_unknown_words(w2v, vocab)
        W, word_idx_map = get_W(w2v)
        rand_vecs = {}
        add_unknown_words(rand_vecs, vocab)
        W2, _ = get_W(rand_vecs)

        # generate numpy file
        text, tag = make_idx_data_cv(revs, word_idx_map, word_vector=W, max_l=100, k=300, filter_h=5)

        #training with cnn
        print("\n\n\n")
        print("train file id " + id_str)
        print("\n\n")
        x_train = text
        y_train = tag

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

    print("\n\nprogram finished")
