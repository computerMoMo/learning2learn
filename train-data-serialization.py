# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import sys
import cPickle
import re
import random
import os
from collections import defaultdict


def build_data_cv(file_pos, file_neg, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []

    vocab = defaultdict(float)
    try:
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
    except:
        revs = []
        print(file_pos+ " doesn't exist")
        return revs, vocab

    try:
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
    except:
        revs = []
        print(file_neg + " doesn't exist")
        return revs, vocab

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
    Loads 300x1 word vecs from Google word2vec
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

def get_idx_from_sent(sent, word_idx_map, word_vector, max_l=100, k=300, filter_h=5):
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

def make_idx_data_cv(revs, word_idx_map, word_vector, max_l=100, k=300, filter_h=5):
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

def w2v_mean(word_vector):
    mean_array = np.array(word_vector)
    return np.mean(mean_array,dtype=np.float32)

def learn2learn_add_zeros(word_vector, max_sent_len=100):
    mean_x=[]
    while len(mean_x) < max_sent_len:
        mean_x.append(w2v_mean(word_vector[0]))
    return mean_x

def learn2learn_sent_mean(sent, word_idx_map, word_vector, max_sent_len=100, w2vDim=300):
    mean_x = []
    # pad = filter_h - 1
    # for i in xrange(pad):
    #     mean_x.append(w2v_mean(word_vector[0]))
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            mean_x.append(w2v_mean(word_vector[word_idx_map[word]]))
            if len(mean_x) == max_sent_len:
                break
    while len(mean_x) < max_sent_len:
        mean_x.append(w2v_mean(word_vector[0]))
    return mean_x

def learn2learn_data_bulid(revs, word_idx_map, word_vector, max_text_len=2500, max_sent_len=100, w2vDim=300):
    x_target = []
    for rev in revs:
        sent_mean = learn2learn_sent_mean(rev['text'], word_idx_map, word_vector, max_sent_len, w2vDim)
        x_target.append(sent_mean)
        if len(x_target) == max_text_len:
            break
    while len(x_target) < max_text_len:
        x_target.append(learn2learn_add_zeros(word_vector, max_sent_len))
    return x_target

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
    train_file_path = '/home/zmh/exp_data/learning2learn/train-data/'
    w2v_file = "/home/zmh/exp_data/word_vectors/GoogleNews-vectors-negative300.bin"
    BestLearnRateRead = open('best_learn_rate.txt', 'r')
    TrainFileIdList = []
    TrainBestRateList = []
    for Line in BestLearnRateRead.readlines():
        LineList = Line.split(':')
        TrainFileIdList.append(LineList[0])
        LineList[1] = LineList[1].strip('\r\n')
        TrainBestRateList.append(LineList[1])

#   read train files
    x_target_text_train = []
    y_target_text_train = []
    x_target_text_test = []
    y_target_text_test = []
    i = 0
    for FileId in TrainFileIdList:
        print("FileId:", FileId)
        revs, vocab = build_data_cv(train_file_path+'pos-'+FileId+'.txt', train_file_path+'neg-'+FileId+'.txt')
        if len(revs) == 0:
            continue
        print("load w2v")
        w2v = load_bin_vec(w2v_file, vocab)
        print("generate numpy")
        W, word_idx_map = get_W(w2v)
    #   generate numpy
        x_target_sent = learn2learn_data_bulid(revs, word_idx_map, word_vector=W, max_text_len=2500,
                                               max_sent_len=100, w2vDim=300)
        if i < 150:
            x_target_text_train.append(x_target_sent)
            y_target_text_train.append(TrainBestRateList[i])
        else:
            x_target_text_test.append(x_target_sent)
            y_target_text_test.append(TrainBestRateList[i])
        i += 1

    x_target_text_train_npy = np.array(x_target_text_train)
    x_target_text_test_npy = np.array(x_target_text_test)
    y_target_text_train_npy = np.array(y_target_text_train)
    y_target_text_test_npy = np.array(y_target_text_test)

    np.save('x_train_l2l.npy', x_target_text_train_npy)
    print('x_target_text_train_npy:', x_target_text_train_npy.shape)
    np.save('x_test_l2l.npy', x_target_text_test_npy)
    print('x_target_text_test_npy:', x_target_text_test_npy.shape)

    np.save('y_train_l2l.npy', y_target_text_train_npy)
    print('y_target_text_train_npy:', y_target_text_train_npy.shape)
    np.save('y_test_l2l.npy', y_target_text_test_npy)
    print('y_target_text_test_npy:t', y_target_text_test_npy.shape)
