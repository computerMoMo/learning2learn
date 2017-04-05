# -*- coding:utf-8 -*-
from __future__ import print_function
import sys
import re
import numpy as np
import gensim
from collections import defaultdict

# word vectors object
global word_vectors


def build_data_cv(file_pos, file_neg, cv=10, clean_string=True):
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


def clean_str(string):
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
    return string.strip()


def read_word_vectors(sent, max_len=100, word_dim=300):
    sent_x_target = []
    word_list = sent.split()
    for word in word_list:
        if word in word_vectors.vocab:
            sent_x_target.append(word_vectors.word_vec(word))
        else:
            vector_random = np.random.uniform(-0.25, 0.25, word_dim)
            sent_x_target.append(vector_random)
        if len(sent_x_target) >= max_len:
            break
    zero_vector = np.zeros(shape=(word_dim,))
    while len(sent_x_target) < max_len:
        sent_x_target.append(zero_vector)
    return sent_x_target


def get_numpy(revs, max_len=100, word_dim=300):
    x_target = []
    y_target = []
    for rev in revs:
        sent_x_target = read_word_vectors(rev['text'], max_len, word_dim)
        x_target.append(sent_x_target)
        y_target.append(rev['y'])

    x_target = np.array(x_target, dtype=np.float32)
    y_target = np.array(y_target, dtype=np.uint8)
    return x_target, y_target


if __name__ == '__main__':
    # load word2vec model
    print("load word2vec model")
    word_vectors = gensim.models.keyedvectors.KeyedVectors.\
        load_word2vec_format('/home/zmh/exp_data/word_vectors/GoogleNews-vectors-negative300.bin', binary=True)

    # open train files
    print("read train files")
    train_file_path = '/home/zmh/exp_data/learning2learn/train-data/'
    revs, vocab = build_data_cv(train_file_path + "pos-0" + ".txt", train_file_path + "neg-0" + ".txt", cv=10,
                                clean_string=True)
    # revs, vocab = build_data_cv(train_file_path+'test-pos.txt', train_file_path+'test-neg.txt', cv=10,
    #                             clean_string=True)
    # generate numpy
    print("generate numpy")
    x_target_array, y_target_array = get_numpy(revs, 100, 300)
    print(x_target_array.shape)
    print(y_target_array.shape)
    np.save('exp-data/x-train.npy', x_target_array)
    np.save('exp-data/y-train.npy', y_target_array)
