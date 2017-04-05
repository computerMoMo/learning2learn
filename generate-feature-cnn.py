# -*- coding:utf-8 -*-
from __future__ import print_function
import sys
import re
import numpy as np
import gensim
import keras
from collections import defaultdict
from keras.models import Model
from keras.models import load_model

# word vectors object
global word_vectors


def build_data_cv(file_pos, file_neg, cv=10, clean_string=True):
    revs = []
    vocab = defaultdict(float)

    try:
        with open(file_pos, "rb") as f:
            for str_line in f:
                str_rev = []
                str_rev.append(str_line.strip())
                if clean_string:
                    orig_rev = clean_str(" ".join(str_rev))
                else:
                    orig_rev = " ".join(str_rev).lower()
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                datum = {"y": 0,
                         "text": orig_rev,
                         "num_words": len(orig_rev.split()),
                         "split": np.random.randint(0, cv)}
                revs.append(datum)
    except:
        print(file_pos + " doesn't exist")
        revs = []
        return revs, vocab

    try:
        with open(file_neg, "rb") as f:
            for str_line in f:
                str_rev = []
                str_rev.append(str_line.strip())
                if clean_string:
                    orig_rev = clean_str(" ".join(str_rev))
                else:
                    orig_rev = " ".join(str_rev).lower()
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                datum = {"y": 1,
                         "text": orig_rev,
                         "num_words": len(orig_rev.split()),
                         "split": np.random.randint(0, cv)}
                revs.append(datum)
    except:
        print(file_neg + " doesn't exist")
        revs = []
        return revs, vocab

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


def get_numpy(revs, max_len=100, word_dim=300, max_sentences=2500):
    x_target = []
    y_target = []
    for rev in revs:
        sent_x_target = read_word_vectors(rev['text'], max_len, word_dim)
        x_target.append(sent_x_target)
        y_target.append(rev['y'])
        if len(x_target) == max_sentences:
            break
    while len(x_target) < max_sentences:
        x_target.append(np.zeros(shape=(100, 300)))

    x_target = np.array(x_target, dtype=np.float32)
    y_target = np.array(y_target, dtype=np.uint8)
    return x_target, y_target


if __name__ == '__main__':
    word_dim = 300
    sentence_max_len = 100
# load word2vec model
    print("load word2vec model")
    w2v_file_path = '/home/zmh/exp_data/word_vectors/'
    word_vectors = gensim.models.keyedvectors.KeyedVectors.\
        load_word2vec_format(w2v_file_path+'GoogleNews-vectors-negative300.bin', binary=True)

# open train files
    train_file_path = '/home/zmh/exp_data/learning2learn/train-data/'
    best_learn_rate_reader = open('best_learn_rate.txt', 'r')
    train_file_id_list = []
    train_file_rate_list = []
    for line in best_learn_rate_reader.readlines():
        line_list = line.split(':')
        train_file_id_list.append(line_list[0])
        line_list[1] = line_list[1].strip('/r/n')
        train_file_rate_list.append(line_list[1])

# generate features

    # target numpy
    x_total_train = []
    y_total_train = []
    x_total_test = []
    y_total_test = []

    # load cnn model
    text_cnn_model_path = 'exp-data/text-cnn-model.h5'
    text_cnn_model = load_model(text_cnn_model_path)
    layer_name = 'Dense1'
    middle_layer_model = Model(inputs=text_cnn_model.input, outputs=text_cnn_model.get_layer(layer_name).output)
    print('generate feature with cnn')
    train_file_sum = 0
    for file_id in train_file_id_list:
        print('\n\nfile id:', file_id)
        text_revs, text_vocab = build_data_cv(train_file_path+'pos-'+file_id+'.txt', train_file_path+'neg-'+file_id+'.txt')
        if len(text_revs) == 0:
            train_file_sum += 1
            continue
        # generate text numpy
        print("generate numpy")
        x_text_array, y_text_array = get_numpy(text_revs, sentence_max_len, word_dim)
        print('x_text_array shape', x_text_array.shape)

        # generate text feature with cnn
        x_text_array_train = x_text_array.reshape(x_text_array.shape[0], x_text_array.shape[1], word_dim, 1)
        middle_layer_model_output = middle_layer_model.predict(x_text_array_train, batch_size=128)
        text_feature_array = np.asarray(middle_layer_model_output, dtype=np.float32)
        print('text_feature_array shape', text_feature_array.shape)

        # add to total numpy
        if train_file_sum < 150:
            x_total_train.append(text_feature_array)
            y_total_train.append(train_file_rate_list[train_file_sum])
        else:
            x_total_test.append(text_feature_array)
            y_total_test.append(train_file_rate_list[train_file_sum])
        train_file_sum += 1

    # save numpy
    x_total_train_array = np.asarray(x_total_train, dtype=np.float32)
    y_total_train_array = np.asarray(y_total_train, dtype=np.float32)
    x_total_test_array = np.asarray(x_total_test, dtype=np.float32)
    y_total_test_array = np.asarray(y_total_test, dtype=np.float32)

    print("\n\n\nx_total_train shape:", x_total_train_array.shape)
    np.save('x_cnn_feature_train.npy', x_total_train_array)
    print("y_total_train shape:", y_total_train_array.shape)
    np.save('y_cnn_feature_train.npy', y_total_train_array)
    print("x_total_test shape:", x_total_test_array.shape)
    np.save('x_cnn_feature_test.npy', x_total_test_array)
    print("y_total_test shape:", y_total_test_array.shape)
    np.save('y_cnn_feature_test.npy', y_total_test_array)
    print("\n\n\nprogram completed")
