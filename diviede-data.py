import sys
import re
import os
import random

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

    if(int(sys.argv[1]) % 2 == 0):
        print "original data path:/opt/exp_data/learning2learn/text-data/aclimdb/train"
        read_filepath_pos = "/opt/exp_data/learning2learn/text-data/aclimdb/train/pos/"
        read_filepath_neg = "/opt/exp_data/learning2learn/text-data/aclimdb/train/neg/"
    else:
        print "original data path:/opt/exp_data/learning2learn/text-data/aclimdb/test"
        read_filepath_pos = "/opt/exp_data/learning2learn/text-data/aclimdb/test/pos/"
        read_filepath_neg = "/opt/exp_data/learning2learn/text-data/aclimdb/test/neg/"

    write_filepath = "/opt/exp_data/learning2learn/text-data/train-data/"

    w_pos = open(write_filepath+'pos-'+sys.argv[1]+'.txt','w')
    w_neg = open(write_filepath + 'neg-' + sys.argv[1] + '.txt', 'w')

    # read pos
    i = 0
    print"generate pos datafile:"+"pos-"+sys.argv[1]+".txt"
    for filename in os.listdir(read_filepath_pos):
        with open(read_filepath_pos+filename, 'r') as f_r:
            for each_line in f_r.readlines():
                each_line = clean(each_line)+'\n'
                r = random.randint(1,10)
                if r == 5:
                    w_pos.write(each_line)
                    i += 1
    print "pos datafile total size: "+str(i) + "lines"

    #read neg
    i = 0
    print"generate neg datafile:" + "neg-" + sys.argv[1] + ".txt"
    for filename in os.listdir(read_filepath_neg):
        with open(read_filepath_neg + filename, 'r') as f_r:
            for each_line in f_r.readlines():
                each_line = clean(each_line) + '\n'
                r = random.randint(1, 10)
                if r == 5:
                    w_neg.write(each_line)
                    i += 1
    print "neg datafile total size: " + str(i) + "lines"

    print "done!"