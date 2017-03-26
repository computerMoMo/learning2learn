# -*- coding: gb2312 -*-
import sys
import os

if __name__ == '__main__':
    # file_name = sys.argv[1]
    # file_name = 'E:\\my_project\\learning2learn\\exp-data\\result\\testresult-300-new.txt'
    file_path = 'E:\\my_project\\learning2learn\\exp-data\\result\\'
    write_file_name = 'best_learn_rate.txt'
    f_w = open(write_file_name, 'w')
    for file_name in os.listdir(file_path):
        res_read = open(file_path+file_name, 'r')
        str_line = res_read.readline()

        while True:
            # id_line = res_read.readline()
            if not str_line:
                break
            str_line = str_line.strip('\n')
            id_line_list = str_line.split('-')
            file_id = id_line_list[1]
            for _ in range(0, 34):
                str_line = res_read.readline()
                if not str_line:
                    break
            str_line = res_read.readline()
            str_line = str_line.strip('\n')
            str_line_list = str_line.split('：')
            if len(str_line_list) >= 2:
                acc_list = str_line_list[1].split(' ')
                acc = acc_list[0]
            else:
                acc = '-1'
            if acc != '-1':
                f_w.write(file_id+':'+acc+'\n')
            while True:
                str_line = res_read.readline()
                if not str_line:
                    break
                if len(str_line) > 1:
                    break
        res_read.close()

    f_w.close()
    print "done"
