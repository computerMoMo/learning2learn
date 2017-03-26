import sys
import os

reload(sys)
import codecs


def ReadFile(filePath, encoding="gbk"):
    with codecs.open(filePath, "r", encoding) as f:
        return f.read()


def WriteFile(filePath, u, encoding="utf-8"):
    with codecs.open(filePath, "w", encoding) as f:
        f.write(u)


def GBK_2_UTF(src, dst):
    content = ReadFile(src, encoding="gb2312")
    WriteFile(dst, content, encoding="utf-8")


if __name__ == '__main__':
    read_filepath = "/opt/exp_data/learning2learn/text-data/aclimdb/train/pos/"
    write_filepath = "train-pos/"
    # for i in range(12500):
    #     if i%100 == 0:
    #         print i
    #     filename = str(i) + "_*"
    #     file_trans_name = filename+"_utf8"
    #     GBK_2_UTF(read_filepath+filename+'.txt', write_filepath+file_trans_name+'.txt')
    i=0
    for filename in os.listdir(read_filepath):
        i+=1
        if i%100 == 0:
            print i
        GBK_2_UTF(read_filepath+filename, write_filepath+filename+'_utf8')
        # with open(read_filepath+filename,'r') as f_read:
        #     with codecs.open(write_filepath+filename+'_utf8','w','utf-8') as f_write:
        #         f_write.write(f_read.read())
        #         f_read.close()
        #         f_write.close()

    print "trans ok!"