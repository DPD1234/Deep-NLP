# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import jieba
from collections import Counter
import re


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                L.append(os.path.join(root, file))
    return L


def getCharacter(files_path, stopwords_path):
    with open(stopwords_path, encoding='utf-8') as f:
        con = f.readlines()
        stop_words = set()  # 集合可以去重
        for i in con:
            i = i.replace("\n", "")  # 去掉读取每一行数据的\n
            stop_words.add(i)
    text_all = str('')
    word_list = []
    for i in range(1, len(files_path)):
        path = files_path[i]
        with open(path, 'r', errors='ignore') as f:
            data = f.read()
        text_all += (data + ' ')
    text_all = "".join(re.findall('[\u4e00-\u9fa5]+', text_all, re.S))
    for word in jieba.lcut(text_all):
        if word not in stop_words:
            word_list.append(word)
    word_freq_list = Counter(word_list)
    word_freq_list = word_freq_list.items()
    return word_freq_list


def draw_zipf(word_freq_list):
    word_freq_list = sorted(word_freq_list, key=lambda x: x[1], reverse=True)
    ranks = []
    freqs = []
    for rank, value in enumerate(word_freq_list):  # 0 ('的', 87343)
        ranks.append(rank + 1)
        freqs.append(value[1])
        rank += 1
    plt.loglog(ranks, freqs)
    plt.xlabel('汉字频数', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.ylabel('汉字名次', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.grid(True)
    plt.show()


files_path = file_name(r'E:\Deeplearning\Zipf law')
word_freq = getCharacter(files_path, stopwords_path='E:\Deeplearning\Zipf law\DLNLP2023-main\cn_stopwords.txt')
draw_zipf(word_freq)



