# -*- coding: utf-8 -*-
import math
import os
import matplotlib.pyplot as plt
import jieba
from collections import Counter
import re


def split_string(string):
    result = []
    for char in string:
        result.append(char)
    return result


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                L.append(os.path.join(root, file))
    return L


def get_wordlist(files_path, stopwords_path):
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
    return word_list


def get_charlist(files_path, stopwords_path):
    with open(stopwords_path, encoding='utf-8') as f:
        con = f.readlines()
        stop_words = set()  # 集合可以去重
        for i in con:
            i = i.replace("\n", "")  # 去掉读取每一行数据的\n
            stop_words.add(i)
    text_all = str('')
    char_list = []
    for i in range(1, len(files_path)):
        path = files_path[i]
        with open(path, 'r', errors='ignore') as f:
            data = f.read()
        text_all += (data + ' ')
    text_all = "".join(re.findall('[\u4e00-\u9fa5]+', text_all, re.S))
    character_all = split_string(text_all)
    for character in character_all:
        if character not in stop_words:
            char_list.append(character)
    return char_list


def get_unigram_list(cutword_list):
    res = cutword_list
    word_num = len(res)
    res_freq = Counter(res)
    res_freq = res_freq.most_common()
    return res, res_freq, word_num


def get_bigram_list(cutword_list):
    if len(cutword_list) <= 1:
        return []
    res = []
    for i in range(len(cutword_list) - 1):
        res.append(cutword_list[i] + "s" + cutword_list[i + 1])
    word_num = len(res)
    res_freq = Counter(res)
    res_freq = res_freq.most_common()
    return res, res_freq, word_num


def get_trigram_list(cutword_list):
    if len(cutword_list) <= 2:
        return []
    res = []
    for i in range(len(cutword_list) - 2):
        res.append(cutword_list[i] + cutword_list[i + 1] + "s" + cutword_list[i + 2])
    word_num = len(res)
    res_freq = Counter(res)
    res_freq = res_freq.most_common()
    return res, res_freq, word_num


def get_1wordsame_list(bigram_list):
    res = []
    for bi_words in bigram_list:
        res.append(bi_words.split("s")[0])
    res_freq = Counter(res)
    res_freq = dict(res_freq.most_common())
    return res, res_freq


def get_2wordsame_list(trigram_list):
    res = []
    for tri_words in trigram_list:
        res.append(tri_words.split("s")[0])
    res_freq = Counter(res)
    res_freq = dict(res_freq.most_common())
    return res, res_freq


def cal_entropy_unigram(uniword_num, unigram_freq):
    entropy = 0
    for uni_word in unigram_freq:
        p = uni_word[1] / uniword_num
        entropy -= p*math.log(p, 2)
    return entropy


def cal_entropy_bigram(biword_num, bigram_freq, same1word_freq):
    entropy = 0
    for bi_word in bigram_freq:
        p12 = bi_word[1] / biword_num
        same1word = bi_word[0].split("s")[0]
        p2_1 = bi_word[1] / same1word_freq[same1word]
        entropy -= p12*math.log(p2_1, 2)
    return entropy


def cal_entropy_trigram(triword_num, trigram_freq, same2word_freq):
    entropy = 0
    for tri_word in trigram_freq:
        p123 = tri_word[1] / triword_num
        same2word = tri_word[0].split("s")[0]
        p3_12 = tri_word[1] / same2word_freq[same2word]
        entropy -= p123*math.log(p3_12, 2)
    return entropy


if __name__ == '__main__':
    files_path = file_name(r'E:\Deeplearning\Zipf law')
    word_list = get_wordlist(files_path, stopwords_path='E:\Deeplearning\Zipf law\DLNLP2023-main\cn_stopwords.txt')
    char_list = get_charlist(files_path, stopwords_path='E:\Deeplearning\Zipf law\DLNLP2023-main\cn_stopwords.txt')
    unigram_charlist, unigram_charfreq, unichar_num_sum = get_unigram_list(char_list)
    unigram_wordlist, unigram_wordfreq, uniword_num_sum = get_unigram_list(word_list)
    bigram_charlist, bigram_charfreq, bichar_num_sum = get_bigram_list(char_list)
    bigram_wordlist, bigram_wordfreq, biword_num_sum = get_bigram_list(word_list)
    trigram_charlist, trigram_charfreq, trichar_num_sum = get_trigram_list(char_list)
    trigram_wordlist, trigram_wordfreq, triword_num_sum = get_trigram_list(word_list)
    same1char_list, same1char_freq = get_1wordsame_list(bigram_charlist)
    same1word_list, same1word_freq = get_1wordsame_list(bigram_wordlist)
    same2char_list, same2char_freq = get_2wordsame_list(trigram_charlist)
    same2word_list, same2word_freq = get_2wordsame_list(trigram_wordlist)
    entropy = cal_entropy_unigram(bichar_num_sum, bigram_charfreq)
    print("以字为单位使用一元模型计算中文平均信息熵的结果为：", entropy, "比特/词")
    entropy = cal_entropy_unigram(biword_num_sum, bigram_wordfreq)
    print("以词为单位使用一元模型计算中文平均信息熵的结果为：", entropy, "比特/词")
    entropy = cal_entropy_bigram(bichar_num_sum, bigram_charfreq, same1char_freq)
    print("以字为单位使用二元模型计算中文平均信息熵的结果为：", entropy, "比特/词")
    entropy = cal_entropy_bigram(biword_num_sum, bigram_wordfreq, same1word_freq)
    print("以词为单位使用二元模型计算中文平均信息熵的结果为：", entropy, "比特/词")
    entropy = cal_entropy_trigram(trichar_num_sum, trigram_charfreq, same2char_freq)
    print("以字为单位使用三元模型计算中文平均信息熵的结果为：", entropy, "比特/词")
    entropy = cal_entropy_trigram(triword_num_sum, trigram_wordfreq, same2word_freq)
    print("以词为单位使用三元模型计算中文平均信息熵的结果为：", entropy, "比特/词")




















