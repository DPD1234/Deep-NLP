import jieba
import os
import numpy as np
from gensim import corpora, models
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score


def para_t_freq(lda_word, cor_word):
    para_topic = []
    for i in cor_word:
        topic = lda_word.get_document_topics(i)
        init = np.zeros(lda_word.num_topics)
        for ii, value in topic:
            init[ii] = value
        para_topic.append(init)
    return para_topic


def content_deal(content):
    ad = '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com'
    content = content.replace(ad, '')
    content = content.replace("\u3000", '')
    return content


def read_novel(path, stop_word_list):
    file_list = os.listdir(path)
    word_list = {}
    char_list = {}
    for file in file_list:
            novel_path = r"novel/" + file
            char = []
            word = []

            with open(novel_path, 'r', encoding='gb18030') as f:
                content = f.read()
                word_list0 = content_deal(content)
                # 大于500词的段落
                for para in word_list0.split('\n'):
                    if len(para) < 1:
                        continue
                    char.append([char for char in para if char not in stop_word_list and char != ' '])
                    word.append([word for word in jieba.lcut(para) if word not in stop_word_list and word != ' '])
                    file_name = os.path.splitext(file)[0]
                    f.close()
            char_list[file_name] = char
            word_list[file_name] = word

    return char_list, word_list

def fenci_method(data_list,data_label_ids):
    fenci_word = []
    fenci_word_label = []
    fenci_char = []
    fenci_char_label = []

    for index, text in enumerate(data_list):
        fenci = [word for word in jieba.lcut(sentence=text) if word not in stop_word_list]
        fenci_word.append(fenci)
        fenci_word_label.append(data_label_ids[index])
        # 字模式
        t = []
        for word1 in fenci:
            t.extend([char for char in word1])
        fenci_char.append(t)
        fenci_char_label.append(data_label_ids[index])
    return fenci_word[:100],fenci_word_label[:100],fenci_char[:100],fenci_char_label[:100]

def dic_matrix(fenci_word,fenci_char):
    dic_word = corpora.Dictionary(fenci_word)
    cor_word = [dic_word.doc2bow(i) for i in fenci_word]
    dic_char = corpora.Dictionary(fenci_char)
    cor_char = [dic_char.doc2bow(i) for i in fenci_char]
    return dic_word,cor_word,dic_char,cor_char


def evenly_sample_data(word_labels, total_samples):
    label_counts = np.bincount(word_labels)  # 统计每个标签出现的次数
    num_labels = len(label_counts)
    # 计算每个标签需要取得的数据数量
    samples_per_label = total_samples // num_labels
    # 初始化结果列表
    sampled_data = []
    # 遍历每个标签，取得相应数量的数据
    for label in range(num_labels):
        label_indices = [index for index, value in enumerate(word_labels) if value == label]
        if len(label_indices) == 0:
            continue
        label_data = np.random.choice(label_indices, samples_per_label, replace=True)  # 从当前标签的数据中随机抽取
        sampled_data.extend(label_data)
    # 如果总数不够，随机从所有数据中取补足数量
    while len(sampled_data) < total_samples:
        random_index = np.random.randint(len(word_labels))
        sampled_data.append(random_index)
    return sampled_data

if __name__ == '__main__':
    stop_word_file = r"cn_stopwords.txt"
    ll = r"novel/"
    # 读取停词列表
    stop_word_list = []
    with open(stop_word_file, 'r', encoding='utf-8') as f:
        for line in f:
            stop_word_list.append(line.strip())
    stop_word_list.extend("\u3000")
    stop_word_list.extend(['～', ' ', '没', '听', '一声', '道', '见', '中', '便', '说', '一个', '说道'])

    with open('word_dict.pkl', 'rb') as f:
        word_dict = pickle.load(f)
    with open('char_dict.pkl', 'rb') as f:
        char_dict = pickle.load(f)
    book_names = os.listdir(ll)
    for i in range(len(book_names)):
        book_names[i] = book_names[i].split('.')[0]
    book_names_id = {name: i for i, name in enumerate(book_names)}
    # data_label_ids = [book_names_id[label] for label in data_label]
    tokens = 20
    topics = 200
    "词"
    word_corpus = []
    word_labels = []
    for file_name, para in word_dict.items():
        word = []
        for words in para:
            word.extend(words)
            if len(word) < tokens:
                continue
            else:
                word_corpus.append(word[:tokens])
                word_labels.append(file_name)
                word = []
    word_label_ids = [book_names_id[label] for label in word_labels]
    sample_labels = evenly_sample_data(word_label_ids, 1000)

    word_cor = [word_corpus[i] for i in sample_labels]
    word_label = [word_label_ids[i] for i in sample_labels]
    id2word = corpora.Dictionary(word_cor)
    train_word, test_word, train_word_labels, test_word_labels = train_test_split(word_cor, word_label,
                                                                                  test_size=0.1, random_state=42)
    train_cor = [id2word.doc2bow(text) for text in train_word]
    test_cor = [id2word.doc2bow(text) for text in test_word]
    lda_model = models.ldamodel.LdaModel(corpus=train_cor, num_topics=topics, id2word=id2word,
                                         random_state=100, chunksize=800, passes=10,
                                         alpha='auto', per_word_topics=True, dtype=np.float64)
    train_para_topic = para_t_freq(lda_model, train_cor)
    test_para_topic = para_t_freq(lda_model, test_cor)
    classifier = SVC()
    accuracy = np.mean(cross_val_score(classifier, train_para_topic+test_para_topic, train_word_labels+test_word_labels, cv=10))
    print("word Accuracy:", accuracy)
    "字"
    char_corpus = []
    char_labels = []
    for file_name, para in char_dict.items():
        char = []
        for chars in para:
            char.extend(chars)
            if len(char) < tokens:
                continue
            else:
                char_corpus.append(char[:tokens])
                char_labels.append(file_name)
                char = []
    char_label_ids = [book_names_id[label] for label in char_labels]
    sample_labels = evenly_sample_data(char_label_ids, 1000)

    char_cor = [char_corpus[i] for i in sample_labels]
    char_label = [char_label_ids[i] for i in sample_labels]
    id2word = corpora.Dictionary(char_cor)
    train_char, test_char, train_char_labels, test_char_labels = train_test_split(char_cor, char_label,
                                                                                  test_size=0.1, random_state=42)
    train_cor = [id2word.doc2bow(text) for text in train_char]
    test_cor = [id2word.doc2bow(text) for text in test_char]
    lda_model = models.ldamodel.LdaModel(corpus=train_cor, num_topics=topics, id2word=id2word,
                                         random_state=100, chunksize=800, passes=10,
                                         alpha='auto', per_word_topics=True, dtype=np.float64)
    train_para_topic = para_t_freq(lda_model, train_cor)
    test_para_topic = para_t_freq(lda_model, test_cor)
    classifier = SVC()
    accuracy = np.mean(
        cross_val_score(classifier, train_para_topic + test_para_topic, train_char_labels + test_char_labels, cv=10))
    print("char Accuracy:", accuracy)