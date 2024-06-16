import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torchtext.datasets import Multi30k
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
# import spacy
import numpy as np
import random
import math
import jieba
import os
import pickle
import time
import matplotlib.pyplot as plt


class LSTM(torch.nn.Module):
    def __init__(self, hidden_size1, hidden_size2, vocab_size, input_size, num_layers):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, input_size, max_norm=1)
        self.max_len = 30
        self.lstm1 = torch.nn.LSTM(input_size, hidden_size1, num_layers, batch_first=True, bidirectional=False)
        self.lstm2 = torch.nn.LSTM(hidden_size1, hidden_size2, num_layers, batch_first=True, bidirectional=False)
        self.dropout = torch.nn.Dropout(0.1)
        self.line = torch.nn.Linear(hidden_size2 * self.max_len, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embed(x)
        output1, _ = self.lstm1(x)
        output, _ = self.lstm2(output1)
        out_d_reshaped = output.reshape(output.shape[0], (output.shape[1] * output.shape[2]))
        line_o = self.line(out_d_reshaped)
        pred = self.softmax(line_o)
        # print(pred.shape)
        return pred


class GenerateTaskModel(nn.Module):
    def __init__(self, d_model=512, vocab_size=10):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer = nn.Transformer(d_model=d_model, batch_first=True, num_encoder_layers=2, num_decoder_layers=2,
                                          dim_feedforward=512)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=0.1)
        self.predictor = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])
        src_key_padding_mask = self.get_key_padding_mask(src)
        tgt_key_padding_mask = self.get_key_padding_mask(tgt)
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_mask = tgt_mask.to(device)
        src_key_padding_mask = src_key_padding_mask.to(device)
        tgt_key_padding_mask = tgt_key_padding_mask.to(device)
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return out

    def get_key_padding_mask(self, tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask


class TextTensorDataset(Dataset):

    def __init__(self, all_words, maxlen, word_indices):
        sentences = []
        next_word = []
        for i in range(0, len(all_words) - maxlen):
            sentences.append(all_words[i: i + maxlen])
            next_word.append(all_words[i + maxlen])
        print('提取的句子总数:', len(sentences))

        self.inputs = np.zeros((len(sentences), maxlen), dtype='float32')  # 先将每个inputs切成30个词的句子列表，然后将句子中的词转化成index索引
        self.labels = np.zeros((len(sentences)), dtype='float32')
        for i, sentence in enumerate(sentences):
            for t, word in enumerate(sentence):
                self.inputs[i, t] = word_indices[word]
            self.labels[i] = word_indices[next_word[i]]

    def __getitem__(self, item):

        # x = np.expand_dims(self.inputs[item], axis=0)
        # y = np.expand_dims(self.labels[item], axis=0)

        return self.inputs[item], self.labels[item]

    def __len__(self):
        return len(self.inputs)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def content_deal(content):
    ad = '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com'  #去除无意义的广告词
    content = content.replace(ad, '')
    content = content.replace("\u3000", '')
    content = content.replace("=", '')
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


def find_key(dic, value):
    if value not in dic.values():
        return None
    for key in dic:
        if dic[key] == value:
            return key


if __name__ == '__main__':
    stop_word_file = r"cn_stopwords.txt"
    punctuation_file = r"cn_punctuation.txt"
    ll = r"novel/"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    mode = 'Test'

    """语料库预处理，第一次运行，之后可省略"""
    # # 读取停词列表
    stop_word_list = []
    with open(stop_word_file, 'r', encoding='utf-8') as f:
        for line in f:
            stop_word_list.append(line.strip())
    stop_word_list.extend("\u3000")
    stop_word_list.extend(['～', ' ', '没', '听', '一声', '道', '见', '中', '便', '说', '一个', '说道'])
    with open(punctuation_file, 'r', encoding='utf-8') as f:
        for line in f:
            stop_word_list.append(line.strip())
    # 读取段落
    # 处理前删除文件夹内inf.txt
    # char_dict, word_dict = read_novel(ll, stop_word_list)
    # with open('word_dict.pkl', 'wb') as f:
    #     pickle.dump(word_dict, f)
    # with open('char_dict.pkl', 'wb') as f:
    #     pickle.dump(char_dict, f)
    """语料库预处理，第一次运行，之后可省略"""

    # """直接读取保存的数据"""
    with open('word_dict.pkl', 'rb') as f:
        word_dict = pickle.load(f)
    with open('char_dict.pkl', 'rb') as f:
        char_dict = pickle.load(f)
    data_source = ['射雕英雄传']
    for id, wd in enumerate(word_dict):
        if wd in data_source:
            my_dict = word_dict[wd]
            all_words = []
            maxlength = 0
            for sentence in my_dict:
                all_words += sentence
                if len(sentence) > maxlength:
                    maxlength = len(sentence)
            words = sorted(list(set(all_words)))
            word_indices = dict((word, words.index(word)) for word in words)
            if mode == 'Train':
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                mydataset = TextTensorDataset(all_words=all_words, maxlen=30, word_indices=word_indices)
                train_loader = DataLoader(mydataset, batch_size=1024, shuffle=True)
                hidden_size1, hidden_size2, vocab_size, input_size, num_layers = 256, 128, len(words), 128, 2
                model = LSTM(hidden_size1, hidden_size2, vocab_size, input_size, num_layers).to(device)
                criterion = nn.NLLLoss().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
                model.train()
                epoch_num = 25
                for epoch in range(epoch_num):
                    total_loss = 0
                    items = 0
                    for x, target in train_loader:
                        x = Variable(torch.LongTensor(x.numpy())).cuda()  # torch.Size([1024, 30, 1])
                        pred = model(x)
                        # print('pred shape ',  pred.shape)
                        target = Variable(target.view(-1)).cuda()
                        # print('target shape ',  target.shape)
                        loss = criterion(pred, target.long())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss
                        items += 1
                        print('Epoch {}, Step {} Train Loss {}'.format(epoch, items, total_loss))
                torch.save(model, 'seq2seq.pth')
            else:
                model = torch.load('seq2seq.pth')
                device = torch.device("cpu")
                model.to(device)
                model.eval()
                begin_sentence = all_words[1000:1030]
                print('起始句：' + ''.join(begin_sentence))
                src_seq = []
                for word in begin_sentence:
                    src_seq.append(word_indices[word])
                num_words = 100
                generated_seq = []
                for i in range(num_words):
                    src_seq_tensor = torch.LongTensor([src_seq])
                    out = model(src_seq_tensor)
                    y = torch.argmax(out)
                    generated_seq.append(y)
                    src_seq.append(y)
                    src_seq = src_seq[1:]
                doc_generated = []
                for i in range(len(generated_seq)):
                    word = find_key(word_indices, generated_seq[i])
                    doc_generated.append(word)
                print(''.join(doc_generated))




































