import codecs
import re
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

tag2id = {'': 0,
          'B_ns': 1,
          'B_nr': 2,
          'B_nt': 3,
          'M_nt': 4,
          'M_nr': 5,
          'M_ns': 6,
          'E_nt': 7,
          'E_nr': 8,
          'E_ns': 9,
          'o': 0}

id2tag = {0: '',
          1: 'B_ns',
          2: 'B_nr',
          3: 'B_nt',
          4: 'M_nt',
          5: 'M_nr',
          6: 'M_ns',
          7: 'E_nt',
          8: 'E_nr',
          9: 'E_ns',
          10: 'o'}


def vocab_label(input_path):
    with open(input_path, 'wb') as f:
        pickle.dump([tag2id, id2tag], f)


def word_tag(input_path, outpath):
    input_data = codecs.open(input_path, 'r', 'utf-8')
    output_data = codecs.open(outpath, 'w', 'utf-8')
    for line in input_data.readlines():
        line = line.strip().split()
        if len(line) == 0:
            continue
        for word in line:
            word = word.split('/')
            if word[1] != 'o':
                if len(word[0]) == 1:
                    output_data.write(word[0] + '/B_' + word[1] + " ")
                elif len(word[0]) == 2:
                    output_data.write(word[0][0] + '/B_' + word[1] + " ")
                    output_data.write(word[0][1] + '/E_' + word[1] + " ")
                else:
                    output_data.write(word[0][0] + '/B_' + word[1] + " ")
                    for j in word[0][1: len(word[0]) - 1]:
                        output_data.write(j + '/M_' + word[1] + " ")
                    output_data.write(word[0][-1] + '/E_' + word[1] + " ")
            else:
                for j in word[0]:
                    output_data.write(j + '/o' + " ")
        output_data.write('\n')

    input_data.close()
    output_data.close()


def data_label(path):
    data = []
    label = []
    input_data = codecs.open(path, 'r', 'utf-8')
    for line in input_data.readlines():
        line = re.split('[，。；！：？、’‘“”]/[o]', line.strip())
        for sen in line:
            sen = sen.strip().split()
            if len(sen) == 0:
                continue
            data_ = []
            label_ = []
            num_not_o = 0
            for word in sen:
                word = word.split('/')
                data_.append(word[0])
                label_.append(tag2id[word[1]])
                if word[1] != 'o':
                    num_not_o += 1
            if num_not_o != 0:
                data.append(data_)
                label.append(label_)
    input_data.close()
    return data, label


def vocab_build(corpus_path, word2id_path, min_count):
    data, _ = data_label(corpus_path)
    word2id = {}
    id2word = {}
    for sen in data:
        for word in sen:
            # if word.isdigit():
            #     word = '<NUM>'
            # elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            #     word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]
            else:
                word2id[word][1] += 1
    low_fre_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq <= min_count and word != '<NUM>' and word != '<ENG>':
            low_fre_words.append(word)
    for word in low_fre_words:
        del word2id[word]
    # print(low_fre_words)
    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0
    id2word = {word2id[word]: word for word in word2id.keys()}
    with open(word2id_path, 'wb') as f:
        pickle.dump([word2id, id2word], f)
    return word2id, id2word


def word_padding(data_pad, max_len):
    seq_list = []
    for sen in data_pad:
        seq_list.append(word2id[sen])
    list_ = []
    print('seq_list:', seq_list)
    if len(seq_list) >= max_len:
        return seq_list[:max_len]
    seq_list.extend([0] * (max_len - len(seq_list)))
    print('new:', len(seq_list))
    print('new:', seq_list)
    return seq_list

def label_padding(ids, max_len):
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))
    return ids


def data_build(data, label, data_path, max_len):
    x = []
    for data_ in data:
        x.append(word_padding(data_, max_len=max_len))
    print('x:', x)
    y = []
    for label_ in label:
        y.append(label_padding(label_, max_len=max_len))
    print('y:', y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=43)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=43)
    print('x_train:', x_train)
    print('Finished creating the data generator.')
    with open(data_path, 'wb') as f:
        pickle.dump([x_train, y_train, x_test, y_test, x_valid, y_valid], f)
        # pickle.dump(y_train, f)
        # pickle.dump(x_test, f)
        # pickle.dump(y_test, f)
        # pickle.dump(x_valid, f)
        # pickle.dump(y_valid, f)
    print('** Finished saving the data.')


if __name__ == '__main__':
    input_path = 'MSRA/train1.txt'
    output_path = 'MSRA/wordtag.txt'
    word_tag(input_path, output_path)
    data, label = data_label(output_path)
    print(data[0:5])
    print(label[0:5])
    label2id_path = 'MSRA/label2id2label.pkl'
    # id2label_path = 'MSRA/id2label.pkl'
    vocab_label(label2id_path)
    word2id_path = 'MSRA/vocab_word2id2word.pkl'
    # id2word_path = 'MSRA/vocab_id2word.pkl'
    word2id, id2word = vocab_build(output_path, word2id_path, min_count=0)
    print(word2id)
    print(id2word)
    data_path = 'MSRA/data.pkl'
    data_build(data, label, data_path, max_len=20)
