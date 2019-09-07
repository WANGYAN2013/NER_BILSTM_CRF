import pickle
import random
import tensorflow as tf
import logging
from model_lstm_crf import Model
import numpy as np
import re


logging.basicConfig(level=logging.INFO, format="%(asctime)s  - %(message)s")
logger = logging.getLogger(__name__)


def sentence2id(sent, word2id):
    sentence_id = []
    for word in sent:
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def batch_yield(inputs, targets, batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield np.array(inputs)[excerpt], np.array(targets)[excerpt]  # 提取相应的样本数据和标签数据


def read_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def calculate(x, y, id2word, id2tag, res=[]):
    entity = []
    for i in range(len(x)):  # for every sen
        for j in range(len(x[0])):  # for every word
            if x[i][j] == 0 or y[i][j] == 0:
                continue
            if id2tag[y[i][j]][0] == 'B':
                entity = [id2word[x[i][j]] + '/' + id2tag[y[i][j]]]
            elif id2tag[y[i][j]][0] == 'M' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]] + '/' + id2tag[y[i][j]])
            elif id2tag[y[i][j]][0] == 'E' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]] + '/' + id2tag[y[i][j]])
                entity.append(str(i))
                entity.append(str(j))
                res.append(entity)
                entity = []
            else:
                entity = []
    return res


def get_entity(x, y, id2tag):
    entity = ""
    res = []
    for i in range(len(x)):  # for every sen
        for j in range(len(x[0])):  # for every word
            if y[i][j] == 0:
                continue
            if id2tag[y[i][j]][0] == 'B':
                entity = id2tag[y[i][j]][1:] + ':' + x[i][j]
            elif id2tag[y[i][j]][0] == 'M' and len(entity) != 0:
                entity += x[i][j]
            elif id2tag[y[i][j]][0] == 'E' and len(entity) != 0:
                entity += x[i][j]
                res.append(entity)
                entity = []
            else:
                entity = []
    return res


def write_entity(outp, x, y, id2tag):
    '''
    注意，这个函数每次使用是在文档的最后添加新信息。
    '''
    entity = ''
    for i in range(len(x)):
        if y[i] == 0:
            continue
        if id2tag[y[i]][0] == 'B':
            entity = id2tag[y[i]][2:] + ':' + x[i]
        elif id2tag[y[i]][0] == 'M' and len(entity) != 0:
            entity += x[i]
        elif id2tag[y[i]][0] == 'E' and len(entity) != 0:
            entity += x[i]
            outp.write(entity + ' ')
            entity = ''
        else:
            entity = ''
    return


def param(epochs, lr, embedding_dim, sen_len, batch_size, embedding_size, tag_size, pretrained, clip_grad, dropout_keep_embed, dropout_keep_bilstm):
    config = {}
    config["epochs"] = epochs
    config["lr"] = lr
    config["embedding_dim"] =embedding_dim
    config["sen_len"] = sen_len
    config["batch_size"] = batch_size
    config["embedding_size"] = embedding_size
    config["tag_size"] = tag_size
    config["pretrained"] = pretrained
    config["clip_grad"] = clip_grad
    config["dropout_keep_embed"] = dropout_keep_embed
    config["dropout_keep_bilstm"] = dropout_keep_bilstm
    return config


def padding(ids, max_len):
    if len(ids) >= max_len:
        return ids[:max_len]
    else:
        ids.extend([0] * (max_len - len(ids)))
        return ids


def train(model, sess, saver, epochs, batch_size, x_train, y_train, x_test, y_test, id2word, id2tag):
    batch_num = int(len(x_train) / batch_size)
    batch_num_test = int(len(x_test) / batch_size)
    for epoch in range(epochs):
        for batch in range(batch_num):
        # for batch, (x_, y_) in enumerate(batch_yield(x_train, y_train, batch_size=batch_size, shuffle=True)):
            start = batch * batch_size % int(len(x_train))
            end = min(start + batch_size, int(len(x_train)))
            # x_batch, y_batch = x_, y_
            x_batch, y_batch = np.array(x_train)[start: end], np.array(y_train)[start: end]
            # print('yes')
            feed_dict = {model.input_data: x_batch, model.labels: y_batch}
            pre, _ = sess.run([model.viterbi_sequence, model.train_op], feed_dict)
            acc = 0
            if batch % 200 == 0:
                for i in range(len(y_batch)):
                    for j in range(len(y_batch[0])):
                        if y_batch[i][j] == pre[i][j]:
                            acc += 1
                logging.info({'Acc', float(acc) / (len(y_batch) * len(y_batch[0]))})
        path_name = "./model/model" + str(epoch) + ".ckpt"
        print(path_name)
        if epoch % 3 == 0:
            saver.save(sess, path_name)
            print("model has been saved")
            entityres = []
            entityall = []
            for batch in range(batch_num):
                start__ = batch * batch_size % int(len(x_train))
                end__ = min(start__ + batch_size, int(len(x_train)))
                x_batch, y_batch = np.array(x_train)[start__: end__], np.array(y_train)[start__: end__]
                # print x_batch.shape
                feed_dict = {model.input_data: x_batch, model.labels: y_batch}
                pre = sess.run([model.viterbi_sequence], feed_dict)
                pre = pre[0]
                entityres = calculate(x_batch, pre, id2word, id2tag, entityres)
                entityall = calculate(x_batch, y_batch, id2word, id2tag, entityall)
            jiaoji = [i for i in entityres if i in entityall]
            if len(jiaoji) != 0:
                zhun = float(len(jiaoji)) / len(entityres)
                zhao = float(len(jiaoji)) / len(entityall)
                print("train")
                logging.info({'epoch': epoch+1, 'Precision': zhun, 'Recall': zhao, 'F1': (2 * zhun * zhao) / (zhun + zhao)})
            else:
                print("P:", 0)

            entityres = []
            entityall = []
            for batch in range(batch_num_test):
                start_ = batch * batch_size % int(len(x_test))
                end_ = min(start_ + batch_size, int(len(x_test)))
                x_batch, y_batch = np.array(x_test)[start_: end_], np.array(y_test)[start_: end_]
                feed_dict = {model.input_data: x_batch, model.labels: y_batch}
                pre = sess.run([model.viterbi_sequence], feed_dict)
                pre = pre[0]
                entityres = calculate(x_batch, pre, id2word, id2tag, entityres)
                entityall = calculate(x_batch, y_batch, id2word, id2tag, entityall)
            jiaoji = [i for i in entityres if i in entityall]
            if len(jiaoji) != 0:
                zhun = float(len(jiaoji)) / len(entityres)
                zhao = float(len(jiaoji)) / len(entityall)
                print("test")
                logging.info({'epoch': epoch+1, 'Precision': zhun, 'Recall': zhao,  'F1': (2 * zhun * zhao) / (zhun + zhao)})
            else:
                print("P:", 0)


def test_input(model, sess, word2id, id2tag, batch_size, max_len):
    while True:
        text = input("Enter your input: ")
        text = re.split(u'[，。！？、‘’“”（）]', text)
        text_id = []
        for sen in text:
            word_id = []
            for word in sen:
                if word in word2id:
                    word_id.append(word2id[word])
                else:
                    word_id.append(word2id['<UNK>'])
            text_id.append(padding(word_id, max_len=max_len))
        zero_padding = []
        zero_padding.extend([0] * max_len)
        text_id.extend([zero_padding] * (batch_size - len(text_id)))
        feed_dict = {model.input_data: text_id}
        pre = sess.run([model.viterbi_sequence], feed_dict)
        entity = get_entity(text, pre[0], id2tag)
        print('result:')
        for i in entity:
            print(i)


if __name__ == '__main__':
    data_path = 'D:\\Git\\实体识别\\my_ner\\MSRA\\data.pkl'
    data_ = read_data(data_path)
    x_train = data_[0]
    y_train = data_[1]
    x_test = data_[2]
    y_test = data_[3]
    x_valid = data_[4]
    y_valid = data_[5]

    word2id_path = 'D:\\Git\\实体识别\\my_ner\\MSRA\\vocab_word2id2word.pkl'
    word2id_ = read_data(word2id_path)
    word2id = word2id_[0]
    id2word = word2id_[1]

    tag2id_path = 'D:\\Git\\实体识别\\my_ner\\MSRA\\label2id2label.pkl'
    tag2id_ = read_data(tag2id_path)
    tag2id = tag2id_[0]
    id2tag = tag2id_[1]

    print("train len:", len(x_train))
    print("test len:", len(x_test))
    print("word2id len", len(word2id))
    print('Creating the data generator ...')

    # data_train = batch_yield(inputs=x_train, targets=y_train, batch_size=config['batch_size'], shuffle=True)
    # data_valid = batch_yield(inputs=x_valid, targets=y_valid, batch_size=config['batch_size'], shuffle=False)
    # data_test = batch_yield(inputs=x_test, targets=y_test, batch_size=config['batch_size'],  shuffle=False)

    print('Finished creating the data generator.')
    #
    # print("begin to train...")
    # config = param(epochs=31, lr=0.001, embedding_dim=100, batch_size=32, sen_len=len(x_train[0]), embedding_size=len(word2id)+1, tag_size=len(tag2id), pretrained=False, clip_grad=5.0, dropout_keep_embed=0.5, dropout_keep_bilstm=0.5)
    # model = Model(config, embedding_pretrained=[])
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     saver = tf.train.Saver()
    #     train(model, sess, saver, epochs=config['epochs'], batch_size=config['batch_size'], x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, id2word=id2word, id2tag=id2tag)

    print("begin to test...")
    config = param(epochs=31, lr=0.001, embedding_dim=100, batch_size=32, sen_len=len(x_train[0]), embedding_size=len(word2id)+1, tag_size=len(tag2id), pretrained=False, clip_grad=5.0, dropout_keep_embed=0.5, dropout_keep_bilstm=1.0)
    model = Model(config, embedding_pretrained=[])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./model/max_len_20')
        if ckpt is None:
            print('Model not found, please train your model first')
        else:
            path = ckpt.model_checkpoint_path
            print('loading pre-trained model from %s.....' % path)
            saver.restore(sess, path)
            test_input(model, sess, word2id, id2tag, batch_size=config['batch_size'], max_len=len(x_train[0]))




