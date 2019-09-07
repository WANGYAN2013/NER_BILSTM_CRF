import tensorflow as tf
from tensorflow.contrib import crf
import numpy as np


class Model:
    def __init__(self, config, embedding_pretrained):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.embedding_size = config["embedding_size"]
        self.embedding_dim = config["embedding_dim"]
        self.sen_len = config["sen_len"]
        self.tag_size = config["tag_size"]
        self.pretrained = config["pretrained"]
        self.clip_grad = config["clip_grad"]
        self.dropout_keep_embed = config["dropout_keep_embed"]
        self.dropout_keep_bilstm = config["dropout_keep_bilstm"]
        self.embedding_pretrained = embedding_pretrained
        self.input_data = tf.placeholder(tf.int32, shape=[self.batch_size, self.sen_len], name="input_data")
        self.labels = tf.placeholder(tf.int32, shape=[self.batch_size, self.sen_len], name="labels")
        self._bulid_net()

    def _bulid_net(self):
        word_embeddings = tf.get_variable(name='word_embeddings', shape=[self.embedding_size, self.embedding_dim])
        if self.pretrained:
            word_embeddings = tf.Variable(self.embedding_pretrained,
                                          expected_shape=[self.embedding_size, self.embedding_dim],
                                          name='word_embeddings')
        input_embed = tf.nn.embedding_lookup(word_embeddings, self.input_data)
        input_embed = tf.nn.dropout(input_embed, self.dropout_keep_embed)

        lstm_fw = tf.nn.rnn_cell.LSTMCell(self.embedding_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_bw = tf.nn.rnn_cell.LSTMCell(self.embedding_dim, forget_bias=1.0, state_is_tuple=True)
        (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, input_embed,
                                                                         dtype=tf.float32,
                                                                         time_major=False,
                                                                         scope=None)
        bilstm_out = tf.concat([output_fw, output_bw], axis=-1)
        bilstm_out = tf.nn.dropout(bilstm_out, self.dropout_keep_bilstm)
        # fully connected layer
        W = tf.get_variable(name='W', shape=[self.batch_size, 2 * self.embedding_dim, self.tag_size], dtype=tf.float32)
        b = tf.get_variable(name='b', shape=[self.batch_size, self.sen_len, self.tag_size])
        bilstm_out = tf.tanh(tf.matmul(bilstm_out, W) + b)

        # CRF layer
        log_likelihood, self.transition_params = crf.crf_log_likelihood(bilstm_out, self.labels,
                                                                        tf.tile(np.array([self.sen_len]),
                                                                                np.array([self.batch_size])))
        loss = tf.reduce_mean(-log_likelihood)
        # Compute the viterbi sequence and score (used for prediction and test time).
        self.viterbi_sequence, viterbi_score = crf.crf_decode(bilstm_out, self.transition_params,
                                                              tf.tile(np.array([self.sen_len]),
                                                                      np.array([self.batch_size])))
        # train_op
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optim = tf.train.AdamOptimizer(self.lr)
        grads_and_vars = optim.compute_gradients(loss)
        grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
        self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

        self.train_op = optim.minimize(loss)
