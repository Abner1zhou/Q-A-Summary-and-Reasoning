# encoding: utf-8
"""
@author: 周世聪
@contact: abner1zhou@gmail.com 
@file: 03_encoder&decoder.py
@time: 2019/12/1 下午2:12 
@desc: 编写seq2seq所需要的encoder层和decoder层
"""
import tensorflow as tf


class Encoder:
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_size = batch_sz
        self.enc_units = enc_units
        self.embedding_dim = tf.keras.layers.Embedding(vocab_size, embedding_dim, trainable=False)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequence=False, return_state=True)

    def call(self, x, hidden):
        x = self.embedding_dim(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder:
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    def call(self, x, hidden, enc_output):
        x = self.embedding(x)
        output, state = self.gru(enc_output, initial_state=hidden)
        return output, state


