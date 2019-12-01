# encoding: utf-8
"""
@author: caopeng 
@contact: abner1zhou@gmail.com 
@file: 03_encoder&decoer.py 
@time: 2019/12/1 下午2:12 
@desc: 编写seq2seq所需要的encoder层和decoder层
"""
import tensorflow as tf


class Encoder:
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_size = batch_sz
        self.enc_units = enc_units
        self.embedding_dim = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        gpus = tf.config.experimental.list_logical_devices("GPU")
        if gpus:
            self.gru = tf.keras.layers.CuDNNGRU(self.enc_units,
                                                return_sequences=True,
                                                return_state=True,
                                                recurrent_initializer='glorot_uniform')
        else:
            self.gru = tf.keras.layers.GRU(self.enc_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')

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
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            self.gru = tf.keras.layers.CuDNNGRU(self.enc_units,
                                                return_sequences=True,
                                                return_state=True,
                                                recurrent_initializer='glorot_uniform')
        else:
            self.gru = tf.keras.layers.GRU(self.enc_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')

    def call(self, x, hidden, enc_output, context_vector):
        output, state = self.gru(enc_output, initial_state=hidden)
        return output, state


