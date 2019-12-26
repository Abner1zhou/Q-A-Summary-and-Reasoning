# encoding: utf-8 
"""
@author: 周世聪
@contact: abner1zhou@gmail.com 
@file: 04_seq2seq_model.py 
@time: 2019/12/4 下午9:19 
@desc: 结合上面三次作业制作seq2seq模型，并进行训练
"""
from utils import config
from utils.data_processing import load_dataset
from utils.wv_loader import load_vocab, load_embedding_matrix

import tensorflow as tf
import time
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

train_x, train_y, test_x = load_dataset()
vocab, reverse_vocab = load_vocab(config.vocab_path)
embedding_matrix = load_embedding_matrix()

# 训练集的长度
BUFFER_SIZE = len(train_x)

# 输入的长度
max_length_inp = train_x.shape[1]
# 输出的长度
max_length_targ = train_y.shape[1]

BATCH_SIZE = 10

# 训练一轮需要迭代多少步
steps_per_epoch = len(train_x)//BATCH_SIZE

# 词向量维度
embedding_dim = 200
# 隐藏层单元数
units = 256

# 词表大小
vocab_size = len(vocab)

# 构建训练集
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        # 初始化隐藏层
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        # Dense implements the operation: output = activation(dot(input, kernel) + bias)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query为上次的GRU隐藏层
        # values为编码器的编码结果enc_output
        # 在seq2seq模型中，St是后面的query向量，而编码过程的隐藏状态hi是values。
        # St表示的是Ht, Decoder hidden output. Hi is encoder hidden output
        # Encoder得到的hidden是一个二维向量，dec_output是三维向量，所以需要对hidden增维
        # tf.expand_dims(query, 1)  1表示增维的位置
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # 计算注意力权重值
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # 用于处理attention
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # 使用上次的隐藏层（第一次使用编码器隐藏层）、编码器输出计算注意力权重
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # 将上一循环的预测结果跟注意力权重值结合在一起作为本次的GRU网络输入
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        # tf.reshape(tensor, shape, name=None)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, pad_index))

    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        # 1. 构建encoder
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        # 2. 复制
        dec_hidden = enc_hidden
        # 3. <START> * BATCH_SIZE
        dec_input = tf.expand_dims([vocab['<START>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # decoder(x, hidden, enc_output)
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
pad_index = vocab['<PAD>']

encoder = Encoder(vocab_size, embedding_dim, embedding_matrix, units, BATCH_SIZE)
attention_layer = BahdanauAttention(units)
decoder = Decoder(vocab_size, embedding_dim, embedding_matrix, units, BATCH_SIZE)

checkpoint_dir = 'database/checkpoints/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    # 初始化隐藏层
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        #
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 1 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))