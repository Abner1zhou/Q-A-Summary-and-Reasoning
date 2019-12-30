# encoding: utf-8 
"""
@author: 周世聪
@contact: abner1zhou@gmail.com 
@file: seq2seq_train.py 
@time: 2019/12/26 下午9:54 
@desc: 基于Paddle 的seq2seq训练
"""
import paddle
from paddle import fluid
from paddle.fluid import layers
import numpy as np
import pandas as pd
import six
import time

from utils.config import *
from utils.data_processing import load_dataset
from utils.wv_loader import load_vocab, load_embedding_matrix


train_x, train_y, test_x, train_lbl, trg_sequence_length, vocab, reverse_vocab, embedding_matrix = load_dataset()


# 训练集的长度
BUFFER_SIZE = len(train_x)
# 输入的长度
max_length_inp = train_x.shape[1]
# 输出的长度
max_length_targ = train_y.shape[1]
# 词表大小
vocab_size = len(vocab)
beam_size = 4
bos_id = vocab['<START>']
eos_id = vocab['<STOP>']
tar_max_length = len(train_y)

# 加载预训练词向量
w_param_attrs = fluid.ParamAttr(
    name="emb_weight",
    initializer=fluid.initializer.NumpyArrayInitializer(embedding_matrix, ),
    trainable=False)


def encoder(trg_inp, hidden_dim):
    # 使用GRUCell构建前向RNN
    encoder_fwd_cell = layers.GRUCell(hidden_size=hidden_dim)
    src_embedding = fluid.embedding(input=trg_inp,
                                    size=embedding_matrix.shape,
                                    param_attr= w_param_attrs,
                                    is_sparse=False)
    # 前向encoder
    encoder_fwd_output, fwd_state = layers.rnn(
        cell=encoder_fwd_cell,
        inputs=src_embedding,
        time_major=False,
        is_reverse=False)
    # 使用GRUCell构建反向RNN
    encoder_bwd_cell = layers.GRUCell(hidden_size=hidden_dim)
    encoder_bwd_output, bwd_state = layers.rnn(
        cell=encoder_bwd_cell,
        inputs=src_embedding,
        time_major=False,
        is_reverse=True)
    # 拼接前向与反向GRU的编码结果得到h， shape = [batch, sen_length, units*2]
    encoder_output = layers.concat(
        input=[encoder_fwd_output, encoder_bwd_output], axis=2)
    encoder_state = layers.concat(input=[fwd_state, bwd_state], axis=1)
    return encoder_output, encoder_state


class DecoderCell(layers.RNNCell):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.gru_cell = layers.GRUCell(hidden_size)

    def attention(self, hidden, encoder_output, encoder_output_proj):
        # 定义attention用以计算context，即 c_i，这里使用Bahdanau attention机制
        decoder_state_proj = layers.unsqueeze(
            layers.fc(hidden, size=self.hidden_size, bias_attr=False), [1])
        # encoder + decoder
        mixed_state = fluid.layers.elementwise_add(encoder_output_proj,
                                                   layers.expand(decoder_state_proj,
                                                                 [1, layers.shape(decoder_state_proj)[1], 1]))
        attn_scores = layers.squeeze(
            layers.fc(input=mixed_state,
                    size=1,
                    num_flatten_dims=2,
                    bias_attr=False), [2])
        attn_scores = layers.softmax(attn_scores)
        context = layers.reduce_sum(layers.elementwise_mul(encoder_output,
                                                        attn_scores,
                                                        axis=0),
                                    dim=1)
        return context

    def call(self,
             step_input,
             hidden,
             encoder_output,
             encoder_output_proj):
        # Bahdanau attention
        context = self.attention(hidden, encoder_output, encoder_output_proj)
        step_input = layers.concat([step_input, context], axis=1)
        # GRU
        output, new_hidden = self.gru_cell(step_input, hidden)
        return output, new_hidden


def decoder(encoder_output,
            encoder_output_proj,
            encoder_state,
            trg=None,
            is_train=True):
    # 定义 RNN 所需要的组件
    decoder_cell = DecoderCell(hidden_size=units)
    decoder_initial_states = layers.fc(encoder_state,
                                    size=units,
                                    act="tanh")
    trg_embeder = lambda x: fluid.embedding(
                                            input=x,
                                            size=embedding_matrix.shape,
                                            param_attr= w_param_attrs,
                                            is_sparse=False)
    output_layer = lambda x: layers.fc(x,
                                       size=embedding_matrix.shape[0],
                                       num_flatten_dims=len(x.shape) - 1,
                                       param_attr=fluid.ParamAttr(name=
                                                                  "output_w"))
    if is_train:  # 训练
        # 训练时使用 `layers.rnn` 构造由 `cell` 指定的循环神经网络
        # 循环的每一步从 `inputs` 中切片产生输入，并执行 `cell.call`
        decoder_output, _ = layers.rnn(
            cell=decoder_cell,
            inputs=trg_embeder(trg),
            initial_states=decoder_initial_states,
            time_major=False,
            encoder_output=encoder_output,
            encoder_output_proj=encoder_output_proj)
        decoder_output = output_layer(decoder_output)
    else:  # 基于 beam search 的预测生成
        # beam search 时需要将用到的形为 `[batch_size, ...]` 的张量扩展为 `[batch_size* beam_size, ...]`
        encoder_output = layers.BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_output, beam_size)
        encoder_output_proj = layers.BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_output_proj, beam_size)
        # BeamSearchDecoder 定义了单步解码的操作：`cell.call` + `beam_search_step`
        beam_search_decoder = layers.BeamSearchDecoder(cell=decoder_cell,
                                                       start_token=bos_id,
                                                       end_token=eos_id,
                                                       beam_size=beam_size,
                                                       embedding_fn=trg_embeder,
                                                       output_fn=output_layer)
        # 使用 layers.dynamic_decode 动态解码
        # 重复执行 `decoder.step()` 直到其返回的表示完成状态的张量中的值全部为True或解码步骤达到 `max_step_num`
        decoder_output, _ = layers.dynamic_decode(
            decoder=beam_search_decoder,
            inits=decoder_initial_states,
            max_step_num=tar_max_length,
            output_time_major=False,
            encoder_output=encoder_output,
            encoder_output_proj=encoder_output_proj)

    return decoder_output


def model_func(inputs, is_train=True):
    """
    inputs: [src, trg]
    """
    # 源语言输入
    src = inputs[0]

    # 编码器
    encoder_output, encoder_state = encoder(src, units)

    encoder_output_proj = layers.fc(input=encoder_output,
                                    size=units,
                                    num_flatten_dims=2,
                                    bias_attr=False)

    # 目标语言输入，训练时有、预测生成时无该输入
    trg = inputs[1] if is_train else None

    # 解码器
    output = decoder(encoder_output=encoder_output,
                     encoder_output_proj=encoder_output_proj,
                     encoder_state=encoder_state,
                     trg=trg,
                     is_train=is_train)
    return output


def data_func(is_train=True):
    # 源语言source数据
    src = fluid.data(name="src", shape=[None, None], dtype="int64")

    inputs = [src]
    # 训练时还需要目标语言target和label数据
    if is_train:
        trg = fluid.data(name="trg", shape=[None, None], dtype="int64")
        lbl = fluid.data(name="lbl", shape=[None, None], dtype="int64")
        trg_sequence_length = fluid.data(name="trg_sequence_length",
                                         shape=[None],
                                         dtype="int64")
        inputs += [trg, lbl, trg_sequence_length]
    # data loader
    loader = fluid.io.DataLoader.from_generator(feed_list=inputs,
                                                capacity=8,
                                                iterable=True,
                                                use_double_buffer=True)
    return inputs, loader


def loss_func(logits, label, trg_sequence_length):
    probs = layers.softmax(logits)
    # 使用交叉熵损失函数
    loss = layers.cross_entropy(input=probs, label=label)
    # 根据长度生成掩码，并依此剔除 padding 部分计算的损失
    trg_mask = layers.sequence_mask(trg_sequence_length,
                                    maxlen=layers.shape(logits)[1],
                                    dtype="float32")
    avg_cost = layers.reduce_sum(loss * trg_mask) / layers.reduce_sum(trg_mask)
    return avg_cost


def optimizer_func():
    # 设置梯度裁剪
    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0))
    # 定义先增后降的学习率策略
    lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(units, 1000)
    return fluid.optimizer.Adam(
        learning_rate=lr_decay,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=1e-4))


def reader_creator(trainX, trainY, trainLBL, train_Y_length):
    data = zip(trainX, trainY, trainLBL, train_Y_length)
    def reader():
        for x_ids, y_ids, trainLBL, length in data:
            yield x_ids, y_ids, trainLBL, length
    return reader


def test_reader(test_x):
    def reader():
        for x_ids in test_x:
            yield x_ids
    return reader


def input_func(batch_size, is_train=True):
    if is_train:
        data_generator = fluid.io.shuffle(
            reader_creator(train_x, train_y, train_lbl, trg_sequence_length),
            buf_size=100)
    else:
        data_generator = test_reader(test_x)

    batch_generator = fluid.io.batch(data_generator, batch_size=batch_size)

    def _generator():
        for batch in batch_generator():
            if is_train:
                batch_src = np.array([ins[0] for ins in batch])
                inputs = [batch_src]
                batch_trg = np.array([ins[1] for ins in batch])
                batch_lbl = np.array([ins[2] for ins in batch])
                batch_trl = np.array([ins[3] for ins in batch])
                inputs += [batch_trg, batch_lbl, batch_trl]
            else:
                batch_src = np.array([ins for ins in batch])
                inputs = [batch_src]
            yield inputs

    return _generator


train_prog = fluid.Program()
startup_prog = fluid.Program()
with fluid.program_guard(train_prog, startup_prog):
    with fluid.unique_name.guard():
        # 训练时：
        # inputs = [src, src_sequence_length, trg, trg_sequence_length, label]
        inputs, loader = data_func(is_train=True)
        logits = model_func(inputs, is_train=True)
        loss = loss_func(logits, inputs[2], inputs[3])
        optimizer = optimizer_func()
        optimizer.minimize(loss)


# 设置训练设备
use_cuda = True
places = fluid.cuda_places() if use_cuda else fluid.cpu_places()
# 设置数据源
loader.set_batch_generator(input_func(BATCH_SIZE, True),
                           places=places)
# 定义执行器，初始化参数并绑定Program
exe = fluid.Executor(places[0])
exe.run(startup_prog)
# fluid.io.load_params(exe, model_save_dir, main_program=train_prog)
prog = fluid.CompiledProgram(train_prog).with_data_parallel(
    loss_name=loss.name)


batch_start_time = time.time()
for pass_id in six.moves.xrange(EPOCH_NUM):
    batch_id = 0
    for data in loader():
        loss_val  = exe.run(prog, feed=data, fetch_list=[loss])[0]
        if batch_id % 50 == 0:
            print('pass_id: %d, batch_id: %d, loss: %f' %
                    (pass_id, batch_id, loss_val))
            print("batch running time:{}".format(time.time()-batch_start_time))
            batch_start_time = time.time()
        batch_id += 1
    # 保存模型
    fluid.io.save_params(exe, model_save_dir, main_program=train_prog)
    loader.set_batch_generator(input_func(64, True),
                            places=places)
    print("model save success")