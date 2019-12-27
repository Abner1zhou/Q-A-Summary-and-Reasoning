# encoding: utf-8 
"""
@author: 周世聪
@contact: abner1zhou@gmail.com 
@file: data_processing.py 
@time: 2019/12/4 下午9:23 
@desc: 数据处理函数，包括分词，词向量构建，embedding_matrix，训练集 测试集创建
"""

import pandas as pd
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import jieba
import re
import numpy as np

from utils import multi_cpus
from utils import config, file_utils
from utils.config import WV_TRAIN_EPOCHS


def clean_sentence(sentence):
    """
    特殊符号去除
    使用正则表达式去除无用的符号、词语
    """
    if isinstance(sentence, str):
        return re.sub(
            r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好|nan',
            '', sentence)
    else:
        return ''


def get_stop_words(stop_words_path):
    """
    处理停用词表
    :param stop_words_path: 停用词表路径
    :return: 停用词表list
    """
    stop = []
    with open(stop_words_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            stop.append(line.strip())
    return stop


# cut函数，分别对question，dialogue，report进行切词
def cut_words(sentences):
    # 清除无用词
    sentence = clean_sentence(sentences)
    # 切词，默认精确模式，全模式cut参数cut_all=True
    words = jieba.cut(sentence)
    # 过滤停用词
    stop_words = get_stop_words(config.stop_word_path)
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)


def cut_data_frame(df):
    """
    数据集批量处理方法
    :param df: 数据集
    :return:处理好的数据集
    """
    # 批量预处理 训练集和测试集
    for col_name in ['Brand', 'Model', 'Question', 'Dialogue']:
        df[col_name] = df[col_name].apply(cut_words)

    if 'Report' in df.columns:
        # 训练集 Report 预处理
        df['Report'] = df['Report'].apply(cut_words)
    return df


def get_segment(data_path, is_train=False):
    """
    将输入的数据集切词并返回
    :param is_train: 是否训练数据集
    :param data_path: 输入csv格式数据集
    :return: 返回一个切词完毕的数据集
    """
    df = pd.read_csv(data_path)
    if is_train:
        df.dropna(subset=['Report'], inplace=True)
    # 1.切词
    seg_df = multi_cpus.parallelize(df, cut_data_frame)
    # 对切词完的数据进行拼接,这部分用来训练词向量
    seg_df["Data_X"] = seg_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    if "Report" in seg_df.columns:
        seg_df['merged'] = seg_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)  # axis 横向拼接
        seg_df['Data_Y'] = seg_df[['Report']]
    else:
        seg_df['merged'] = seg_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)  # axis 横向拼接

    return seg_df


def get_w2v(merger_seg_path):
    w2v_model = word2vec.Word2Vec(LineSentence(merger_seg_path)
                                  , min_count=5
                                  , workers=6
                                  # 忽略词频小于5的单词
                                  , size=200)
    return w2v_model


def get_max_len(data):
    """
    获得合适的最大长度值
    :param data: 待统计的数据  train_df['Question']
    :return: 最大长度值
    """
    max_lens = data.apply(lambda x: x.count(' ') + 1)
    return int(np.mean(max_lens) + 2 * np.std(max_lens))


def pad_proc(sentence, max_len, vocab):
    """
    # 填充字段
    < start > < end > < pad > < unk > max_lens
    """
    # 0.按空格统计切分出词
    words = sentence.strip().split(' ')
    # 1. 截取规定长度的词数
    words = words[:max_len]
    # 2. 填充< unk > ,判断是否在vocab中, 不在填充 < unk >
    sentence = [word if word in vocab else '<UNK>' for word in words]
    # 3. 填充< start > < end >
    sentence = ['<START>'] + sentence + ['<STOP>']
    # 4. 判断长度，填充　< pad >
    sentence = sentence + ['<PAD>'] * (max_len - len(words))
    return ' '.join(sentence)


def translate_data(sentence, vocab):
    """
    把句子中的单词转换为index
    :param sentence: 一行数据
    :param vocab: 词表
    :return: 由index组成过的句子
    """
    words = sentence.split(' ')
    ids = [vocab[word] if word in vocab else vocab['<UNK>'] for word in words]
    return ids


def build_dataset(train_df_path, test_df_path):
    """
    主函数，用于构建数据集
    :param train_df_path: 训练数据集地址
    :param test_df_path:  测试数据集地址
    :return:
    """
    # 1. 切词
    train_seg_df = get_segment(train_df_path, is_train=True)
    test_seg_df = get_segment(test_df_path)
    # 2. 合并数据集
    merged_df = pd.concat([train_seg_df[['merged']], test_seg_df[['merged']]], axis=0)
    merged_df.to_csv(config.merger_seg_path, header=False, index=False)
    train_seg_df['X'] = train_seg_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    test_seg_df['X'] = test_seg_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    train_seg_df['Y'] = train_seg_df[['Report']]
    # 3. 计算词向量
    print("getting word vector...")
    w2v_model = get_w2v(config.merger_seg_path)
    # 4. 构建词表
    print("getting vocab...")
    vocab = {word: index for index, word in enumerate(w2v_model.wv.index2word)}
    # 5. 计算数据集长度，用来给encoder设置合理的单元数
    train_x_max_len = get_max_len(train_seg_df['X'])
    train_y_max_len = get_max_len(train_seg_df['Y'])
    test_x_max_len = get_max_len(test_seg_df['X'])
    trg_sequence_length = train_seg_df['Y'].apply(lambda x: len(x))
    # 6. 添加填充字段
    train_seg_df['X'] = train_seg_df['X'].apply(lambda x: pad_proc(x, train_x_max_len, vocab))
    train_seg_df['Y'] = train_seg_df['Y'].apply(lambda x: pad_proc(x, train_y_max_len, vocab))
    test_seg_df['X'] = test_seg_df['X'].apply(lambda x: pad_proc(x, test_x_max_len, vocab))

    train_seg_df['X'].to_csv(config.train_x_path, header=False, index=False)
    train_seg_df['Y'].to_csv(config.train_y_path, header=False, index=False)
    test_seg_df['X'].to_csv(config.test_x_path, header=False, index=False)
    # 7. 再次训练词向量，把填充字段加进去
    print('start retrain w2v model')
    w2v_model.build_vocab(LineSentence(config.train_x_path), update=True)
    w2v_model.train(LineSentence(config.train_x_path), epochs=WV_TRAIN_EPOCHS, total_examples=w2v_model.corpus_count)
    print('1/3')
    w2v_model.build_vocab(LineSentence(config.train_y_path), update=True)
    w2v_model.train(LineSentence(config.train_y_path), epochs=WV_TRAIN_EPOCHS, total_examples=w2v_model.corpus_count)
    print('2/3')
    w2v_model.build_vocab(LineSentence(config.test_x_path), update=True)
    w2v_model.train(LineSentence(config.test_x_path), epochs=WV_TRAIN_EPOCHS, total_examples=w2v_model.corpus_count)
    # 保存
    w2v_model.save(config.w2v_model_path)
    # 8. 重新构建词表
    vocab = {word: index for index, word in enumerate(w2v_model.wv.index2word)}
    reverse_vocab = {index: word for index, word in enumerate(w2v_model.wv.index2word)}
    # 保存词表
    print("Saving vocab...")
    file_utils.save_dict(config.vocab_path, vocab)
    file_utils.save_dict(config.reverse_vocab_path, reverse_vocab)
    # 9. 保存embedding matrix
    embedding_matrix = w2v_model.wv.vectors
    np.savetxt(config.embedding_matrix_path, embedding_matrix, fmt='%0.8f')
    # 10. 把单词转换为index
    # 原数据为【方向机 重 助力 泵 方向机 都 换 新 都...】 转换成 [398, 985, 244, 229, 398...]
    print("Translating the sentences...")
    train_x_ids = train_seg_df['X'].apply(lambda x: translate_data(x, vocab))
    train_y_ids = train_seg_df['Y'].apply(lambda x: translate_data(x, vocab))
    test_x_ids = test_seg_df['X'].apply(lambda x: translate_data(x, vocab))
    # 11. 把数据保存为numpy格式
    train_x = np.array(train_x_ids.tolist())
    train_y = np.array(train_y_ids.tolist())
    test_x = np.array(test_x_ids.tolist())
    trg_sequence_length = np.array(trg_sequence_length.tolist())
    np.savetxt(config.train_x_path, train_x)
    np.savetxt(config.train_y_path, train_y)
    np.savetxt(config.test_x_path, test_x)
    np.savetxt(config.trg_sequence_length_path, trg_sequence_length)

    return train_x, train_y, test_x, trg_sequence_length


def load_dataset():
    """
    :return: 加载处理好的数据集
    """
    train_x = np.loadtxt(config.train_x_path)
    train_y = np.loadtxt(config.train_y_path)
    test_x = np.loadtxt(config.test_x_path)
    trg_sequence_length = np.loadtxt(config.trg_sequence_length_path)
    train_x = train_x.astype('int64')
    train_y = train_y.astype('int64')
    test_x = test_x.astype('int64')
    trg_sequence_length = trg_sequence_length.astype('int64')
    return train_x, train_y, test_x, trg_sequence_length


def main():
    train_x, train_y, test_x, trg_sequence_length = build_dataset(config.train_data_path, config.test_data_path)


if __name__ == '__main__':
    main()









