# encoding: utf-8 
"""
@author: 周世聪
@contact: abner1zhou@gmail.com 
@file: wv_loader.py 
@time: 2019/12/8 下午4:08 
@desc: 读取word vector 相关数据集
"""
from gensim.models.word2vec import LineSentence, Word2Vec
import numpy as np

from utils.config import embedding_matrix_path


def load_vocab(file_path):
    """
    读取字典
    :param file_path: 文件路径
    :return: 返回读取后的字典
    """
    vocab = {}
    reverse_vocab = {}
    for line in open(file_path, "r", encoding='utf-8').readlines():
        word, index = line.strip().split("\t")
        index = int(index)
        vocab[word] = index
        reverse_vocab[index] = word
    return vocab, reverse_vocab


def load_embedding_matrix():
    """
    加载 embedding_matrix_path
    """
    return np.loadtxt(embedding_matrix_path)
