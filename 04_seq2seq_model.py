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

train_x, train_y, test_x = load_dataset()
vocab, reverse_vocab = load_vocab(config.vocab_path)
embedding_matrix = load_embedding_matrix()
