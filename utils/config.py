# -*- coding:utf-8 -*-
import os
import pathlib

# 获取项目根目录
root = pathlib.Path(os.path.abspath(__file__)).parent.parent

# 训练数据路径
train_data_path = os.path.join(root, 'database', 'AutoMaster_TrainSet.csv')
# 测试数据路径
test_data_path = os.path.join(root, 'database', 'AutoMaster_TestSet.csv')
# 停用词路径
stop_word_path = os.path.join(root, 'database', 'StopWords.txt')

# 自定义切词表
user_dict = os.path.join(root, 'database', 'user_dict.txt')

# 预处理后的训练数据
train_seg_path = os.path.join(root, 'database', 'train_seg_data.csv')
# 预处理后的测试数据
test_seg_path = os.path.join(root, 'database', 'test_seg_data.csv')
# 合并训练集测试集数据
merger_seg_path = os.path.join(root, 'database', 'merged_train_test_seg_data.csv')
# word2vec 模板
w2v_model_path = os.path.join(root, 'database/wv', 'word2vec.model')
# FastText 模板
ft_model_path = os.path.join(root, 'database/wv', 'FastText.model')
# train_X
train_x_path = os.path.join(root, 'database', 'train_x_df.csv')
# train_Y
train_y_path = os.path.join(root, 'database', 'train_y_df.csv')
# test_Y
test_x_path = os.path.join(root, 'database', 'text_x_df.csv')
# 词表保存路径
vocab_path = os.path.join(root, 'database', 'wv', 'vocab.txt')
reverse_vocab_path = os.path.join(root, 'database', 'wv', 'reverse_vocab.txt')
# embedding matrix
embedding_matrix_path = os.path.join(root, 'database', 'wv', 'embedding_matrix.csv')

# 设置常量
# 训练轮数
WV_TRAIN_EPOCHS = 10

