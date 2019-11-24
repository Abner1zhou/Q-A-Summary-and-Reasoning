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

