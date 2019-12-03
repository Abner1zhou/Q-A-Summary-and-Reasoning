# encoding: utf-8
"""
@author: caopeng 
@contact: abner1zhou@gmail.com 
@file: 02_get_embedding_matirx_2.py 
@time: 2019/12/1 下午7:44 
@desc: 使用FastText来获得词向量结果
"""
from gensim.models import fasttext
from gensim.models.word2vec import LineSentence
from utils import config

model = fasttext.FastText(sentences=LineSentence(config.merger_seg_path), workers=6, min_count=5, size=200)
model.save(config.ft_model_path)
"""
测试数据集，结果：
[('吉利车', 0.9441115856170654), ('帝豪', 0.8785934448242188), 
('帝豪车', 0.8539795875549316), ('路霸', 0.8515048623085022), 
('远景', 0.8433764576911926), ('长安', 0.8415511250495911), 
('博瑞', 0.8405277729034424), ('GS', 0.829689621925354), 
('哈佛', 0.8269863128662109), ('东南', 0.8233577609062195)]
"""
# model = fasttext.FastText.load(config.ft_model_path)
# print(model.wv.most_similar("吉利"))
