from utils.config import *
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
import gensim
"""
将第一步创建好的vocab转换成word2vec模型
主要用到gensim
"""
model = word2vec.Word2Vec(LineSentence(merger_seg_path), workers=8, min_count=5, size=200)
model.save("database/wv/word2vec.model")
