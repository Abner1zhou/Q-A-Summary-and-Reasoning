import gensim
from gensim.models import word2vec
from utils.config import *
import numpy as np

# 导入已经创建好的word vector 模板
wv_model = word2vec.Word2Vec.load(w2v_model_path)

save_embedding_matrix_path = "database/embedding_matrix.txt"

# 创建方法一
embedding_matrix_01 = wv_model.wv.vectors


# 创建方法二
def get_embedding_matrix(model):
    vocab_size = len(model.wv.vocab)
    embedding_dim = model.wv.vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for i in range(vocab_size):
        embedding_matrix[i, :] = model.wv[model.wv.index2word[i]]
        embedding_matrix = embedding_matrix.astype("float32")
    # 检查数据维度是不是正确
    assert embedding_matrix.shape == (vocab_size, embedding_dim)
    return embedding_matrix


embedding_matrix_02 = get_embedding_matrix(wv_model)
print("The two embedding_matrix is equal : \n{}".format(embedding_matrix_01 == embedding_matrix_02))

np.savetxt(save_embedding_matrix_path, embedding_matrix_02, fmt='%0.8f')


