import pandas as pd
import collections
import jieba
import re
from utils.multi_cpus import parallelize

TRAIN_PATH = "/home/abner/PycharmProjects/NLP_Pro_01/database/AutoMaster_TrainSet.csv"
TEST_PATH = "/home/abner/PycharmProjects/NLP_Pro_01/database/AutoMaster_TestSet.csv"
STOP_WORDS = '/home/abner/PycharmProjects/NLP_Pro_01/database/StopWords.txt'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 去除空白的行
train_df.dropna(subset=['Question', 'Dialogue', 'Report'], how='any', inplace=True)
test_df.dropna(subset=['Question', 'Dialogue'], how='any', inplace=True)

# 设置停用词表，去掉无关的单词
stop = []
with open(STOP_WORDS, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop.append(line.strip())


def clean_sentence(sentence):
    '''
    特殊符号去除
    使用正则表达式去除无用的符号、词语
    '''
    if isinstance(sentence, str):
        return re.sub(
            r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
            '', sentence)
    else:
        return ''


# cut函数，分别对question，dialogue，report进行切词
def cut_words(sentences, stop=stop):
    # 清除无用词
    sentence = clean_sentence(sentences)
    # 切词，默认精确模式，全模式cut参数cut_all=True
    words = jieba.cut(sentence)
    # 过滤停用词
    words = [w for w in words if w not in stop]
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


train_df = parallelize(train_df, cut_data_frame);
test_df = parallelize(test_df, cut_data_frame);
train_df.to_csv('database/train_seg_data.csv',index=None,header=True)
test_df.to_csv('database/test_seg_data.csv',index=None,header=True)

