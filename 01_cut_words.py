import pandas as pd
import collections
import jieba


TRAIN_PATH = "/home/abner/PycharmProjects/NLP_Pro_01/database/AutoMaster_TrainSet.csv"
TEST_PATH = "/home/abner/PycharmProjects/NLP_Pro_01/database/AutoMaster_TestSet.csv"
STOP_WORDS = '/home/abner/PycharmProjects/NLP_Pro_01/database/StopWords.txt'

df = pd.read_csv(TRAIN_PATH)
df = df.dropna()
# 设置停用词表，去掉无关的单词
stop = []
with open(STOP_WORDS, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop.append(line.strip())


# cut函数，分别对question，dialogue，report进行切词
def cut_words(sentences, stop=stop):
    words_df = []
    for line in sentences:
        words = jieba.cut(line)
        for w in words:
            if w not in stop:
                words_df.append(w)
    return words_df


question_words = cut_words(df.Question)
dialogue_words = cut_words(df.Dialogue)
report_words = cut_words(df.Report)

vocab = []
vocab = question_words + dialogue_words + report_words

vocab = collections.Counter(vocab)
# 结果保存为dict
vocab_dict = dict(vocab)
with open('/home/abner/PycharmProjects/NLP_Pro_01/database/vocab_dict.txt', 'w') as f:
    f.write((str(vocab_dict)))

