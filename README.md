# 问答摘要与推理

项目链接：<https://aistudio.baidu.com/aistudio/competition/detail/3>

这是我学习NLP的项目，根据这个项目的要求一点一点学习需要使用的工具、算法。

由于没有做个软件项目，所以文件命名、摆放可能不太规范，请谅解。

欢迎指正～

已完成PaddlePaddle版本的seq2seq + attention



数据处理归类到了 [utils/data_processing.py](https://github.com/Abner1zhou/Q-A-Summary-and-Reasoning/blob/master/utils/data_processing.py)

### 如何运行

utils.data_processing 来处理训练数据集和测试数据集



直接运行[**seq2seq_train.py**](https://github.com/Abner1zhou/Q-A-Summary-and-Reasoning/blob/paddle/seq2seq_train.py)可以训练数据集



### 不同方案分数记录：

seq2seq + attention   最高分只能到29分左右

### 存在以下问题：

1. 无法解决OOV问题
2. 预测出来的句子会出现词语重复的问题