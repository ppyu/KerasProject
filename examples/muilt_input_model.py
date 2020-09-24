# -*- coding: utf-8 -*-
"""
@File   : muilt_input_model.py
@Author : Pengy
@Date   : 2020/9/24
@Description : 使用keras的函数式API来构建多输入模型
"""

# 一个非常简单的多输入模型示例——一个问答（QA）模型
# 典型的问答模型有两个输入：一个自然语言处理描述的问题(question)和一个文章片段(passage)。 文章片段用于对问题进行回答。

from keras.models import Model
from keras import layers
from keras import Input
from keras import utils
import numpy as np

passage_vocab_size = 10000
question_vocab_size = 10000
answer_vocab_size = 500

# 构建文章片段的子模型结构
passage_input = Input(shape=(None,), dtype='int32', name='passage')
embedded_passage = layers.Embedding(passage_vocab_size, 64)(passage_input)
encoded_passage = layers.LSTM(32)(embedded_passage)

# 构建问题的子模型结构
question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(question_vocab_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

# 将文章与问题的子结构连接起来
concatenated = layers.concatenate([encoded_passage, encoded_question], axis=-1)
answer = layers.Dense(answer_vocab_size, activation='softmax')(concatenated)

# 构建keras模型
model = Model([passage_input, question_input], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# 打印模型结构
model.summary()

# 使用numpy生成模拟数据并输入该双输入模型中去
num_samples = 1000  # 样本数
max_length = 100  # 文本最大长度
# 生成模拟文章
passage = np.random.randint(1, passage_vocab_size, size=(num_samples, max_length))
# 生成模拟问题
question = np.random.randint(1, question_vocab_size, size=(num_samples, max_length))
# 生成模拟答案
answer = np.random.randint(1, answer_vocab_size, size=(num_samples))
# 答案是one-hot编码，将其由整数转换至one-hot(为了softmax)
answer = utils.to_categorical(answer, answer_vocab_size)

# 将数据喂给模型训练，两种方式：1.使用输入组成的列表（注意顺序）；2.使用输入与其名字组成的字典（只有对输入进行命名之后才能使用这种方法）
model.fit([passage, question], answer, epochs=10, batch_size=128)
# model.fit({'passage': passage, "question": question}, answer, epochs=10, batch_size=128)
