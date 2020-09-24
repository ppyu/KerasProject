# -*- coding: utf-8 -*-
"""
@File   : muilt_output_model.py
@Author : Pengy
@Date   : 2020/9/24
@Description : 使用keras的函数式API来构建多输出模型
"""

# 一个非常简单的多输出模型示例——同时预测数据的不同性质
# 比如输入某个人的一系列社交媒体发帖，对这个人的性别、年龄和收入水平等进行预测。

from keras import layers
from keras import Input
from keras.models import Model
from keras import utils
import numpy as np

vocab_size = 50000
num_income_group = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(vocab_size, 256)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
# x = layers.MaxPooling1D(5)(x)
# x = layers.Conv1D(256, 5, activation='relu')(x)
# x = layers.Conv1D(256, 5, activation='relu')(x)
# x = layers.MaxPooling1D(5)(x)
# x = layers.Conv1D(256, 5, activation='relu')(x)
# x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

# 年龄预测
age_prediction = layers.Dense(1, name='age')(x)
# 收入预测
income_prediction = layers.Dense(num_income_group, activation='softmax', name='income')(x)
# 性别预测
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

# 构建keras模型
model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])

# 每种预测需要不同的损失函数
# 如年龄预测是标量任务，性别预测是二分类任务。
# 必须将损失合并成单个标量

# 多输出模型的编译选项：多重损失
# 两种方式指定不同输出的损失函数：1.列表；2.字典
# model.compile(optimizer='rmsprop', loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
# model.compile(optimizer='rmsprop',
#               loss={'age': 'mse', 'income': 'categorical_crossentropy', 'gender': 'binary_crossentropy'})

# 严重不平衡的损失贡献会导致模型表示针对单个损失值最大的任务优先进行优化，而不考虑其他任务的优化。
# 比如，用于年龄回归任务的均方误差（MSE）损失通常在3~5左右，而用于性别分类任务的交叉熵损失值可能低至0.1。
# 为了平衡损失贡献，需要为各个损失函数分配不同的权重。

# model.compile(optimizer='rmsprop', loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
#               loss_weights=[0.25, 1., 10.])
model.compile(optimizer='rmsprop',
              loss={'age': 'mse', 'income': 'categorical_crossentropy', 'gender': 'binary_crossentropy'},
              loss_weights={'age': 0.25, 'income': 1., 'gender': 10.})

model.summary()

# 生成模拟数据
num_samples = 1000
max_length = 100
posts = np.random.randint(1, vocab_size, size=(num_samples, max_length))
age_targets = np.random.randint(12, 50, size=(num_samples))
income_targets = utils.to_categorical(np.random.randint(0, num_income_group, size=(num_samples)), num_income_group)
gender_targets = np.random.randint(0, 2, size=(num_samples))

# 将数据喂给模型训练，两种方式：1.使用输出组成的列表（注意顺序）；2.使用输出与其名字组成的字典（只有对输出进行命名之后才能使用这种方法）
model.fit(posts, [age_targets, income_targets, gender_targets], epochs=10, batch_size=64)
# model.fit(posts, {"age_prediction": age_targets, "income_prediction": income_targets,
#                   "gender_prediction": gender_targets}, epochs=10, batch_size=64)
