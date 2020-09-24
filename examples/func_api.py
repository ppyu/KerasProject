# -*- coding: utf-8 -*-
"""
@File   : func_api.py
@Author : Pengy
@Date   : 2020/9/24
@Description : 使用函数式API构建keras模型
"""

from keras import layers, Input
from keras.models import Model, Sequential

# seq_model = Sequential()
# seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
# seq_model.add(layers.Dense(32, activation='relu'))
# seq_model.add(layers.Dense(10, activation='softmax'))
#
# seq_model.summary()

# 使用keras的函数式API搭建一个相同的Sequential模型
input_tensor = Input(shape=(64,))  # 输入张量
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)  # 输出张量

# Model类将输入张量和输出张量转换为一个模型
# Keras会在后台检索从input_tensor到output_tensor所包含的每一层，并将这些层组合成一个类图的数据结构，即一个Model。
# 如果试图使用不相关的输入和输出来构建一个模型，会发生Error。
model = Model(input_tensor, output_tensor)
model.summary()
