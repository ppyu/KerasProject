# -*- coding: utf-8 -*-
"""
@File   : load_albert.py
@Author : Pengy
@Date   : 2020/9/28
@Description : Input your description here ... 
"""
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import SpTokenizer
from keras.layers import LSTM, Dense
from keras.models import Model
import numpy as np

config_path = '../Models/albert_base_v2/albert_base/albert_config.json'
checkpoint_path = '../Models/albert_base_v2/albert_base/model.ckpt-best'
vocab_path = '../Models/albert_base_v2/albert_base/30k-clean.vocab'
spm_path = '../Models/albert_base_v2/albert_base/30k-clean.model'

tokenizer = SpTokenizer(spm_path)
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='albert')
model.summary()

token_ids, segment_ids = tokenizer.encode('language model')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))

output = LSTM(64)(model.output)
output = Dense(32)(output)
my_model = Model(model.input, output)
my_model.summary()
