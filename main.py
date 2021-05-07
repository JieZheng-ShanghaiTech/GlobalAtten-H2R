#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/18
# @Author  : Eric
# @Site    : 
# @File    : main.py
# @Software: PyCharm

import os

filePath = '../data'
files = os.listdir(filePath)
# print(files)
files.sort()
# files_clipping = files[0: 3]
for i in files:
    os.system("python train_model.py --model_type=LSTM_general_0.05 --down_sampling --epochs=50 --split=0.05 --method=general --cell_type=" + i)

