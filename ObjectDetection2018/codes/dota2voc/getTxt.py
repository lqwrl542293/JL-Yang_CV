#!/usr/bin/env python
# coding=utf-8
import os
import random

train_val_percent = 0.66
train_percent = 0.5
xmlfilepath = "./Annotations"
txtsavepath = "./ImageSets/Main"
xmlfiles = os.listdir(xmlfilepath)

num = len(xmlfiles)
list = range(num)
tv = int(num * train_val_percent)
tr = int(tv * train_percent)
train_val = random.sample(list, tv)
train = random.sample(train_val, tr)

ftrain_val = open('ImageSets/Main/trainval.txt', 'w')
ftest = open('ImageSets/Main/test.txt', 'w')
ftrain = open('ImageSets/Main/train.txt', 'w')
fval = open('ImageSets/Main/val.txt', 'w')

for i in list:
    name = xmlfiles[i][:-4] + '\n'
    if i in train_val:
        ftrain_val.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrain_val.close()
ftrain.close()
fval.close()
ftest.close()
