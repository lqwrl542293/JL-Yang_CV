#!/usr/bin/env python
# coding: utf-8

"""
	转换脚本需放置到以 DOTA 数据集为根目录的路径, 先对数据进行 split 后执行即可
"""

import glob
from PIL import Image
import os
import shutil

classes = ["soccer-ball-field", "helicopter", "swimming-pool", "roundabout", 
           "large-vehicle", "small-vehicle", "bridge", "harbor", 
           "ground-track-field", "basketball-court", "tennis-court", 
           "baseball-diamond", "storage-tank", "ship", "plane"]
folders = ['train', 'val']

# 生成标注文件
def getLabels(mode):
    if not os.path.exists('./' + mode +'Labels'):
        os.mkdir('./' + mode + 'Labels')
    filelist = glob.glob('./' + mode + 'Split/labelTxt/*.txt')
    for text_ in filelist:
        inputText = open(text_, 'r')
        labels = inputText.readlines()
        inputText.close()
        result = ""
        # 保证路径中不含 "P", 否则将运行出错
        filename = text_[text_.index('P'):-4]
        print(filename)
        img = Image.open('./' + mode + 'Split/images/' + filename + '.png')
        imageSize = img.size

        for i in range(0, len(labels)):
            tmp = labels[i].split(' ')
            if tmp[0][:5] == 'image' or tmp[0][:3] == 'gsd':
                continue
            for j in range(8):
                tmp[j] = float(tmp[j])

            xmax = max(tmp[0], tmp[2], tmp[4], tmp[6])
            xmin = min(tmp[0], tmp[2], tmp[4], tmp[6])
            ymax = max(tmp[1], tmp[3], tmp[5], tmp[7])
            ymin = min(tmp[1], tmp[3], tmp[5], tmp[7])
            category = str(classes.index(tmp[8]))
            dw = 1./imageSize[0]
            dh = 1./imageSize[1]
            x = ((xmax) + (xmin)) / 2.0
            y = (ymax + ymin) / 2.0
            w = xmax - xmin
            h = ymax - ymin
            x = str(x * dw)
            w = str(w * dw)
            y = str(y * dh)
            h = str(h * dh)
            result += "" + category + " " + x + " " + y + " " + w + " " + h + "\n"

        output = mode + 'Labels/' + filename + '.txt'
        f = open(output, 'w')
        f.write(result)
        f.close

# 生成 *.txt
def getTxt(mode):
    filelist = glob.glob(os.getcwd() + '/' + mode + 'Labels/*.txt')
    f = open('./' + mode + '.txt', 'w')
    for text_ in filelist:
        f.write(os.getcwd() + '/' +mode + 'Split/images/' + text_[text_.index('P'):-4] + '.png\n')
    f.close()

# 生成 .names
def getNames():
    f = open('./dota.names', 'w')
    for objClass in classes:
        f.write(objClass + '\n')
    f.close()

# 将生成的 标注文件放入 ***Split/labels
def copyToSplitDir(mode):
    isExist = os.path.exists('./' + mode + 'Split/labels')
    if isExist == True:
        shutil.rmtree('./' + mode + 'Split/labels')
    shutil.copytree('./' + mode + 'Labels', './' + mode + 'Split/labels')


for folder in folders:
    getLabels(folder)
    getTxt(folder)
    copyToSplitDir(folder)
getNames()