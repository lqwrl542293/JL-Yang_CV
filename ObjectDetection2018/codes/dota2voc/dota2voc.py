#!/usr/bin/env python
# coding: utf-8
import glob
from PIL import Image
import os
import shutil

"""
    放到 DOTA 数据集根目录下先split数据集后运行即可
"""

s1="""\t<object>
\t\t<name>{0}</name>
\t\t<pose>Unspecified</pose>
\t\t<truncated>0</truncated>
\t\t<difficult>0</difficult>
\t\t<bndbox>
\t\t\t<xmin>{1}</xmin>
\t\t\t<ymin>{2}</ymin>
\t\t\t<xmax>{3}</xmax>
\t\t\t<ymax>{4}</ymax>
\t\t</bndbox>
\t</object>"""

s2="""<annotation>
\t<folder>VOC2007</folder>
\t<filename>{0}</filename>
\t<path>{1}</path>
\t<source>
\t\t<database>My Database</database>
\t\t<annotation>VOC2007</annotation>
\t\t<image>flickr</image>
\t\t<flickrid>NULL</flickrid>
\t</source>
\t<owner>
\t\t<flickrid>NULL</flickrid>
\t\t<name>J</name>
\t</owner>
\t<size>
\t\t<width>{2}</width>
\t\t<height>{3}</height>
\t\t<depth>3</depth>
\t</size>
\t<segmented>0</segmented>
\t{4}
</annotation>
"""

def createDirs():
    if os.path.exists('./VOCdevkit'):
        confirm = input("VOCdevkit已存在是否删除? y/n: ")
        if confirm == 'y':
            shutil.rmtree('./VOCdevkit')
        else:
            os._exit(1)
    os.makedirs('./VOCdevkit/VOC2007/Annotations')
    os.makedirs('./VOCdevkit/VOC2007/ImageSets/Main')
    os.makedirs('./VOCdevkit/VOC2007/JPEGImages')

def getTxt(mode):
    filelist = glob.glob(os.getcwd() + '/' + mode + 'Split/labelTxt/*.txt')
    f = open('./VOCdevkit/VOC2007/ImageSets/Main/' + mode + '.txt', 'w')
    for text_ in filelist:
        f.write(text_[text_.index('P'):-4] + '\n')
    f.close()

def cpImgs(mode):
    filelist = glob.glob(os.getcwd() + '/' + mode + 'Split/images/*.png')
    for img in filelist:
        shutil.copy(img, './VOCdevkit/VOC2007/JPEGImages/')

def getAnnotations(mode):
    textlist = glob.glob('./' + mode + 'Split/labelTxt/*.txt')
    for text_ in textlist:
        inputText = open(text_, 'r')
        labels = inputText.readlines()
        inputText.close()
        str = ''
        filename = text_[text_.index('P'): -4]
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
            name = tmp[8]
            str += '\n' + s1.format(name, xmin, ymin, xmax, ymax)
            
        imagename = filename + '.png'
        result = s2.format(imagename, os.getcwd() + '/' +  imagename, imageSize[0], imageSize[1], str)
        f = open('./VOCdevkit/VOC2007/Annotations/' + filename + '.xml', 'w')
        f.write(result)
        f.close()


createDirs()
for mode in ['train', 'val']:
    getAnnotations(mode)
    getTxt(mode)
    cpImgs(mode)

