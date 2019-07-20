import os
import numpy as np
import re

try:
    from cv2 import imread, imwrite
except ImportError:
    # Note that, sadly, skimage unconditionally import scipy and matplotlib,
    # so you'll need them if you don't have OpenCV. But you probably have them.
    from skimage.io import imread, imsave
    imwrite = imsave

from os import listdir, getcwd
from os.path import join

classes = ["aeroplane", "ship", "storage_tank", "baseball_diamond", "tennis_court", "basketball_court", "ground_track_field", "harbor", "bridge", "vehicle"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

wd = getcwd()

gt_dir = wd + '/ground_truth'
imageset_dir = wd + '/positive_image_set'
labels_dir = wd + '/labels'
gt_file_list = os.listdir(gt_dir)
sort_result = []
for i in range(0,10):
    sort_result.append([])

for gt_filename in gt_file_list:
    gt_num = int(gt_filename[0:3].lstrip('0')) #parse the number of the gt filename
    whole_gt_filename = os.path.join(gt_dir,gt_filename)
    label_filename = os.path.join(labels_dir, gt_filename[0:3] + '.txt')
    label_file = open(label_filename,'w')
    with open(whole_gt_filename) as gt_file:  #read the gt file
        while True:
            lines = gt_file.readline() #read a line
            if not lines or lines == '\r\n':
                break
            try:
                cls_num = int(lines.split(',')[4])-1
            except:
                continue
            sort_result[cls_num].append(gt_num)
            #TO DO...
            cor_list = [ re.sub("\D", "", s) for s in lines.split(',')  ]
            b = (float(cor_list[0]), float(cor_list[2]), float(cor_list[1]),float(cor_list[3]))
            jpg_filename = gt_filename[0:3] + '.jpg'
            train_pic_name = os.path.join(imageset_dir, jpg_filename)
            img = imread(train_pic_name)
            bb = convert((img.shape[1], img.shape[0]), b)
            label_file.write(str(int(cor_list[4])-1) + " " + " ".join([str(a) for a in bb]) + '\n')

    label_file.close()

sort_result_unique = []
for i in range(0,10):
    sort_result_unique.append([])
    sort_result_unique[i] = np.unique(sort_result[i])

test_rate = 0.2
train_set = []
val_set = []
for i in range(0,10):
    num_sort_result_unique = int(len(sort_result_unique[i]) * test_rate)
    for j in range(0,len(sort_result_unique[i])):
        if j <= num_sort_result_unique:
            val_set.append(sort_result_unique[i][j])
        else:
            train_set.append(sort_result_unique[i][j])

train_set = np.unique(train_set)
val_set = np.unique(val_set)

common_list = [val for val in train_set if val in val_set]
train_set = list(set(train_set).difference(set(common_list)))

print(len(train_set))
print('\n')
print(len(val_set))
print('\n')
print(len(list(set(train_set) | (set(val_set)))))

train_set_file = open('train.txt','w')
for i in range(0,len(train_set)):
    str_i = str(train_set[i])
    str_i = str_i.zfill(3) + '.jpg'
    train_pic_name = os.path.join(imageset_dir,str_i)
    train_set_file.write(train_pic_name)
    train_set_file.write('\n')

train_set_file.close()

val_set_file = open('val.txt','w')
for i in range(0,len(val_set)):
    str_i = str(val_set[i])
    str_i = str_i.zfill(3) + '.jpg'
    val_pic_name = os.path.join(imageset_dir,str_i)
    val_set_file.write(val_pic_name)
    val_set_file.write('\n')

val_set_file.close()
