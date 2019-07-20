import glob
from PIL import Image
import os
import re
import shutil
import numpy as np

"""
需放到西工大数据集根目录执行
若要转换为标准 VOC 格式, 将含有 zfill() 函数的行取消注释, 并注释被替代的行即可
"""

classes = ["aeroplane", "ship", "storage_tank", "baseball_diamond",
           "tennis_court", "basketball_court", "ground_track_field",
           "harbor", "bridge", "vehicle"]

s1 = """\t<object>
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

s2 = """<annotation>
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


def create_dirs():
    if os.path.exists('./VOCdevkit'):
        confirm = input("VOCdevket exists, delete it? (y/n) ")
        if confirm == 'y':
            shutil.rmtree('./VOCdevkit')
        else:
            os._exit(1)
    os.makedirs('./VOCdevkit/VOC2007/Annotations')
    os.makedirs('./VOCdevkit/VOC2007/ImageSets/Main')
    os.makedirs('./VOCdevkit/VOC2007/JPEGImages')


def get_annotations():
    sort_result = []
    for i in range(0, 10):
        sort_result.append([])

    text_list = glob.glob('./ground_truth/*.txt')
    for text in text_list:
        input_text = open(text, 'r')
        labels = input_text.readlines()
        input_text.close()
        tmp_str = ''
        filename = text[-7:-4]
        img_size = Image.open('./positive_image_set/' + filename + '.jpg').size

        for i in range(0, len(labels)):
            cor_list = [re.sub('\D', '', s) for s in labels[i].split(',')]
            if cor_list == ['']:
                continue

            # b (xmin, xmax, ymin, ymax)
            b = (int(cor_list[0]), int(cor_list[2]), int(cor_list[1]), int(cor_list[3]))
            cls_num = int(cor_list[4]) - 1
            name = classes[cls_num]
            sort_result[cls_num].append(filename)
            tmp_str += '\n' + s1.format(name, b[0], b[2], b[1], b[3])

        img_name = filename + '.jpg'
        # img_name = filename.zfill(6) + '.jpg'
        result = s2.format(img_name, os.getcwd() + '/' + img_name, img_size[0], img_size[1], tmp_str)
        f = open('./VOCdevkit/VOC2007/Annotations/' + filename + '.xml', 'w')
        # f = open('./VOCdevkit/VOC2007/Annotations/' + filename.zfill(6) + '.xml', 'w')
        f.write(result)
        f.close()
    return sort_result


def cp_imgs():
    filelist = glob.glob('./positive_image_set/*.jpg')
    for img in filelist:
        new_path = './VOCdevkit/VOC2007/JPEGImages/' + img[-7:-4] + '.jpg'
        # new_path = './VOCdevkit/VOC2007/JPEGImages/' + img[-7:-4].zfill(6) + '.jpg'
        shutil.copy(img, new_path)


def get_text(test_rate, sort_result):
    sort_result_unique = []
    train_set = []
    test_set = []
    for i in range(0, 10):
        sort_result_unique.append([])
        sort_result_unique[i] = np.unique(sort_result[i])

    for i in range(0, 10):
        num_sort_result_unique = int(len(sort_result_unique[i]) * test_rate)
        for j in range(0, len(sort_result_unique[i])):
            if j <= num_sort_result_unique:
                test_set.append(sort_result_unique[i][j])
            else:
                train_set.append(sort_result_unique[i][j])

    train_set = np.unique(train_set)
    test_set = np.unique(test_set)

    common_list = [val for val in train_set if val in test_set]
    train_set = list(set(train_set).difference(set(common_list)))
    print("length of train set: " + str(len(train_set)))
    print("length of val set: " + str(len(test_set)))
    print("total length: " + str(len(list(set(train_set) | (set(test_set))))))

    with open('./VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'w') as file:
        for i in range(0, len(train_set)):
            str_i = str(train_set[i])
            # str_i = str_i.zfill(6)
            file.write(str_i + '\n')

    with open('./VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'w') as file:
        for i in range(0, len(test_set)):
            str_i = str(test_set[i])
            # str_i = str_i.zfill(6)
            file.write(str_i + '\n')


if __name__ == '__main__':
    create_dirs()
    sort_result = get_annotations()
    cp_imgs()
    get_text(0.2, sort_result)
