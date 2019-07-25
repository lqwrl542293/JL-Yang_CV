import re
import random
import os.path
from glob import glob
import tensorflow as tf
from model import Model
import tensorflow.contrib.slim as slim
import scipy.misc
from util import *
import numpy as np
from time import time
def label2annotation(pic):
    # input:  8 bits picture
    # output: 24 bits picture

    #Potsdam dataset only

    new_pic = np.zeros([512, 512, 3])
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if pic[i][j] == 0:
                new_pic[i][j][0] = 255
                new_pic[i][j][1] = 255
                new_pic[i][j][2] = 255

            elif pic[i][j] == 1:
                new_pic[i][j][0] = 0
                new_pic[i][j][1] = 0
                new_pic[i][j][2] = 255

            elif pic[i][j] == 2:
                new_pic[i][j][0] = 0
                new_pic[i][j][1] = 255
                new_pic[i][j][2] = 255
            elif pic[i][j] == 3:
                new_pic[i][j][0] = 0
                new_pic[i][j][1] = 255
                new_pic[i][j][2] = 0

            elif pic[i][j] == 4:
                new_pic[i][j][0] = 255
                new_pic[i][j][1] = 255
                new_pic[i][j][2] = 0

            elif pic[i][j] == 5:
                new_pic[i][j][0] = 255
                new_pic[i][j][1] = 0
                new_pic[i][j][2] = 0

    return new_pic.astype(np.uint8)


def pre_process(input, gt, image_shape):
    """
    Transforming the size of input image and ground truth image
    :param input: High-resolution input image
    :param gt: Ground truth label
    :return: Scaled input values.
    """
    middle = tf.image.resize_images(input, (int(image_shape[0] / 2), int(image_shape[1] / 2)),
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    low = tf.image.resize_images(input, (int(image_shape[0] / 4), int(image_shape[1] / 4)),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    gt3 = tf.image.resize_images(gt, (int(image_shape[0] / 2), int(image_shape[1] / 2)),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    gt2 = tf.image.resize_images(gt, (int(image_shape[0] / 8), int(image_shape[1] / 8)),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    gt1 = tf.image.resize_images(gt, (int(image_shape[0] / 16), int(image_shape[1] / 16)),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return middle, low, gt3, gt2, gt1


if __name__ == '__main__':
    target_dir = '/server_space/jiangyl/zhu_useful/Potsdam_512_full/test/gt'
    image_shape = (512, 512)
    target = './text_output/'
    # input and gt placeholder
    image_input = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="img")
    correct_label4 = tf.placeholder(tf.float32, shape=(None, None, None, 6), name="label4")
    cms = np.zeros([6,6])
    val = os.listdir(target_dir)
    # model
    model = Model()
    oas = 0
    middle_input, low_input, correct_label3, correct_label2, correct_label1 = pre_process(image_input, correct_label4,                                                                                          image_shape)
    count = 0
    total_time = 0
    cls1, cls2, cls3, cls4 = model.build_model(low_input, middle_input, image_input)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, "xxx") #dir of model
        image_paths = glob(os.path.join('xxx', '*.bmp')) #dir of testing set
        for image_file in image_paths:
            image = scipy.misc.imresize(scipy.misc.imread(image_file, mode='RGB'), image_shape)
            image = np.expand_dims(image, 0)
            time_0 = time()
            count = count+1
            im_softmax = sess.run(
                [cls4],
                {image_input: image})
#            mask = im_softmax
            time_1 = time()-time_0
            labels = np.argmax(im_softmax[0], axis=3)[0]
            mask = labels
            a = image_file.split('/')
            gt = np.load(os.path.join(target_dir,a[7][:-3]+'npy'))
            gt = np.transpose(gt,(1,2,0))
            oa,cm = calculate_metrics(mask,gt,6)
            oas = oa+oas
            cms = cm + cms
            kappa = conf_mat2_kappa(cms)

            img = label2annotation(labels)
            total_time = total_time+time_1
            result = open(os.path.join(target,'xxx'),'w') #dir to save accuracy result
            print >> result, 'OA:', oas/count
            print >> result, 'confusion_matrix:', cms
            print >> result, 'kappa:',kappa
            print >> result, 'time:',total_time
            print(time_1)
            scipy.misc.imsave('xxx'+os.path.basename(image_file).split('.')[0] + ".bmp", img, "bmp") #save testing result pics
