
#modify from https://github.com/k0suke-murakami/ICNet/blob/master/main.py

import re
import random
import os.path
from glob import glob
import tensorflow as tf
from model import Model
import tensorflow.contrib.slim as slim
import scipy.misc
import numpy as np
import os
from time import time
log_dir = 'xxx'# dir of the log

def transform_annotation(pic):
    label_map = np.zeros([512, 512, 6])
    label_map = np.load(pic + '.npy')
    label_map = label_map.transpose(1,2,0)
    return label_map.astype(np.uint8)


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    middle_shape = (int(image_shape[0] / 2), int(image_shape[1] / 2))
    low_shape = (int(image_shape[0] / 4), int(image_shape[1] / 4))
    gt2_shape = (int(image_shape[0] / 8), int(image_shape[1] / 8))
    gt1_shape = (int(image_shape[0] / 16), int(image_shape[1] / 16))
    image_paths = glob(os.path.join(data_folder, 'img', '*.bmp'))
    random.shuffle(image_paths)
    all_images = []
    all_gts = []
    for image_file in image_paths:
        # reading images
        gt_image_file = os.path.join(data_folder, 'gt', str(os.path.basename(image_file).split('.')[0]) + '.npy')
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
#        gt = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
        gt_image = transform_annotation(gt_image_file[:-4])
        # add to batch list
        print('after_transform_anno',gt_image.shape)
        all_images.append(image)
        all_gts.append(gt_image)

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for i in range(batch_i, batch_i + batch_size):
                images.append(all_images[i])
                gt_images.append(all_gts[i])
            yield np.array(images), np.array(gt_images)

    return get_batches_fn


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


def train_nn():
    """
    Training ICNet
    :return:
    """
    # define variables
    num_classes = 6
    learning_rate = 0.001
    epochs = 100
    batch_size = 2
    image_shape =(512, 512)

    # input and gt placeholder
    image_input = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="img")
    correct_label4 = tf.placeholder(tf.float32, shape=(None, None, None, num_classes), name="label4")

    # model
    model = Model()
    middle_input, low_input, correct_label3, correct_label2, correct_label1 = pre_process(image_input, correct_label4,
                                                                                          image_shape)
    cls1, cls2, cls3, cls4 = model.build_model(low_input, middle_input, image_input)

    # loss and train_op
    loss1 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=cls1, labels=correct_label1))
    loss2 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=cls2, labels=correct_label2))
    loss3 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=cls3, labels=correct_label3))
    loss4 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=cls4, labels=correct_label4))
    # cross_entropy_loss = loss1+loss2+loss3+loss4
    cross_entropy_loss = loss2 + loss3 + loss4
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)

    # get data function
    get_batches_fn = gen_batch_function('xxx', image_shape) #the dir of training set

    # train
    max_loss = 99999999
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep = 1000)
        tic = time()
        for epoch in range(epochs):
            for image, gt4 in get_batches_fn(batch_size):
                l1, l2, l3, l4, _ = sess.run([loss1, loss2, loss3, loss4, train_op],
                                             feed_dict={image_input: image,
                                                        correct_label4: gt4})
            loss = l2 + l3 + l4
            print("epoch: %d, loss1: %e, loss2: %e, loss3: %e, loss4: %e, overall: %e" % (epoch, l1, l2, l3, l4, loss))
            if loss < max_loss:
               # saver.save(sess, "./model/512_potsdam_model")
                max_loss = loss
            logs = open(os.path.join(log_dir,'again_logs.txt'),'a')
            logs.write('****************\n')
            logs.write('epoch:%d\n'%epoch)
            logs.write('time:%s\n'%(str(time()-tic)))
            logs.write('loss1:%f\n'%(l1))
            logs.write('loss2:%f\n'%(l2))
            logs.write('loss3:%f\n'%(l3))
            logs.write('loss4:%f\n'%(l4))
            logs.write('overall loss:%f\n'%(loss))

            logs.flush()


            #strategy of model saving

            if epoch >0 and epoch <=50:
                if epoch % 2 == 0:
                    saver.save(sess, "./model/again/512_potsdam_model%d"%epoch)
            if epoch > 50 and epoch <=150:
                if epoch % 5 == 0:
                    saver.save(sess, "./model/again/512_potsdam_model%d"%epoch)
            if epoch >150 and epoch <=300:
                if epoch % 3 == 0:
                    saver.save(sess, "./model/512_potsdam_model%d"%epoch)
            if epoch > 300 <= 600:
                if epoch %2 == 0:
                    saver.save(sess, "./model/512_potsdam_model%d"%epoch)
            if epoch > 600:
                break
    logs.close()
if __name__ == '__main__':
    train_nn()
