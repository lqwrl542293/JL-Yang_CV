#from https://github.com/k0suke-murakami/ICNet/blob/master/model.py
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import numpy as np


class Model:
    # build icnet model class
    def resize_nn(self, x, ratio):
        """
        Resizing image mianly for upsample
        :param x: Input TF Tensor
        :param ratio: Ratio for resizing, must be int
        :return: Output for resized image(layer)
        """
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return slim.conv2d_transpose(x, x.shape[3], kernel_size=3, stride=2)

    def conv(self, x, num_out_layers, kernel_size=3, stride=1, activation_fn=tf.nn.elu,
             normalizer_fn=None):
        """
        Convolution layer
        :param x: Input TF Tensor
        :param num_out_layers: Number of output filters
        :param kernel_size: Convolution kernel size
        :param stride: Stride for convolution
        :return: Convolution layer
        """
        return slim.conv2d(x, num_out_layers, kernel_size, stride, 'SAME',
                           activation_fn=activation_fn, normalizer_fn=normalizer_fn)

    def max_pool(self, x):
        """
        Max Pooling layer
        :param x: Input TF Tensor
        :return: Pooling layer
        """
        return slim.max_pool2d(x, [2, 2])

    def conv_block(self, x, num_out):
        """
        Convolution block
        :param x: Input TF Tensor
        :param num_out_layers: Number of output filters
        :return: Output of TF Tensor
        """
        with tf.variable_scope("block1"):
            conv1 = self.conv(x, num_out)
            pool1 = self.max_pool(conv1)
        with tf.variable_scope("block2"):
            conv2 = self.conv(pool1, num_out * 2)
            pool2 = self.max_pool(conv2)
        with tf.variable_scope("block3"):
            conv3 = self.conv(pool2, num_out * 4)
            pool3 = self.max_pool(conv3)
        return pool3

    def dilate_conv(self, x, ratio, num_in, num_out):
        """
        Dilate Convolution
        :param x: Input TF Tensor
        :param ratio: Number of dilation
         :param num_in_layers: Number of input filters
        :param num_out_layers: Number of output filters
        :return: Dilated layer
        """
        dilate_filter = tf.Variable(tf.truncated_normal([3, 3, num_in, num_out], stddev=0.01))
        conv1 = tf.nn.atrous_conv2d(x, filters=dilate_filter, rate=ratio, padding='SAME')
        return conv1

    def cff_block(self, small, big, num_in_layers):
        """
        Cascade Feature Fusion Block
        :param small: Input Smaller TF Tensor
        :param big: Input Bigger TF Tensor
         :param num_in_layers: Number of input filters
        :return: Fused Layer and Classfier Layer
        """
        # both small and big has to have same depth layers
        upsample1 = self.resize_nn(small, 2)
        upsample2 = self.conv(upsample1, num_in_layers * 2, normalizer_fn=slim.batch_norm)

        # projection conv 1x1
        projec = self.conv(big, num_in_layers * 2, kernel_size=1, normalizer_fn=slim.batch_norm)

        elementwise_sum1 = tf.add(upsample2, projec)
        elementwise_sum2 = tf.nn.relu(elementwise_sum1)

        classifier = self.conv(upsample1, 6, kernel_size=1)
        softmax = tf.nn.softmax(classifier)
        return elementwise_sum2, classifier

    def build_model(self, low, middle, high):
        """
        Build ICNet
        :param low: Low Resolution Input TF Tensor
        :param middle: Middle Resolution Input TF Tensor
         :param high: High Resolution Input TF Tensor
        :return: classifier Layer * 4
        """

        with tf.variable_scope("shared") as scope:
            low = self.conv_block(low, 32)
            scope.reuse_variables()
            middle = self.conv_block(middle, 32)
        with tf.variable_scope("dilated"):
            low = self.dilate_conv(low, 3, 32 * 4, 32 * 4 * 2)
            # reduce conv
            low = self.conv(low, 32 * 4)
        with tf.variable_scope("cff1"):
            integrate1, classifier1 = self.cff_block(low, middle, 32 * 4)

        with tf.variable_scope("high_res"):
            high = self.conv_block(high, 32 * 2)

        with tf.variable_scope("cff2"):
            integrate2, classifier2 = self.cff_block(integrate1, high, 32 * 4 * 2)

        with tf.variable_scope("decode"):
            upsample1 = self.resize_nn(integrate2, 2)
            upsample2 = self.resize_nn(upsample1, 2)
            pre_cls3 = self.conv(upsample2, 6, kernel_size=1)
            classifier3 = tf.nn.softmax(pre_cls3)
            projec = self.conv(upsample2, 32 * 4 * 2, kernel_size=1)
            upsample3 = self.resize_nn(upsample2, 2)
            pre_cls4 = self.conv(upsample3, 6, kernel_size=1)
            classifier4 = tf.nn.softmax(pre_cls4)
        return classifier1, classifier2, classifier3, classifier4

