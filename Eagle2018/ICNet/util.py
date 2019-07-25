# @Author : Qingtian Zhu
# @File : util.py
# @Version : 1.0
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
import numpy as np


# To satisfy API requirement of sklearn.metrics, no need to invoke this method.
def one_hot2label_map(pred):
    label_map = np.zeros([pred.shape[0], pred.shape[1]])
    hot_index = np.where(pred == 1)
    a = hot_index[0]
    b = hot_index[1]
    c = hot_index[2]
    for i in range(hot_index[0].shape[0]):
        label_map[a[i]][b[i]] = c[i]
    return label_map


def conf_mat2_kappa(mat):
    p_0 = 0
    p_e = 0
    sum = np.sum(mat)
    mat_ = mat.transpose(1, 0)
    for i in range(mat.shape[0]):
        p_0 = p_0 + mat[i][i]
        p_e = p_e + np.sum(mat[i])*np.sum(mat_[i])
    p_0 = p_0 / sum
    p_e = p_e / sum / sum
    return (p_0-p_e)/(1-p_e)


# y_pred is label map and y_true is one hot
def calculate_metrics(y_pred, y_true, num_of_class):
    normalized_size = y_pred.shape[0] * y_pred.shape[1]
#    print( y_pred.shape)
 #   print(y_true.shape)
    assert normalized_size == y_true.shape[0] * y_true.shape[1],\
        'y_pred and y_true have different size!'
    y_true = one_hot2label_map(y_true).reshape(normalized_size)
    y_pred = y_pred.reshape(normalized_size)
    label_set = np.arange(num_of_class)
    return accuracy_score(y_true, y_pred), confusion_matrix(y_true, y_pred, labels=label_set)


# It's a test case, plz ignore it.
def test():
    y_pred = np.zeros([2, 2])
    y_pred[0][0] = 0
    y_pred[0][1] = 1
    y_pred[1][0] = 2
    y_pred[1][1] = 2

    y_true = np.zeros([2, 2, 3])
    y_true[0][0][0] = 1
    y_true[0][1][2] = 1
    y_true[1][0][2] = 1
    y_true[1][1][2] = 1
    oa, conf_mat, kappa = calculate_metrics(y_pred, y_true, 3)
    print(oa)
    print(conf_mat)
    print(kappa)
