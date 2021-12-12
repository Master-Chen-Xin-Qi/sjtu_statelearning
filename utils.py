# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :statlearning-sjtu-2021
# @File     :utils
# @Date     :2021/12/7 16:17
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""
import numpy as np
import os
import torch
import joblib
from sklearn.metrics import f1_score
from config import train_folder, test_folder, data_len, data_width, label_train, save_path


# 载入所有训练数据和标签
def get_data(folder_name):
    data = np.zeros((1, data_len, data_width))
    label = np.zeros((1, data_len, data_width))
    label_dict = train_label_dict()
    for (root, _, files) in os.walk(folder_name):
        for filename in files:
            file = os.path.join(root, filename)
            tmp_data = np.load(file).reshape(1, data_len, data_width)
            tmp_label = get_train_label(filename, label_dict)
            data = np.vstack((data, tmp_data))
            label = np.vstack((label, tmp_label))
    data = data[1:]
    label = label[1:]
    np.save('./data_npy/train_data.npy', data)
    np.save('./data_npy/train_label', label)
    return data, label


# 载入测试集数据
def get_test_data(folder_name):
    data = np.zeros((1, data_len, data_width))
    for (root, _, files) in os.walk(folder_name):
        for filename in files:
            file = os.path.join(root, filename)
            tmp_data = np.load(file).reshape(1, data_len, data_width)
            data = np.vstack((data, tmp_data))
    data = data[1:]
    np.save('./data_npy/test_data.npy', data)
    return data


# 根据.csv文件找到训练集对应的标签
def get_train_label(file_name, label_dict):
    label = label_dict[file_name]
    label_np = np.array([label]*data_len*data_width).reshape(1, data_len, data_width)
    return label_np


# 将csv文件中每一行对应为一个字典
def train_label_dict():
    label_dict = dict()
    fid = open(label_train, 'r')
    for line in fid:
        if 'id' in line:
            continue
        line = line.split(',')
        line[1] = int(line[1].replace('\n', ''))
        label_dict[line[0]] = line[1]
    return label_dict


def get_device(gpu):
    "get device (CPU or GPU)"
    if gpu is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device


def stat_acc_f1(label, results_estimated):
    # label = np.concatenate(label, 0)
    # results_estimated = np.concatenate(results_estimated, 0)
    f1 = f1_score(label, results_estimated, average='macro')
    acc = np.sum(label == results_estimated) / label.size
    return acc, f1


def dl_acc_f1(label, results_estimated):
    label_estimated = np.argmax(results_estimated, 1)
    f1 = f1_score(label, label_estimated, average='macro')
    acc = np.sum(label == label_estimated) / label.size
    return acc, f1

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


# 对整体数据做归一化
def total_min_max(data, max_data, min_data):
    from sklearn import preprocessing
    data = (data-min_data)/(max_data-min_data)
    return data


if __name__ == "__main__":
    train_data, train_label = get_data(train_folder)
    test_data = get_test_data(test_folder)
    # max_pos = np.unravel_index(np.argmax(train_data), train_data.shape)
    # max_data = train_data[max_pos[0], max_pos[1], max_pos[2]]
    # max_pos = np.unravel_index(np.argmax(test_data), train_data.shape)
    # test_max = test_data[max_pos[0], max_pos[1], max_pos[2]]
    # if test_max > max_data:
    #     max_data = test_max
    #
    # min_pos = np.unravel_index(np.argmin(train_data), train_data.shape)
    # min_data = train_data[min_pos[0], min_pos[1], min_pos[2]]
    # min_pos = np.unravel_index(np.argmin(test_data), train_data.shape)
    # test_min = test_data[min_pos[0], min_pos[1], min_pos[2]]
    # if test_min < min_data:
    #     min_data = test_min
    #
    # train_data = (train_data - min_data) / (max_data - min_data)
    # test_data = (test_data - min_data) / (max_data - min_data)
    # np.save('./data_npy/train_data.npy', train_data)
    # np.save('./data_npy/test_data.npy', test_data)
