# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :statlearning-sjtu-2021
# @File     :123
# @Date     :2021/12/11 19:07
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""
import numpy as np
import pandas as pd
import os
from config import result_path
arg = 'mlp'
svm = np.load('./results/svm_result.npy')
kneighbor = np.load('./results/k-neighbor_result.npy')
rf = np.load('./results/rf_result.npy')
mlp = np.load('./results/mlp_result.npy')
data = np.load('./data_npy/test_data.npy')


print('finish')