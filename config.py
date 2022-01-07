# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :statlearning-sjtu-2021
# @File     :config
# @Date     :2021/12/7 16:21
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""

train_folder = './train'
test_folder = './test'
data_len = 15  # 数据的长度
data_width = 100  # 数据的宽度
label_train = './label_train.csv'
batch_size = 128
lr = 1e-4  # 学习率
epoch = 1000  # MLP训练的轮数
ml_epoch = 300  # 机器学习算法的迭代次数
save_path = './saved'  # 模型和结果保存位置
result_path = './results'  # 测试集预测结果保存
arg_list = ['rf', 'svm', 'k-neighbor', 'mlp']  # 四种模型，分别是随机森林、支持向量机、KNN、MLP
