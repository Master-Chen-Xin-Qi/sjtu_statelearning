# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :statlearning-sjtu-2021
# @File     :train
# @Date     :2021/12/7 16:49
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""

import numpy as np
import torch.optim
from torch.utils.data import DataLoader
import argparse
import joblib
from sklearn.model_selection import StratifiedKFold

import train
from utils import get_device, stat_acc_f1
from model import PrepareDataset, choose_model
from sklearn.model_selection import train_test_split
from config import batch_size, lr, save_path, arg_list, data_len, data_width, ml_epoch


if __name__ == "__main__":
    data, label = np.load('./data_npy/train_data.npy'), np.load('./data_npy/train_label.npy')

    parser = argparse.ArgumentParser(description='Four Training Models')
    parser.add_argument('-m', '--model', type=str, default='mlp', choices=arg_list)
    args = parser.parse_args()
    arg = args.model
    model = choose_model(arg)
    if arg == 'mlp':
        train_data, vali_data, train_label, vali_label = train_test_split(data, label, test_size=0.2, random_state=0)
        dataset_train = PrepareDataset(train_data, train_label)
        dataset_vali = PrepareDataset(vali_data, vali_label)
        train_loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
        vali_loader = DataLoader(dataset_vali, shuffle=True, batch_size=batch_size)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.001)  # L2正则化参数为0.001
        trainer = train.Trainer(model=model, optimizer=optimizer, save_path=save_path, save_name=arg, device=get_device(None))
        trainer.train(train_loader, vali_loader, model_file=None)
    else:
        data = data.reshape(-1, data_len*data_width)
        label = label[:, 0, 0]
        skf_cv = StratifiedKFold(n_splits=10, shuffle=True)
        best_acc = 0
        for e in range(ml_epoch):
            i = 0
            for train_index, vali_index in skf_cv.split(data, label):
                X_train, X_vali = data[train_index], data[vali_index]
                y_train, y_vali = label[train_index], label[vali_index]
                model.fit(X_train, y_train)
                train_predict = model.predict(X_train)
                train_acc, train_f1 = stat_acc_f1(y_train, train_predict)
                predict = model.predict(X_vali)
                acc, f1 = stat_acc_f1(y_vali, predict)
                i += 1
                print('Epoch %d/%d 第%d折交叉验证，训练集 Acc: %f, 验证集 Acc: %f' % (e, ml_epoch, i, train_acc, acc))
                if acc > best_acc:
                    best_acc = acc
                    save_model = save_path+'/'+arg+'.m'
                    joblib.dump(model, save_model)
                    print('Already save model %s' % save_model)
                    save_describe = save_path+'/'+arg+'.txt'
                    f = open(save_describe, 'w')
                    f.write('Epoch %d/%d 第%d折交叉验证，训练集 Acc: %f, 验证集 Acc: %f' % (e, ml_epoch, i, train_acc, acc))
                    f.close()
