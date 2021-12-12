# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :statlearning-sjtu-2021
# @File     :test
# @Date     :2021/12/11 18:37
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""

import numpy as np
import argparse
import joblib
import torch
import pandas as pd
from model import choose_model
from config import save_path, result_path, arg_list, data_len, data_width
import os


if __name__ == '__main__':
    test_data = np.load('./data_npy/test_data.npy')
    parser = argparse.ArgumentParser(description='Four Training Models')
    parser.add_argument('-m', '--model', type=str, default='mlp', choices=arg_list)
    args = parser.parse_args()
    arg = args.model
    print(arg)
    if arg == 'mlp':
        save_model = save_path + '/' + arg + '.pt'
        model = choose_model(arg)
        model.load_state_dict(torch.load(save_model))
        model.eval()
        test_data = torch.from_numpy(test_data.astype('float32'))
        results = model(test_data).detach().numpy()
        results = np.argmax(results, -1)
    else:
        test_data_reshape = test_data.reshape(-1, data_len*data_width)
        save_model = save_path + '/' + arg + '.m'
        model = joblib.load(save_model)
        results = model.predict(test_data_reshape)
    np.save(result_path+'/'+arg+'_result.npy', results)
    # 保存为csv文件
    test_file = []
    for (root, _, files) in os.walk('./test'):
        for filename in files:
            test_file.append(filename)
    dataframe = pd.DataFrame({'id': test_file, 'category': results})
    dataframe.to_csv(result_path + '/' + arg + "_test.csv", index=False, sep=',')
