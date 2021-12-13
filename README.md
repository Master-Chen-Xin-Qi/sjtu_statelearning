# Readme

## 数据获取

- 在`utils.py`中注释部分为对原始数据载入并处理，保存在了`data_npy`文件夹中，分别有：`train_data.npy`、`train_label.npy`、`test_data.npy`

## 训练模型

- 在`main.py`中更改**arg**参数即可以切换训练模型，可选择的模型在`config.py`中的`arg_list`中，有随机森林、支持向量机、K近邻、多层感知机四种模型
- 设定好arg之后运行`main.py`即可开始训练，训练最优模型与相关信息保存在了`saved`文件夹中，以模型的名字命名

## 得到测试结果

- 在`test.py`中设置相同的**arg**再运行即可在`results`文件夹中得到相应的预测结果