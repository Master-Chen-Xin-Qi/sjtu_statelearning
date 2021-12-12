# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :statlearning-sjtu-2021
# @File     :train
# @Date     :2021/12/8 11:42
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""

import torch
import torch.nn as nn
import time
from utils import dl_acc_f1
from config import epoch, save_path


def func_evaluate(label, predicts):
    stat = dl_acc_f1(label.cpu().numpy(), predicts.cpu().numpy())
    return stat


class Trainer(object):
    def __init__(self, model, optimizer, save_path, save_name, device):
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.save_name = save_name
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def load(self, model_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file + '.pt', map_location=self.device))

    def func_loss(self, batch, vali_flag=False):
        data, label = batch
        label = label[:, 0, 0]
        predict = self.model(data)
        loss = self.criterion(predict, label)
        if vali_flag:
            return loss.mean().cpu().numpy(), predict, label
        return loss, predict, label

    def train(self, train_loader, vali_loader, model_file=None):
        self.load(model_file)
        # self.model = self.model.to(self.device)
        best_acc = 0.1
        for e in range(epoch):
            total_loss = 0.0
            total_time = 0.0
            self.model.train()
            train_predict = []
            train_label = []
            for i, batch in enumerate(train_loader):
                # batch = [b.to(self.device) for b in batch]
                start_time = time.time()
                loss, predict, label = self.func_loss(batch)
                loss.backward()
                self.optimizer.step()
                total_time += time.time() - start_time
                total_loss += loss
                train_predict.append(predict)
                train_label.append(label)
            train_label = torch.cat(train_label, 0).cpu().numpy()
            train_predict = torch.cat(train_predict, 0).detach().numpy()
            train_acc, train_f1 = dl_acc_f1(train_label, train_predict)
            loss_eva, predicts, labels = self.validate(vali_loader, model_file)
            acc, f1 = dl_acc_f1(labels, predicts)
            print('Epoch %d/%d : Train Loss: %5.4f Acc: %5.4f. Validate Loss: %5.4f Acc: %5.4f.'
                  % (e + 1, epoch, total_loss / len(train_loader), train_acc, loss_eva, acc))
            if acc > best_acc:
                best_acc = acc
                save_model = self.save_path + '/' + self.save_name + '.pt'
                torch.save(self.model.state_dict(), save_model)  # 保存最优模型
                print('Already saved model %s!' % save_model)
                save_describe = save_path + '/' + 'mlp.txt'
                f = open(save_describe, 'w')
                f.write('Epoch %d/%d Best Train Loss %5.4f Acc: %5.4f. Validate Loss %5.4f Accuracy %5.4f'
                  % (e + 1, epoch, total_loss / len(train_loader), train_acc, loss_eva, acc))
                f.close()
        print('The Total Epoch have been reached.')

    def validate(self, vali_loader, model_file):
        self.model.eval()  # evaluation mode
        self.load(model_file)
        total_loss = 0.0
        time_sum = 0.0
        result = []
        labels = []
        for batch in vali_loader:
            # batch = [b.to(self.device) for b in batch]
            with torch.no_grad():  # evaluation without gradient calculation
                start_time = time.time()
                loss, predict, label = self.func_loss(batch, vali_flag=True)
                time_sum += time.time() - start_time
                total_loss += loss.item()
                result.append(predict)
                labels.append(label)
        return total_loss / len(vali_loader), torch.cat(result, 0).cpu().numpy(), torch.cat(labels, 0).cpu().numpy()

