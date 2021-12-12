# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :statlearning-sjtu-2021
# @File     :model
# @Date     :2021/12/7 20:53
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""
import argparse

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
from config import arg_list, data_len, data_width
from utils import split_last, merge_last


# 制作数据集
class PrepareDataset(Dataset):
    def __init__(self, data, label):
        # super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        instance = self.data[index]
        return torch.from_numpy(instance).float(), torch.from_numpy(np.array(self.label[index])).long()

    def __len__(self):
        return len(self.data)


# 训练模型主体
def choose_model(arg):
    if arg not in arg_list:
        raise Exception("Model not in list!")
    if arg == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100)
        # model.fit(trainX, trainY, train_wt)
    elif arg == 'svm':
        from sklearn import svm
        model = svm.SVC()
        # model = svm.SVC(C=3, kernel='rbf', gamma=10, decision_function_shape='ovr')
        # model.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
    elif arg == 'k-neighbor':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=10)
    elif arg == 'mlp':
        model = MLP(in_channel=100, label_num=20)
    else:
        # Bert参数
        parser1 = argparse.ArgumentParser(description='Bert')
        parser1.add_argument('--n_heads', type=int, default=4)
        parser1.add_argument('--n_layer', type=int, default=4)
        parser1.add_argument('--hidden', type=int, default=72)
        parser1.add_argument('--hidden_ff', type=int, default=144)
        parser1.add_argument('--seq_len', type=int, default=100)
        parser1.add_argument('--feature_num', type=int, default=15)
        parser1.add_argument('--emb_norm', type=bool, default=True)
        cfg = parser1.parse_args()
        # GRU参数
        parser2 = argparse.ArgumentParser(description='GRU')
        parser2.add_argument('--seq_len', type=int, default=100)
        parser2.add_argument('--input', type=int, default=15)
        parser2.add_argument('--num_rnn', type=int, default=2)
        parser2.add_argument('--num_layers', default=[2, 1])
        parser2.add_argument('--rnn_io', default=[[6, 30], [30, 40]])
        parser2.add_argument('--num_linear', type=int, default=1)
        parser2.add_argument('--linear_io', type=int, default=[[40, 20]])
        parser2.add_argument('--activ', type=bool, default=False)
        parser2.add_argument('--dropout', type=bool, default=True)
        mfg = parser2.parse_args()
        model = Transformer(cfg, mfg)
    return model


class MLP(nn.Module):
    def __init__(self, in_channel, label_num, dropout=0.8):
        super(MLP, self).__init__()
        self.in_channel = in_channel
        self.label_num = label_num
        self.layer1 = nn.Linear(in_channel, 128)
        self.bn1 = nn.BatchNorm1d(data_len)
        self.relu1 = nn.ReLU()
        self.pooling1 = nn.AdaptiveAvgPool1d(30)
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(data_len)
        self.pooling2 = nn.AdaptiveAvgPool1d(20)
        self.layer4 = nn.Linear(64*data_len, label_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.bn1(self.relu1(self.layer1(x)))
        x = self.dropout(x)
        # x = self.pooling1(x)
        x = self.bn2(self.relu2(self.layer2(x)))
        x = self.dropout(x)
        # x = self.pooling2(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.layer4(x)
        return x


class Embeddings(nn.Module):

    def __init__(self, cfg, pos_embed=None):
        super().__init__()
        # Original BERT Embedding
        # self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.hidden) # token embedding

        # factorized embedding
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden) # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNorm(cfg)
        self.emb_norm = cfg.emb_norm

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)

        # factorized embedding
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg, mfg):
        super().__init__()
        # Original BERT not used parameter-sharing strategies

        # To used parameter-sharing strategies
        self.embed = Embeddings(cfg)
        self.n_layers =cfg.n_layer
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        self.classifier = ClassifierGRU(mfg, input=cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        x = self.embed(x)
        for _ in range(self.n_layers):
            x = self.attn(x)
            x = self.norm1(x + self.proj(x))
            x = self.norm2(x + self.pwff(x))
        x = self.classifier(x)
        return x


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        #scores = self.drop(F.softmax(scores, dim=-1))
        scores = F.softmax(scores, dim=-1)  # scores[x][y]中每一行的和为1
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(cfg.hidden), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class ClassifierGRU(nn.Module):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        for i in range(cfg.num_rnn):
            if input is not None and i == 0:
                self.__setattr__('gru' + str(i), nn.GRU(input, cfg.rnn_io[i][1], num_layers=cfg.num_layers[i], batch_first=True))
            else:
                self.__setattr__('gru' + str(i),
                                 nn.GRU(cfg.rnn_io[i][0], cfg.rnn_io[i][1], num_layers=cfg.num_layers[i],
                                         batch_first=True))
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_rnn = cfg.num_rnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_rnn):
            rnn = self.__getattr__('gru' + str(i))
            h, _ = rnn(h)
            if self.activ:
                h = F.relu(h)
        h = h[:, -1, :]
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class FCN_LSTM(nn.Module):
    def __init__(self):
        super(FCN_LSTM, self).__init__()



