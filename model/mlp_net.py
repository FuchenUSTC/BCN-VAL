# -*- encoding: utf-8 -*-
# auth: Fuchen Long
# mail: longfc.ustc@gmail.com
# date: 2022/04/10
# desc: feature input action recognition

import torch
import torch.nn as nn
import numpy as np
from .model_factory import register_model

__all__ = ['MLP_Net']


class MLP_Net(nn.Module):

    def __init__(self, block, layers, pooling_arch, num_classes=200, dropout_ratio=0.5, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, deep_stem=False, clip_length=4):
        super(MLP_Net, self).__init__()
        self.input_dim = 4096
        self.iter_dim = 2048
        self.linear1 = nn.Linear(self.input_dim, self.iter_dim)
        self.bn1 = nn.BatchNorm1d(self.iter_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(self.iter_dim, self.iter_dim)
        self.bn2 = nn.BatchNorm1d(self.iter_dim)
        self.relu2  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.fc = nn.Linear(self.iter_dim, num_classes)
        
    def forward(self, x):
        bsz = x.shape[0]
        x = x.reshape(bsz, -1)
        x = self.linear1(x)
        x - self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x - self.bn2(x)
        x = self.relu2(x)
        x = self.drop(x)
        x = self.fc(x)
        return x


def _mlp_net(arch, block, layers, pooling_arch, **kwargs):
    model = MLP_Net(block, layers, pooling_arch, **kwargs)
    return model


@register_model
def mlp_net(pooling_arch, **kwargs):
    return _mlp_net('mlp_net', None, [2, 2, 2, 2], pooling_arch, **kwargs)
