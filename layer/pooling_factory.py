import torch
import torch.nn as nn

from layer.BP import *
from layer.IBP import *
from layer.CBP import *
from layer.SAP import *
from layer.MHAP import *


class PoolingAverage(nn.Module):
    def __init__(self, input_dim=2048):
        super(PoolingAverage, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = input_dim

    def forward(self, x):
        x = torch.flatten(self.pool(x.view(x.size(0), x.size(1), x.size(2), -1)), 1)
        return x


class PoolingBP(nn.Module):
    def __init__(self, input_dim=2048):
        super(PoolingBP, self).__init__()
        self.dr = 128
        self.pool = BP(input_dim=input_dim, dr=self.dr)
        self.output_dim = self.dr * self.dr

    def forward(self, x):
        x = torch.flatten(self.pool(x.view(x.size(0), x.size(1), x.size(2), -1)), 1)
        return x


class PoolingCBP(nn.Module):
    def __init__(self, input_dim=2048):
        super(PoolingCBP, self).__init__()
        self.pool = CBP(thresh=1e-8, projDim=16384, input_dim=input_dim)
        self.output_dim = 16384

    def forward(self, x):
        x = torch.flatten(self.pool(x.view(x.size(0), x.size(1), x.size(2), -1)), 1)
        return x


class PoolingIBP(nn.Module):
    def __init__(self, input_dim=2048):
        super(PoolingIBP, self).__init__()
        self.multiplier = 8
        self.pool = IBP(input_dim=input_dim, multiplier=self.multiplier)
        self.output_dim = input_dim * self.multiplier

    def forward(self, x):
        x = torch.flatten(self.pool(x.view(x.size(0), x.size(1), x.size(2), -1)), 1)
        return x


class PoolingSAP(nn.Module):
    def __init__(self, input_dim=2048):
        super(PoolingSAP, self).__init__()
        self.dr = 512
        self.pool = SAP(input_dim=input_dim, dr=self.dr)
        self.output_dim = input_dim

    def forward(self, x):
        x = torch.flatten(self.pool(x.view(x.size(0), x.size(1), x.size(2), -1)), 1)
        return x


class PoolingMHAP(nn.Module):
    def __init__(self, input_dim=2048):
        super(PoolingMHAP, self).__init__()
        self.num_head = 8
        self.pool = MHAP(input_dim=input_dim, num_head=self.num_head)
        self.output_dim = input_dim * self.num_head

    def forward(self, x):
        x = torch.flatten(self.pool(x.view(x.size(0), x.size(1), x.size(2), -1)), 1)
        return x


pooling_dict = {
    'PoolingAverage': PoolingAverage,
    'PoolingBP': PoolingBP,
    'PoolingCBP': PoolingCBP,
    'PoolingIBP': PoolingIBP,
    'PoolingSAP': PoolingSAP,
    'PoolingMHAP': PoolingMHAP
}


def get_pooling_by_name(pooling_name):
    return pooling_dict.get(pooling_name)