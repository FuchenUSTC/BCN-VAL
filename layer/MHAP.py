import torch
import torch.nn as nn

from layer.SAP import *


class MHAP(nn.Module):
    def __init__(self, input_dim=2048, num_head=8):
        super(MHAP, self).__init__()
        self.num_head = num_head

        assert num_head == 8

        self.pool1 = SAP(input_dim=input_dim, dr=512)
        self.pool2 = SAP(input_dim=input_dim, dr=512)
        self.pool3 = SAP(input_dim=input_dim, dr=512)
        self.pool4 = SAP(input_dim=input_dim, dr=512)
        self.pool5 = SAP(input_dim=input_dim, dr=512)
        self.pool6 = SAP(input_dim=input_dim, dr=512)
        self.pool7 = SAP(input_dim=input_dim, dr=512)
        self.pool8 = SAP(input_dim=input_dim, dr=512)


    def forward(self, x):
        x = torch.cat((self.pool1(x), self.pool2(x), self.pool3(x), self.pool4(x), self.pool5(x), self.pool6(x), self.pool7(x), self.pool8(x)), dim=1)
        return x