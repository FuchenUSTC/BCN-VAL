import torch
import torch.nn as nn

import math


class SAP(nn.Module):
    def __init__(self, input_dim=2048, dr=512):
        super(SAP, self).__init__()
        self.dr = dr

        self.conv_dr1 = nn.Sequential(
            nn.Conv2d(input_dim, self.dr, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.dr),
            nn.ReLU(inplace=True)
        )
        self.conv_dr2 = nn.Sequential(
            nn.Conv2d(input_dim, self.dr, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.dr),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _nonlocalpool(self, x):
         batchSize, dim, h, w = x.data.shape

         x_dr1 = self.conv_dr1(x)
         x_dr1 = x_dr1.reshape(batchSize, self.dr, h * w)

         x_dr2 = self.conv_dr2(x)
         x_dr2 = self.pool(x_dr2)
         x_dr2 = x_dr2.reshape(batchSize, self.dr, 1)

         x_dp = x_dr1.transpose(1, 2).bmm(x_dr2) / math.sqrt(self.dr)
         x_dp = self.softmax(x_dp)

         x = x.reshape(batchSize, dim, h * w)
         x = x.bmm(x_dp)
         return x

    def forward(self, x):
        x = self._nonlocalpool(x)
        x = x.view(x.size(0), -1)
        return x