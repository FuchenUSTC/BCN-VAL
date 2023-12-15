import torch
import torch.nn as nn


class BP(nn.Module):
    def __init__(self, input_dim=2048, dr=128):
        super(BP, self).__init__()
        self.dr = dr
        self.thresh = 1e-8

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
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _bilinearpool(self, x):
         batchSize, dim, h, w = x.data.shape

         x_dr1 = self.conv_dr1(x)
         x_dr2 = self.conv_dr2(x)
         x_dr1 = x_dr1.reshape(batchSize, self.dr, h * w)
         x_dr2 = x_dr2.reshape(batchSize, self.dr, h * w)
         x = 1. / (h * w) * x_dr1.bmm(x_dr2.transpose(1, 2))
         return x

    def _signed_sqrt(self, x):
         x = torch.mul(x.sign(), torch.sqrt(x.abs()+self.thresh))
         return x

    def _l2norm(self, x):
         x = nn.functional.normalize(x)
         return x

    def forward(self, x):
        x = self._bilinearpool(x)
        x = x.view(x.size(0), -1)
        x = self._l2norm(x)
        x = self._signed_sqrt(x)
        return x