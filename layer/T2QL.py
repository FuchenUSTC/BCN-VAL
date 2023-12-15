# -*- encoding: utf-8 -*-
# auth: Fuchen Long
# mail: longfc.ustc@gmail.com
# date: 2021/04/19
# desc: Twin-turbo sub sum up to query


import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict

class T2QL(nn.Module):

    def __init__(self, mapping_file, e=0, reduction='mean', 
                 prob_thres=0.008, pseudo_conf_thres=0.20):
        super().__init__()
        self._build_mapping(mapping_file)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.e = e
        self.reduction = reduction
        self.prob_thres = prob_thres
        self.pseudo_conf_thres = pseudo_conf_thres

    def _build_mapping(self, mapping_file):
        self.sub_main_mapping = {}
        self.main_sub_mapping = defaultdict(list)
        df_map = pd.read_csv(mapping_file,sep='\n',header=None)
        for idx, row in df_map.iterrows():
            main = row[0]
            sub = idx
            self.sub_main_mapping[sub] = main
            self.main_sub_mapping[main].append(sub)

    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors

        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1

        Returns:
            return one hot format labels in shape [batchsize, classes]
        """

        one_hot = torch.zeros(labels.size(0), classes)

        # labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)

        one_hot.scatter_add_(1, labels, value_added)

        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth

        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / (length - 1)

        return one_hot.to(target.device)

    def _get_prob_sum(self, target_main, score_sub):
        sub_index_list = self.main_sub_mapping[target_main]
        score_sub_np = score_sub.cpu().numpy().tolist()
        prob_sum = 0
        for index in sub_index_list:
            prob_sum += score_sub_np[index]
        return prob_sum

    def _sub2main_corrected_label(self, target, classes, x, x_sub):
        batch_size = target.size(0)
        score_main = self.softmax(x).detach()
        score_sub = self.softmax(x_sub).detach()
        pseudo_label_main = torch.argmax(score_main, dim=1)  
        label1_index, value1 = [],[]
        label2_index, value2 = [],[]
        for index in range(batch_size):
            target_main = target[index].data.cpu().numpy().tolist()
            pseudo_main = pseudo_label_main[index].data.cpu().numpy().tolist()
            prob_target_main = self._get_prob_sum(target_main, score_sub[index])
            prob_pseudo_main = self._get_prob_sum(pseudo_main, score_sub[index])
            lambda_target = 1.0
            lambda_correct = 0.0
            if target_main != pseudo_main and prob_pseudo_main >= self.pseudo_conf_thres:
                lambda_target = prob_target_main
                lambda_correct = prob_pseudo_main
                lambda_sum = lambda_target + lambda_correct
                lambda_target /= lambda_sum
                lambda_correct /= lambda_sum
            label1_index.append(target_main)
            value1.append(lambda_target)
            label2_index.append(pseudo_main)
            value2.append(lambda_correct)            
        one_hot = torch.zeros(target.size(0), classes)
        value1_added = torch.Tensor(value1).view(target.size(0),1)
        value2_added = torch.Tensor(value2).view(target.size(0),1)
        labels1 = torch.LongTensor(label1_index).view(target.size(0),1)
        labels2 = torch.LongTensor(label2_index).view(target.size(0),1)
        one_hot.scatter_add_(1, labels1, value1_added)
        one_hot.scatter_add_(1, labels2, value2_added)
        return one_hot.to(target.device)
                
    def forward(self, x, target, x_sub):
        # check the dimension
        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                             .format(x.size(0), target.size(0)))
        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                             .format(x.size(0)))
        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                             .format(x.size()))

        corrected_target = self._sub2main_corrected_label(target, x.size(1), x, x_sub)
        x = self.log_softmax(x)
        loss = torch.sum(-x * corrected_target, dim=1)

        # reduction check 
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')

