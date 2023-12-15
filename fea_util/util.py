import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
from PIL import ImageFilter


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if not isinstance(output, torch.Tensor):
        output = torch.from_numpy(output)
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target)

    num_classes = output.size(1)
    topk_refine = [min(k, num_classes - 1) for k in topk]

    maxk = max(topk_refine)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk_refine:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1. / batch_size))
    return res


def accuracy_each_cate(output, target):
    """Computes the precision@k for each category"""
    if not isinstance(output, torch.Tensor):
        output = torch.from_numpy(output)
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target)

    num_classes = output.size(1)
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    cate_acc = []
    for cate_id in range(num_classes):
        mask = (target == cate_id)
        correct_k = (correct * mask).reshape(-1).float().sum(0)
        cate_acc.append(correct_k.mul_(1. / mask.sum()))
    return cate_acc


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0)


class DistributedShuffle:
    @staticmethod
    def forward_shuffle(x, epoch):
        """ forward shuffle, return shuffled batch of x from all processes.
        epoch is used as manual seed to make sure the shuffle id in all process is same.
        """
        x_all = dist_collect(x)
        forward_inds, backward_inds = DistributedShuffle.get_shuffle_ids(x_all.shape[0], epoch)

        forward_inds_local = DistributedShuffle.get_local_id(forward_inds)

        return x_all[forward_inds_local], backward_inds

    @staticmethod
    def backward_shuffle(x, backward_inds, return_local=True):
        """ backward shuffle, return data which have been shuffled back
        x is the shared data, should be local data
        if return_local, only return the local batch data of x.
            otherwise, return collected all data on all process.
        """
        x_all = dist_collect(x)
        if return_local:
            backward_inds_local = DistributedShuffle.get_local_id(backward_inds)
            return x_all[backward_inds], x_all[backward_inds_local]
        else:
            return x_all[backward_inds]

    @staticmethod
    def get_local_id(ids):
        return ids.chunk(dist.get_world_size())[dist.get_rank()]

    @staticmethod
    def get_shuffle_ids(bsz, epoch):
        """generate shuffle ids for ShuffleBN"""
        torch.manual_seed(epoch)
        # global forward shuffle id  for all process
        forward_inds = torch.randperm(bsz).long().cuda()

        # global backward shuffle id
        backward_inds = torch.zeros(forward_inds.shape[0]).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)

        return forward_inds, backward_inds


def set_bn_train(model):
    def set_bn_train_helper(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    model.eval()
    model.apply(set_bn_train_helper)


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def compute_top1_acc(all_scores, all_labels, args):
    top1_hit = 0
    top1_idx = np.argmax(all_scores, axis=1)
    for i in range(top1_idx.shape[0]):
        pred = int(top1_idx[i])
        label = int(all_labels[i])
        if pred == label: top1_hit += 1
    return top1_hit*1.0 / top1_idx.shape[0]


# config
def add_config(args, name, config):
    if isinstance(config, dict):
        for key in config.keys():
            add_config(args, key, config[key])
    else:
        setattr(args, name, config)


def merge_config(conf1, conf2):
    if isinstance(conf1, dict) and isinstance(conf2, dict):
        new_config = {}
        key_list = list(set(conf1.keys()).union(set(conf2.keys())))

        for key in key_list:
            if (key in conf1) and (key in conf2): # union of c1 & c2
                new_config[key] = merge_config(conf1.get(key), conf2.get(key))
            else:
                new_config[key] = conf1.get(key) if key in conf1 else conf2.get(key)
        return new_config
    else:
        return conf1 if conf2 is None else conf2


# frozen bn
def frozen_bn(model):
    first_bn = True
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            if first_bn:
                first_bn = False
                print('Skip frozen first bn layer: ' + name)
                continue
            m.eval()
            try:
                m.weight.requires_grad = False
                m.bias.requires_grad = False
            except: continue
