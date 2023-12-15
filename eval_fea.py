import argparse
import os
import time
import json
import numpy as np
import yaml

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


from fea_util.util import accuracy, merge_config, add_config
from fea_util.logger import setup_logger
from layer.LSR import *
from dataset.video_dataset import VideoFeaValDataset


import model as model_factory
from layer.pooling_factory import get_pooling_by_name


def parse_option():
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--config_file', type=str, required=True, help='path of config file (yaml)')
    parser.add_argument('--local_rank', type=int, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    # load config file, default + base + exp
    config_default = yaml.load(open('./config/default.yml', 'r'),Loader=yaml.FullLoader)
    config_exp = yaml.load(open(args.config_file, 'r'),Loader=yaml.FullLoader)
    if 'base' in config_exp:
        config_base = yaml.load(open(config_exp['base'], 'r'),Loader=yaml.FullLoader)
    else:
        config_base = None
    config = merge_config(merge_config(config_default, config_base), config_exp)
    args.C = config
    add_config(args, 'root', config)
    return args


def get_val_loader(args):
    test_dataset = VideoFeaValDataset(list_file=args.eva_list_file, root_path=args.eva_root_path, 
                                        transform=None, clip_length=None, 
                                        num_steps=None, num_segments=None, 
                                        format=args.format,
                                        use_mean_data=args.use_mean_data,
                                        clip_stride=args.clip_stride)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    return test_loader    


def build_model(args):
    model = model_factory.get_model_by_name(net_name=args.net_name, pooling_arch=get_pooling_by_name(args.pooling_name),
                              num_classes=args.num_classes, dropout_ratio=args.dropout_ratio, 
                              clip_length=None).cuda()
    if args.eval_model:
        load_pretrained(args, model)
    return model


def load_pretrained(args, model):
    ckpt = torch.load(args.eval_model, map_location='cpu')
    if 'model' in ckpt:
        state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
    else:
        state_dict = ckpt

    [misskeys, unexpkeys] = model.load_state_dict(state_dict, strict=False)
    logger.info('Missing keys: {}'.format(misskeys))
    logger.info('Unexpect keys: {}'.format(unexpkeys))
    logger.info("==> loaded checkpoint '{}'".format(args.eval_model))


def main(args):
    val_loader = get_val_loader(args)
    model = build_model(args)
    model.eval()

    # print network architecture
    if global_rank == 0 or dist.get_rank() == 0:
        logger.info(model)

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=True,
                                        find_unused_parameters=True)

    # evaluation on test data
    tic_val = time.time()
    eva_accuracy = eval(0, val_loader, model, args)
    t1 = eva_accuracy[0].cuda().data.cpu().item()
    t3 = eva_accuracy[1].cuda().data.cpu().item()
    t5 = eva_accuracy[2].cuda().data.cpu().item()               
    logger.info('Val top1 accuracy {:.4f}, top3 accuracy: {:.4f}, top5: {:.4f} val time {:.2f}'.format(t1, t3, t5, time.time() - tic_val))


def eval(epoch, val_loader, model, args):
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    all_scores = np.zeros([len(val_loader) * args.batch_size, args.num_classes], dtype=np.float)
    all_labels = np.zeros([len(val_loader) * args.batch_size], dtype=np.float)
    top_idx = 0
    with torch.no_grad():
        logger.info('==> Begin Validating, clip stride: {}, val video clip num {}'.format(args.clip_stride, len(val_loader.dataset)))
        for idx, (x, label) in enumerate(val_loader):
            if idx > 0 and idx % args.eval_freq == 0:
                temp_acc = accuracy(all_scores[:top_idx,:], all_labels[:top_idx], topk=(1, 3, 5))
                top1_accuracy = temp_acc[0].cuda().data.cpu().item()
                top3_accuracy = temp_acc[1].cuda().data.cpu().item()
                top5_accuracy = temp_acc[2].cuda().data.cpu().item()                
                logger.info('{}/{}, top1: {:.4f}, top3: {:.4f}, top5: {:.4f}'.format(idx, len(val_loader), top1_accuracy, top3_accuracy, top5_accuracy))
            x = x.cuda(non_blocking=True)
            x = x.to(torch.float32)
            bsz = x.size(0)
            score = model(x)
            if isinstance(score, list):
                score_numpy = (softmax(score[0]).data.cpu().numpy() + softmax(score[1]).data.cpu().numpy()) / 2
            else:
                score_numpy = softmax(score).data.cpu().numpy()
            label_numpy = label.data.cpu().numpy()
            all_scores[top_idx: top_idx + bsz, :] = score_numpy
            all_labels[top_idx: top_idx + bsz] = label_numpy
            top_idx += bsz
    all_scores = all_scores[:top_idx, :]
    all_labels = all_labels[:top_idx]
    # compute the accuracy
    acc = accuracy(all_scores, all_labels, topk=(1, 3, 5))
    return acc


def set_evn(args):
    global_rank = 0
    args.distributed = torch.cuda.device_count() > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        global_rank = torch.distributed.get_rank()
    return global_rank


if __name__ == '__main__':
    opt = parse_option()
    global_rank = set_evn(opt)
    cudnn.benchmark = True

    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir, distributed_rank=global_rank, name="bcn-ac")
    if global_rank == 0 or dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "train_val_3d.config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))
    main(opt)
