import torch.utils.data
import os
import random
import torch
import numpy as np
import lmdb
import io
from PIL import Image
import pandas as pd

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, list_file, transform, root_path, clip_length=1, num_steps=1, num_segments=1, num_channels=3,
                 format='LMDB'):
        super(VideoDataset, self).__init__()
        self.list_file = list_file
        self.transform = transform
        self.root_path = root_path
        self.clip_length = clip_length
        self.num_steps = num_steps
        self.num_segments = num_segments
        self.num_channels = num_channels
        self.format = format

        self.samples = self._load_list(list_file)

    def _load_list(self, list_root):
        with open(list_root, 'r') as f:
            samples = f.readlines()
        return samples

    def _parse_rgb_lmdb(self, video_path, offsets):
        """Return the clip buffer sample from video lmdb."""
        lmdb_env = lmdb.open(os.path.join(self.root_path, video_path), readonly=True)

        with lmdb_env.begin() as lmdb_txn:
            image_list = []
            for offset in offsets:
                for frame_id in range(offset + 1, offset + self.num_steps * self.clip_length + 1, self.num_steps):
                    bio = io.BytesIO(lmdb_txn.get('{:06d}.jpg'.format(frame_id).encode()))
                    image = Image.open(bio).convert('RGB')
                    image_list.append(image)
        lmdb_env.close()
        return image_list

    def _parse_rgb_frame(self, video_path, offsets):
        """Return the clip buffer sample from video frames."""
        image_list = []
        for offset in offsets:
            for frame_id in range(offset + 1, offset + self.num_steps * self.clip_length + 1, self.num_steps):
                image = Image.open(os.path.join(self.root_path, video_path, '{:05d}'.format(frame_id) + '.jpg')).convert('RGB')
                image_list.append(image)
        return image_list

    def _parse_csv_feature(self, video_path):
        feature = pd.read_csv(os.path.join(self.root_path, video_path), sep='\n', header=None).values
        return feature

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        raise NotImplementedError



class VideoFeaTrainDataset(torch.utils.data.Dataset):
    def __init__(self, list_file, transform, root_path, clip_length=1, num_steps=1, num_segments=1, num_channels=3,
                 format='LMDB', use_mean_data=False):
        super(VideoFeaTrainDataset, self).__init__()
        self.list_file = list_file
        self.transform = transform
        self.root_path = root_path
        self.clip_length = clip_length
        self.num_steps = num_steps
        self.num_segments = num_segments
        self.num_channels = num_channels
        self.format = format
        self.use_mean_data = use_mean_data
        self.samples, self.sample_labels = self._load_csv(list_file)

    def _load_csv(self, list_file):
        df_values = pd.read_csv(list_file)
        samples = df_values['video_name'].values
        sample_labels = df_values['class'].values
        return samples, sample_labels

    def __len__(self):
        return len(self.samples)

    def _parse_npy_feature(self, video_npy_path):
        feature = np.load(video_npy_path)
        if self.use_mean_data: feature = np.mean(feature, axis=0)
        return feature

    def __getitem__(self, item):
        vname = self.samples[item]
        label = self.sample_labels[item]
        video_npy_path = os.path.join(self.root_path, '{}.npy'.format(vname))
        feature = self._parse_npy_feature(video_npy_path)
        return feature, label
    

class VideoFeaValDataset(torch.utils.data.Dataset):
    def __init__(self, list_file, transform, root_path, clip_length=1, num_steps=1, num_segments=1, num_channels=3,
                 format='LMDB', use_mean_data=False, clip_stride=-1):
        self.list_file = list_file
        self.transform = transform
        self.root_path = root_path
        self.clip_length = clip_length
        self.num_steps = num_steps
        self.num_segments = num_segments
        self.num_channels = num_channels
        self.format = format
        self.use_mean_data = use_mean_data
        self.clip_stride = clip_stride
        self.samples, self.sample_labels, self.clip_num = self._load_csv(list_file)
        if self.clip_stride > -1:
            self.vid_idx, self.fea_idx = 0, 0
            self.keep_fea = np.load(os.path.join(self.root_path, '{}.npy'.format(self.samples[0])))

    def _load_csv(self, list_file):
        df_values = pd.read_csv(list_file)
        samples = df_values['video_name'].values
        sample_labels = df_values['class'].values
        clip_num = df_values['clip_num'].values
        return samples, sample_labels, clip_num
    
    def __len__(self):
        if self.clip_stride > -1:
            return sum(self.clip_num) // self.clip_stride
        return len(self.samples)
    
    def _parse_npy_feature(self, video_npy_path):
        feature = np.load(video_npy_path)
        if self.clip_stride == -1: feature = np.mean(feature, axis=0)
        return feature

    def __getitem__(self, item):
        if self.clip_stride == -1:
            vname = self.samples[item]
            label = self.sample_labels[item]
            video_npy_path = os.path.join(self.root_path, '{}.npy'.format(vname))
            feature = self._parse_npy_feature(video_npy_path)
        else:
            if item == 0: self.keep_fea = np.load(os.path.join(self.root_path, '{}.npy'.format(self.samples[0])))
            label = self.sample_labels[self.vid_idx]
            seed = random.randint(0, 2)
            fea_len = self.keep_fea.shape[0]
            if seed == 0: before_mean_fea = self.keep_fea[:fea_len//10*9,:]
            elif seed == 1: before_mean_fea = self.keep_fea[fea_len//10:,:]
            else: before_mean_fea = self.keep_fea
            if before_mean_fea.shape[0] == 0: before_mean_fea = self.keep_fea
            feature = np.mean(before_mean_fea, axis=0)
            if feature.shape[0] != 4096: feature = np.mean(self.keep_fea, axis=0)
            if self.fea_idx + self.clip_stride >= self.clip_num[self.vid_idx]:
                self.vid_idx += 1
                self.keep_fea = np.load(os.path.join(self.root_path, '{}.npy'.format(self.samples[self.vid_idx])))
                self.fea_idx = -self.clip_stride
            self.fea_idx += self.clip_stride
        return feature, label