from torch.utils.data import Dataset

import os
import pickle
import numpy as np
import torch
import glob
import os
import smplx
from tqdm import tqdm
import pickle


class grab(Dataset):
    def __init__(self, path_to_data, input_n, output_n, split=0):
        tra_val_test = ['train', 'val', 'test']
        subset_split = tra_val_test[split]
        self.data_path = os.path.join(path_to_data, subset_split)
        self.seq_len = input_n + output_n  # 采样的长度
        self.input_n = input_n
        self.output_n = output_n
        #
        # 数据的长度
        self.data_dir = os.listdir(self.data_path)
        self.lenth = len(self.data_dir)

        # 数据的长度

    # 得到长度
    def __len__(self):
        return self.lenth

    # 得到input ，target，all
    def __getitem__(self, item):
        # load data from file
        seq = os.path.join(self.data_path, self.data_dir[item])
        f = open(seq, 'rb')
        data = pickle.load(f, encoding='latin1')

        joints = data['joints']

        # 将数据拼接在一起
        # pad_idx = np.repeat([self.input_n - 1], self.output_n)
        # i_idx = np.append(np.arange(0, self.input_n), pad_idx)
        self.input_joints = joints[:self.input_n]
        self.target = joints[self.output_n:, ]
        return self.input_joints, self.target
