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

        # load data from file
        subset_dir = [x for x in os.listdir(self.data_path)
                      if os.path.isdir(self.data_path + '/' + x)]
        sampled_seq = []
        print(subset_dir)
        # 之后改为for循环
        for subset in subset_dir:
            # subset = subset_dir[0]
            # 数据输入
            # if subset != subset_dir[0]:
            #     continue
            print('-- processing subset {:s}'.format(subset))
            seqs = glob.glob(os.path.join(self.data_path, subset, '*.pkl'))
            # print(len(seqs))
            for seq in tqdm(seqs):
                f = open(seq, 'rb')
                data = pickle.load(f, encoding='latin1')

                joints = data['joints']
                num_frames = len(joints)

                seq_len = input_n + output_n
                # 堆叠data 产生窗口
                fs = np.arange(0, num_frames - seq_len + 1)
                fs_sel = fs
                for i in np.arange(seq_len - 1):
                    fs_sel = np.vstack((fs_sel, fs + i + 1))
                fs_sel = fs_sel.transpose()
                seq_sel = joints[fs_sel, :]
                # 将数据拼接在一起
                if len(sampled_seq) == 0:
                    sampled_seq = seq_sel
                else:
                    sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)
        self.input_joints = sampled_seq[:, i_idx]
        print("input", self.input_joints.shape)
        self.target = sampled_seq
        print("target", self.target.shape)
        # 数据的长度

    # 得到长度
    def __len__(self):
        return np.shape(self.input_joints)[0]

    # 得到input ，target，all
    def __getitem__(self, item):
        return self.input_joints[item], self.target[item]
