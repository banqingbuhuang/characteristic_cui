from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import print_function, absolute_import, division
import json
import glob
import os
import smplx
import numpy as np
from tqdm import tqdm
import pickle as pkl
from util.utils import params2torch
from util.utils import parse_npz
from util.utils import to_cpu

# 目标帧率30
TARGET_FPS = 30


def data_batch(split=0):
    grab_path = "/data/xt/dataset/GRAB"
    out_path = "/data/xt/dataset/GRAB_batch"
    tra_val_test = ['train', 'val', 'test']

    subset_split = tra_val_test[split]
    print(subset_split)
    data_path = os.path.join(grab_path, subset_split)
    out_path = os.path.join(out_path, subset_split)
    subset_dir = [x for x in os.listdir(data_path)
                  if os.path.isdir(data_path + '/' + x)]
    input_n = 30
    output_n = 30
    print(subset_dir)
    # 之后改为for循环
    for subset in subset_dir:
        # subset = subset_dir[0]
        # 数据输入
        # if subset != subset_dir[0]:
        #     continue
        print('-- processing subset {:s}'.format(subset))
        seqs = glob.glob(os.path.join(data_path, subset, '*.pkl'))
        # print(len(seqs))
        for seq in tqdm(seqs):
            idx = 0
            f = open(seq, 'rb')
            data = pkl.load(f, encoding='latin1')
            seq = seq.replace('\\', '/')
            script_name = seq.split('/')[-1].split('.')[0]
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
            for a in range(seq_sel.shape[0]):
                data_out = {'joints': seq_sel[a]}
                # 这里创建数据输出文件夹
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                out_path_end = os.path.join(out_path, subset + script_name + str(idx) + '.pkl')
                idx += 1
                print(out_path_end)
                with open(out_path_end, 'wb') as fout:
                    pkl.dump(data_out, fout)


def main(split=0):
    model_path = "C://Gtrans/body_models/vPOSE/models"
    grab_path = "C:/Gtrans/dataset/GRAB_unzip/grab"
    out_path = "C:/Gtrans/dataset/GRAB"
    tra_val_test = ['train', 'val', 'test']

    subset_split = tra_val_test[split]
    print(subset_split)
    data_path = os.path.join(grab_path, subset_split)
    out_path = os.path.join(out_path, subset_split)
    print(out_path)
    subset_dir = [x for x in os.listdir(data_path)
                  if os.path.isdir(data_path + '/' + x)]

    print(subset_dir)
    # 之后改为for循环
    for subset in subset_dir:
        # subset = subset_dir[0]
        # 数据输入
        print('-- processing subset {:s}'.format(subset))
        seqs = glob.glob(os.path.join(data_path, subset, '*.npz'))
        # print(len(seqs))
        for seq in tqdm(seqs):
            seq = seq.replace('\\', '/')
            script_name = seq.split('/')[-1].split('.')[0]
            bdata = parse_npz(seq)
            sbj_params = params2torch(bdata.body.params)
            fullpose = sbj_params['fullpose'].numpy()
            framerate = bdata['framerate']
            gender = bdata['gender']
            n_comps = bdata.n_comps
            skip = 1

            sbj_m = smplx.create(model_path=model_path,
                                 model_type='smplx',
                                 gender=gender,
                                 # use_pca=False,
                                 num_pca_comps=n_comps,
                                 # v_template=sbj_vtemp,
                                 batch_size=fullpose.shape[0])
            # sbj_parms = params2torch(sbj_params)
            verts_sbj = to_cpu(sbj_m(**sbj_params).vertices)
            # body_data['verts'].append(verts_sbj)
            print(verts_sbj.shape)
            joints_sbj = to_cpu(sbj_m(**sbj_params).joints)[:, :54]
            if framerate % TARGET_FPS == 0:
                skip = int(framerate / TARGET_FPS)
            fullpose = fullpose[::skip]
            joints_sbj = joints_sbj[::skip]
            # num_frames = len(fullpose)
            # seq_len = 60
            # fs = np.arange(0, num_frames - seq_len + 1)  # 窗口滑动
            # fs_sel = fs
            # for i in np.arange(seq_len - 1):
            #     fs_sel = np.vstack((fs_sel, fs + i + 1))  # 沿着竖直方向将矩阵堆叠起来将fs_sel和fs+i+1放到一起
            # fs_sel = fs_sel.transpose()  # eg.((35, 1704)变成(1704, 35))
            # full_pose_sam = fullpose[fs_sel, :].numpy()
            # joints_sam=joints_sbj[fs_sel,:].numpy()
            # for a in range(full_pose_sam.shape[0]):
            data_out = {'gender': gender, 'fullpose': fullpose, 'joints': joints_sbj}
            # 这里创建数据输出文件夹
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out_path_end = os.path.join(out_path, subset + script_name + '.pkl')
            print(out_path_end)
            with open(out_path_end, 'wb') as fout:
                pkl.dump(data_out, fout)


if __name__ == '__main__':
    # main(split=2)

    data_batch(1)
    # pre_cmu("C:/Gtrans/dataset/CMU Panoptic/")
