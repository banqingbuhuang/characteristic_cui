# coding=utf-8
import numpy
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import pandas as pd
import os
from util.grab import grab
# import model.Predict_imu as nnmodel
import model_others.GCN_DCT as nnmodel
from util.opt import Options
from util import utils_utils as utils
from util import loss_func


def main(opt):
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n
    print(" torch.cuda.is_available()", torch.cuda.is_available())
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    script_name = os.path.basename(__file__).split('.')[0]

    # new_3:将up变成LSTM
    script_name += "eval_gcn_dct_in{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.all_n)
    # 返回main_in10_out25_dctn35
    print(">>> creating model")

    model = nnmodel.GCN(input_feature=all_n, hidden_feature=256, p_dropout=0.5,
                        num_stage=12, node_n=54 * 3)

    if is_cuda:
        model = model.to(device)

    model_path_len = "/home/xt/EID/checkpoint/test_GCN/ckpt_main_GCN_DCTgcn_dct_n30_out30_dctn60_best.pth.tar"
    print(">>> loading ckpt len from '{}'".format(model_path_len))

    if is_cuda:
        ckpt = torch.load(model_path_len)
    else:
        ckpt = torch.load(model_path_len, map_location='cpu')

    start_epoch = ckpt['epoch']
    print(">>>  start_epoch", start_epoch)
    model.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} )".format(start_epoch))
    # data loading
    print(">>> loading data")
    test_dataset = grab(path_to_data=opt.remote_grab_dir, input_n=input_n, output_n=output_n,
                        split=2)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,  # 128
        shuffle=False,
        num_workers=2,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=True)
    ret_log = np.array([1])
    head = np.array(['epoch'])
    mean_err, body_err, lhand_err, rhand_err = eval(train_loader=test_loader, model=model, device=device)
    mean, body_ = 'mean_', 'body_'
    lhand_, rhand_ = 'lhand_', 'rhand_'
    ret_log = np.append(ret_log, [mean_err, body_err, lhand_err, rhand_err])
    head = np.append(head,
                     [mean + '3', mean + '6', mean + '9',
                      mean + '12', mean + '15',
                      mean + '18', mean + '21',
                      mean + '24', mean + '27', mean + '30'
                      ])
    head = np.append(head,
                     [body_ + '3', body_ + '6', body_ + '9',
                      body_ + '12', body_ + '15',
                      body_ + '18', body_ + '21',
                      body_ + '24', body_ + '27', body_ + '30'
                      ])
    head = np.append(head,
                     [lhand_ + '3', lhand_ + '6', lhand_ + '9',
                      lhand_ + '12', lhand_ + '15',
                      lhand_ + '18', lhand_ + '21',
                      lhand_ + '24', lhand_ + '27', lhand_ + '30'
                      ])
    head = np.append(head,
                     [rhand_ + '3', rhand_ + '6', rhand_ + '9',
                      rhand_ + '12', rhand_ + '15',
                      rhand_ + '18', rhand_ + '21',
                      rhand_ + '24', rhand_ + '27', rhand_ + '30'
                      ])
    df = pd.DataFrame(np.expand_dims(ret_log, axis=0))  # DataFrame是Python中Pandas库中的一种数据结构，它类似excel，是一种二维表。

    df.to_csv(opt.ckpt + '/' + script_name + '.csv', header=head, index=False)


def eval(train_loader, model, device):
    print("进入test")
    N = 0
    #
    eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
    model.eval()
    t_posi = np.zeros(len(eval_frame))  # 6位
    body_posi = np.zeros(len(eval_frame))  # 6位
    lhand_posi = np.zeros(len(eval_frame))  # 6位
    rhand_posi = np.zeros(len(eval_frame))  # 6位
    with torch.no_grad():
        for i, (input_pose, target_pose) in enumerate(train_loader):
            # print("input_pose", input_pose.shape)
            n = input_pose.shape[0]
            input_pose=get_dct(input_pose)
            if torch.cuda.is_available():
                input_pose = input_pose.to(device).float()
                target_pose = target_pose.to(device).float()
            out_pose = model(input_pose)
            pred_3d, targ_3d = get_idct(y_out=out_pose, out_joints=target_pose, device=device)
            for k in np.arange(0, len(eval_frame)):  # 6
                j = eval_frame[k]+30

                test_out, test_joints = pred_3d[:, j, :, :], targ_3d[:, j, :, :]
                t_posi[k] += loss_func.joint_loss(test_out, test_joints).cpu().data.numpy() * n * 100
                body_posi[k] += loss_func.joint_loss(test_out[:, :24], test_joints[:, :24]).cpu().data.numpy() * n * 100
                lhand_posi[k] += loss_func.joint_loss(test_out[:, 24:39],
                                                      test_joints[:, 24:39]).cpu().data.numpy() * n * 100
                rhand_posi[k] += loss_func.joint_loss(test_out[:, 39:],
                                                      test_joints[:, 39:]).cpu().data.numpy() * n * 100
            N += n

    return t_posi / N, body_posi / N, lhand_posi / N, rhand_posi / N

# 一维DCT变换
def get_dct_matrix(N):
    dct_m = np.eye(N)  # 返回one-hot数组
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)  # 2/35开更
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)  # 矩阵求逆
    return dct_m, idct_m

def get_dct(out_joints):
    batch, frame, node, dim = out_joints.data.shape
    dct_m_in, _ = get_dct_matrix(frame)
    input_joints = out_joints.transpose(0, 1).reshape(frame, -1).contiguous()
    input_dct_seq = np.matmul((dct_m_in[0:frame, :]), input_joints)
    input_dct_seq = torch.as_tensor(input_dct_seq)
    input_joints = input_dct_seq.reshape(frame, batch, -1).permute(1, 2, 0).contiguous()
    return input_joints


def get_idct(y_out, out_joints, device):
    batch, frame, node, dim = out_joints.data.shape
    _, idct_m = get_dct_matrix(frame)
    idct_m = torch.from_numpy(idct_m).float().to(device)
    outputs_t = y_out.view(-1, frame).transpose(1, 0)
    # 50,32*54*3
    outputs_p3d = torch.matmul(idct_m[:, 0:frame], outputs_t)
    outputs_p3d = outputs_p3d.reshape(frame, batch, -1, dim).contiguous().transpose(0, 1)
    # 32,162,50
    pred_3d = outputs_p3d
    targ_3d = out_joints
    return pred_3d, targ_3d


if __name__ == "__main__":
    option = Options().parse()
    main(option)
