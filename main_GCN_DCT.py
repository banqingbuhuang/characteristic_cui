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
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr
    ckpt = opt.ckpt + '_GCN'
    device_ids = [0, 1, 2]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    is_cuda = torch.cuda.is_available()
    print(">>>is_cuda ", device)
    print(">>>lr_now ", lr_now)
    script_name = os.path.basename(__file__).split('.')[0]
    # new_2:测试total_capture
    script_name += "gcn_dct_n{:d}_out{:d}_dctn{:d}".format(input_n, output_n, all_n)
    # 返回main_in10_out25_dctn35
    print(">>> creating model")
    # 将adjs放入cuda
    # in_features, adjs, node_n, dim, p_dropout,
    # 维度为dim dim=3表示三维位置，dim=9表示rotation matrix
    model = nnmodel.GCN(input_feature=all_n, hidden_feature=256, p_dropout=0.5,
                        num_stage=12, node_n=54 * 3)

    if is_cuda:
        model = model.to(device)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # data loading
    print(">>>  err_best", err_best)
    print(">>> loading data")
    train_dataset = grab(path_to_data=opt.remote_grab_dir, input_n=input_n, output_n=output_n,
                         split=0)
    val_dataset = grab(path_to_data=opt.remote_grab_dir, input_n=input_n, output_n=output_n,
                       split=1)
    test_dataset = grab(path_to_data=opt.remote_grab_dir, input_n=input_n, output_n=output_n,
                        split=2)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=16,  # batch 32
        shuffle=True,  # 在每个epoch开始的时候，对数据进行重新排序
        num_workers=2,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        # 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
        pin_memory=True)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=128,  # 128
        shuffle=False,
        num_workers=1,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=True)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,  # 128
        shuffle=False,
        num_workers=2,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=True)
    print(">>> train data {}".format(train_dataset.__len__()))  # 32178
    print(">>> validation data {}".format(val_dataset.__len__()))  # 1271
    print(">>> test data {}".format(test_dataset.__len__()))
    for epoch in range(start_epoch, opt.epochs):
        if (epoch + 1) % opt.lr_decay == 0:  # lr_decay=2学习率延迟
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)  # lr_gamma学习率更新倍数0.96
        print('=====================================')
        print('>>> epoch: {} | lr: {:.6f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch
        Ir_now, t_l, = train(train_loader, model, optimizer,
                             device=device,
                             lr_now=lr_now, max_norm=opt.max_norm)
        print("train_loss:", t_l)
        ret_log = np.append(ret_log, [lr_now, t_l])
        head = np.append(head, ['lr', 't_l'])
        v_loss = val(val_loader, model, device=device)
        print("val_loss:", v_loss)
        ret_log = np.append(ret_log, [v_loss])
        head = np.append(head, ['v_loss'])

        test_loss = test(test_loader, model=model, device=device)
        position = 'po_'
        # eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 49]
        ret_log = np.append(ret_log, test_loss)
        head = np.append(head,
                         [position + '3', position + '6', position + '9',
                          position + '12', position + '15',
                          position + '18', position + '21',
                          position + '24', position + '27', position + '30'
                          ])
        if not np.isnan(v_loss):  # 判断空值 只有数组数值运算时可使用如果v_e不是空值
            is_best = v_loss < err_best  # err_best=10000
            err_best = min(v_loss, err_best)
        else:
            is_best = False
        ret_log = np.append(ret_log, is_best)  # 内容
        head = np.append(head, ['is_best'])  # 表头
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))  # DataFrame是Python中Pandas库中的一种数据结构，它类似excel，是一种二维表。
        if not os.path.exists(ckpt):
            os.makedirs(ckpt)
        if epoch == start_epoch:
            df.to_csv(ckpt + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(ckpt + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        if is_best:
            file_name = ['ckpt_' + str(script_name) + '_best.pth.tar', 'ckpt_']
            utils.save_ckpt({'epoch': epoch + 1,
                             'lr': lr_now,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()},
                            ckpt_path=ckpt,
                            file_name=file_name)


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


def train(train_loader, model, optimizer, device, lr_now, max_norm):
    print("进入train")
    # 初始化
    t_l = utils.AccumLoss()
    model.train()
    for i, (input_pose, target_pose) in enumerate(train_loader):
        model_input = get_dct(input_pose)
        n = input_pose.shape[0]  # 16
        if torch.cuda.is_available():
            model_input = model_input.to(device).float()
            target_pose = target_pose.to(device).float()
        out_pose = model(model_input)
        pred_3d, targ_3d = get_idct(y_out=out_pose, out_joints=target_pose, device=device)

        loss = loss_func.joint_loss(pred_3d, targ_3d)
        optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0.
        loss.backward()
        if max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()  # 则可以用所有Variable的grad成员和lr的数值自动更新Variable的数值
        t_l.update(loss.cpu().data.numpy() * n, n)
    return lr_now, t_l.avg


def val(train_loader, model, device):
    print("进入train")
    # 初始化
    t_l = utils.AccumLoss()
    model.eval()
    for i, (input_pose, target_pose) in enumerate(train_loader):
        model_input = get_dct(input_pose)
        n = input_pose.shape[0]  # 16
        if torch.cuda.is_available():
            model_input = model_input.to(device).float()
            target_pose = target_pose.to(device).float()
        out_pose = model(model_input)
        pred_3d, targ_3d = get_idct(y_out=out_pose, out_joints=target_pose, device=device)

        loss = loss_func.joint_loss(pred_3d, targ_3d)
        t_l.update(loss.cpu().data.numpy() * n, n)
    return t_l.avg


def test(train_loader, model, device):
    print("进入test")
    N = 0
    #
    eval_frame = [32, 35, 38, 41, 44, 47, 50, 53, 56, 59]
    model.eval()
    t_posi = np.zeros(len(eval_frame))  # 6位
    with torch.no_grad():
        for i, (input_pose, target_pose) in enumerate(train_loader):
            model_input = get_dct(input_pose)
            n = input_pose.shape[0]
            if torch.cuda.is_available():
                model_input = model_input.to(device).float()
                target_pose = target_pose.to(device).float()
            out_pose = model(model_input)
            pred_3d, targ_3d = get_idct(y_out=out_pose, out_joints=target_pose, device=device)
            for k in np.arange(0, len(eval_frame)):  # 6
                j = eval_frame[k]

                test_out, test_joints = pred_3d[:, j, :, :], targ_3d[:, j, :, :]
                t_posi[k] += loss_func.joint_loss(test_out, test_joints).cpu().data.numpy() * n * 100
            N += n

    return t_posi / N


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
            n = input_pose.shape[0]
            if torch.cuda.is_available():
                input_pose = input_pose.to(device).float()
                target_pose = target_pose.to(device).float()
            out_pose = model(input_pose)
            for k in np.arange(0, len(eval_frame)):  # 6
                j = eval_frame[k]

                test_out, test_joints = out_pose[:, j, :, :], target_pose[:, j, :, :]
                t_posi[k] += loss_func.joint_loss(test_out, test_joints).cpu().data.numpy() * n * 100
                body_posi[k] += loss_func.joint_loss(test_out[:, :24], test_joints[:, :24]).cpu().data.numpy() * n * 100
                lhand_posi[k] += loss_func.joint_loss(test_out[:, 24:39],
                                                      test_joints[:, 24:39]).cpu().data.numpy() * n * 100
                rhand_posi[k] += loss_func.joint_loss(test_out[:, 39:],
                                                      test_joints[:, 39:]).cpu().data.numpy() * n * 100
            N += n

    return t_posi / N, body_posi / N, lhand_posi / N, rhand_posi / N


if __name__ == "__main__":
    option = Options().parse()
    main(option)
    # test()
# 查看网络的每一个参数
# print("=============更新之后===========")
# for name, parms in model.named_parameters():
#     print('-->name:', name)
#     print('-->para:', parms)
#     print('-->grad_requirs:', parms.requires_grad)
#     print('-->grad_value:', parms.grad)
#     print("===")
# print(optimizer)
# print("max_norm", max_norm)
