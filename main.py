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

from util.opt import Options
from util.grab_batch import grab
from util import loss_func, utils
from model_att.model import Network_arch as ATTmodel


def main(opt):
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr
    job = 0
    ckpt = opt.ckpt
    script_name = os.path.basename(__file__).split('.')[0]
    # new_2:测试total_capture
    script_name += "_n{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.all_n)
    dropout = 0.1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    is_cuda = torch.cuda.is_available()
    # create model
    model = ATTmodel(input_f=input_n, output_f=output_n, model_dim=256, pos_encoding_params=(10, 500))
    if is_cuda:
        model = model.to(device)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    train_dataset = grab(path_to_data=opt.remote_grab_batch_dir, input_n=input_n, output_n=output_n,
                         split=0)
    val_dataset = grab(path_to_data=opt.remote_grab_batch_dir, input_n=input_n, output_n=output_n,
                       split=1)
    test_dataset = grab(path_to_data=opt.remote_grab_batch_dir, input_n=input_n, output_n=output_n,
                        split=2)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=16,  # batch 32
        shuffle=True,  # 在每个epoch开始的时候，对数据进行重新排序
        num_workers=3,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        # 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
        pin_memory=True)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=128,  # 128
        shuffle=False,
        num_workers=2,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
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
        test_loss = test(train_loader=test_loader, model=model, device=device)
        mean, body_ = 'mean_', 'body_'
        lhand_, rhand_ = 'lhand_', 'rhand_'
        ret_log = np.append(ret_log, test_loss)
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
        file_name = ['ckpt_' + str(script_name) + '_best.pth.tar', 'ckpt_']
        utils.save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         # 'err': test_e[0],
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        ckpt_path=ckpt,
                        file_name=file_name)


def train(train_loader, model, optimizer, device, lr_now, max_norm):
    print("进入train")
    # 初始化
    t_l = utils.AccumLoss()
    model.train()
    for i, (input_pose, target_pose) in enumerate(train_loader):

        n = input_pose.shape[0]  # 16
        if torch.cuda.is_available():
            input_pose = input_pose.to(device).float()
            target_pose = target_pose.to(device).float()
        out_pose = model(input_pose)
        loss = loss_func.joint_diff_loss(out_pose, target_pose)
        optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0.
        loss.backward()
        if max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()  # 则可以用所有Variable的grad成员和lr的数值自动更新Variable的数值
        t_l.update(loss.cpu().data.numpy() * n, n)
    return lr_now, t_l.avg


def val(train_loader, model, device):
    print("进入val")
    # 初始化
    t_l = utils.AccumLoss()
    model.eval()
    for i, (input_pose, target_pose) in enumerate(train_loader):
        n = input_pose.shape[0]  # 16
        if torch.cuda.is_available():
            input_pose = input_pose.to(device).float()
            target_pose = target_pose.to(device).float()
        out_pose = model(input_pose)
        loss = loss_func.joint_diff_loss(out_pose, target_pose)
        t_l.update(loss.cpu().data.numpy() * n, n)
    return t_l.avg


def test(train_loader, model, device):
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


if __name__ == '__main__':
    option = Options().parse()
    main(option)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
