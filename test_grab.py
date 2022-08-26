import numpy as np
import os
import smplx
import torch.nn as nn
from util.meshviewer import Mesh, MeshViewer, colors, points2sphere
from util.utils import parse_npz
from util.utils import params2torch
from util.utils import to_cpu
from util.utils import euler
from torch.utils.data import DataLoader
from util.opt import Options
import torch.nn.functional as F
from util.grab import grab
from util import loss_func, utils


class model_l(nn.Module):
    def __init__(self, input_frame_len, output_frame_len, input_num, out_num, dropout):
        super().__init__()
        self.ll = nn.Linear(input_num
                            , out_num)
        self.dropout = dropout
        self.out_num = out_num

    def forward(self, x):
        y = self.ll(x)
        y = F.dropout(y, self.dropout, training=self.training)
        return y


def main(opt):
    model_path = "C://Gtrans/body_models/vPOSE/models"
    model = model_l(input_frame_len=opt.all_n, output_frame_len=opt.all_n, input_num=165, out_num=165,
                    dropout=opt.dropout)
    test_dataset = grab(path_to_data=opt.data_grab_dir, input_n=opt.input_n, output_n=opt.output_n,
                        split=2)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,  # 128
        shuffle=False,
        num_workers=0,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=False)
    model_path_len = './checkpoint/test/ckpt_main_n30_out30_dctn609_best.pth.tar'
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    for i, (input_pose, target_pose, gender) in enumerate(test_loader):
        out_pose = model(input_pose)
        sbj_fullpose = out_pose.reshape(-1, 165)
        tar_fullpose = target_pose.reshape(-1, 165)
        print("out_shape", sbj_fullpose.shape)
        print("tar_shape", tar_fullpose.shape)
        body_data_out = {
            'global_orient': sbj_fullpose[:, :3], 'body_pose': sbj_fullpose[:, 3:66],
            'right_hand_pose': sbj_fullpose[:, 120:165], 'left_hand_pose': sbj_fullpose[:, 75:120],
        }
        body_data_tar = {
            'global_orient': tar_fullpose[:, :3], 'body_pose': tar_fullpose[:, 3:66],
            'right_hand_pose': tar_fullpose[:, 120:165], 'left_hand_pose': tar_fullpose[:, 75:120],
        }
        print(gender[0])
        gender = gender[0]
        T = sbj_fullpose.shape[0]
        mv = MeshViewer(offscreen=False)
        # set the camera pose
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
        camera_pose[:3, 3] = np.array([-.5, -4., 1.5])
        mv.update_camera_pose(camera_pose)
        sbj_m = smplx.create(model_path=model_path,
                             model_type='smplx',
                             gender=gender,
                             # num_pca_comps=45,
                             use_pca=False,
                             # v_template=sbj_vtemp,
                             batch_size=T)

        verts_out = to_cpu(sbj_m(**body_data_out).vertices)
        verts_tar = to_cpu(sbj_m(**body_data_tar).vertices)
        skip_frame = 1
        print(T)
        for frame in range(0, T, skip_frame):
            s_mesh = Mesh(vertices=verts_out[frame], faces=sbj_m.faces, vc=colors['pink'], smooth=True)
            # s_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['body'][frame] > 0)
            s_mesh_1 = Mesh(vertices=verts_tar[frame] + 2, faces=sbj_m.faces, vc=colors['pink'], smooth=True)
            # s_mesh_1.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['body'][frame] > 0)
            # joints_mesh = points2sphere(joints_sbj[frame], radius=0.008, vc=colors['blue'])
            mv.set_static_meshes([s_mesh + s_mesh_1])
        mv.close_viewer()


if __name__ == '__main__':
    option = Options().parse()
    main(option)
