import torch


def poses_loss(y_out, out_poses):
    y_out = y_out.reshape(-1, 3)
    out_poses = out_poses.reshape(-1, 3)
    loss = torch.mean(torch.norm(y_out - out_poses, 2, 1))
    return loss


def joint_loss(y_out, out_poses):
    y_out = y_out.reshape(-1, 3)
    out_poses = out_poses.reshape(-1, 3)
    loss = torch.mean(torch.norm(y_out - out_poses, 2, 1))
    return loss


def joint_diff_loss(y_out, out_poses):
    """

    :param y_out: batch,frames, 54,3
    :param out_poses:
    :return:
    """
    hbody = y_out[:, :, :24]
    lhand = y_out[:, :, 24:39]
    rhand = y_out[:, :, 39:]
    hbody_t = out_poses[:, :, :24]
    lhand_t = out_poses[:, :, 24:39]
    rhand_t = out_poses[:, :, 39:]
    hand_length = torch.mean(torch.norm(out_poses[:, :, 20] - out_poses[:, :, 28], 2, 2))
    body_length = torch.mean(torch.norm(out_poses[:, :, 1] - out_poses[:, :, 4], 2, 2))
    lhand_loss = torch.mean(torch.norm((lhand - lhand_t) / hand_length, 2, 3))
    rhand_loss = torch.mean(torch.norm((rhand - rhand_t) / hand_length, 2, 3))
    body_loss = torch.mean(torch.norm((hbody - hbody_t) / body_length, 2, 3))
    # print("hand_length", hand_length, body_length, lhand_loss, rhand_loss, body_loss)
    return lhand_loss + rhand_loss + body_loss


if __name__ == "__main__":
    input = torch.randn(128, 30, 54, 3)

    out_pose = torch.randn(128, 30, 54, 3)
    joint_diff_loss(input, out_pose)
