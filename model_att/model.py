#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import  # 绝对导入，导入的包选择为系统的包，而不是自己定义的
from __future__ import print_function  # 解决python版本的问题

'''
如果某个版本中出现了某个新的功能特性，而且这个特性和当前版本中使用的不兼容，
也就是它在该版本中不是语言标准，那么我如果想要使用的话就需要从future模块导入。
加上这些，如果你的python版本是python2.X，你也得按照python3.X那样使用这些函数。
'''
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import util.positionEncoding as PositionEncodings
import util.utils as utils


class Inner_att(nn.Module):
    def __init__(self,
                 d_model=256,
                 head_num=4,
                 dim_ffn=2048,
                 dropout=0.2,
                 init_fn=utils.normal_init_):
        super(Inner_att, self).__init__()
        self._model_dim = d_model
        self._dim_ffn = dim_ffn
        self._relu = nn.ReLU()
        self._dropout_layer = nn.Dropout(dropout)
        self.inner_att = nn.MultiheadAttention(d_model, head_num, dropout=dropout)
        self._linear1 = nn.Linear(d_model, self._dim_ffn)
        self._linear2 = nn.Linear(self._dim_ffn, self._model_dim)
        self._norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self._norm2 = nn.LayerNorm(d_model, eps=1e-5)

        utils.weight_init(self._linear1, init_fn_=init_fn)
        utils.weight_init(self._linear2, init_fn_=init_fn)

    def forward(self, source_seq, pos_encoding):
        query = source_seq + pos_encoding
        key = query
        value = source_seq
        attn_output, attn_weights = self.inner_att(
            query,
            key,
            value,
            need_weights=True
        )
        norm_attn_ = self._dropout_layer(attn_output) + source_seq
        norm_attn = self._norm2(norm_attn_)

        output = self._linear1(norm_attn)
        output = self._relu(output)
        output = self._dropout_layer(output)
        output = self._linear2(output)
        output = self._dropout_layer(output) + norm_attn_
        return output


class Interaction_att(nn.Module):
    def __init__(self,
                 d_model=256,
                 head_num=4,
                 dim_ffn=2048,
                 dropout=0.2,
                 init_fn=utils.normal_init_):
        super(Interaction_att, self).__init__()
        self._model_dim = d_model
        self._dim_ffn = dim_ffn
        self._relu = nn.ReLU()
        self._dropout_layer = nn.Dropout(dropout)
        self.inner_att = nn.MultiheadAttention(d_model, head_num, dropout=dropout)
        self._linear1 = nn.Linear(d_model, self._dim_ffn)
        self._linear2 = nn.Linear(self._dim_ffn, self._model_dim)
        self._norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self._norm2 = nn.LayerNorm(d_model, eps=1e-5)

        utils.weight_init(self._linear1, init_fn_=init_fn)
        utils.weight_init(self._linear2, init_fn_=init_fn)

    def forward(self, x, y):
        query = x
        key = y
        value = y
        attn_output, attn_weights = self.inner_att(
            query,
            key,
            value,
            need_weights=True
        )
        norm_attn_ = self._dropout_layer(attn_output) + query
        norm_attn = self._norm2(norm_attn_)

        output = self._linear1(norm_attn)
        output = self._relu(output)
        output = self._dropout_layer(output)
        output = self._linear2(output)
        output = self._dropout_layer(output) + norm_attn_
        return output


class Network_arch(nn.Module):
    def __init__(self,
                 input_f,
                 output_f,
                 model_dim,
                 pos_encoding_params
                 ):
        super(Network_arch, self).__init__()
        self.temporal = Temporal_block(input_frames=input_f, out_frames=output_f, model_dim=model_dim,
                                       pos_encoding_params=pos_encoding_params)
        self.spatial = Spatial_block(input_frame=input_f, model_dim=model_dim,
                                     pos_encoding_params=pos_encoding_params)

    def forward(self, x):
        """
        :param x:  batch,input_frames,node,3
        :return: batch,output_frames,node*3
        """
        y_t = self.temporal(x)  # output_frames,batch,node,3;
        y_s = self.spatial(x)  # node*3,batch,output_frames;
        return y_t.transpose(0, 1) + y_s


class Temporal_block(nn.Module):
    def __init__(self, input_frames, out_frames, model_dim, pos_encoding_params):
        super(Temporal_block, self).__init__()
        self.body_dim = 72
        self.hand_dim = 45
        head_num = 4
        self.body_encoder = Encoder(input_frames, pose_dim=self.body_dim, model_dim=model_dim,
                                    pos_encoding_params=pos_encoding_params)
        self.lhand_encoder = Encoder(input_frames, pose_dim=self.hand_dim, model_dim=model_dim,
                                     pos_encoding_params=pos_encoding_params)
        self.rhand_encoder = Encoder(input_frames, pose_dim=self.hand_dim, model_dim=model_dim,
                                     pos_encoding_params=pos_encoding_params)
        self._inter_body_lhand = Interaction_att(d_model=model_dim, head_num=head_num)
        self._inter_body_rhand = Interaction_att(d_model=model_dim, head_num=head_num)
        self._inter_lhand_body = Interaction_att(d_model=model_dim, head_num=head_num)
        self._inter_lhand_rhand = Interaction_att(d_model=model_dim, head_num=head_num)
        self._inter_rhand_lhand = Interaction_att(d_model=model_dim, head_num=head_num)
        self._inter_rhand_body = Interaction_att(d_model=model_dim, head_num=head_num)
        self.body_decoder = Decoder(out_frames, pose_dim=self.body_dim, model_dim=model_dim,
                                    pos_encoding_params=pos_encoding_params)
        self.lhand_decoder = Decoder(out_frames, pose_dim=self.hand_dim, model_dim=model_dim,
                                     pos_encoding_params=pos_encoding_params)
        self.rhand_decoder = Decoder(out_frames, pose_dim=self.hand_dim, model_dim=model_dim,
                                     pos_encoding_params=pos_encoding_params)

    def forward(self, x):
        """

        :param x: batch,input_frames,node,3
        :return: batch,output_frames,node,3
        """
        batch, frames, _, _ = x.shape
        input = x.transpose(0, 1)
        hbody = self.body_encoder(input[:, :, :24].reshape(frames, batch, -1))
        lhand = self.lhand_encoder(input[:, :, 24:39].reshape(frames, batch, -1))
        rhand = self.rhand_encoder(input[:, :, 39:].reshape(frames, batch, -1))
        hbody = hbody + self._inter_body_lhand(hbody, lhand) + self._inter_body_rhand(hbody, rhand)
        lhand = lhand + self._inter_lhand_rhand(lhand, rhand) + self._inter_lhand_body(lhand, hbody)
        rhand = rhand + self._inter_rhand_lhand(rhand, lhand) + self._inter_rhand_body(rhand, hbody)
        hbody = self.body_decoder(hbody)
        lhand = self.lhand_decoder(lhand)
        rhand = self.rhand_decoder(rhand)
        return torch.cat([hbody, lhand, rhand], dim=2).reshape(frames, batch, -1, 3)


class Spatial_block(nn.Module):
    def __init__(self, input_frame, model_dim, pos_encoding_params):
        super(Spatial_block, self).__init__()

        self.body_dim = 72
        self.hand_dim = 45
        head_num = 4
        self.body_encoder = Encoder(self.body_dim, pose_dim=input_frame, model_dim=model_dim,
                                    pos_encoding_params=pos_encoding_params)
        self.lhand_encoder = Encoder(self.hand_dim, pose_dim=input_frame, model_dim=model_dim,
                                     pos_encoding_params=pos_encoding_params)
        self.rhand_encoder = Encoder(self.hand_dim, pose_dim=input_frame, model_dim=model_dim,
                                     pos_encoding_params=pos_encoding_params)
        self._inter_body_lhand = Interaction_att(d_model=model_dim, head_num=head_num)
        self._inter_body_rhand = Interaction_att(d_model=model_dim, head_num=head_num)
        self._inter_lhand_body = Interaction_att(d_model=model_dim, head_num=head_num)
        self._inter_lhand_rhand = Interaction_att(d_model=model_dim, head_num=head_num)
        self._inter_rhand_lhand = Interaction_att(d_model=model_dim, head_num=head_num)
        self._inter_rhand_body = Interaction_att(d_model=model_dim, head_num=head_num)
        self.body_decoder = Decoder(self.body_dim, pose_dim=input_frame, model_dim=model_dim,
                                    pos_encoding_params=pos_encoding_params)
        self.lhand_decoder = Decoder(self.hand_dim, pose_dim=input_frame, model_dim=model_dim,
                                     pos_encoding_params=pos_encoding_params)
        self.rhand_decoder = Decoder(self.hand_dim, pose_dim=input_frame, model_dim=model_dim,
                                     pos_encoding_params=pos_encoding_params)

    def forward(self, x):
        """

        :param x: batch,input_frames,node,3
        :return: batch,output_frames,node,3
        """
        batch, frames, _, _ = x.shape
        hbody = x[:, :, :24].reshape(batch, frames, -1).permute(2, 0, 1)
        lhand = x[:, :, 24:39].reshape(batch, frames, -1).permute(2, 0, 1)
        rhand = x[:, :, 39:].reshape(batch, frames, -1).permute(2, 0, 1)
        hbody = self.body_encoder(hbody)
        lhand = self.lhand_encoder(lhand)
        rhand = self.rhand_encoder(rhand)
        hbody = hbody + self._inter_body_lhand(hbody, lhand) + self._inter_body_rhand(hbody, rhand)
        lhand = lhand + self._inter_lhand_rhand(lhand, rhand) + self._inter_lhand_body(lhand, hbody)
        rhand = rhand + self._inter_rhand_lhand(rhand, lhand) + self._inter_rhand_body(rhand, hbody)
        hbody = self.body_decoder(hbody)
        lhand = self.lhand_decoder(lhand)
        rhand = self.rhand_decoder(rhand)
        output = torch.cat([hbody, lhand, rhand], dim=0).permute(1, 2, 0).reshape(batch, frames, -1, 3)
        return output


class Encoder(nn.Module):
    def __init__(self, input_n=30,
                 pose_dim=72,
                 model_dim=256,
                 encoder_layers=4,
                 encoder_head_num=4,  # encoder中selfAttention的head数量
                 pos_encoding_params=(1000, 1),
                 dropout=0.5):
        super(Encoder, self).__init__()

        self.input_n = input_n
        self.model_dim = model_dim
        self._pos_encoding_params = pos_encoding_params
        self.encoder_layers = encoder_layers
        self._pos_encoder = PositionEncodings.PositionEncodings1D(
            num_pos_feats=self.model_dim,
            temperature=self._pos_encoding_params[0],
            alpha=self._pos_encoding_params[1]
        )
        encoder_pos_encodings = self._pos_encoder(input_n).view(
            input_n, 1, self.model_dim)
        self._encoder_pos_encodings = nn.Parameter(
            encoder_pos_encodings, requires_grad=False)
        self._relu = nn.ReLU()
        self._dropout_layer = nn.Dropout(dropout)
        self.HomoLinear = nn.Linear(in_features=pose_dim, out_features=self.model_dim)
        self.INATs = []
        # 有几个剩余块，就是几个隐含层，hidden_feature 输入，hidden_feature输出
        for i in range(encoder_layers):
            self.INATs.append(
                Inner_att(d_model=self.model_dim, head_num=encoder_head_num, dim_ffn=2048, dropout=dropout))

        self.INATs = nn.ModuleList(self.INATs)

        self.init_position_encodings()

    def init_position_encodings(self):
        src_len = self.input_n
        # when using a token we need an extra element in the sequence
        encoder_pos_encodings = self._pos_encoder(src_len).view(
            src_len, 1, self.model_dim)
        self._encoder_pos_encodings = nn.Parameter(
            encoder_pos_encodings, requires_grad=False)

    def forward(self, x):
        y = self._dropout_layer(self._relu(self.HomoLinear(x)))
        for i in range(self.encoder_layers):
            y = self.INATs[i](y, self._encoder_pos_encodings)
        return y


class Decoder(nn.Module):
    def __init__(self, input_n=30,
                 pose_dim=72,
                 model_dim=256,
                 decoder_layers=4,
                 decoder_head_num=4,  # encoder中selfAttention的head数量
                 pos_encoding_params=(1000, 1),
                 dropout=0.5):
        super(Decoder, self).__init__()
        self.input_n = input_n
        self.model_dim = model_dim
        self._pos_encoding_params = pos_encoding_params
        self._pos_decoder = PositionEncodings.PositionEncodings1D(
            num_pos_feats=self.model_dim,
            temperature=self._pos_encoding_params[0],
            alpha=self._pos_encoding_params[1]
        )
        self._relu = nn.ReLU()
        self._dropout_layer = nn.Dropout(dropout)
        self.restoreLinear = nn.Linear(in_features=self.model_dim, out_features=pose_dim)

        self.decoder_layers = decoder_layers
        self.INATs = []
        # 有几个剩余块，就是几个隐含层，hidden_feature 输入，hidden_feature输出
        for i in range(decoder_layers):
            self.INATs.append(
                Inner_att(d_model=self.model_dim, head_num=decoder_head_num, dim_ffn=2048, dropout=dropout))

        self.INATs = nn.ModuleList(self.INATs)
        self.init_position_encodings()

    def init_position_encodings(self):
        src_len = self.input_n
        # when using a token we need an extra element in the sequence
        decoder_pos_encodings = self._pos_decoder(src_len).view(
            src_len, 1, self.model_dim)
        self._decoder_pos_encodings = nn.Parameter(
            decoder_pos_encodings, requires_grad=False)

    def forward(self, x):
        y = x
        for i in range(self.decoder_layers):
            y = self.INATs[i](y, self._decoder_pos_encodings)
        y = self._dropout_layer(self._relu(self.restoreLinear(y)))
        return y


class Inro_attention(nn.Module):
    def __init__(self):
        super(Inro_attention, self).__init__()
    # def forward(self,x,y):


# if __name__ == "__main__":
    # # model = Encoder(input_n=30, pose_dim=72, model_dim=256, encoder_layers=4, encoder_head_num=4,
    # #                 pos_encoding_params=(10, 500), dropout=0.2)
    # model = Network_arch(input_f=30, output_f=30, model_dim=256, pos_encoding_params=(10, 500))
    # input = torch.randn(128, 30, 54, 3)
    #
    # output = model(input)
    #
    # print("output", output.shape)
