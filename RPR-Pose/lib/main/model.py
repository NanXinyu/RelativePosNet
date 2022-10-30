# --------------------------------------------------
# 2022/10/25
# Written by Xinyu Nan (nan_xinyu@126.com)
# --------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import logging
from tkinter import Y

import torch
import torch.nn as nn
from einops import rearrange
# from config.base import _C as cfg

logger = logging.getLogger(__name__)

class SqueezeExcitation(nn.Module):
    '''
    input: [batch_size, channels, height, width]
    output: [batch_size, channels, height, width]
    '''
    def __init__(
        self,
        in_channels,
        squeeze_channels,
        activation = nn.ReLU,
        scale_activation = nn.Sigmoid
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def forward(self, x):
        # print("SE:")
        # print("in = {}".format(x.shape))
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        # print("out = {}".format(x.shape))
        return scale * x

class Conv2d_BN_ReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        groups = 1,
    ):
        super(Conv2d_BN_ReLU, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv2d_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias = False)
       
        self.norm_layer = nn.BatchNorm2d(out_channels)
       
        self.activation_layer = nn.ReLU(inplace = True)
        
    def forward(self, x):
        # print("Conv2d:")
        # print("in = {}".format(x.shape))
        x = self.conv2d_layer(x) 
        x = self.norm_layer(x)
        x = self.activation_layer(x)
        # print("out = {}".format(x.shape))
        return x

class Conv3d_BN_ReLU(nn.Module):
    def __init__(
        self,
        in_channels,  # cfg.MODEL.NUM_JOINTS
        out_channels, # cfg.MODEL.NUM_JOINTS
        kernel_size,  # [1, 3, 3]
        stride,       # [1, 2, 2]
        groups = 1,
        ):
        super(Conv3d_BN_ReLU, self).__init__()
        self.padding = [(kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2]

        self.conv3d_layer = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, groups = groups, bias = False)

        self.norm_layer = nn.BatchNorm3d(out_channels)

        self.activation_layer = nn.ReLU(inplace = True)
    
    def forward(self, x):
        # print("Conv3d")
        # print("in = {}".format(x.shape))
        x = self.conv3d_layer(x)
        x = self.norm_layer(x)
        x = self.activation_layer(x)
        # print("out = {}".format(x.shape))
        return x

class JointsClassifer(nn.Module):
    def __init__(
        self,
        img_channels,    # cfg.MODEL.IMG_CHANNELS
        hidden_channels, # cfg.MODEL.HIDDEN_SIZE [64, 128, 256]
        num_joints,      # cfg.MODEL.NUM_JOINTS
    ):
        super(JointsClassifer, self).__init__()
        
        layers = []
        layers.append(Conv2d_BN_ReLU(img_channels, hidden_channels[0], kernel_size=1))
        
        for i in range(len(hidden_channels)-1):
            in_planes = hidden_channels[i]
            out_planes = hidden_channels[i+1]
            layers.append(Conv2d_BN_ReLU(in_planes, in_planes, kernel_size=3, stride=2, groups=in_planes))
            layers.append(Conv2d_BN_ReLU(in_planes, out_planes, kernel_size=1, stride=1))
        
        layers.append(SqueezeExcitation(out_planes, out_planes//4))
        layers.append(Conv2d_BN_ReLU(out_planes, num_joints, kernel_size=1))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # print("Joints Classifier:")
        # print("enter size:{}".format(x.shape))
        x = self.classifier(x)
        # print("leave size:{}".format(x.shape))
        return x

class PatchesBlock(nn.Module):
    def __init__(
        self,
        # in_size,
        # patch_size, # cfg.MODEL.PATCH_SIZE
        # num_joints,
        num_patches,
    ):
        super().__init__()
        # self.p_h = patch_size[0]
        # self.p_w = patch_size[1]
        # self.n_h = in_size[0] // self.p_h
        # self.n_w = in_size[1] // self.p_w
        # self.new_channels = self.n_h * self.n_w
        # self.depth = num_joints
        # self.conv3d_layer1 = Conv3d_BN_ReLU(self.new_channels, self.new_channels, 1, 1)
        # self.conv3d_layer2 = Conv3d_BN_ReLU(self.new_channels, self.new_channels, [1, 3, 3], [1, 2, 2], self.new_channels)
        # self.pool_layer = nn.AdaptiveAvgPool3d((self.depth, 1, 1))
        self.conv3d_layer = Conv3d_BN_ReLU(num_patches, num_patches, [1, 3, 3], [1, 2, 2], num_patches)
        # self.conv3d_layer2 = Conv3d_BN_ReLU(num_patches, num_patches, [1, 1, 1], 1)

    def forward(self, x):
        # print("Patches Block:")
        # print("enter size:{}".format(x.shape))
        # x = rearrange(x, 'n c (n_h p_h) (n_w p_w) -> n (n_h n_w) c p_h p_w', p_h = self.p_h, n_h = self.n_h, p_w = self.p_w, n_w = self.n_w) 
        x = self.conv3d_layer(x)
        # y = self.conv3d_layer2(x)
        # x = x + y 
        # print("leave size: {}".format(x.shape))
        return x
        

class PatchesPosNet(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        self.p_h, self.p_w = cfg.MODEL.PATCH_SIZE
        self.channels = cfg.MODEL.NUM_PATCHES
        self.depth = cfg.MODEL.NUM_JOINTS
        self.Classifier = JointsClassifer(cfg.MODEL.IMG_CHANNELS, cfg.MODEL.HIDDEN_CHANNELS, cfg.MODEL.NUM_JOINTS)
        self.Predictor = PatchesBlock(self.channels)
        self.pool_layer = nn.AdaptiveAvgPool3d((self.depth, 1, 1))

      
    def forward(self, x):
        x = self.Classifier(x)
        h, w = x.shape[2], x.shape[3]
        assert h == self.heatmap_size[0] and w == self.heatmap_size[1], \
            'Heatmap Size must be {0} and {1} but not {2} and {3}'.format(self.heatmap_size[0], self.heatmap_size[1], h, w)
        n_h, n_w = h // self.p_h, w // self.p_w
        # print("Before Stacking = {}".format(x.shape))
        x = rearrange(x, 'n c (n_h p_h) (n_w p_w) -> n (n_h n_w) c p_h p_w', p_h = self.p_h, n_h = n_h, p_w = self.p_w, n_w = n_w) 
        assert n_h * n_w == self.channels, \
            'Number of Patches should be {0}*{1}={2}'.format(n_h, n_w, n_h * n_w)
        # print("After Stacking = {}".format(x.shape))
        # print("x is:")
        # print(x)
        x = self.Predictor(x)
        x = self.pool_layer(x)
        # print("After Pooling = {}".format(x))
        x = x.reshape(-1, self.channels, self.depth)
        # print("After = {}".format(x))
        x = rearrange(x, 'n c d -> n d c')
        # print("result:{}".format(x.shape))
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.normal_(m.weight, std=0.001)
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def get_pose_net(cfg, is_train):
    model = PatchesPosNet(cfg)
    if is_train:
        model.init_weights()
    return model