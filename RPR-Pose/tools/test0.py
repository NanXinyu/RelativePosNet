from __future__ import absolute_import
from __future__ import  division
from __future__ import print_function
from unittest.mock import patch



import torch 
import torch.nn as nn
import numpy as np

import _init_paths
from einops import rearrange
from main.model import get_pose_net
from config import cfg, update_config
from main.base import inference
from utils.transforms import flip_back_output
import torch.nn.functional as F
from utils.utils import create_logger
#from dataset.data import Data
import os.path as osp
def generate_token_target(joints):
        '''
            :param joints: [num_joints, 3]
            :param joints_vis: [num_joints, 3]
            :return: target, target_weight(1: visible, 0:invisible)
        '''
        num_joints = 17
        num = 64
        patch = [32, 32]
        sigma = 2
        target = np.zeros((17, 64), dtype = np.float32)
        
       
        div = np.full((num), 8, dtype = np.float32) 
        offset_x = np.full((num), patch[1] * 0.5, dtype = np.float32)
        offset_y = np.full((num), patch[0] * 0.5, dtype = np.float32)
        
        for joint_id in range(num_joints):

            mu_x = np.full((num), joints[joint_id][0], dtype = np.float32)
            print("mu_x:{}".format(mu_x))
            mu_y = np.full((num), joints[joint_id][1], dtype = np.float32)
            print("mu_y:{}".format(mu_y))
            
            x = (np.arange(0, num, 1, np.float32) % div) * patch[1] + offset_x
            print("x={}".format(x-mu_x))
            y = (np.arange(0, num, 1, np.float32) // div) * patch[0] + offset_y
            print("y={}".format(y-mu_x))

            target[joint_id] = (np.exp(-0.5 * (((x - mu_x)**2 + (y - mu_y)**2) / sigma**2)))/(np.pi*2*(sigma**2))



        return target

if __name__ == "__main__":
    '''
    whether normalization?
    '''
    '''
    test generate patch target
    '''
    # print("input:")
    # x = np.arange(0, cfg.MODEL.NUM_PATCHES, 1)
    # print(x)
    # div = 8
    # x_x = x %8
    # print("x")
    # print(x_x)
    # print("y")
    # x_y = x // 8
    # print(x_y)
    '''
    stacking test:
    '''
    # x = torch.rand(2, 3, 4, 4)
    # print("in:")
    # print(x)
    # p_h, p_w, n_w, n_h = 2, 2, 2, 2
    # x = rearrange(x, 'n c (n_h p_h) (n_w p_w) -> n (n_h n_w) c p_h p_w', p_h = p_h, n_h = n_h, p_w = p_w, n_w = n_w) 
    # print("out:")
    # print(x)
    # pool = nn.AdaptiveAvgPool3d((3, 1, 1))
    # x = pool(x).reshape(-1, 4, 3)
    # x = rearrange(x, 'n c d -> n d c')
    # print("result:")
    # print(x)
    '''
    model test:
    '''
    # input = torch.rand(32, 3, 256, 256)
    # model = get_pose_net(cfg, True)
    # output = model(input)
    '''
    inference:
    '''
    # print(output.shape)
    # output = F.softmax(output, dim = 2)
    # mavels = np.max(output.detach().numpy(), axis = 2)
    # coordinate = torch.ones([input.size(0), output.size(1),2])
    # coordinate[:,:,0], coordinate[:,:,1] = inference(output, cfg)
    # pred_x, pred_y = inference(output, cfg)
    # #assert pred_x == coordinate[:,:,0] and pred_y == coordinate[:,:,1], \
    # #    '???'
    # print("mavels shape = {}".format(mavels.shape))
    # print("mavels = {}".format(mavels))
    # print("coordinate = {}".format(coordinate.shape))
    # print("pred_x = {}".format(pred_x.shape))
    # print("pred_y = {}".format(pred_y.shape))

    '''
    config dir
    '''
    update_config(cfg)
    print(cfg.CUR_DIR)
    print(cfg.ROOT_DIR)
    print(cfg.OUTPUT_DIR)
    print(cfg.LOG_DIR)
    print(cfg.DATA_DIR)
    print(cfg.DATASET_ROOT)
    logger, l, r = create_logger(cfg)


