# -*- coding: utf-8 -*-
import random
import os
import scipy.io as sio
import numpy as np
import argparse
import torch
import cv2
import os


def args_initialize():
    parser = argparse.ArgumentParser(description="新")
    
    # specifications for model
    parser.add_argument('-model_name','--model_name', choices=['RCAN','EDSR','VDSR'], default='RCAN')
    # specifications for training
    parser.add_argument('-train','--train', action='store_true', default=False)   
    parser.add_argument('-bt','--batch_size', type=int, default=1)
    parser.add_argument('-num_epoch','--num_epoch', type=int, default=200)
    parser.add_argument('-continue_training','--continue_training',action='store_true', default=False)
    parser.add_argument('-pretrain_model','--pretrain_model')  
    parser.add_argument('-save_model_name','--save_model_name', type=str)
    parser.add_argument('-patch_size','--patch_size', type=int, default=128)

    parser.add_argument('-n_feats','--n_feats', type=int, default=64,help='number of feature maps') 
    parser.add_argument('-n_resblocks','--n_resblocks', type=int, default=8, help='number of residual blocks')
    parser.add_argument('-n_colors','--n_colors', type=int, default=3, help='number of color channels to use')    
    parser.add_argument('-loss','--loss', type=str, choices=['L1Loss','MSELoss'], default='MSELoss',help='loss function configuration')
    parser.add_argument('-optim','--optim',type=str)
                    
    # specifications for testing
    parser.add_argument('-test','--test', action='store_true', default=False)     
    parser.add_argument('-test_path','--test_path', default='./yourImages')
    parser.add_argument('-test_model','--test_model', type=str)  
    parser.add_argument('-test_save_path','--test_save_path', type=str)       
    
    # specifications for data and models
    parser.add_argument('-xfactor','--step', type=int, default=1)
    parser.add_argument('-scale','--scale', type=int, default=1)
    parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')    
    # Hardware specifications
    parser.add_argument('-cpu','--cpu', action='store_true', default=False)                     
    parser.add_argument('-gpu','--gpu', action='store_true', default=False)
    parser.add_argument('-GPU_ID','--GPU_ID', type=int, default=0)
    
    # other specifications
    parser.add_argument('-clear_valid_and_test_folder', action='store_true', default=False)

    args = parser.parse_args()
    args = check_core(args)
    args = check_path(args)
    if args.clear_valid_and_test_folder:
        clear_folder()
    print(args)
    return args
    
def check_core(args): #设置cpu/gpu device 四式改一20/08/24
    if args.cpu:
        args.device = torch.device('cpu')
    if args.gpu:
        assert torch.cuda.is_available()==True,'\'args.gpu\' is True, but cuda is not available.'
        args.device = torch.device('cuda:'+str(args.GPU_ID))
    assert (args.cpu ^ args.gpu)==True,'One of the options: \'args.cpu\' and \'args.gpu\' should have one True.' 
    return args
        
def check_path(args): # 检查路径
    if args.test:
        assert os.path.isdir(args.test_path),'folder \'args.test_path\' is not found.'   
        if args.test_save_path[-1]=='/': args.test_save_path.rstrip('/')
    if args.save_model_name == None:
        args.save_model_name = args.model_name
    return args

def clear_folder():     #是否更新数据集 
    os.system('rm -rf ./valid/*')
    os.system('rm -rf ./test/*')
    print('clear valid and test folder is done.')

def get_patch(*args, patch_size=128, scale=1, multi=False):
    ih, iw = args[0].shape[2:]
    p = scale if multi else 1
    tp = p * patch_size
    ip = tp
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = ix, iy
    ret = [
        args[0][:,:,iy:iy + ip, ix:ix + ip],
        *[a[:,:,ty:ty + tp, tx:tx + tp] for a in args[1:]]
    ]
    return ret
