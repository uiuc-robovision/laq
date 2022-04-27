import torch
from torch.utils import data
import torch.nn as nn
import numpy as np
import os
from os import path
from PIL import Image
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import random
import glob

def imageNetTransformPIL():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def imageNetTransformPILDownsample():
    return transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Grayscale()
    ])

def imageNetTransformPILVisNav(size=224):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform = imageNetTransformPIL()
transformDownsample = imageNetTransformPILDownsample()
transformVisNav = imageNetTransformPILVisNav()

def ToTensor():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

transformToTensor = ToTensor()

class DopamineDatasetSeedNoSticky(data.Dataset):
    def __init__(self, train=True):
        
        self.data_folder = 'batch_rl_data/imgs/'        

        self.train = train
        if self.train:
            self.num_samples = 1000000 - 50000 - 5
        else:
            self.num_samples = 50000
        
        self.acts = np.load('batch_rl_data/actions.npy')

    def __len__(self):
        'Denotes the total number of samples'
        return self.num_samples
    
    def get_act_and_reset(self, action_ckpt):
        buf = OutOfGraphReplayBuffer((210,128,3),4,int(1e2),32)
        buf.load('batch_rl_data/replay_logs', action_ckpt)
        actions = buf._store['action']
        return actions

    def __getitem__(self, index, multiplier_=None):
                           
        img_num = index
        if self.train:
            if index >= 475000:
                img_num += 50000
        else:
            img_num += 475000
        if multiplier_ == None:
            multiplier_ = random.randint(0,4)
        img_num += multiplier_ * 1000000
            
        if index < 1000000:
            o_tm1_1 = transformToTensor(Image.open(self.data_folder + f"{img_num:06d}.png"))
            o_tm1_2 = transformToTensor(Image.open(self.data_folder + f"{img_num+1:06d}.png"))
            o_tm1_3 = transformToTensor(Image.open(self.data_folder + f"{img_num+2:06d}.png"))
            o_tm1_4 = transformToTensor(Image.open(self.data_folder + f"{img_num+3:06d}.png"))
            o_t = transformToTensor(Image.open(self.data_folder + f"{img_num+4:06d}.png"))
        else:
            o_tm1_1 = transformToTensor(Image.open(self.data_folder + f"{img_num:07d}.png"))
            o_tm1_2 = transformToTensor(Image.open(self.data_folder + f"{img_num+1:07d}.png"))
            o_tm1_3 = transformToTensor(Image.open(self.data_folder + f"{img_num+2:07d}.png"))
            o_tm1_4 = transformToTensor(Image.open(self.data_folder + f"{img_num+3:07d}.png"))
            o_t = transformToTensor(Image.open(self.data_folder + f"{img_num+4:07d}.png"))
        o_tm1 = torch.cat([o_tm1_1, o_tm1_2, o_tm1_3, o_tm1_4], dim=0)
               
        # actions
        a_tm1 = int(self.acts[img_num+3])
        
        return o_tm1, a_tm1, o_t, multiplier_, index # 0, 0
