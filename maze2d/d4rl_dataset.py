import torch
from torch.utils import data
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import random
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler


class d4rlDataset(data.Dataset):
    def __init__(self, train=True):
        
        self.samples = pd.read_feather('data.feather')
        self.samples = self.samples.to_numpy()
        self.len = len(self.samples)
        
        if train:
            self.samples = np.concatenate([self.samples[:int(0.45*self.len)], self.samples[int(0.55*self.len):]], 0)
            self.len *= 0.9
        else:
            self.samples = self.samples[int(0.45*self.len):int(0.55*self.len)]
            self.len *= 0.1

    def __len__(self):
        
        return int(self.len)

    def __getitem__(self, index):
        
        curr_state = self.samples[index][1:5]
        next_state = self.samples[index][5:9]
        action     = self.samples[index][-1]
        
        return curr_state, action, next_state

