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


class kitchenDataset(data.Dataset):
    def __init__(self, train=True, normalize=False):
        
        self.samples = h5py.File('data.hdf5')['observations']
        self.rewards = h5py.File('data.hdf5')['rewards']
        self.len = len(self.samples)-1

        if normalize:
            scaler = StandardScaler()
            scaler.fit(self.samples)
            self.samples = scaler.transform(self.samples)
        
        if train:
            self.samples = np.concatenate([self.samples[:int(0.45*self.len)], self.samples[int(0.55*self.len):]], 0)
            self.len *= 0.9
        else:
            self.samples = self.samples[int(0.45*self.len):int(0.55*self.len)]
            self.len *= 0.1

    def __len__(self):
        
        return int(self.len)

    def __getitem__(self, index):

        while self.rewards[index] == 1:
            index = random.randint(0, int(self.len)-1) 
        
        curr_state = self.samples[index]
        next_state = self.samples[index+1]
        
        return curr_state, next_state

