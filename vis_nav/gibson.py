import torch
from torch.utils import data
from PIL import Image
import csv
import numpy as np
import torchvision.transforms as transforms
from os import path
import random
import matplotlib.pyplot as plt
import pandas as pd


def imageNetTransformPIL(size=224):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform = imageNetTransformPIL()

class BranchingDataset(data.Dataset):
    def __init__(self, file_location):
        self.samples = np.load(file_location, allow_pickle=True)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    # returns before image, after image, action, reward, terminal
    def __getitem__(self, index):
        k = self.samples[index, 0]
        k_plus_one = self.samples[index, 8]
        act = int(self.samples[index, -1])

        # convert to new data dir
        k = k.split('0.2/')[1]
        k_plus_one = k_plus_one.split('0.2/')[1]

        # pick k+1 amongst (k_minus_2, k_minus_1, k_plus_2, k_plus_3, k_plus_4)
        path_ = k.rsplit('/', 1)[0]

        img_num = int(k.rsplit('/', 1)[1])
        offsets = [-2, -1, 2, 3, 4]
        valid_k_plus_x = []
        for offset in offsets:
            path_to_test = f'{path_}/{img_num+offset}'
            if path.exists(path_to_test):
                valid_k_plus_x.append(path_to_test)
        k_plus_x = random.choice(valid_k_plus_x)

        k = transform(Image.open(f'{k}/0.jpg'))
        k_plus_one = transform(Image.open(f'{k_plus_one}/0.jpg'))
        k_plus_x = transform(Image.open(f'{k_plus_x}/0.jpg'))
        rew = 0
        term = 0
        gt = 0
        k_plus_1_1 = k_plus_one
        k_plus_1_2 = k_plus_one
        x = 0

        return k, k_plus_1_1, k_plus_1_2, k_plus_x, act, rew, term, gt, x


