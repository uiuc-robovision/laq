import pandas as pd
import numpy as np
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import csv
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import os
import pdb
import time
from itertools import permutations
from absl import app
from absl import flags
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from torch.utils import data

samples = pd.read_feather('data.feather')
samples = samples.to_numpy()

class ForwardModel(nn.Module):
    def __init__(self, num_actions):
        super(ForwardModel, self).__init__()
        
        activation = nn.ReLU()
        
        embedding_dim = 1
        self.embedding = nn.Embedding(num_embeddings=bottleneck_size, 
                                      embedding_dim=embedding_dim)
        
        readout_dim = 16
        self.action_readout = nn.Sequential(
                nn.Linear(embedding_dim, readout_dim), activation, nn.Dropout(0.5),
                nn.Linear(readout_dim, readout_dim), activation, nn.Dropout(0.5),
                nn.Linear(readout_dim, num_actions))
        
        hidden_size1 = 8
        hidden_size2 = 16
        hidden_size3 = 8
        
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_size1), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size1),
            nn.Linear(hidden_size1, hidden_size2), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size2),
            nn.Linear(hidden_size2, hidden_size3), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size3)
        )
        
        hidden_size4 = 16
        hidden_size5 = 8
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size3 * 2, hidden_size4), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size4),
            nn.Linear(hidden_size4, hidden_size5), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size5),
            nn.Linear(hidden_size5, 4), 
        )
        
        
    def forward(self, curr_states, actions, next_states):

        all_actions = torch.arange(bottleneck_size).cuda()
        encoding = self.embedding(all_actions)
        
        y = encoding
        act1 = encoding
        y = y.detach()

        # readout function
        y = y / y.norm(dim=1, keepdim=True)
        act_readout = self.action_readout(y)
        
        # encoder
        phi_k = self.encoder(curr_states.float())

        assert act1.shape[1] <= phi_k.shape[1]
        rep = int(phi_k.shape[1] / act1.shape[1])
        act1 = act1.repeat(1, rep)
        act1, phi_k = torch.broadcast_tensors(act1.unsqueeze(0), phi_k.unsqueeze(1))
        
        # concat 
        psi = torch.cat((act1, phi_k), 2)
        sz = psi.shape
        psi = psi.view(sz[0]*sz[1], sz[2])
        psi = self.decoder(psi)
        sz2 = psi.shape
        psi = psi.view(sz[0], sz[1], sz2[1])
        return psi, encoding, act_readout


num_actions = 4
bottleneck_size = 8
device = torch.device("cuda")
model = ForwardModel(num_actions).to(device)
dir_ = 'repro' 
model_ = 400000
model_path = f'lam_runs/{dir_}/model-{model_}.pth'
model.load_state_dict(torch.load(model_path))


def loss_fn(mse_loss, psi, o_t, o_tm1, iteration, train=True):
    loss = mse_loss(psi.float(), o_t.unsqueeze(1).repeat(1, bottleneck_size, 1).float())
    loss = loss.view(loss.shape[0], loss.shape[1], -1)
    loss = loss.mean(2)
    _loss, ind = torch.min(loss, 1)
    return ind


batch_size = 256
gt, pred = [], []
mse_loss = nn.MSELoss(reduction='none')
num_iters = int(len(samples) / 256)
for batch_idx in tqdm(range(num_iters)):
    o_tm1 = torch.tensor(samples[batch_idx * batch_size:(batch_idx+1)*batch_size, 1:5])
    o_t   = torch.tensor(samples[batch_idx * batch_size:(batch_idx+1)*batch_size, 5:9])
    a_tm1 = samples[batch_idx * batch_size:(batch_idx+1)*batch_size, -1]
    
    o_tm1, o_t = o_tm1.to(device), o_t.to(device)
    
    # forward pass 
    psi, encoding, y = model(o_tm1, a_tm1, o_t)
    assignment = loss_fn(mse_loss, psi, o_t, o_tm1, 0, train=False)
    
    gt   = np.concatenate([gt, a_tm1], 0)
    pred = np.concatenate([pred, assignment.squeeze().cpu()], 0)


# last sub-batch
o_tm1 = torch.tensor(samples[(batch_idx+1)*batch_size:, 1:5])
o_t   = torch.tensor(samples[(batch_idx+1)*batch_size:, 5:9])
a_tm1 = samples[(batch_idx+1)*batch_size:, -1] # unused

o_tm1, o_t = o_tm1.to(device), o_t.to(device)

# forward pass 
psi, encoding, y = model(o_tm1, a_tm1, o_t)
assignment = loss_fn(mse_loss, psi, o_t, o_tm1, 0, train=False)

gt   = np.concatenate([gt, a_tm1], 0)
pred = np.concatenate([pred, assignment.squeeze().cpu()], 0)
assert len(samples) == len(pred)

data = pd.read_feather('data.feather') 
data['actions'] = pred.astype(int)
data.to_feather('data_latentActs.feather')
