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
from d4rl_dataset import kitchenDataset
from tqdm import tqdm
from sklearn import metrics
import random
import pandas as pd
import h5py

embedding_dim = 64
bottleneck_size = 64
arch_mul = 48
class ForwardModel(nn.Module):
    def __init__(self):
        super(ForwardModel, self).__init__()
        
        activation = nn.ReLU()
        
        embedding_dim = 64
        self.embedding = nn.Embedding(num_embeddings=bottleneck_size, 
                                      embedding_dim=embedding_dim)
                
        hidden_size1 = 8 * arch_mul
        hidden_size2 = 16 * arch_mul
        hidden_size3 = 8 * arch_mul
        
        self.encoder = nn.Sequential(
            nn.Linear(24, hidden_size1), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size1),
            nn.Linear(hidden_size1, hidden_size2), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size2),
            nn.Linear(hidden_size2, hidden_size2), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size2),
            nn.Linear(hidden_size2, hidden_size2), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size2),
            nn.Linear(hidden_size2, embedding_dim), 
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim)
        )
        
        hidden_size4 = 16 * arch_mul
        hidden_size5 = 8 * arch_mul
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_size4), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size4),
            nn.Linear(hidden_size4, hidden_size4), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size4),
            nn.Linear(hidden_size4, hidden_size4), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size4),
            nn.Linear(hidden_size4, hidden_size5), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size5),
            nn.Linear(hidden_size5, 24), 
        )
        
        
    def forward(self, curr_states, next_states):

        all_actions = torch.arange(bottleneck_size).cuda()
        encoding = self.embedding(all_actions)
        
        y = encoding
        act1 = encoding
                
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
        return psi, encoding


torch.cuda.set_device(0)
torch.set_num_threads(1)
device = torch.device("cuda")

model = ForwardModel().to(device)

dir_ = 'repro'
model_ = 3800000
model_path = f'lam_runs/{dir_}/model-{model_}.pth'
model.load_state_dict(torch.load(model_path, device))
model.eval()

def loss_fn(mse_loss, psi, o_t, o_tm1, iteration, train=True):
    # losses
    baseline_loss = mse_loss(o_tm1.float(), o_t.float())
    baseline_loss = baseline_loss.mean()

    loss = mse_loss(psi.float(), o_t.unsqueeze(1).repeat(1, bottleneck_size, 1).float())
    loss = loss.view(loss.shape[0], loss.shape[1], -1)
    loss = loss.mean(2)
    _loss, ind = torch.min(loss, 1)
    return ind

obs = h5py.File('data.hdf5')['observations'][:]
batch_size = 50
actions = []
mse_loss = nn.MSELoss(reduction='none')
for i in tqdm(range(0, obs.shape[0], batch_size)):
    if i+batch_size > obs.shape[0]:
        o_tm1 = torch.from_numpy(obs[i:-1]).to('cuda')
        o_t   = torch.from_numpy(obs[i+1:]).to('cuda')
    else:
        o_tm1 = torch.from_numpy(obs[i:i+batch_size]).to('cuda')
        o_t   = torch.from_numpy(obs[i+1:i+batch_size+1]).to('cuda')

    with torch.no_grad():
        psi, encoding = model(o_tm1, o_t)

    acts = loss_fn(mse_loss, psi, o_t, o_tm1, 0)
    actions.extend(list(acts.squeeze().cpu().detach().numpy()))


# convert to feather
data = h5py.File('data.hdf5')
my_dict = {}
for i in range(24):
    my_dict[f'observations{i}'] = data['observations'][:-1, i]
for i in range(24):
    my_dict[f'next_observations{i}'] = data['next_observations'][:-1, i]
my_dict['reward'] = data['rewards'][:-1]
my_dict['actions'] = np.array(actions).astype(float)
df = pd.DataFrame(data=my_dict)
df.to_feather('data_latentActs.feather')
