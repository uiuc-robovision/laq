import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse
from stable_baselines3 import DDPG
import os
import torch
import numpy as np
from FCQNetwork import FCQNetwork
from tqdm import tqdm

import gym
env = gym.make('CartPole-v0')
env.action_space = gym.spaces.Box(-8.0, 8.0, (24,), np.float32)
env.close()

model1 = DDPG("MlpPolicy", env, verbose=1)
model1 = model1.load('maze2d_ddpg/checkpoints/12000.zip')

dataset = {}
samples = pd.read_feather('data.feather').to_numpy()
dataset['observations'] = samples[:, 1:5]

mean = torch.tensor(dataset['observations'].mean(axis=0)).cuda().float()
stds = torch.tensor(dataset['observations'].std(axis=0)).cuda().float()

import util
from scipy import stats
inds = np.linspace(10000,2400000,240).astype(np.int)
latent_spearmans = []
for ind in tqdm(inds):
    model2 = FCQNetwork(4,8,means=mean,stds=stds)
    lat_params = torch.load(f'configs/experiments/real_data/models/sample{ind}.torch')
    model2.load_state_dict(lat_params['model_state_dict'])
    model2 = model2.cuda()

    from lambda_loader import LambdaLoader
    from torch.utils.data import DataLoader
    dat = LambdaLoader(len(dataset['observations']),lambda x: dataset['observations'][x])
    loader = DataLoader(dat,batch_size=16,shuffle=True)
    it = iter(loader)
    allOb = []
    allVal = []
    val2=[]
    for _ in range(3000):
        obs = next(it)
        obs = obs.cuda()
        with torch.no_grad():
            acts = model1.actor(obs)
            values = model1.critic(obs,acts)[0]
            val2.append(model2(obs).max(axis=-1)[0])
        allOb.append(obs)
        allVal.append(values)
    allOb=torch.cat(allOb,axis=0).cpu()
    allVal=torch.cat(allVal,axis=0).cpu()
    val2=torch.cat(val2,axis=0).cpu()

    latent_spearmans.append(stats.spearmanr(allVal.ravel(),val2.ravel()).correlation)

    # save spearman
    np.save(f'spearmans.npy', latent_spearmans)
    print(latent_spearmans[-1])
