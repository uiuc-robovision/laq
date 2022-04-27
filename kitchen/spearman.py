import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import argparse
import os
import torch
import numpy as np
from FCQNetwork import FCQNetwork
from tqdm import tqdm
from scipy import stats
import gym
import d4rl

env = gym.make('kitchen-partial-v0')
device = torch.device("cpu")

dataset = {}
dataset['observations'] = h5py.File('data.hdf5')['observations'][:]
env.close()

env = gym.make('kitchen-end-effector-v0')
start_pos = env.reset()
def microwave_value(obs):
    assert obs.shape[-1] == 24
    ef_pos = obs[:3]
    md = abs(obs[-8])
    pos = start_pos.copy()
    pos[-8] = obs[-8]
    env.robot.reset(env,pos,np.zeros((30,)))
    env.sim.forward()
    doorsid = env.model.site_name2id('microhandle_site')
    door_site = env.data.site_xpos[doorsid]
    dist = np.linalg.norm(ef_pos - door_site)
    if md >= 0.7:
        return 5
    elif md > 0.01:
        return 1+md
    else:
        return np.exp(-dist)

mean = torch.tensor(dataset['observations'].mean(axis=0)).float()
stds = torch.tensor(dataset['observations'].std(axis=0)).float()

ckpts = np.arange(1, 601)*5000
spearmans2=[]
for idx, ckpt_num in tqdm(enumerate(ckpts)):
    model2 = FCQNetwork(24,64,means=mean,stds=stds)
    lat_params = torch.load(f'configs/experiments/real_data/models/sample{ckpt_num}.torch')
    model2.load_state_dict(lat_params['model_state_dict'])

    from lambda_loader import LambdaLoader
    from torch.utils.data import DataLoader
    dataset = {}
    dataset['observations'] = np.load('kitchen_obs.npy')
    dat = LambdaLoader(len(dataset['observations']),lambda x: dataset['observations'][x])
    loader = DataLoader(dat,batch_size=16,shuffle=True)
    it = iter(loader)
    allVal,val2=[],[]
    for _ in tqdm(range(100)):
        try:
            obs = next(it)
        except:
            break
        obs = obs
        with torch.no_grad():
            values = []
            for idx in range(obs.shape[0]):
                value = microwave_value(obs[idx].cpu())
                values.append(value)
            values = torch.Tensor(values)
            val2.append(model2(obs.float()).max(axis=-1)[0])

        allVal.append(values)
    allVal = torch.stack(allVal[:-1]).cpu()
    val2 = torch.stack(val2[:-1]).cpu()
    print(ckpt_num, stats.spearmanr(allVal.ravel(),val2.ravel()).correlation)
    spearmans2.append(stats.spearmanr(allVal.ravel(),val2.ravel()).correlation)
    np.save(f'spearmans.npy', spearmans2)
