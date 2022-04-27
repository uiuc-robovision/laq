import numpy as np
import util.pd
import d4rl
import gym
import scipy.cluster
import pandas as pd
import argparse
from tqdm import tqdm 

parser = argparse.ArgumentParser(description='process data')
parser.add_argument('-s',
                    '--skip',
                    dest='skip',
                    type=int,
                    default=0,
                    help='frame skip')
parser.add_argument('-n','--num-actions',
                    dest='num_actions',
                    type=int,
                    default=4,
                    help='num_actions')
parser.add_argument('-p',
                    '--pos-only',
                    dest='pos_only',
                    action='store_true',
                    help='and')
parser.add_argument('--cat',
                    action='store_true')
parser.add_argument('env', help='environment')
args = parser.parse_args()

env = gym.make(args.env)
dataset = env.env.get_dataset()

downsample = args.skip + 1
totaldf = None

terminals = [-1]+sorted(np.where(dataset['terminals'])[0]) 
if terminals[-1] != len(dataset['terminals'])-1:
    terminals.append(len(dataset['terminals'])-1)

trajectories = []
for i in range(len(terminals)-1):
    b,a = terminals[i]+1,terminals[i+1]+1
    if a-b > 1:
        data = {'observations': dataset['observations'][b:a,:],
                'actions': dataset['actions'][b:a,:],
                'terminals': dataset['terminals'][b:a]
                }
        trajectories.append(data)

lens = [len(x['observations']) for x in trajectories]
for traj in tqdm(trajectories):
    for offset in range(downsample):
        obs = traj['observations'][offset::downsample,:]
        if args.pos_only:
            obs = obs[:,:2]
        next_obs = obs[1:,:]
        obs = obs[:-1,:]
        df = pd.DataFrame()
        util.pd.multi_add(df,obs,'observations')
        util.pd.multi_add(df,next_obs,'next_observations')
        target = env.unwrapped._target
        reward = (np.linalg.norm(obs[:,:2] - target,axis=1) <= 0.5).astype(np.int32)
        df['reward'] = reward

        # steps for monte carlo values
        reward_pos = np.where(df['reward'])[0]
        dists = np.zeros_like(df['reward']).astype(np.float)
        # print(len(dists))
        
        for i in range(len(dists)):
            relative_pos = reward_pos-i
            if len(reward_pos) == 0 or relative_pos.max() < 0:
                dists[i] = np.nan
            else:
                dists[i] = relative_pos[relative_pos>=0].min()
        df['steps_to_goal'] = dists

        if totaldf is None:
            totaldf = df
        else:
            totaldf = pd.concat([totaldf,df])
            # print(len(totaldf))

if args.cat:
    feats = np.concatenate([util.pd.multi_get(totaldf,'next_observations'),util.pd.multi_get(totaldf,'observations')],axis=1)
else:
    feats = util.pd.multi_get(totaldf,'next_observations')- util.pd.multi_get(totaldf,'observations')

clusters = scipy.cluster.vq.kmeans2(feats,args.num_actions)
discrete_actions = clusters[1]
totaldf['actions'] = discrete_actions
totaldf = totaldf.reset_index()
# sanity check
assert totaldf['reward'].sum() == (totaldf['steps_to_goal'] == 0).sum()

name = (f'{args.env}-skip{args.skip}-actions{args.num_actions}')
if args.pos_only:
    name += '-pos_only'
if args.cat:
    name += '-cat'
totaldf.to_feather(f'maze2d/data.feather')
print(totaldf.actions)
print(totaldf.reward.sum())
print(args.env)
print('saving ', name)

