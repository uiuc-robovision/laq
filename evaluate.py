import gym
import numpy as np
import d4rl
import stable_baselines3
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse
from stable_baselines3 import DDPG
from stable_baselines3 import SAC,PPO,TD3
import os
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from FCQNetwork import FCQNetwork
from stable_baselines3.common.noise import NormalActionNoise
# from franka_value import microwave_value

import argparse
parser = argparse.ArgumentParser(description='')
argparse
parser.add_argument('--env',choices=['ant','kitchen','maze'])
parser.add_argument('--reset',action='store_true')
parser.add_argument('--exp-name',default=None)
parser.add_argument('--model-path',default=None)
parser.add_argument('--seed',default=0,type=int)
parser.add_argument('--num-actions',default=64,type=int)
# parser.add_argument('--reward',action='store_true')
# parser.add_argument('--latent',action='store_true')
parser.add_argument('--latent-norm',action='store_true')
# parser.add_argument('--policy-eval',action='store_true')
# parser.add_argument('--pe-norm',action='store_true')
# parser.add_argument('--bcq',action='store_true')
# parser.add_argument('--bcq-norm',action='store_true')
# parser.add_argument('--d3g',action='store_true')
# parser.add_argument('--d3g-norm',action='store_true')
parser.add_argument('--no-gmm',action='store_true')
# parser.add_argument('--target-ent',default=-3,type=int)
args = parser.parse_args()

exp_name = args.env
if args.seed != 0:
    exp_name += f"_seed{args.seed}"
if args.no_gmm:
    exp_name += f"_nogmm"
if args.exp_name is not None:
    exp_name = args.exp_name
if args.reset:
    os.system(f'rm -rf ./runs/{exp_name}')

import math
import imageio
import pathlib
import gym
from gym.wrappers.frame_stack import FrameStack
from matplotlib import pyplot as plt 

print("EXP: ",exp_name)
log_folder = f'./runs/{exp_name}'
# reproducability snippet
import sys
pathlib.Path(log_folder).mkdir(parents=True, exist_ok=True)
os.system(f'git diff > {log_folder}/code_diff.txt')
os.system(f'git diff --cached >> {log_folder}/code_diff.txt')
# Only works correctly if running in the git root directory
os.system(f'git ls-files --others --exclude-standard | while read -r i; do git diff -- /dev/null "$i"; done >> {log_folder}/code_diff.txt')
os.system(f'echo "Commit: " > {log_folder}/info.txt')
os.system(f'git rev-parse --verify HEAD >> {log_folder}/info.txt')
with open(f"{log_folder}/info.txt", "a") as fil: fil.write(' '.join(sys.argv)) 
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
TEEVAR = Tee(f'{log_folder}/log.txt','w')

import torch
from get_kwargs import get_kwargs
kwargs = {}
env_string = args.env
from get_kwargs import get_kwargs
env_string,kwargs = get_kwargs(args)
print('ENV_STRING:', env_string)
env = gym.make(env_string,**kwargs)
if args.env in ['kitchen','ant']:
    env = FrameStack(env,4)
lr = 3e-4 if args.env == 'ant' else 1e-3 

obs = env.reset()
set_random_seed(args.seed)
env.seed(args.seed)

#maybe need to do this?
# ent_coef = 1.5e-3 if args.latent_norm and args.env=='kitchen' else 'auto'
# roughly 1e-3 for ant if fixed (or 2e-4?)tk
model = SAC("MlpPolicy", env, verbose=1,tensorboard_log=f'runs/{exp_name}/logs',device='cuda:0',learning_rate=lr,tau=0.005,gradient_steps = -1,ent_coef='auto')

if 'kitchen' in env_string:
    total_timesteps = int(1e6)
elif 'ant' in env_string:
    total_timesteps = int(7.5e5)
else:
    total_timesteps = int(2e5)

model.learn(total_timesteps=total_timesteps)
env.close()
os.system(f'echo done > {log_folder}/done.txt')
