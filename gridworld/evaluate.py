import gym
import os
import numpy as np
from matplotlib import pyplot as plt 
import argparse
from tqdm import tqdm
import random
import pandas as pd

cmd_parser = argparse.ArgumentParser(description='args')
cmd_parser.add_argument('value_function_path')
cmd_args = cmd_parser.parse_args()
from gridworld.envs import make_gridworld
env = make_gridworld()
env.reset()
V = np.load(cmd_args.value_function_path)
def peV(s):
    r,c = s%8,s//8
    return V[r-1,c-1]

methods = [("Densified Reward",peV),("Sparse Reward",lambda x: 0)]
rList = []
shapingFunction = peV
latent_value = True
lr = 1
y = .95
base_eps = 0.1
num_episodes = 250
for name,shapingFunction in methods:
    for it in range(5):
        Q = np.zeros([env.observation_space.n,env.action_space.n])
        num_timesteps = 0
        for i in tqdm(range(num_episodes)):
            eps = base_eps*(num_episodes-i)/num_episodes
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            while j < 1000:
                j+=1
                num_timesteps += 1
                if random.random() <= eps:
                    a = env.action_space.sample()
                else:
                    a = np.argmax(Q[s,:])
                s1,r,d,_ = env.step(a)
                if latent_value:
                    if d:
                        r = 5
                    else:
                        r = r + shapingFunction(s1)*y - shapingFunction(s)
                Q[s,a] = r + y*np.max(Q[s1,:])
                rAll += r
                s = s1
                if d == True:
                    break
            rList.append((num_timesteps,1 if d else 0,it,y**j*int(d),j,name))

df = pd.DataFrame(columns=["Timestep",'Success',"Run","Return","length","Method"],data=rList).infer_objects()
import scipy.interpolate
def interpolate_lines(data,num_points,ranges=None):
    if ranges is None:
        ranges = data[:,:,0].min(),data[:,:,0].max()
    points = np.linspace(ranges[0],ranges[1],endpoint=True,num=500)
    new_lines = []
    for line in data:
        interp = scipy.interpolate.interp1d(line[:,0],line[:,1],fill_value=(line[0,1],line[-1,1]),bounds_error=False)
        new_lines.append(np.stack((points,interp(points)),axis=1))
    return np.stack(new_lines)

plt.clf()
ranges = df['Timestep'].min(),df['Timestep'].max()
for method,_ in methods:
    data = np.stack([df.query(f'Run == { i } and Method == @method')[['Timestep','Return']] for i in range(5)])
    new_data = interpolate_lines(data,500,ranges=ranges)
    nd =new_data.mean(axis=0)
    plt.plot(nd[:,0],nd[:,1],label=method)
plt.legend()
plt.savefig('./gridworld_plot.png')
print("Written visualization './gridworld_plot.png'")
