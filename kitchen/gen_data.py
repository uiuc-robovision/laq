import gym
import d4rl  # Import required to register environments
import numpy as np
import numpy as np
env = gym.make('kitchen-partial-v0')
dataset = env.get_dataset()
dataset = d4rl.qlearning_dataset(env)
observations = []
def convert_obs(ob):
    env.robot.reset(env,ob[:30],np.zeros((30,)))
    env.sim.forward()
    efsid = env.model.site_name2id('end_effector')
    ef_pos = env.data.site_xpos[efsid]
    return np.concatenate((ef_pos,ob[9:30]))
new_obs = []
new_next_obs = []
rewards = []
from tqdm import tqdm
for i in tqdm(range(len(dataset['observations']))):
    obs = dataset['observations'][i]
    robs = obs[:30]
    reward = abs(robs[-8]) >= 0.7
    new_obs.append(convert_obs(obs))
    new_next_obs.append(convert_obs(dataset['next_observations'][i]))
    rewards.append(reward)

new_obs = np.stack(new_obs)
new_next_obs = np.stack(new_next_obs)
rewards = np.stack(rewards)
rewards = rewards.astype(np.float32)

import h5py
fil = h5py.File(f"kitchen/data.hdf5", "w")
fil.create_dataset('observations',data=new_obs)
fil.create_dataset('next_observations',data=new_next_obs)
fil.create_dataset('rewards',data=rewards)
fil.close()
