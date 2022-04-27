import gym
import os
import numpy as np
import random
from gym_minigrid.wrappers import *
from collections import defaultdict
from util import argmax

class WithDiagonal(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = gym.spaces.Discrete(8)

    def step(self, action):
        # print("action:", action,self.cur_pos)
        newActs = [(0, 1), (2,1), (2, 3), (3, 0)]
        if action < 4:
            obs,rew,done,info = self.env.step(action)
            self.cur_pos = info['pos']
            return obs,rew,done,info

        na = action - 4
        # print('na: ',na)
        for i,a in enumerate(newActs[na]):
            obs, rew, done, info = self.env.step(a)
            if done: break
        self.cur_pos = info['pos']
        return obs, rew, done, info

    def reset(self):
        self.cur_pos = (1,1)
        return self.env.reset()

class NewMaze(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        sh = env.observation_space['image'].shape[:2]
        self.observation_space = gym.spaces.Discrete(sh[0] * sh[1])
        self.action_space = gym.spaces.Discrete(4)
        self.cols = sh[1]
        self.orientation = 0

    def step(self, action):
        if self.cur_pos == (6, 6):
            return None, 1, True, {'pos': self.cur_pos}

        while self.orientation != action:
            obs, rew, done, info = self.env.step(0)
            x, y = np.where(obs['image'][:, :, 0] == 10)
            self.cur_pos = (x, y)
            self.orientation = obs['image'][x, y, 2]
            rew = 1 if (x.item(), y.item()) == self.goal_loc else 0
            if done: return (x * self.cols + y).item(), 0, False, {'pos':(x,y)}

        obs, rew, done, info = self.env.step(2)
        x, y = np.where(obs['image'][:, :, 0] == 10)
        self.cur_pos = (x, y)
        rew = 1 if (x.item(), y.item()) == self.goal_loc else 0
        return (x * self.cols + y).item(), 0, False, {'pos':(x,y)}

    def reset(self):
        obs = self.env.reset()
        x, y = np.where(obs['image'][:, :, 0] == 10)
        self.cur_pos = (x, y)
        self.orientation = obs['image'][x, y, 2]

        loc = np.where(obs['image'][:, :, 0] == 8)
        self.goal_loc = loc[0].item(), loc[1].item()
        return (x * self.cols + y).item()


class ActionRefinement(gym.ActionWrapper):
    def __init__(self, env, mult, pr=1):
        super().__init__(env)
        self.mult = mult
        self.action_space = gym.spaces.Discrete(env.action_space.n * mult)
        self.pr = pr

    def action(self, act):
        ba = act // self.mult
        # return ba
        if random.random() <= self.pr:
            return ba
        else:
            return (ba + 2) % 4

def make_gridworld():
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = WithDiagonal(NewMaze(FullyObsWrapper(env)))
    return env

