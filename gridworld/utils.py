import gym
import os
import numpy as np
import random
import scipy.stats
from tqdm import tqdm

def qlearn(nobs,nact,data,epochs=3,lr=0.3):
    Q = np.zeros([nobs, nact])
    y = .95
    for _ in range(epochs):
        for row in tqdm(data):
            [state,act,ns,done,rew] = row
            if done:
                Q[state,act] = Q[state,act] + lr * (rew - Q[state,act])
            else:
                Q[state,act] = Q[state,act] + lr * (rew + y * np.max(Q[ns,:]) - Q[state,act])
    return Q, (Q.max(axis=1).reshape(8, 8))[1:-1,1:-1]

