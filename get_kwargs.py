import gym
import numpy as np
import d4rl
from FCQNetwork import FCQNetwork
import torch
 
def get_kwargs(args):
    if 'kitchen' in args.env:
        return get_kwargs_kitchen(args)
    else:
        return get_kwargs_other(args)

def get_kwargs_kitchen(args):
    env_string = 'kitchen-no-kettle-end-effector-hook-microwave-sparse-term-v0'
    kwargs = {}
    obs = np.load('kitchen/kitchen_obs.npy')
    mean = torch.tensor(obs.mean(axis=0)).cuda().float()
    stds = torch.tensor(obs.std(axis=0)).cuda().float()
    if args.model_path is not None:
        network = FCQNetwork(24,args.num_actions,means=mean,stds=stds,tcc=args.latent_norm).cuda()
        network.load_state_dict(torch.load(args.model_path)['model_state_dict'])
        network.cuda()
        def rew(x):
            with torch.no_grad():
                res = network(torch.tensor([x]).float().cuda())
                return res.max().item()
        kwargs['value_augment'] = rew
    if not args.no_gmm:
        import sklearn.mixture
        import pickle
        with open('kitchen/gm2.pkl', 'rb') as f: gm = pickle.load(f)
        if 'value_augment' in kwargs:
            ova = kwargs['value_augment']
            def rew2(dat,ova=ova,gm=gm):
                in_dist = gm.score_samples(dat[None,:3])[0] > -1.837
                # clip ova for gmm
                return int(in_dist) * min(ova(dat),1)
        else:
            def rew2(dat,gm=gm):
                in_dist = gm.score_samples(dat[None,:3])[0] > -1.837
                return 0 if in_dist else -1
        kwargs['value_augment'] = rew2
    return env_string,kwargs

def get_kwargs_other(args):
    if args.model_path is None:
        # sparse reward variant
        env_string = 'antmaze-medium-super-close-diverse-v0' if args.env == 'ant' else 'maze2d-medium-dense-fixed-v1' 
        return env_string,{}
    if 'ant' in args.env:
        env_string = 'antmaze-medium-super-close-valdif-diverse-v0'
    else:
        env_string = 'maze2d-medium-valdif-fixed-v1'
    kwargs = {}
    dataset = {}
    dataset['observations'] = np.load('maze2d/maze2d_obs.npy')
    mean = torch.tensor(dataset['observations'].mean(axis=0)).cuda().float()
    stds = torch.tensor(dataset['observations'].std(axis=0)).cuda().float()
    model2 = FCQNetwork(4,8,means=mean,stds=stds,tcc=args.latent_norm)
    lat_params = torch.load(args.model_path)
    model2.load_state_dict(lat_params['model_state_dict'])
    model2 = model2.cuda()
    def rew(ins,init_pos=None):
        return model2(ins.unsqueeze(0).float()).max(axis=-1)[0].item()
    kwargs['learned_reward_model'] = rew
    return env_string,kwargs
