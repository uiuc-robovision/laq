import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import torch
from util.plt import show
import torch.optim as optim
from FCQNetwork import FCQNetwork
from torch.utils import data
import util
import os
from matplotlib import pyplot as plt
from torch.utils.data import WeightedRandomSampler
import util.plt
from util.pd import multi_get
import torchvision.utils
from tqdm import tqdm
from lambda_loader import LambdaLoader
import pandas as pd
import random
import gym

REACHER_DATASETS = ['reacher', 'reacher_replay', 'reacher_replay_override']
GIBSON_DATASETS = [
    'gibson', 'gibson_medium_40k', 'gibson_medium', 'gibson_medium_inverse',
    'gibson_medium_noisy_40k', 'gibson_medium_noisy',
    'gibson_medium_noisy_value', 'gibson_medium_noisy_40k_value',
    'gibson_branch', 'multi_target', 'multi_target_longer', 'explore',
    'branch_updated', 'gibson_medium_inverse_train'
]

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1024, rlimit[1]))


# computes KL divergence of KL(q,p), which is backwards from how the variables normally, Maybe??
def KLD(mu_p, sig_p, mu_q, sig_q):
    KLD = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) +
                 ((mu_p - mu_q) / sig_p).pow(2))
    return KLD


def build_model(config,inputs=4,actions=None,positional_encoding=False,arch_dim=128,tcc=False):
    if config.D4RL:
        return FCQNetwork(inputs,actions,positional_encoding=positional_encoding,arch_dim=arch_dim,tcc=tcc).cuda()

    sigmoid = config.LOSS_CLIP == 'sigmoid'
    if config.VALUE_LEARNING or config.ONE_ACTION:
        actions = 1
    elif config.SLAM_ACTIONS:
        actions = config.NUM_SLAM_ACTIONS
        print("Using ", actions, "actions (SLAM)")
    else:
        if config.NUM_LATENT_ACTIONS > 0:
            actions = config.NUM_LATENT_ACTIONS
        else:
            actions = 3
    model = HabitatDQNMultiAction(
        actions,
        5,
        extra_capacity=(config.ARCHITECTURE == 'extra_capacity'),
        panorama=(config.PANORAMA or config.PREVIOUS_IMAGES),
        gaussian=config.DISTRIBUTIONAL,
        multi_task=config.MULTI_TASK)
    return model.to(config.device)


def load_model_number(config, number):
    model = build_model(config)
    model_loc = f'{config.folder}/models/sample{number}.torch'
    snapshot = torch.load(model_loc, map_location=config.device)
    print(f'Loading model from: {model_loc}')
    model.load_state_dict(snapshot['model_state_dict'])
    return model


def loopLoader(loader):
    i = iter(loader)
    while True:
        try:
            yield next(i)
        except StopIteration:
            print("reset iterator")
            i = iter(loader)


def visualize_d4rl(config,model,dataset,sample_number,num=10000):
    model.eval()
    num=10000
    inds = np.random.randint(0,len(dataset),(num,))
    locs = dataset[inds][0][:,0:2]
    vels = dataset[inds][0][:,2:]
    res = []
    for ins in tqdm(util.chunks(inds,16)):
        obs = torch.tensor(dataset[ins][0],device='cuda')
        res.append(model(obs).detach().cpu())
    vals = torch.cat(res).max(axis=2).values[:,0].cpu().detach().numpy()
    plt.clf()
    plt.scatter(locs[:,0],locs[:,1],c=vals)
    plt.colorbar()
    directory = f'{config.folder}/maps'
    if not os.path.isdir(directory):
        os.system(f'mkdir {directory}')
    plt.savefig(f'{directory}/%07d.png' % sample_number)
    config.writer.add_figure(f"value_map", plt.gcf(), global_step=sample_number)

def visualize_d4rl_generalization(config,model,dataset,sample_number,num=500000):
    maxes = dataset.max(axis=0)
    mins = dataset.min(axis=0)
    point_scale = maxes-mins
    random_points =np.random.rand(num,maxes.shape[0])
    points = random_points*point_scale+mins
    results = []
    for vs in tqdm(util.chunks(points,16)):
        results.append(model(torch.tensor(vs).cuda().float()).detach())

    catRes = torch.cat(results).max(axis=2)[0][:,0].cpu()
    scale = 100
    grid = np.zeros((scale,scale))
    for p,v in zip(random_points,catRes):
        loc = (p*scale).astype(np.int)
        grid[loc[1],loc[0]] = max(v,grid[loc[1],loc[0]])

    grid = np.flip(grid,axis=0)
    plt.clf()
    plt.imshow(grid)
    plt.colorbar()
    config.writer.add_figure(f"OOD_value_map", plt.gcf(), global_step=sample_number)


def visualize_reacher_model(config, dataset, model, sample_number):
    vals = []
    device = next(model.parameters()).device
    for dat in tqdm(dataset):
        pre, _, _, _, _, _, pre_tip_pos = dat
        val = model(torch.tensor([pre]).to(device)).max().item()
        vals.append([pre_tip_pos[0], pre_tip_pos[1], val])
    vals = np.stack(vals)
    plt.clf()
    sp = plt.scatter(vals[:, 0], vals[:, 1], c=vals[:, 2])
    plt.colorbar(sp)
    directory = f'{config.folder}/maps'
    if not os.path.isdir(directory):
        os.system(f'mkdir {directory}')
    plt.savefig(f'{directory}/{sample_number}.png')
    config.writer.add_figure(f"value_map",
                             plt.gcf(),
                             global_step=sample_number)


def visualize_house(config, model, house, floor, sample_number, gt=False):
    print(f'render sample: {sample_number} on {house.name}{floor}')
    if gt:
        forward_args = {'gt_head': True}
    else:
        forward_args = {}
    figs = build_map_gibson(config,
                            model,
                            house,
                            floor,
                            forward_args=forward_args)
    images = np.array([util.plt.fig2img(x) for x in figs])[..., :-1]
    # make grid and convert to torch format
    grid = torchvision.utils.make_grid(torch.tensor(images).permute(
        0, 3, 1, 2),
                                       nrow=5)
    name = f"value_map_{house.name}{floor}({house.data['split_tiny']})"
    if gt:
        name += "_multitask_gt"
    config.writer.add_image(name, grid, global_step=sample_number)


def render_visualization(config, sample_number):
    print(f'render sample: {sample_number}')
    vis_maps = build_map(config, sample_number)
    figs = []
    for i, vis_map in enumerate(vis_maps):
        fig = plt.Figure()
        ax = fig.subplots()
        vis_map = util.habitat.crop(vis_map)
        pos = ax.imshow(vis_map)
        fig.colorbar(pos, ax=ax)
        figs.append(fig)

    agg_map_uncropped = np.stack(vis_maps).max(0)
    agg_map = util.habitat.crop(agg_map_uncropped)
    fig = plt.Figure()
    ax = fig.subplots()
    pos = ax.imshow(agg_map)
    fig.colorbar(pos, ax=ax)
    figs.insert(0, fig)
    print(f'writing sample_number: {sample_number}')
    directory = f'{config.folder}/maps'
    if not os.path.isdir(directory):
        os.system(f'mkdir {directory}')
    np.save(f'{directory}/sample_number{sample_number}', agg_map_uncropped)
    plt.imsave(f'{directory}/sample_number{sample_number}_cropped.png',
               agg_map)
    # convert to images, remove alpha channel
    images = np.array([util.plt.fig2img(x) for x in figs])[..., :-1]
    # make grid and convert to torch format
    grid = torchvision.utils.make_grid(
        torch.tensor(images).permute(0, 3, 1, 2))
    config.writer.add_image(f"value_map", grid, global_step=sample_number)


def run_train(config, resume_from=-1):
    AVERAGED = config.AVERAGED_DQN_K > 0
    ENSEMBLE = config.ENSEMBLE_COUNT > 0
    torch.set_num_threads(1)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Using: {config.device}")
    houses_to_render = [("Allensville", 0), ("Beechwood", 1), ("Darden", 0), ("Merom", 1)]
    if config.DATASET == 'branch_updated':
        houses_to_render = [("Arkansaw", 1)]

    params = {'batch_size': 16, 'num_workers': 8, 'drop_last': True}
    if config.D4RL:
        print(f'Loading from {config.DATASET}')
        dat = pd.read_feather(config.DATASET)
        d4rl_observations = util.pd.multi_get(dat,'observations')
        d4rl_actions = dat['actions'].max()+1
        normalization = {}
        normalization['means'] = torch.tensor(d4rl_observations.mean(axis=0)).cuda().float()
        normalization['stds'] = torch.tensor(d4rl_observations.std(axis=0)).cuda().float()
        
        def lam(ind,dat=dat):
            line = dat.iloc[ind]
            before = multi_get(line,'observations').astype(np.float32)
            after = multi_get(line,'next_observations').astype(np.float32)
            act = line['actions'].astype(np.int64)
            rew = line['reward']
            term = rew
            ground_truth = np.NaN
            valid_mask = np.NaN
            if config.ONE_ACTION or config.VALUE_LEARNING:
                act = 0
            if config.VALUE_LEARNING:
                ground_truth = np.power(config.GAMMA,line['steps_to_goal'])
            return before, after, act, rew, term, ground_truth, valid_mask

        dataset = LambdaLoader(len(dat),lam)
        d4rl_inputs = dataset[0][0].shape[0]

    if not config.D4RL:
        print(f'Load data from {datafile}')
        print(f'Reward Ration: {dataset.reward_percentage()}')

    test_size = int(params['batch_size'] * 10)
    train_data, test_data = data.random_split( dataset, [len(dataset) - test_size, test_size])
    training_generator = data.DataLoader(train_data, **params, shuffle=True)
    eval_generator = data.DataLoader(test_data, **params, shuffle=False)

    # threshold at roughly 5 meters
    if config.DATASET in REACHER_DATASETS:
        known_gt_data = ReacherDataset(
            datafile,
            gamma=config.GAMMA,
            known=True,
            has_reward=(config.DATASET == 'reacher_replay'
                        or config.DATASET == 'reacher_replay_override'))
    elif config.DATASET == 'nobias':
        known_gt_data = HabitatQRandomizedDataset(datafile,
                                                  gamma=config.GAMMA,
                                                  panorama=config.PANORAMA,
                                                  known=True)

    print(len(train_data))

    # Training
    kwargs = {}
    if config.D4RL:
        kwargs['inputs'] = d4rl_inputs
        kwargs['actions'] = d4rl_actions
        kwargs['arch_dim'] = config.ARCH_DIM
        kwargs['tcc'] = config.TCC
        if config.POSITIONAL_ENCODING:
            # L = 10
            kwargs['inputs'] = d4rl_inputs*20
            kwargs['positional_encoding'] = True

    
    if AVERAGED or ENSEMBLE:
        models = [build_model(config,**kwargs) for _ in range(config.AVERAGED_DQN_K)]
        for m in models[1:]: m.eval()
    else:
        model = build_model(config,**kwargs)
        target_net = build_model(config,**kwargs)

    if config.D4RL:
        if AVERAGED:
            for m in models:
                m.means = normalization['means']
                m.stds = normalization['stds']
        else:
            model.means = normalization['means']
            model.stds = normalization['stds']
            target_net.means = normalization['means']
            target_net.stds = normalization['stds']

    if AVERAGED:
        # gradients only to first model
        optimizer = optim.Adam(models[0].parameters(), lr=config.LEARNING_RATE)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    def process_batch(batch,
                      compare_ground_truth=False,
                      batch_number=None,
                      forward_args={}):
        # batch x classes x actions
        before, after, act, rew, term, ground_truth, valid_mask = [
            x.to(config.device) for x in batch
        ]

        if AVERAGED:
            before_values = models[0](before, **forward_args)
        else:
            before_values = model(before, **forward_args)

        if config.D4RL:
            classes = 1
        else:
            classes = 5

        action_indices = act.view(-1, 1).repeat(1, classes)

        if config.DISTRIBUTIONAL and not compare_ground_truth:
            action_indices = action_indices.unsqueeze(2).repeat(1, 1,
                                                                2).unsqueeze(2)
        else:
            action_indices = action_indices.unsqueeze(2)


        # get the current predictions for q values of the actions taken
        # (batch,)
        Q_b = before_values.gather(2, action_indices).squeeze()


        if not compare_ground_truth:
            if AVERAGED:
                res = [m(after,**forward_args) for m in models[1:]]
                if config.AVERAGED_DQN_MIN:
                    after_values = torch.stack(res).min(axis=0)[0].detach()
                else:
                    after_values = torch.stack(res).mean(axis=0).detach()
                model_after_values = after_values
            else:
                after_values = target_net(after, **forward_args)
                # use his for double dqn
                model_after_values = model(after, **forward_args)
                # use this for regular dqn
                # model_afkter_values = after_values

            #select the best action acording to the online model
            # -1 x classes
            if config.DISTRIBUTIONAL:
                # select means if gaussian distributional
                model_after_values = model_after_values[..., 0]

            best_action_indices = model_after_values.argmax(-1)
            # best_actions = model_after_values.argmax(1,-1).view(-1, 1)

            # get q values of best action from next state
            # detach so no gradients flow back
            if config.DISTRIBUTIONAL:
                best_action_indices = best_action_indices.unsqueeze(2).repeat(
                    1, 1, 2).unsqueeze(2)
            else:
                best_action_indices = best_action_indices.unsqueeze(2)
            # -1 x classes
            Q_a = after_values.gather(2,
                                      best_action_indices).detach().squeeze()

            # remove after value for terminal states, so
            # only the reward is fitted
            if len(Q_a.shape) == 3:
                # extra dimension for distributional
                term=term.unsqueeze(2)
            Q_a = Q_a * (1 - term.float())

            if config.DISTRIBUTIONAL:
                # Q_a has after means and variances with terminal states masked out to 0
                Q_a[..., 0] += rew.float()
                mu_p = Q_a[..., 0] * config.GAMMA
                if config.LOSS_CLIP == 'rect':
                    mu_p = torch.clamp(mu_p, max=1, min=0)
                mu_q = Q_b[..., 0]

                mask = term == 1
                extended_mask = mask.repeat(1, 1, 2)
                extended_mask[:, :, 0] = False
                if config.LOG_SIGMA:
                    Q_a[extended_mask] = -10
                    if config.NO_VAR_DECAY:
                        sig_p = torch.exp(Q_a[..., 1])
                    else:
                        sig_p = torch.exp(Q_a[..., 1]) * config.GAMMA
                    sig_q = torch.exp(Q_b[..., 1])
                else:
                    Q_a[extended_mask] = config.SOFTPLUS_REWARD_VARIANCE
                    if config.NO_VAR_DECAY:
                        sig_p = (F.softplus(Q_a[..., 1]) +
                                 config.SOFTPLUS_LOWER_BOUND)
                    else:
                        sig_p = (F.softplus(Q_a[..., 1]) +
                                 config.SOFTPLUS_LOWER_BOUND) * config.GAMMA
                    sig_q = (F.softplus(Q_b[..., 1]) +
                             config.SOFTPLUS_LOWER_BOUND)

                if config.KL_BACKWARDS:
                    losses = KLD(mu_q, sig_q, mu_p, sig_p)
                else:
                    losses = KLD(mu_p, sig_p, mu_q, sig_q)
            else:  #Standard q learning
                if config.LINEAR:
                    learn_targets = rew.float() + (Q_a - 0.1)
                else:
                    learn_targets = rew.float() + config.GAMMA * Q_a
                if config.LOSS_CLIP == 'rect':
                    learn_targets = torch.clamp(learn_targets, max=1, min=0)
                losses = (0.5 * (Q_b - learn_targets)**2)
            if config.REMOVE_BEFORE_REWARD:
                losses = losses * valid_mask
        else:
            # for ground truth values
            if config.VALUE_LEARNING:
                mask = (1 - torch.isnan(ground_truth).int())
                gt = ground_truth.clone()
                gt[torch.isnan(ground_truth)] = 0
                losses = 0.5 * (Q_b * mask - gt.float())**2
            else:
                losses = 0.5 * (Q_b - ground_truth.float())**2
        loss = losses.mean()
        if torch.isnan(loss):
            # import pdb
            # pdb.set_trace()
            loss = torch.tensor(1, dtype=torch.float32)
            print('loss is nan')
        return loss

    metrics = {
        'losses': [],
        'eval_losses': [],
    }
    train_iterator = loopLoader(training_generator)
    if config.MULTI_TASK:

        def multi_loader():
            while True:
                yield next(train_iterator) + [False]
                yield next(multi_task_gt_iterator) + [True]

        iterator = multi_loader()
    else:
        iterator = train_iterator

    os.system(f'mkdir {config.folder}/models')

    if len(config.MODEL_INITILIZATION):
        snapshot = torch.load(config.MODEL_INITILIZATION,
                              map_location=config.device)
        print(f'Initilization model from: {config.MODEL_INITILIZATION}')
        # taking the weights for forward action only
        snapshot['model_state_dict']['top.4.weight'] = snapshot[
            'model_state_dict']['top.4.weight'].view(5, 3, -1)[:, 0, :]
        snapshot['model_state_dict']['top.4.bias'] = snapshot[
            'model_state_dict']['top.4.bias'].view(5, 3)[:, 0]
        model.load_state_dict(snapshot['model_state_dict'])

    # sample_number = -1
    sample_number =0 
    if resume_from > -1:
        # model_loc = f'{config.folder}/models/epoch{resume_from}.torch'
        model_loc = f'{config.folder}/models/sample{resume_from}.torch'
        snapshot = torch.load(model_loc, map_location=config.device)
        print(f'Loading model from: {model_loc}')
        model.load_state_dict(snapshot['model_state_dict'])
        optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        sample_number = resume_from + 1
    # test loading gt net
    if config.BOOTSTRAP:
        print('\n\nBOOTSTRAP\n\n')
        model_loc = f'logs/trained_gt_0.99/models/epoch99.torch'
        snapshot = torch.load(model_loc, map_location=config.device)
        print(f'Loading model from: {model_loc}')
        model.load_state_dict(snapshot['model_state_dict'])
        optimizer.load_state_dict(snapshot['optimizer_state_dict'])

    if config.AVERAGED_DQN_K == 0:
        target_net.load_state_dict(model.state_dict())
        target_net.eval()


    running_loss = None
    # for epoch in range(resume_from+1,100):
    if resume_from > -1:
        step_limit = 1e7
    else:
        step_limit = config.NUM_STEPS
    while sample_number < step_limit:
        # update target network if needed
        sample_number += 1
        # visualize_house(config,model,"Beechwood",sample_number)
        if sample_number % config.TARGET_UPDATE_INTERVAL == 0 and not AVERAGED:
            target_net.load_state_dict(model.state_dict())

        # set training mode
        if AVERAGED:
            models[0].set_train()
        else:
            model.set_train()

        optimizer.zero_grad()

        batch = next(iterator)
        ground_truth = config.TRAIN_ON_GROUND_TRUTH
        forward_args = {}
        if config.MULTI_TASK:
            ground_truth = batch[-1]
            batch = batch[:-1]
            if ground_truth:
                forward_args = {'gt_head': True}
        loss = process_batch(batch,
                             compare_ground_truth=ground_truth,
                             batch_number=sample_number,
                             forward_args=forward_args)
        loss.backward()
        if AVERAGED:
            # copy model params before gradient step on model 0
            for i in range(len(models)-1,0,-1):
                state = models[i-1].state_dict()
                models[i].load_state_dict(state)
        optimizer.step()
        # check for nan after optimizer step
        if not AVERAGED and torch.isnan(next(model.parameters())).any().item():
            import pdb
            pdb.set_trace()
        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = running_loss * 0.99 + loss.item() * 0.01
        print(
            f'\rbatch:{sample_number}/{config.NUM_STEPS} avg_loss: {running_loss}',
            end="")

        # if sample_number % 100 == 0:
        if sample_number % 1000 == 0:
            # config.writer.add_scalar('avg_q_loss/train', running_loss,
            config.writer.add_scalar('train_running_loss', running_loss,
                                     sample_number)
            config.writer.add_scalar('train_loss', loss.item(),
                                     sample_number)
            losses = [
                # process_batch(batch, compare_ground_truth=True).detach()
                process_batch(batch, compare_ground_truth=False).detach()
                for batch in eval_generator
            ]
            eval_loss = torch.stack(losses).mean().item()
            # config.writer.add_scalar('eval_loss_clipped', eval_loss,
            config.writer.add_scalar('val_loss', eval_loss,
                                     sample_number)
            

        # checkpoint and eval
        if sample_number % config.CHECKPOINT_INTERVAL == 0:
            mod = models[0] if AVERAGED else model
            torch.save(
                {
                    'sample_number': sample_number,
                    'model_state_dict': mod.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, f'{config.folder}/models/sample{sample_number}.torch')

            if config.DATASET in GIBSON_DATASETS or config.DATASET == 'real' or config.DATASET == 'real_new' or config.DATASET == 'real-550k':
                gts = [False]
                if config.MULTI_TASK:
                    gts = [False, True]
                '''
                for gt in gts:
                    for house, floor in houses_to_render:
                        visualize_house(config,
                                        model,
                                        house,
                                        floor,
                                        sample_number,
                                        gt=gt)
                '''
            elif config.D4RL:
                if AVERAGED:
                    visualize_d4rl(config,models[0],dataset,sample_number)
                    visualize_d4rl_generalization(config,models[0],d4rl_observations,sample_number)
                else:
                    visualize_d4rl(config,model,dataset,sample_number)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train q network')
    parser.add_argument('-g',
                        '--gpu',
                        dest='gpu',
                        default='0',
                        help='which gpu to run on')

    parser.add_argument('-r',
                        '--resume',
                        dest='resume',
                        action="store_true",
                        help='resume from last epoch?')

    parser.add_argument('-d',
                        '--delete',
                        dest='delete',
                        action="store_true",
                        help='delete stored tensorboard data')
    parser.add_argument('--vis',
            action='store_true',
            help='test vis')

    parser.add_argument('config', help='folder containing config file')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from experiment_config import ExperimentConfig

    config = ExperimentConfig(args.config,
                              device='cuda',
                              remove=args.delete,
                              resume=args.resume)

    with open(f'{config.folder}/log', "w") as text_file:
        text_file.write(f"Running with config ({str(config.cfg)})")

    if args.resume:
        models = os.popen(f'ls {config.folder}/models').read().split()

        if len(models) == 0:
            run_train(config)
        else:
            latest_model = max([int(n[6:-6]) for n in models])
            print(f"Resuming from: {latest_model}")
            run_train(config, latest_model)
    else:
        run_train(config)
