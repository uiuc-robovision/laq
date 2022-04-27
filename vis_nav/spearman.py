import torch
from torch.utils import data
from gibson import GibsonDataset
from HabitatDQNMultiAction import HabitatDQNMultiAction
from scipy import stats
from tqdm import tqdm
import numpy as np

def build_model(sigmoid=False, actions=3, extra_capacity=False, panorama=False, tcc=False, device=None):
    sigmoid = sigmoid 
    model = HabitatDQNMultiAction(
        actions,
        5,
        extra_capacity=extra_capacity, 
        panorama=panorama, 
        tcc=tcc)
    return model.to(device)

torch.cuda.set_device(0)
device = torch.device("cuda")
params = {'batch_size': 16, 'num_workers': 8, 'drop_last': True}

def loopLoader(loader):
    i = iter(loader)
    while True:
        try:
            yield next(i)
        except StopIteration:
            print("reset iterator")
            i = iter(loader)
            
model = build_model(extra_capacity=True, device=device)
model_loc = f'ground_truth_actions/models/sample300000.torch'
snapshot = torch.load(model_loc, map_location=device)
print(f'Loading model from: {model_loc}')
model.load_state_dict(snapshot['model_state_dict'])
model.eval()

ckpt_nums = np.arange(60)*500+500
spearmans_latent = []
for ckpt in tqdm(ckpt_nums):
    datafile = 'data.npy'
    dataset = GibsonDataset(datafile,
                            panorama=False,
                            class_label='all',
                            inverse_action=True,
                            random_actions=False,
                            value_learning=False,
                            reward_dist=1,
                            one_action=False,
                            prev_images=False)

    train_data = dataset
    training_generator = data.DataLoader(train_data, **params, shuffle=False)
    iterator = loopLoader(training_generator)
    
    msePixelEnumerateBranching30k_model = build_model(actions=8, extra_capacity=True, device=device)
    model_loc = f'configs/experiments/real_data/models/sample{ckpt}.torch'
    snapshot = torch.load(model_loc, map_location=device)
    print(f'Loading model from: {model_loc}')
    msePixelEnumerateBranching30k_model.load_state_dict(snapshot['model_state_dict'])
    msePixelEnumerateBranching30k_model.eval()


    Q_bs, Q_bs_msePixelEnumerateBranching30k = [], []
    for i in tqdm(range(200)):
        batch = next(iterator)

        forward_args = {}
        before, after, act, rew, term, ground_truth, valid_mask = [
                    x.to(device) for x in batch
                ]
        classes = 5

        before_values = model(before, **forward_args)
        Q_b, _ = torch.max(before_values, 2)
        Q_b = Q_b[:, 0]
        Q_b = Q_b.flatten()
        Q_bs.append(Q_b.cpu().detach())

        before_values_msePixelEnumerateBranching30k = msePixelEnumerateBranching30k_model(before, **forward_args)
        Q_b_msePixelEnumerateBranching30k, _ = torch.max(before_values_msePixelEnumerateBranching30k, 2)
        Q_b_msePixelEnumerateBranching30k = Q_b_msePixelEnumerateBranching30k[:, 0]
        Q_b_msePixelEnumerateBranching30k = Q_b_msePixelEnumerateBranching30k.flatten()
        Q_bs_msePixelEnumerateBranching30k.append(Q_b_msePixelEnumerateBranching30k.cpu().detach())

    Q_bs = torch.stack(Q_bs).flatten()
    Q_bs_msePixelEnumerateBranching30k = torch.stack(Q_bs_msePixelEnumerateBranching30k).flatten()
    spearmans_latent.append(stats.spearmanr(Q_bs.cpu().detach(), Q_bs_msePixelEnumerateBranching30k.cpu().detach()).correlation)
    np.save(f'spearmans.npy', np.array(spearmans_latent))
    print(spearmans_latent[-1])
