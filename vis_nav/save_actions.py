import numpy as np
import torch
import torch.nn as nn
import model
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"]='0'
data = np.load('data.npy')

def imageNetTransformPIL(size=224):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform = imageNetTransformPIL()

def load_images(paths):
    return torch.stack([transform(Image.open(path[23:] +'/0.jpg')) for path in paths])


model = model.model()
dir_ = 'repro'
model_path = f'lam_runs/{dir_}/model-6100.pth'
model.load_state_dict(torch.load(model_path,map_location='cpu'))
model.eval()
model.to('cuda')


def EnumerateLoss(psi, phi_k_plus_one, phi_k_plus_x, iteration=0, train=True):
    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(psi, phi_k_plus_one.unsqueeze(1).repeat(1, 8, 1, 1, 1))
    
    loss = loss.view(loss.shape[0], loss.shape[1], -1)
    loss = loss.mean(2)
    _loss, ind = torch.min(loss, 1)
    
    return ind


zeros = torch.zeros(50, 3, 224, 224).to('cuda')
actions = []
for i in tqdm(range(0, 40000, 100)):
    be = load_images(data[i:i+100,0]).to('cuda')
    ae = load_images(data[i:i+100,8]).to('cuda')
    
    with torch.no_grad():
        psi, phi_k_plus_one, phi_k_plus_x, _, _, _ = model(zeros,be,ae,zeros,zeros)
    
    acts = EnumerateLoss(psi, phi_k_plus_one, phi_k_plus_x)
    
    actions.extend(list(acts.squeeze().cpu().detach().numpy()))


actions = np.array(actions)
data[:, -1] = actions.astype('U77')
np.save('data_latentActs.npy', data)
