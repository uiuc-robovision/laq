import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import csv
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import os
import pdb
import time
from itertools import permutations
from absl import app
from absl import flags
import matplotlib.pyplot as plt
from gibson import BranchingDataset
from pathlib import Path
import sklearn
from sklearn import metrics
import random

FLAGS = flags.FLAGS
flags.DEFINE_integer('bottleneck_size', 8, 'Output dimension of CNN')
flags.DEFINE_integer('batch_size', 32, 'batch_size')
flags.DEFINE_integer('gpu', 5, 'Which GPU to use.')
flags.DEFINE_float('weight_decay', 0.00001, 'Weight decay in optimizer')
flags.DEFINE_float('scale', 10.0, 'Scale for one-hot things')
flags.DEFINE_string('model_path', '0', 'path from which to load model')
flags.DEFINE_boolean('annealing', True, 'gradually decrease freq of funny assignment')
flags.DEFINE_integer('annealing_freq', 1250, 'reduce anneal freq after x steps')
flags.DEFINE_integer('cnt', 2, 'cnt limit: min number of samples per bin in funny assignment')
flags.DEFINE_float('lr', 0.001, 'Scale for one-hot things')
flags.DEFINE_integer('step_size', 30, 'cnt limit: min number of samples per bin in funny assignment')
flags.DEFINE_integer('seed', -1, 'cnt limit: min number of samples per bin in funny assignment')
flags.DEFINE_string('logdir', 'repro', 'Name of tensorboard logdir')
flags.DEFINE_string('logdir_prefix', 'lam_runs/', 'Name of tensorboard logdir')


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        # dynamics model resnet
        self.resnet18_dynamics = models.resnet18(pretrained=True)
        self.modules_dynamics = list((self.resnet18_dynamics).children())[:-2]
        self.resnet18_dynamics = nn.Sequential(*self.modules_dynamics)
        
        self.resnet18_dynamics_psi = models.resnet18(pretrained=True)
        self.modules_dynamics_psi = list((self.resnet18_dynamics_psi).children())[:-2]
        self.resnet18_dynamics_psi = nn.Sequential(*self.modules_dynamics_psi)
 
        # freeze_resnet_phi
        (self.resnet18_dynamics).eval()
        for param in (self.resnet18_dynamics).parameters():
            param.requires_grad = False
                
        embedding_dim = 32
        self.embedding = nn.Embedding(num_embeddings=8,
                                      embedding_dim=embedding_dim)

        self.action_readout = nn.Sequential(
                nn.Linear(embedding_dim, 64), nn.ReLU(), nn.BatchNorm1d(64),
                nn.Linear(64, 64), nn.ReLU(), nn.BatchNorm1d(64),
                nn.Linear(64, 3))


        self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=256, stride=1, kernel_size=3),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(in_channels=256, out_channels=128, stride=1, kernel_size=3),
                nn.ReLU(),
                nn.BatchNorm2d(128)
            )

        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=256, out_channels=128, stride=2, kernel_size=4, padding=(1, 1)),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(in_channels=128, out_channels=128, stride=2, kernel_size=6, padding=(1, 1)),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(in_channels=128, out_channels=128, stride=2, kernel_size=6, padding=(2, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(in_channels=128, out_channels=128, stride=2, kernel_size=6, padding=(2, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(in_channels=128, out_channels=128, stride=2, kernel_size=8, padding=(3, 3)),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(in_channels=128, out_channels=3, stride=2, kernel_size=8, padding=(3, 3))
            )


    def forward(self, act, k, k_plus_one, k_plus_one_two, k_plus_x):

        # freeze resnets
        self.resnet18_dynamics.eval()

        all_actions = torch.arange(8).cuda()
        encoding = self.embedding(all_actions)

        y = encoding
        act1 = FLAGS.scale*encoding
        y = y.detach()

        # readout function
        y = y / y.norm(dim=1, keepdim=True)
        act_readout = self.action_readout(y)

        # encoder
        phi_k = self.resnet18_dynamics_psi(k)
        phi_k = self.encoder(phi_k)

        # upsample action
        act1 = act1.unsqueeze(2)
        act1 = act1.unsqueeze(3)
        act1 = act1.repeat(1, 4, 3, 3)

        # concat
        act1, phi_k = torch.broadcast_tensors(act1.unsqueeze(0), phi_k.unsqueeze(1))
        psi = torch.cat((act1, phi_k), 2)
        
        # decoder
        sz = psi.shape
        psi = psi.view(sz[0]*sz[1], sz[2], sz[3], sz[4])

        psi = self.decoder(psi)
        sz2 = psi.shape
        psi = psi.view(sz[0], sz[1], sz2[1], sz2[2], sz2[3])

        phi_k_plus_one = k_plus_one
        phi_k_plus_x = k_plus_x
        
        factor = 200 
        psi = factor*psi / psi.view(psi.shape[0], -1).norm(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        phi_k_plus_one = factor*phi_k_plus_one / phi_k_plus_one.view(phi_k_plus_one.shape[0], -1).norm(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        phi_k_plus_x = factor*phi_k_plus_x / phi_k_plus_x.view(phi_k_plus_x.shape[0], -1).norm(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        return psi, phi_k_plus_one, phi_k_plus_x, act_readout, encoding, encoding



def EnumerateLoss(psi, phi_k_plus_one, phi_k_plus_x, iteration=0, train=True):
    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(psi, phi_k_plus_one.unsqueeze(1).repeat(1, 8, 1, 1, 1))

    loss = loss.view(loss.shape[0], loss.shape[1], -1)
    loss = loss.mean(2)
    _loss, ind = torch.min(loss, 1)
    loss_np = loss.cpu().detach().numpy()
    ind_np = loss_np.argsort(0)
    taken = []

    do_ = True
    if FLAGS.annealing:
        anneal_every = int(iteration / FLAGS.annealing_freq) + 1
        if iteration % anneal_every != 0:
            do_ = False

    if train and do_:
        for i in range(loss_np.shape[1]):
            cnt = 0
            for j in ind_np[:,i]:
                if j not in taken:
                    ind[j] = i
                    taken += [j]
                    cnt += 1
                if cnt == FLAGS.cnt:
                    break

    loss = loss * torch.nn.functional.one_hot(ind, num_classes=FLAGS.bottleneck_size).float()
    loss = loss.sum(1)
    loss = loss.mean()

    return loss, ind


def vis_tensor(imgs):
    mu = torch.Tensor([0.485, 0.456, 0.406]).float().reshape((1,3,1,1)).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).float().reshape((1,3,1,1)).cuda()
    imgs = torch.clip(imgs * std + mu, 0, 1)
    return imgs

def train(model, device, train_loader, optimizer, epoch, val_loader, writer, iteration):

    model.train()

    print_every = 20
    val_every = 1000
    iteration_ = iteration

    train_loss = 0
    train_loss_ce = 0
    train_loss_encoding = 0
    train_acc = 0
    loss = 0
    iter_ = 0
    bins = [-2, -1, 0, 1, 2, 3, 4]
    hist = np.zeros((len(bins), 2))
    hist_val = np.zeros((len(bins), 2))
    hist_gs = np.zeros((FLAGS.bottleneck_size))
    hist_gs_val = np.zeros((FLAGS.bottleneck_size))
    purity = np.zeros((FLAGS.bottleneck_size, 3))
    encodings = []
    acts = []
    ys = []

    optimizer.zero_grad()
    
    for batch_idx, (k, k_plus_one, k_plus_one_two, k_plus_x, act, rew, term, gt, x) in enumerate(train_loader):

        k, k_plus_one, k_plus_one_two, k_plus_x, act = k.to(device), k_plus_one.to(device), k_plus_one_two.to(device), k_plus_x.to(device), act.to(device)
        
        # forward pass 
        psi, phi_k_plus_one, phi_k_plus_x, y, encoding, gumbel_softmax = model(act, k, k_plus_one, k_plus_one_two, k_plus_x) 
        encodings.append(encoding.detach().cpu().numpy()) 
        acts.append(act.detach().cpu().numpy()) 
        ys.append(y.detach().cpu().numpy())
        
        # losses
        loss, ind = EnumerateLoss(psi, phi_k_plus_one, phi_k_plus_x, iteration=iter_)
        train_loss += loss.item()

        # backprop
        loss.backward()
        if (batch_idx+1)%(256/FLAGS.batch_size) == 0:
            optimizer.step()
            optimizer.zero_grad()

        iter_ += 1

        if batch_idx % print_every == 0 and batch_idx != 0:
            
            iteration_ += 1

            # loss per batch
            train_loss /= (print_every + 1)
              
            # purity
            val_loss, purity = get_purity(val_loader, model, device)
            model.train()

            # save to tensorboard
            writer.add_scalar('Loss/train_--_mse_loss', train_loss, iteration_*print_every)
            writer.add_scalar('Loss/val_--_mse_loss', val_loss, iteration_*print_every)
            writer.add_scalar('Purity', purity, iteration_*print_every)

            # save image
            gt = k_plus_one[:1,...]
            pred = psi[:1, ind[0], ...]
            gt_pred = torch.cat([gt, pred], -1)
            writer.add_image('vis/gt_pred', vis_tensor(gt_pred)[0,...], iteration*print_every)
            
            # reset
            train_loss = 0
            train_loss_ce = 0
            train_loss_encoding = 0
            train_acc = 0
            encodings = []
            acts = []
            ys = []
            purity = np.zeros((FLAGS.bottleneck_size, 3))
            
            # save model
            if iteration_ % 100 == 0:
                file_name = os.path.join(FLAGS.logdir_prefix, FLAGS.logdir, 'model-{:d}.pth'.format(iteration_))
                Path(os.path.join(FLAGS.logdir_prefix, FLAGS.logdir)).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), file_name)

    return iteration_



def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def get_purity(val_loader, model, device):
    model.eval()
    val_loss, purity = 0, 0
    pred_, gt_ = [], []
    with torch.no_grad():
        for batch_idx, (k, k_plus_one, k_plus_one_two, k_plus_x, act, rew, term, gt, x) in enumerate(val_loader):
            k, k_plus_one, k_plus_one_two, k_plus_x, act = k.to(device), k_plus_one.to(device), k_plus_one_two.to(device), k_plus_x.to(device), act.to(device)

            # forward pass 
            psi, phi_k_plus_one, phi_k_plus_x, y, encoding, gumbel_softmax = model(act, k, k_plus_one, k_plus_one_two, k_plus_x) 
            
            # losses
            loss, ind = EnumerateLoss(psi, phi_k_plus_one, phi_k_plus_x, train=False)
            val_loss += loss.item()

            pred_.extend(list(ind.cpu().numpy()))
            gt_.extend(list(act.cpu().numpy()))

            if batch_idx == 25:
                break

        val_loss /= 26
        purity = purity_score(gt_, pred_)

        return val_loss, purity


def validate(model, device, val_loader, train_loader, print_every, hist_val, hist_gs_val):

    model.eval()

    val_loss = 0
    val_loss_ce = 0
    val_loss_encoding = 0
    val_acc = 0

    j = 0

    perms = list(permutations([0, 1, 2]))
    train_acc_permutation = [0] * len(perms)
    val_acc_permutation = [0] * len(perms)

    with torch.no_grad():
        for batch_idx, (k, k_plus_one, k_plus_one_two, k_plus_x, act, rew, term, gt, x) in enumerate(val_loader):

            k, k_plus_one, k_plus_one_two, k_plus_x, act = k.to(device), k_plus_one.to(device), k_plus_one_two.to(device), k_plus_x.to(device), act.to(device)
        
            # forward pass 
            psi, phi_k_plus_one, phi_k_plus_x, y, encoding, gumbel_softmax = model(act, k, k_plus_one, k_plus_one_two, k_plus_x)

            # losses
            triplet_loss, _ = EnumerateLoss(psi, phi_k_plus_one, phi_k_plus_x, train=False)
            val_loss += triplet_loss
            
            hist_val = 0
            hist_gs_val = 0

            if FLAGS.bottleneck_size == 3:

                # accuracy_permutation
                for i, perm in enumerate(perms):

                    # convert pred to this permutation
                    pred = torch.index_select(encoding, 1, (torch.LongTensor(list(perm))).cuda())

                    pred = pred.argmax(dim=1, keepdim=True)
                    val_acc_permutation[i] += pred.eq(act.view_as(pred)).sum().item()

            j += 1
            if j == print_every:
                break

    val_loss /= print_every
    val_loss_ce /= print_every
    val_loss_encoding /= print_every
    val_acc /= (print_every * FLAGS.batch_size)

    val_acc_permutation_ = max(val_acc_permutation)
    val_acc_permutation_ /= (print_every * FLAGS.batch_size)

    j = 0

    with torch.no_grad():
        for batch_idx, (k, k_plus_one, k_plus_one_two, k_plus_x, act, rew, term, gt, x) in enumerate(train_loader):

            k, k_plus_one, k_plus_one_two, k_plus_x, act = k.to(device), k_plus_one.to(device), k_plus_one_two.to(device), k_plus_x.to(device), act.to(device)
            
            # forward pass 
            psi, phi_k_plus_one, phi_k_plus_x, y, encoding, gumbel_softmax = model(act, k, k_plus_one, k_plus_one_two, k_plus_x)

            if FLAGS.bottleneck_size == 3:

                # accuracy_permutation
                for i, perm in enumerate(perms):

                    # convert pred to this permutation
                    pred = torch.index_select(encoding, 1, (torch.LongTensor(list(perm))).cuda())

                    pred = pred.argmax(dim=1, keepdim=True)
                    train_acc_permutation[i] += pred.eq(act.view_as(pred)).sum().item()

            j += 1
            if j == print_every:
                break

    train_acc_permutation_ = max(train_acc_permutation)
    train_acc_permutation_ /= (print_every * FLAGS.batch_size)

    return val_loss, val_loss_ce, val_loss_encoding, val_acc, train_acc_permutation_, val_acc_permutation_, hist_val, hist_gs_val, score_class_0, score_class_1, score_class_2, score_k_minus_2, score_k_minus_1, score_k_minus_0, score_k_plus_2, score_k_plus_3, score_k_plus_4

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

def main(argv):

    if FLAGS.seed != -1:
        set_seed(FLAGS.seed)

    torch.cuda.set_device(FLAGS.gpu)
    torch.set_num_threads(1)

    train_dataset = BranchingDataset('data.npy')
    train_loader = data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_dataset = BranchingDataset('data.npy')
    val_loader = data.DataLoader(val_dataset, batch_size=FLAGS.batch_size, drop_last=True)

    device = torch.device("cuda")
    model_ = model().to(device)
    optimizer = torch.optim.Adam(model_.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    
    if FLAGS.model_path != '0':
        model_.load_state_dict(torch.load(FLAGS.model_path))
        print("loaded model!")
  
    writer = SummaryWriter(FLAGS.logdir_prefix + FLAGS.logdir)
    scheduler = StepLR(optimizer, step_size=FLAGS.step_size, gamma=0.1)
    iteration = 0

    for epoch in range(1, 100):
        print("Train Epoch: ", epoch)
        iteration = train(model_, device, train_loader, optimizer, epoch, val_loader, writer, iteration)
        scheduler.step()

        
def calc_pr(gt, out, wt=None):
  if wt is None:
    wt = np.ones((gt.size,1))

  gt = gt.astype(np.float64).reshape((-1,1))
  wt = wt.astype(np.float64).reshape((-1,1))
  out = out.astype(np.float64).reshape((-1,1))

  gt = gt*wt
  tog = np.concatenate([gt, wt, out], axis=1)*1.
  ind = np.argsort(tog[:,2], axis=0)[::-1]
  tog = tog[ind,:]
  cumsumsortgt = np.cumsum(tog[:,0])
  cumsumsortwt = np.cumsum(tog[:,1])
  prec = cumsumsortgt / cumsumsortwt
  rec = cumsumsortgt / np.sum(tog[:,0])

  ap = voc_ap(rec, prec)
  return ap, rec, prec

def voc_ap(rec, prec):
  rec = rec.reshape((-1,1))
  prec = prec.reshape((-1,1))
  z = np.zeros((1,1)) 
  o = np.ones((1,1))
  mrec = np.vstack((z, rec, o))
  mpre = np.vstack((z, prec, z))
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])

  I = np.where(mrec[1:] != mrec[0:-1])[0]+1;
  ap = 0;
  for i in I:
    ap = ap + (mrec[i] - mrec[i-1])*mpre[i];
  return ap

if __name__ == '__main__':
    app.run(main)
