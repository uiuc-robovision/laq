import torch, os
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
from d4rl_dataset import kitchenDataset 
from tqdm import tqdm
from sklearn import metrics
import random

FLAGS = flags.FLAGS

flags.DEFINE_integer('gpu', 4, 'Which GPU to use.')
flags.DEFINE_string('logdir', 'repro', 'Name of tensorboard logdir')
flags.DEFINE_string('logdir_prefix', 'lam_runs/', 'Name of tensorboard logdir')
flags.DEFINE_integer('batch_size', 128, 'batch_size')
flags.DEFINE_float('weight_decay', 0.00001, 'Weight decay in optimizer')
flags.DEFINE_float('lr', 1e-4, 'Learning rate')
flags.DEFINE_integer('lr_decay_every', 500000, 'Learning rate')
flags.DEFINE_integer('step_size', 1, 'number of epochs to wait before dropping lr')
flags.DEFINE_string('model_path', '0', 'path from which to load model')
flags.DEFINE_integer('bottleneck_size', 64, 'size of bottleneck')
flags.DEFINE_boolean('no_greedy_assignment', False, 'if want to turn off greedy assignment')
flags.DEFINE_boolean('annealing', True, 'gradually decrease freq of funny assignment')
flags.DEFINE_integer('annealing_freq', 5000, 'reduce anneal freq after x steps')
flags.DEFINE_integer('stop_after', 10000000, 'number of iterations to stop training after')
flags.DEFINE_integer('iteration_start', 0, 'number of iterations to start at if loading model')
flags.DEFINE_integer('cnt', 2, 'cnt limit: min number of samples per bin in funny assignment')
flags.DEFINE_integer('embedding_dim', 64, 'embedding dimension')
flags.DEFINE_integer('arch_mul', 48, 'multiply hidden dimensions by')
flags.DEFINE_boolean('set_seed', True, 'set seed')

class ForwardModel(nn.Module):
    def __init__(self):
        super(ForwardModel, self).__init__()
        
        activation = nn.ReLU()
        
        embedding_dim = FLAGS.embedding_dim
        self.embedding = nn.Embedding(num_embeddings=FLAGS.bottleneck_size, 
                                      embedding_dim=embedding_dim)

        hidden_size1 = 8 * FLAGS.arch_mul
        hidden_size2 = 16 * FLAGS.arch_mul
        hidden_size3 = 8 * FLAGS.arch_mul
        
        self.encoder = nn.Sequential(
            nn.Linear(24, hidden_size1), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size1),
            nn.Linear(hidden_size1, hidden_size2), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size2),
            nn.Linear(hidden_size2, hidden_size2), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size2),
            nn.Linear(hidden_size2, hidden_size2), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size2),
            nn.Linear(hidden_size2, FLAGS.embedding_dim), 
            nn.ReLU(),
            nn.BatchNorm1d(FLAGS.embedding_dim)
        )
        
        hidden_size4 = 16 * FLAGS.arch_mul
        hidden_size5 = 8 * FLAGS.arch_mul
        
        self.decoder = nn.Sequential(
            nn.Linear(FLAGS.embedding_dim * 2, hidden_size4), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size4),
            nn.Linear(hidden_size4, hidden_size4), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size4),
            nn.Linear(hidden_size4, hidden_size4), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size4),
            nn.Linear(hidden_size4, hidden_size5), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size5),
            nn.Linear(hidden_size5, 24), 
        )
        
        
    def forward(self, curr_states, next_states):

        all_actions = torch.arange(FLAGS.bottleneck_size).cuda()
        encoding = self.embedding(all_actions)
        
        y = encoding
        act1 = encoding
        
        phi_k = self.encoder(curr_states.float())

        assert act1.shape[1] <= phi_k.shape[1]
        rep = int(phi_k.shape[1] / act1.shape[1])
        act1 = act1.repeat(1, rep)
        act1, phi_k = torch.broadcast_tensors(act1.unsqueeze(0), phi_k.unsqueeze(1))
        
        psi = torch.cat((act1, phi_k), 2)
        sz = psi.shape
        psi = psi.view(sz[0]*sz[1], sz[2])
        psi = self.decoder(psi)
        sz2 = psi.shape
        psi = psi.view(sz[0], sz[1], sz2[1])
        return psi, encoding
    
def log_aps(writer, iteration, acts, encodings, ys):
    acts = np.concatenate(acts, 0)
    encodings = np.concatenate(encodings, 0)
    ys = np.concatenate(ys, 0)
    num_actions = ys.shape[1]
    aps = np.zeros((num_actions, num_actions))
    aps_y = np.zeros((num_actions, num_actions))
    for i in range(num_actions):
      for j in range(num_actions):
        ap, _, __ = calc_pr(acts == j, encodings[:,i])
        aps[i,j] = ap[0]
    aps = np.max(aps, 0)
    for i in range(num_actions):
      for j in range(num_actions):
        ap, _, __ = calc_pr(acts == j, ys[:,i])
        aps_y[i,j] = ap[0]
    aps_y = np.max(aps_y, 0)
            
    for i, p in enumerate(aps):
        writer.add_scalar('aps/train_{:02d}'.format(i), p, iteration)
        print(f'                   aps/{i:02d} [{iteration:6d}]: {p:0.8f}')
    
    for i, p in enumerate(aps_y):
        writer.add_scalar('aps_y/train_{:02d}'.format(i), p, iteration)
        print(f'                 aps_y/{i:02d} [{iteration:6d}]: {p:0.8f}')


def log(writer, optimizer, iteration,
        losses, baseline_losses, assignments):
    print('')
    
    ks = ['lr', 'mse_loss', 'baseline_loss']
    vs = [optimizer.param_groups[0]['lr'], 
          np.mean(losses), np.mean(baseline_losses)]
    
    for k, v in zip(ks, vs):
        print('{:>25s} [{:6d}]: {:0.8f}'.format(k, iteration, v))
        writer.add_scalar(f'loss/{k}', v, iteration)
    
    assignments = np.concatenate(assignments, 0)
    writer.add_histogram('vis/selection_histogram', assignments.ravel(), iteration)
    dist, _ = np.histogram(assignments, bins=np.arange(FLAGS.bottleneck_size+1))
    dist = dist / np.sum(dist)
    print('{:>25s} [{:6d}]: {:s}'.format('selection_distribution', iteration, str(dist)))

def loss_fn(mse_loss, psi, o_t, o_tm1, iteration, train=True):
    # losses
    baseline_loss = mse_loss(o_tm1.float(), o_t.float())
    baseline_loss = baseline_loss.mean()
    
    loss = mse_loss(psi.float(), o_t.unsqueeze(1).repeat(1, FLAGS.bottleneck_size, 1).float())
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
        
    if train and not FLAGS.no_greedy_assignment and do_:
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
    
    return loss, baseline_loss, ind
        

def train(model, optimizer, epoch, device, train_loader, val_loader,
          train_writer, val_writer, iteration, scheduler):
    np.set_printoptions(precision=3, suppress=True)

    model.train()

    print_every = 2500
    val_every = 50000
    save_every = 200000

    train_loss, train_act_loss, train_baseline_loss, train_acc, train_loc_l1, \
        train_loc_loss = [], [], [], [], [], []
    encodings, acts, ys, assignments = [], [], [], []
    
    mse_loss = nn.MSELoss(reduction='none')
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()

    for batch_idx, (o_tm1, o_t) in tqdm(enumerate(train_loader)):
            
        o_tm1, o_t = o_tm1.to(device), o_t.to(device)
                    
        # forward pass 
        psi, encoding = model(o_tm1, o_t)
        
        # losses
        loss, baseline_loss, assignment = loss_fn(mse_loss, psi, o_t, o_tm1, iteration, train=True)

        # compute the y and the loc_readout using the assignment
        encoding = encoding[assignment, :]

        # Interpretable metrics
        encodings.append(encoding.detach().cpu().numpy()) 
        assignments.append(assignment.detach().cpu().numpy())
        
        train_loss += [loss.item()]
        train_baseline_loss += [baseline_loss.item()]

        # backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        iteration += 1
       
        if iteration % print_every == 0 and batch_idx != 0:
            log(train_writer, optimizer, iteration, train_loss, train_baseline_loss, assignments)
                        
            train_loss, train_act_loss, train_baseline_loss, train_acc, train_loc_l1, \
                train_loc_loss = [], [], [], [], [], []
            assignments, encodings, acts, ys = [], [], [], []
            
        # save model
        if iteration % save_every == 0:
            file_name = os.path.join(FLAGS.logdir_prefix, FLAGS.logdir, 'model-{:d}.pth'.format(iteration))
            torch.save(model.state_dict(), file_name)
        
        # validation
        if iteration % val_every == 0:
            val_loss,  val_baseline_loss, \
            val_encodings, val_o_t, val_psi, val_assignments \
                = validate(model, device, val_loader, 25)
            
            model.train()
            
            log(val_writer, optimizer, iteration, val_loss, val_baseline_loss, val_assignments)
            
            val_encodings = np.concatenate(val_encodings, 0)
            val_writer.add_histogram('vis/encoding_histogram', 
                                     val_encodings.ravel(), iteration)
            val_writer.add_histogram('vis/encoding_argmax_histogram', 
                                     val_encodings.argmax(axis=1), iteration)
            
        if iteration % FLAGS.lr_decay_every == 0:
            scheduler.step()
            print("done step")

        if iteration == FLAGS.stop_after:
            return -1
            
    return iteration


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def validate(model, device, val_loader, print_every):
    model.eval()
    j = 0
    
    train_loss, train_act_loss, train_baseline_loss, train_acc = [], [], [], []
    assignments, encodings, acts, ys = [], [], [], []
    _pred, _gt = [], []
    
    mse_loss = nn.MSELoss(reduction='none')
    cross_entropy_loss = nn.CrossEntropyLoss()

    with torch.no_grad():
        
        for batch_idx, (o_tm1, o_t) in enumerate(val_loader):
            
            o_tm1, o_t = o_tm1.to(device), o_t.to(device)

            # forward pass 
            psi, encoding = model(o_tm1, o_t)
            loss, baseline_loss, assignment = loss_fn(mse_loss, psi, o_t, o_tm1, 0, train=False)

            # compute the y and the loc_readout using the assignment
            encoding = encoding[assignment, :]
            encodings.append(encoding.detach().cpu().numpy()) 
            assignments.append(assignment.detach().cpu().numpy())
            
            train_loss += [loss.item()]
            train_baseline_loss += [baseline_loss.item()]
            
            j += 1
            if j == print_every:
                break
    
    return train_loss, train_baseline_loss, \
            encodings, o_t, psi, assignments 


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

    if FLAGS.set_seed:
        set_seed()

    torch.cuda.set_device(FLAGS.gpu)
    torch.set_num_threads(1)
    
    train_dataset = kitchenDataset(train=True)
    train_loader = data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, 
                                   shuffle=True, num_workers=2, drop_last=True)
    
    val_dataset = kitchenDataset(train=False)
    val_loader = data.DataLoader(val_dataset, batch_size=FLAGS.batch_size,  
                                 num_workers=0, drop_last=True)
    
    # Get total number of actions.
    print('bottleneck_size: ', FLAGS.bottleneck_size)

    device = torch.device("cuda")
    model = ForwardModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, 
                                 weight_decay=FLAGS.weight_decay)
 
    if FLAGS.model_path != '0':
        model.load_state_dict(torch.load(FLAGS.model_path, device))
        print("loaded model!")

    train_writer = SummaryWriter(FLAGS.logdir_prefix + FLAGS.logdir + '/train/', flush_secs=60)
    val_writer = SummaryWriter(FLAGS.logdir_prefix + FLAGS.logdir + '/val/', flush_secs=60)
       
    scheduler = StepLR(optimizer, step_size=FLAGS.step_size, gamma=0.1)
    iteration = FLAGS.iteration_start

    for epoch in range(1, 1000000):
        print("Train Epoch: ", epoch)
        iteration = train(model, optimizer, epoch, device, train_loader,
                          val_loader, train_writer, val_writer, iteration,
                          scheduler)
        if iteration == -1:
            break



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

