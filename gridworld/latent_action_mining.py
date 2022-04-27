import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import csv
import numpy as np
import os
import pdb
import time
from absl import app
from absl import flags
import matplotlib.pyplot as plt
# from rl_unplugged import atari_shuffle_test
from tqdm import tqdm
import pathlib
from sklearn import metrics

FLAGS = flags.FLAGS

flags.DEFINE_integer('gpu', 4, 'Which GPU to use.')
flags.DEFINE_string('logdir', 'runs_debug', 'Name of tensorboard logdir')
flags.DEFINE_string('logdir_prefix', './output-grid/', 'Name of tensorboard logdir')
flags.DEFINE_integer('batch_size', 64, 'batch_size')
flags.DEFINE_float('weight_decay', 0.00001, 'Weight decay in optimizer')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('step_size', 1, 'number of epochs to wait before dropping lr')
flags.DEFINE_float('m', 1.0, 'Multiplier for cross entropy loss between y and act')
flags.DEFINE_string('model_path', '0', 'path from which to load model')
flags.DEFINE_boolean('ignore_act', False, 'whether to ignore actions')
flags.DEFINE_integer('bottleneck_size', 2, 'size of bottleneck')
flags.DEFINE_boolean('use_gt_act', False, 'whether to use gt acts instead of pred')
flags.DEFINE_boolean('pred_multiple_act', False, 'whether to predict multiple actions as opposed to one')
flags.DEFINE_boolean('softmax2d', False, 'whether to use a 2d softmax')
flags.DEFINE_boolean('vertical_loss', False, 'whether to have mse loss only on vertical strip of agent in Freeway')
flags.DEFINE_boolean('last_four_frames', False, 'whether to use the last four frames instead of last two')
flags.DEFINE_boolean('diff', False, 'whether to predict the difference as opposed to I_t+1')
flags.DEFINE_boolean('bigger_spatial_dim', False, 'instead of convolving to 64x7x7, convolve to 64x16x16')
flags.DEFINE_boolean('diff_loss', False, 'loss just on non-zero pixels in difference')
flags.DEFINE_boolean('leaky_relu', False, 'leaky_relu as opposed to relu')
flags.DEFINE_boolean('batch_norm', False, 'whether to use batch norm')
flags.DEFINE_boolean('no_greedy_assignment', False, 'if want to turn off greedy assignment')
flags.DEFINE_boolean('repeat_act', False, 'if want to turn off greedy assignment')

def vis_tensor(imgs):
    mu = torch.Tensor([0.485, 0.456, 0.406]).float().reshape((1,3,1,1)).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).float().reshape((1,3,1,1)).cuda()
    imgs = torch.clip(imgs * std + mu, 0, 1)
    return imgs

class GridDataset(data.Dataset):
    def __init__(self, mode='train'):
        print('Reload')
        dt = np.load('gridworld/data.npy')
        # remove dummy transitions
        dt = dt[dt[:,-1] != 1]
        if mode == 'train':
            dt = dt[:250000,:]
        else:
            dt = dt[250000:,:]
        self.dt = dt

    def __len__(self):
        return self.dt.shape[0]

    def __getitem__(self, index):
        state, act, next_state = self.dt[index, :3]
        state = torch.Tensor([state // 8, state % 8]).float() - 3.5
        next_state = torch.Tensor([next_state // 8, next_state % 8]).float() - 3.5
        action = torch.Tensor([act]).long()
        return state, action, next_state, state, next_state

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

class EnumerateForwardModel(nn.Module):
    def __init__(self, num_actions):
        super(EnumerateForwardModel, self).__init__()
        embedding_dim = 64
        self.embedding = nn.Embedding(num_embeddings=FLAGS.bottleneck_size,
                                      embedding_dim=embedding_dim)
        state_dim = 2
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.BatchNorm1d(64), 
            nn.Linear(64, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 64), nn.ReLU(), nn.BatchNorm1d(64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64+embedding_dim, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, state_dim)
        )

        if FLAGS.batch_norm and FLAGS.bottleneck_size != 1:
            self.action_readout = nn.Sequential(
                nn.Linear(embedding_dim, 64), nn.ReLU(), nn.BatchNorm1d(64),
                nn.Linear(64, 64), nn.ReLU(), nn.BatchNorm1d(64),
                nn.Linear(64, num_actions))
        else:
            self.action_readout = nn.Sequential(
                nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(64, num_actions))
        
        # agent location readoit function
        if FLAGS.batch_norm and FLAGS.bottleneck_size != 1:
            self.agent_location_readout = nn.Sequential(
                nn.Linear(embedding_dim, 64), nn.ReLU(), nn.BatchNorm1d(64),
                nn.Linear(64, 64), nn.ReLU(), nn.BatchNorm1d(64),
                nn.Linear(64, state_dim))
        else:
            self.agent_location_readout = nn.Sequential(
                nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(64, state_dim))

    def forward(self, state, action, next_state):
        all_actions = torch.arange(FLAGS.bottleneck_size).cuda()
        encoding = self.embedding(all_actions)
        
        y = encoding
        act1 = encoding

        y = y.detach()
        y = y / y.norm(dim=1, keepdim=True)  # [1024, 18]
        act_readout = self.action_readout(y)
        loc_readout = self.agent_location_readout(y)

        phi_k = self.encoder(state)
        act1, phi_k = torch.broadcast_tensors(act1.unsqueeze(0), phi_k.unsqueeze(1))

        psi = torch.cat((act1, phi_k), 2)
        sz = psi.shape
        psi = self.decoder(psi.view(-1, sz[-1]))
        sz2 = psi.shape
        psi = psi.view(sz[0], sz[1], sz2[-1])

        return psi, encoding, act_readout, loc_readout
    
def log_aps(writer, iteration, acts, encodings, ys):
    acts = np.concatenate(acts, 0)
    encodings = np.concatenate(encodings, 0)
    ys = np.concatenate(ys, 0)
    num_actions = ys.shape[1]
    aps = np.zeros((num_actions, num_actions))
    aps_y = np.zeros((num_actions, num_actions))
    if not FLAGS.pred_multiple_act and not FLAGS.softmax2d:
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
            
    if not FLAGS.pred_multiple_act and not FLAGS.softmax2d:
        for i, p in enumerate(aps):
            writer.add_scalar('aps/train_{:02d}'.format(i), p, iteration)
            print(f'                   aps/{i:02d} [{iteration:6d}]: {p:0.8f}')
    
    for i, p in enumerate(aps_y):
        writer.add_scalar('aps_y/train_{:02d}'.format(i), p, iteration)
        print(f'                 aps_y/{i:02d} [{iteration:6d}]: {p:0.8f}')


def log(writer, optimizer, iteration,
        losses, baseline_losses, act_losses, train_accs, loc_losses, loc_l1, 
        assignments):
    print('')
    
    ks = ['lr', 'mse_loss', 'baseline_loss', 'action_loss', 'action_acc',
          'loc_loss', 'loc_l1']
    vs = [optimizer.param_groups[0]['lr'], 
          np.mean(losses), np.mean(baseline_losses), 
          np.mean(act_losses), np.mean(train_accs), 
          np.mean(loc_losses), np.mean(loc_l1)]
    
    for k, v in zip(ks, vs):
        print('{:>25s} [{:6d}]: {:0.8f}'.format(k, iteration, v))
        writer.add_scalar(f'loss/{k}', v, iteration)
    
    assignments = np.concatenate(assignments, 0)
    writer.add_histogram('vis/selection_histogram', assignments.ravel(), iteration)
    dist, _ = np.histogram(assignments, bins=np.arange(FLAGS.bottleneck_size+1))
    dist = dist / np.sum(dist)
    print('{:>25s} [{:6d}]: {:s}'.format('selection_distribution', iteration, str(dist)))

def loss_fn(mse_loss, psi, o_t, o_tm1, train=True):
    # losses
    if FLAGS.diff:
        o_t = o_t - o_tm1
    baseline_loss = mse_loss(o_tm1, o_t)
    baseline_loss = baseline_loss.mean()
    loss = mse_loss(psi, o_t.unsqueeze(1).repeat(1, FLAGS.bottleneck_size, 1))
    loss = loss.view(loss.shape[0], loss.shape[1], -1)
    loss = loss.mean(2) # [B x A]
    _, ind = torch.min(loss, 1)
    loss_np = loss.cpu().detach().numpy()
    ind_np = loss_np.argsort(0) 
    taken = []
    if train and not FLAGS.no_greedy_assignment:
        for i in range(loss_np.shape[1]):
            cnt = 0
            for j in ind_np[:,i]:
                if j not in taken:
                    ind[j] = i
                    taken += [j]
                    cnt += 1
                if cnt == 2:
                    break
    loss = loss * torch.nn.functional.one_hot(ind, num_classes=FLAGS.bottleneck_size).float()
    loss = loss.sum(1)
    loss = loss.mean()
    return loss, baseline_loss, ind
        

def train(model, optimizer, epoch, device, train_loader, val_loader,
          train_writer, val_writer, iteration, num_actions):
    np.set_printoptions(precision=3, suppress=True)

    model.train()

    print_every = 200
    val_every = 2000
    save_every = 5000

    train_loss, train_act_loss, train_baseline_loss, train_acc, train_loc_l1, \
        train_loc_loss = [], [], [], [], [], []
    encodings, acts, ys, assignments = [], [], [], []
    
    mse_loss = nn.MSELoss(reduction='none')
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()

    for batch_idx, (o_tm1, a_tm1, o_t, s_tm1, s_t) in tqdm(enumerate(train_loader)):

        o_tm1, a_tm1, o_t = o_tm1.to(device), a_tm1.to(device), o_t.to(device)
        s_tm1, s_t = s_tm1.float().to(device), s_t.float().to(device)
                    
        # forward pass 
        psi, encoding, y, loc_readout = model(o_tm1, a_tm1, o_t)
        
        # losses
        loss, baseline_loss, assignment = loss_fn(mse_loss, psi, o_t, o_tm1, train=True)

        # compute the y and the loc_readout using the assignment
        y = y[assignment]
        encoding = encoding[assignment, :]
        loc_readout = loc_readout[assignment]
        
        loss_act = cross_entropy_loss(y, a_tm1[:,0].to(torch.long))
        loss_loc = mse_loss(loc_readout.squeeze(1), s_t - s_tm1).mean()
        
        # Interpretable metrics
        loc_l1_dist = (loc_readout.squeeze(1) - (s_t - s_tm1)).abs().mean()
        pred = y.argmax(dim=1, keepdim=True)

        encodings.append(encoding.detach().cpu().numpy()) 
        acts.append(a_tm1.detach().cpu().numpy())
        assignments.append(assignment.detach().cpu().numpy())
        ys.append((F.softmax(y, dim=1)).detach().cpu().numpy())
        
        train_loss += [loss.item()]
        train_baseline_loss += [baseline_loss.item()]
        train_act_loss += [loss_act.item()]
        train_loc_loss += [loss_loc.item()]
        
        train_acc += [pred.eq(a_tm1.view_as(pred)).float().mean().item()]
        train_loc_l1 += [loc_l1_dist.item()]
        
        loss += FLAGS.m * loss_act
        loss += FLAGS.m * loss_loc

        # backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        iteration += 1
       
        if batch_idx % print_every == 0 and batch_idx != 0:
            log(train_writer, optimizer, iteration, train_loss, train_baseline_loss, 
                train_act_loss, train_acc,
                train_loc_loss, train_loc_l1, assignments)
            log_aps(train_writer, iteration, acts, encodings, ys)
                        
            train_loss, train_act_loss, train_baseline_loss, train_acc, train_loc_l1, \
                train_loc_loss = [], [], [], [], [], []
            assignments, encodings, acts, ys = [], [], [], []
            
            # gt = o_t[:1,...]
            # pred = psi[:1, assignment[0], ...]
            # gt_pred = torch.cat([gt, pred], -1)
            # train_writer.add_image('vis/gt_pred', vis_tensor(gt_pred)[0,...], iteration)
            
        # save model
        if iteration % save_every == 0:
            file_name = os.path.join(FLAGS.logdir_prefix, FLAGS.logdir, 'model-{:d}.pth'.format(iteration))
            pathlib.Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), file_name)
        
        # validation
        if iteration % val_every == 0:
            val_loss,  val_baseline_loss, val_act_loss, val_act_acc, \
            val_loc_loss, val_loc_l1, \
            val_encodings, val_acts, val_ys, val_o_t, val_psi, val_assignments, purity \
                = validate(model, device, val_loader, 25)
            
            model.train()
            
            log(val_writer, optimizer, iteration, val_loss, val_baseline_loss, 
                val_act_loss, val_act_acc, val_loc_loss, val_loc_l1, val_assignments)
            log_aps(val_writer, iteration, val_acts, val_encodings, val_ys)
            val_writer.add_scalar('loss/purity', purity, iteration)
            
            # gt = val_o_t[:1,...]
            # pred = val_psi[:1, val_assignments[-1][0], ...]
            # gt_pred = torch.cat([gt, pred], -1)
            # val_writer.add_image('vis/gt_pred', vis_tensor(gt_pred)[0,...], iteration)
            
            val_encodings = np.concatenate(val_encodings, 0)
            val_writer.add_histogram('vis/encoding_histogram', 
                                     val_encodings.ravel(), iteration)
            val_writer.add_histogram('vis/encoding_argmax_histogram', 
                                     val_encodings.argmax(axis=1), iteration)
            
    return iteration



def validate(model, device, val_loader, print_every):
    model.eval()
    j = 0
    
    train_loss, train_act_loss, train_baseline_loss, train_acc, train_loc_l1, \
        train_loc_loss = [], [], [], [], [], []
    assignments, encodings, acts, ys = [], [], [], []
    
    mse_loss = nn.MSELoss(reduction='none')
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    _pred, _gt = [], []

    with torch.no_grad():
        
        for batch_idx, (o_tm1, a_tm1, o_t, s_tm1, s_t) in enumerate(val_loader):
            
            o_tm1, a_tm1, o_t = o_tm1.to(device), a_tm1.to(device), o_t.to(device)
            s_tm1, s_t = s_tm1.to(device), s_t.to(device)
                        
            # forward pass 
            psi, encoding, y, loc_readout = model(o_tm1, a_tm1, o_t)
            loss, baseline_loss, assignment = loss_fn(mse_loss, psi, o_t, o_tm1, train=False)

            # compute the y and the loc_readout using the assignment
            y = y[assignment]
            encoding = encoding[assignment, :]
            loc_readout = loc_readout[assignment]

            # purity
            _pred = np.concatenate([_pred, assignment.squeeze().cpu()], 0)
            _gt   = np.concatenate([_gt, a_tm1[:,0].cpu()], 0)

            loss_act = cross_entropy_loss(y, a_tm1[:,0].to(torch.long))
            loss_loc = mse_loss(loc_readout.squeeze(1), s_t - s_tm1).mean()
            
            loc_l1_dist = (loc_readout.squeeze(1) - (s_t - s_tm1)).abs().mean()
            pred = y.argmax(dim=1, keepdim=True)
            
            encodings.append(encoding.detach().cpu().numpy()) 
            assignments.append(assignment.detach().cpu().numpy())
            acts.append(a_tm1.detach().cpu().numpy())
            ys.append((F.softmax(y, dim=1)).detach().cpu().numpy())
            
            train_loss += [loss.item()]
            train_act_loss += [loss_act.item()]
            train_baseline_loss += [baseline_loss.item()]
            train_loc_loss += [loss_loc.item()]
            
            train_loc_l1 += [loc_l1_dist.item()]
            train_acc += [pred.eq(a_tm1.view_as(pred)).float().mean().item()]
            
            loss += FLAGS.m * loss_act
            loss += FLAGS.m * loss_loc

            j += 1
            if j == print_every:
                break
    
    purity = purity_score(_gt, _pred)
    
    return train_loss, train_baseline_loss, train_act_loss, train_acc, \
            train_loc_loss, train_loc_l1, encodings, acts, ys, o_t, psi, \
            assignments, purity 

def main(argv):
    batch_size = FLAGS.batch_size
    tmp_path = 'tmp/atari'

    torch.cuda.set_device(FLAGS.gpu)
    torch.set_num_threads(1)
        
    train_dataset = GridDataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, 
                                   shuffle=True, num_workers=0, drop_last=True)
    
    val_dataset = GridDataset('val')
    val_loader = data.DataLoader(val_dataset, batch_size=FLAGS.batch_size,  
                                 num_workers=0, drop_last=True)
        
    num_actions = 8
    
    # Get total number of actions.
    print('bottleneck_size: ', FLAGS.bottleneck_size)

    device = torch.device("cuda")
    model = EnumerateForwardModel(num_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, 
                                 weight_decay=FLAGS.weight_decay)
 
    if FLAGS.model_path != '0':
        model.load_state_dict(torch.load(FLAGS.model_path))
        print("loaded model!")

    train_writer = SummaryWriter(FLAGS.logdir_prefix + FLAGS.logdir + '/train/', flush_secs=60)
    print(train_writer)
    val_writer = SummaryWriter(FLAGS.logdir_prefix + FLAGS.logdir + '/val/', flush_secs=60)
       
    scheduler = StepLR(optimizer, step_size=FLAGS.step_size, gamma=0.1)
    iteration = 0

    for epoch in range(1, 20):
        print("Train Epoch: ", epoch)
        iteration = train(model, optimizer, epoch, device, train_loader,
                          val_loader, train_writer, val_writer, iteration,
                          num_actions)
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

