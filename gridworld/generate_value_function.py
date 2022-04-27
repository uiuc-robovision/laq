import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import numpy as np
from absl import app, flags
from gridworld.latent_action_mining import GridDataset, loss_fn, EnumerateForwardModel
from sklearn.cluster import KMeans
from sklearn import metrics

FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'cluster', 'Model to evaluate')
# flags.DEFINE_string('model_path', None, 'Model to evaluate')
# flags.DEFINE_integer('bottleneck_size', 16, '')
# flags.DEFINE_integer('gpu', 4, 'Which GPU to use.')

def clustering(train_data, val_data, bottleneck_size):
    kmeans = KMeans(bottleneck_size, random_state=0).fit(train_data)
    val_cluster_id = kmeans.predict(val_data)
    train_cluster_id = kmeans.predict(train_data)
    return train_cluster_id, val_cluster_id

def single_action(train_dataset, val_dataset):
    # Returns the action labels for the validation dataset
    None

def entropy_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    prob = contingency_matrix / np.sum(contingency_matrix, 0, keepdims=True)
    entropy = np.sum(-prob * np.log(np.maximum(prob, 1e-10)), 0)
    mean_entropy = np.sum(np.sum(contingency_matrix, 0, keepdims=True) / np.sum(contingency_matrix) * entropy)
    return mean_entropy
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def main(_):
    if FLAGS.model == 'cluster' or FLAGS.model == 'cluster_diff':
        dt = np.load('gridworld/data.npy')
        dt = dt[dt[:,-1] != 1] # remove dummy transitions
        train_data = dt[:250000, :]
        val_data = dt[250000:, :]
        state = train_data[:, 0]
        state = np.array([state // 8, state % 8]).T + 0.
        next_state = train_data[:, 2]
        next_state = np.array([next_state // 8, next_state % 8]).T + 0.
        train_action = train_data[:,1]
        
        if FLAGS.model == 'cluster_diff':
            train_feature = next_state - state
        elif FLAGS.model == 'cluster':
            train_feature = np.concatenate([state, next_state], 1)
        
        state = val_data[:, 0]
        state = np.array([state // 8, state % 8]).T + 0.
        next_state = val_data[:, 2]
        next_state = np.array([next_state // 8, next_state % 8]).T + 0.
        val_action = val_data[:,1]
        if FLAGS.model == 'cluster_diff':
            val_feature = next_state - state
        elif FLAGS.model == 'cluster':
            val_feature = np.concatenate([state, next_state], 1)
        
        train_id, val_id = clustering(train_feature, val_feature, FLAGS.bottleneck_size) 
        all_actions = dt[:,1:2]
        all_assignments = np.concatenate([train_id, val_id], 0)
        dt = np.concatenate([dt, all_actions, all_assignments[:,np.newaxis]], 1)
        assert(np.allclose(dt[:,1], all_actions[:,0]))
        out_file_name = FLAGS.model + '_' + str(FLAGS.bottleneck_size) + '.npy'
        np.save(out_file_name, dt)


    
    elif FLAGS.model == 'learned':
        torch.cuda.set_device(FLAGS.gpu)
        torch.set_num_threads(1)
        device = torch.device("cuda")
        
        all_assignments = []
        all_actions = []
        for imset in ['train', 'val']:
            dataset = GridDataset(imset)
            loader = data.DataLoader(dataset, batch_size=FLAGS.batch_size,
                                     num_workers=0, drop_last=False, shuffle=False)
            model = EnumerateForwardModel(8).to(device)
            model.load_state_dict(torch.load(FLAGS.model_path, map_location=device))
            mse_loss = nn.MSELoss(reduction='none')
            model = model.eval()
            assignments = [] 
            actions = []
            with torch.no_grad():
                for batch_idx, (o_tm1, a_tm1, o_t, s_tm1, s_t) in tqdm(enumerate(loader)):

                    o_tm1, a_tm1, o_t = o_tm1.to(device), a_tm1.to(device), o_t.to(device)
                    s_tm1, s_t = s_tm1.to(device), s_t.to(device)

                    # forward pass
                    psi, encoding, y, loc_readout = model(o_tm1, a_tm1, o_t)
                    loss, baseline_loss, assignment = loss_fn(mse_loss, psi, o_t, o_tm1, train=False)
                    assignments.append(assignment.cpu().numpy())
                    actions.append(a_tm1.cpu().numpy())
            assignments = np.concatenate(assignments, 0)
            actions = np.concatenate(actions, 0)
            all_assignments.append(assignments)
            all_actions.append(actions)
    
        train_id, val_id = all_assignments
        train_action, val_action = all_actions
    
        # out_file_name = FLAGS.model_path.split('.')[-1] + '.npy'
        dt = np.load('gridworld/data.npy')
        dt = dt[dt[:,-1] != 1] # remove dummy transitions
        out_file_name = os.path.splitext(FLAGS.model_path)[0]+'.npy'
        all_actions = np.concatenate(all_actions, 0)
        all_assignments = np.concatenate(all_assignments, 0)
        dt = np.concatenate([dt, all_actions, all_assignments[:,np.newaxis]], 1)
        assert(np.allclose(dt[:,1], all_actions[:,0]))
        np.save(out_file_name, dt)
        from gridworld.utils import qlearn
        from gridworld.envs import make_gridworld
        env = make_gridworld()
        dt[:,1] = dt[:,-1]
        fd = np.load('gridworld/data.npy')
        terms = fd[fd[:,-1] == 1]
        terms_full=np.concatenate((terms,np.zeros((terms.shape[0],2))),axis=1)
        dt_full = np.concatenate((dt,terms_full),axis=0).astype(np.int32)
        plainQFull, plainQ = qlearn(env.observation_space.n,env.action_space.n,dt_full[:,:-2],epochs=3)
        np.save(os.path.dirname(FLAGS.model_path)+"/value_function.npy",plainQ)


    print('Purity [val]: ', purity_score(val_action, val_id))
    print('Purity [train]: ', purity_score(train_action, train_id))
    print('entropy [val]: ', entropy_score(val_action, val_id))
    print('entropy [train]: ', entropy_score(train_action, train_id))
    
if __name__ == '__main__':
    app.run(main)
