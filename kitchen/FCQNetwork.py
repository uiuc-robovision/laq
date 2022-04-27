import torch
import torch.nn as nn
import torch.nn.functional as F

class FCQNetwork(nn.Module):
    def __init__(self, in_features, out_features,means=None,stds=None,positional_encoding=False,arch_dim=128,tcc=False):
        super(FCQNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = int(out_features)
        if tcc:
            # self.core = nn.Sequential(nn.Linear(in_features, arch_dim), nn.ReLU(),
            #                           nn.Linear(arch_dim, arch_dim*int(out_features)))
            self.core = nn.Sequential(nn.Linear(in_features, arch_dim), nn.ReLU(),
                                      nn.Linear(arch_dim, arch_dim), nn.ReLU(),
                                      nn.Linear(arch_dim, arch_dim*int(out_features)))
        else:
            self.core = nn.Sequential(nn.Linear(in_features, arch_dim), nn.ReLU(),
                                      nn.Linear(arch_dim, arch_dim), nn.ReLU(),
                                      nn.Linear(arch_dim, int(out_features)))
        self.means = means
        self.stds = stds
        self.positional_encoding = positional_encoding
        self.arch_dim = arch_dim
        self.tcc = tcc

    def forward(self, inputs):
        inp = (inputs-self.means)/self.stds
        if self.positional_encoding:
            scaled = [(inp*(2**l)) for l in range(0,10)]
            scaled = torch.cat(scaled,axis=1)
            inp = torch.cat((torch.sin(scaled),torch.cos(scaled)),axis=1)
        out= self.core(inp)
        if self.tcc:
            out = torch.reshape(out, (out.shape[0], self.out_features, self.arch_dim))
            out = 1 - torch.linalg.vector_norm(out, dim=2)

        # add object class dimention
        return out.unsqueeze(1)

    def set_train(self):
        self.train()
