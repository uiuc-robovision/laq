import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

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
 
        # freeze resnet_dynamics models
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

        scale = 10
        
        # freeze resnets
        self.resnet18_dynamics.eval()

        all_actions = torch.arange(8).cuda()
        encoding = self.embedding(all_actions)

        y = encoding
        act1 = scale*encoding
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
