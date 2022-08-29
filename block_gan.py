import numpy as np
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),  # 0.01
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, n_features, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.spec_norm = SpectralNorm
        self.pool = nn.AvgPool2d(4)

        self.base_disc = nn.Sequential(
            InitResBlock(3, n_features),
            ResBlock(n_features, n_features * 2, stride=2),
            ResBlock(n_features * 2, n_features * 4, stride=2),
            ResBlock(n_features * 4, n_features * 8, stride=2),
        )
        self.d = ResBlock(n_features * 8, n_features * 8)
        self.q = ResBlock(n_features * 8, n_features * 8)

        # final fc layer init + norm
        self.fc = nn.Linear(n_features * 8, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = self.spec_norm(self.fc)

        self.cls = nn.Linear(n_features * 8, z_dim + 3)
        nn.init.xavier_uniform(self.cls.weight.data, 1.)
        self.cls = self.spec_norm(self.cls)

    def forward(self, x):
        h = self.base_disc(x)
        h_d = self.pool(self.d(h))
        h_q = self.pool(self.q(h))
        h_d = h_d.view(-1, h_d.size(1))
        h_q = h_q.view(-1, h_q.size(1))

        pred_d = self.fc(h_d)
        pred_d = F.sigmoid(pred_d)  # if using sigmoid in final layer
        pred_zt = self.cls(h_q)
        pred_z = pred_zt[:, 0:self.z_dim]
        pred_t = pred_zt[:, self.z_dim:]
        return pred_d, pred_z, pred_t


########  util layers for discriminator and generator  ########
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.spec_norm = SpectralNorm

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.spec_norm(self.conv1),
                nn.ReLU(),
                self.spec_norm(self.conv2)
            )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.spec_norm(self.conv1),
                nn.ReLU(),
                self.spec_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        self.bypass = nn.Sequential()
        if in_channels != out_channels:
            self.bypass = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass.weight.data, np.sqrt(2))
            self.bypass = self.spec_norm(self.bypass)
        if stride != 1:
            self.bypass = nn.Sequential(
                self.bypass,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class InitResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitResBlock, self).__init__()
        self.spec_norm = SpectralNorm

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        self.model = nn.Sequential(
            self.spec_norm(self.conv1),
            nn.ReLU(),
            self.spec_norm(self.conv2),
            nn.AvgPool2d(2)
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.spec_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)
