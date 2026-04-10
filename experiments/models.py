"""
SQ-VAE ResNet encoder/decoder for CIFAR-10 (32x32).
Architecture from Takida et al. (2022), github.com/sony/sqvae, networks/net_32.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderVqResnet32(nn.Module):
    def __init__(self, dim_z=64, num_rb=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, dim_z // 2, 4, 2, 1),
            nn.BatchNorm2d(dim_z // 2),
            nn.ReLU(),
            nn.Conv2d(dim_z // 2, dim_z, 4, 2, 1),
            nn.BatchNorm2d(dim_z),
            nn.ReLU(),
            nn.Conv2d(dim_z, dim_z, 3, 1, 1),
            nn.BatchNorm2d(dim_z),
            nn.ReLU(),
            *[ResBlock(dim_z) for _ in range(num_rb)],
        )

    def forward(self, x):
        return self.encoder(x)


class DecoderVqResnet32(nn.Module):
    def __init__(self, dim_z=64, num_rb=2):
        super().__init__()
        self.decoder = nn.Sequential(
            *[ResBlock(dim_z) for _ in range(num_rb)],
            nn.ConvTranspose2d(dim_z, dim_z, 3, 1, 1),
            nn.BatchNorm2d(dim_z),
            nn.ReLU(),
            nn.ConvTranspose2d(dim_z, dim_z // 2, 4, 2, 1),
            nn.BatchNorm2d(dim_z // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(dim_z // 2, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class EncoderVqResnet128(nn.Module):
    """SQ-VAE encoder for 128x128 input (CelebA). Based on net_64.py with num_rb=6.
    Two stride-2 convs: 128x128 -> 64x64 -> 32x32 latent."""
    def __init__(self, dim_z=64, num_rb=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, dim_z // 2, 4, 2, 1),
            nn.BatchNorm2d(dim_z // 2),
            nn.ReLU(),
            nn.Conv2d(dim_z // 2, dim_z, 4, 2, 1),
            nn.BatchNorm2d(dim_z),
            nn.ReLU(),
            nn.Conv2d(dim_z, dim_z, 3, 1, 1),
        )
        self.res = nn.Sequential(*[ResBlock(dim_z) for _ in range(num_rb)])

    def forward(self, x):
        return self.res(self.conv(x))


class DecoderVqResnet128(nn.Module):
    """SQ-VAE decoder for 128x128 output (CelebA). Based on net_64.py with num_rb=6.
    Two stride-2 transposed convs: 32x32 -> 64x64 -> 128x128."""
    def __init__(self, dim_z=64, num_rb=6):
        super().__init__()
        self.res = nn.Sequential(*[ResBlock(dim_z) for _ in range(num_rb - 1)])
        self.convt = nn.Sequential(
            nn.ConvTranspose2d(dim_z, dim_z, 3, 1, 1),
            nn.BatchNorm2d(dim_z),
            nn.ReLU(),
            nn.ConvTranspose2d(dim_z, dim_z // 2, 4, 2, 1),
            nn.BatchNorm2d(dim_z // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(dim_z // 2, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.convt(self.res(x))


class VQVAE(nn.Module):
    def __init__(self, vq_layer, dim_z=64, num_rb=2, arch='32'):
        super().__init__()
        if arch == '32':
            self.encoder = EncoderVqResnet32(dim_z, num_rb)
            self.decoder = DecoderVqResnet32(dim_z, num_rb)
        elif arch == '128':
            self.encoder = EncoderVqResnet128(dim_z, num_rb)
            self.decoder = DecoderVqResnet128(dim_z, num_rb)
        else:
            raise ValueError(f'Unknown arch: {arch}')
        self.vq_layer = vq_layer

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_dict = self.vq_layer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_dict
