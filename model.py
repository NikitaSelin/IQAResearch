import math
import torch
from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, large_kernel_size: int = 9, small_kernel_size: int = 3, n_channels: int = 64, n_blocks: int = 16, scaling_factor: int = 4) -> None:
        super(Generator, self).__init__()

        assert type(scaling_factor) is int and scaling_factor in [2, 4, 8]

        self.conv_block1 = ConvBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size, batch_norm=False, activation='PReLU')
        self.residual_blocks = nn.Sequential(*[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for _ in range(n_blocks)])
        self.conv_block2 = ConvBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=small_kernel_size, batch_norm=True, activation=None)

        n_upsample_blocks = int(math.log2(scaling_factor))
        self.upsample_blocks = nn.Sequential(*[UpSampleBlock(kernel_size=small_kernel_size, n_channels=n_channels) for _ in range(n_upsample_blocks)])

        self.conv_block3 = ConvBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size, batch_norm=False, activation='Tanh')
    
    def forward(self, lr_imgs: torch.Tensor) -> torch.Tensor:
        output = self.conv_block1(lr_imgs)
        residual = output
        output = self.residual_blocks(output)
        output = self.conv_block2(output)
        output += residual
        output = self.upsample_blocks(output)
        sr_imgs = self.conv_block3(output)
        return sr_imgs


class Discriminator(nn.Module):
    def __init__(self, kernel_size: int = 3, n_channels: int = 64, n_blocks: int = 8, fc_size: int = 1024) -> None:
        super(Discriminator, self).__init__()

        in_channels = 3
        conv_blocks = []
        for i in range(n_blocks):
            out_channels = (n_channels if i is 0 else in_channels * 2) if i % 2 is 0 else in_channels
            conv_blocks.append(ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1 if i % 2 is 0 else 2, batch_norm=i is not 0, activation='LeakyReLU'))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(fc_size, 1)
    
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)
        return logit


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, batch_norm: bool = False, activation: str = None) -> None:
        super(ConvBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in ['prelu', 'relu', 'tanh', 'leakyrelu']
        
        layers = []

        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2))

        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv_block(input)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, n_channels: int = 64, kernel_size: int = 3) -> None:
        super(ResidualBlock, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, batch_norm=True, activation='PReLU')
        self.conv_block2 = ConvBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, batch_norm=True, activation=None)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        output = self.conv_block1(input)
        output = self.conv_block2(output)
        output = output + residual
        return output


class UpSampleBlock(nn.Module):
    def __init__(self, n_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> None:
        super(UpSampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.conv(F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)), 0.2, True)
