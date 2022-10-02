import math
import torch
from torch import nn
from torchvision import models
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from functools import reduce

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2, output: (same w, same h, out_channels) """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Decoder_Generator(nn.Module):
    def __init__(self, in_channels, mid_channels, output_shape, bilinear=True):
        super(Decoder_Generator, self).__init__()
        self.output_shape = output_shape
        self.DC1 = DoubleConv(in_channels, mid_channels, mid_channels)
        if bilinear:
            self.UP1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.UP1 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2)
        self.DC2 = DoubleConv(mid_channels*2, mid_channels, mid_channels)
        if bilinear:
            self.UP2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.UP2 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2)
        self.DC3 = DoubleConv(mid_channels, mid_channels, mid_channels)
        self.OutConv = nn.Conv2d(mid_channels, 3, kernel_size=1)

    def forward(self, x1, x2):
        # input is CHW
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x1 = self.UP1(self.DC1(x1))
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.UP2(self.DC2(x))
        diffY = self.output_shape[2] - x.size()[2]
        diffX = self.output_shape[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])        
        out = self.OutConv(self.DC3(x))
        return out
