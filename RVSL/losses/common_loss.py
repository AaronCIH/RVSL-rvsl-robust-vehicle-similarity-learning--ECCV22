import torch
from torch import nn

class MSEloss(nn.Module):
    def __init__(self, use_gpu=True):
        super(MSEloss, self).__init__()
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.mseloss = nn.MSELoss().cuda()
        else:
            self.mseloss = nn.MSELoss()
    
    def forward(self, gx, gt):
        loss = self.mseloss(gx, gt)
        return loss

class SmoothL1loss(nn.Module):
    def __init__(self, use_gpu=True):
        super(SmoothL1loss, self).__init__()
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.smoothl1loss = nn.SmoothL1Loss().cuda()
        else:
            self.smoothl1loss = nn.SmoothL1Loss()
    
    def forward(self, gx, gt):
        loss = self.smoothl1loss(gx, gt)
        return loss

class L1loss(nn.Module):
    def __init__(self, use_gpu=True):
        super(L1loss, self).__init__()
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.L1 = nn.L1Loss().cuda()
        else:
            self.L1 = nn.L1Loss()
    
    def forward(self, gx, gt):
        loss = self.L1(gx, gt)
        return loss