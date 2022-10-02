# by Cihsiang, utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import L1Loss
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, img):
        hor = self.grad_conv_hor()(img)
        vet = self.grad_conv_vet()(img)
        target = torch.autograd.Variable(torch.FloatTensor(img.shape).zero_().cuda())
        loss_hor = L1Loss(reduction='mean')(hor, target)
        loss_vet = L1Loss(reduction='mean')(vet, target)
        loss = loss_hor+loss_vet
        return loss

    # horizontal gradient, the input_channel is default to 3
    def grad_conv_hor(self, ):
        grad = nn.Conv2d(3, 3, (1, 3), stride=1, padding=(0, 1))
        weight = np.zeros((3, 3, 1, 3))
        for i in range(3):
            weight[i, i, :, :] = np.array([[-1, 1, 0]])
        weight = torch.FloatTensor(weight).cuda()
        weight = nn.Parameter(weight, requires_grad=False)
        bias = np.array([0, 0, 0])
        bias = torch.FloatTensor(bias).cuda()
        bias = nn.Parameter(bias, requires_grad=False)
        grad.weight = weight
        grad.bias = bias
        return  grad

    # vertical gradient, the input_channel is default to 3
    def grad_conv_vet(self, ):
        grad = nn.Conv2d(3, 3, (3, 1), stride=1, padding=(1, 0))
        weight = np.zeros((3, 3, 3, 1))
        for i in range(3):
            weight[i, i, :, :] = np.array([[-1, 1, 0]]).T
        weight = torch.FloatTensor(weight).cuda()
        weight = nn.Parameter(weight, requires_grad=False)
        bias = np.array([0, 0, 0])
        bias = torch.FloatTensor(bias).cuda()
        bias = nn.Parameter(bias, requires_grad=False)
        grad.weight = weight
        grad.bias = bias
        return  grad

class DCLoss(nn.Module):
    def __init__(self):
        super(DCLoss, self).__init__()
        self.patch_size = 35
    
    def forward(self, img):
        """
        calculating dark channel of image, the image shape is of N*C*W*H
        """
        maxpool = nn.MaxPool3d((3, self.patch_size, self.patch_size), stride=1, padding=(0, self.patch_size//2, self.patch_size//2))
        dc = maxpool(1-img[:, None, :, :, :]) # size=([1, 1, 1, 384, 384])
        target = torch.ones(dc.shape, dtype=torch.float64).zero_().cuda()
        loss = L1Loss(reduction='mean')(target, dc) #/ (len(img)*dc.shape[3]*dc.shape[4])
        return loss

class MIDCLoss(nn.Module): 
    # Monotonously Increasing Dark Channel Loss.
    def __init__(self, ):
        super(MIDCLoss, self).__init__()

    def forward(self, clear, foggy):
        # dark
        clear_dc, _idx = torch.min(clear, 1)  # [b, w, h]
        clear_dc = torch.unsqueeze(clear_dc, dim=1)   # [b, 1, w, h]        
        foggy_dc, _idx = torch.min(foggy, 1)  # [b, w, h]
        foggy_dc = torch.unsqueeze(foggy_dc, dim=1)   # [b, 1, w, h]

        # binary map
        map = torch.as_tensor(clear_dc > foggy_dc, dtype=torch.float32) 
        DC_c = map*clear_dc
        DC_f = map*foggy_dc

        L_MIDC = L1Loss(reduction='sum')(DC_f, DC_c)/torch.sum(map) #/ (len(img)*dc.shape[3]*dc.shape[4])
        return L_MIDC

class CRLoss(nn.Module): 
    # Colinear Relation Constraint.
    def __init__(self, sz):
        super(CRLoss, self).__init__()
        self.sz = sz
        self.mse = nn.MSELoss(reduction='mean').cuda()

    def forward(self, clear, foggy):
        # dark
        dc, dc_idx = torch.min(foggy, 1)  # [b, w, h]
        dc = torch.unsqueeze(dc, dim=1)   # [b, 1, w, h]
        dark = self.tensor_erode(dc, self.sz) # [b, 1, w, h]
        # AtmLight 
        b, c, h, w = foggy.size()
        imsz = h*w
        numpx = int(max(math.floor(imsz/1000),1))
        darkvec = torch.reshape(dark, (b, imsz))
        imvec = torch.reshape(foggy, (b, 3, imsz))
        indices = torch.argsort(darkvec, dim=1)
        indices = indices[:, imsz-numpx::]
        atmsum = torch.zeros((b,3)).cuda()
        for ind in range(1,numpx):
            for idx in range(b):
                atmsum[idx] = atmsum[idx] + imvec[idx, :, indices[idx, ind]]
        A = atmsum / numpx
        A = torch.reshape(A, (b, 3, 1, 1))
        # Loss 
        # chromaticity of clean, foggy, Airlight
        gamma = torch.div(clear, torch.sum(clear, dim=1).unsqueeze(1).clamp_(1e-6)) # [b,3,w,h]
        sigma = torch.div(foggy, torch.sum(foggy, dim=1).unsqueeze(1).clamp_(1e-6)) # [b,3,w,h]
        A = torch.div(A, torch.sum(A, dim=1).unsqueeze(1).clamp_(1e-6)) # [b,3,1,1]
        # cosine similarity loss, equal to normalize-MSE 
        foggy_diff = F.normalize(sigma - A)
        clear_diff = F.normalize(gamma - A)
        L_hazeline = self.mse(foggy_diff, clear_diff)
        return L_hazeline

    def tensor_erode(self, bin_img, ksize=5):
        # add padding
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=1)
        # unfold to patch
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        # B x C x H x W x k x k, select minnum
        eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
        return eroded   
