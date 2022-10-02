import torch.nn.functional as F
import torch

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .common_loss import L1loss, MSEloss, SmoothL1loss # input: (gx, gt)
from .prior_loss import TVLoss, DCLoss, MIDCLoss, CRLoss
from .discriminator_loss import Discriminator, weights_init_normal

###################################################################
## Stage 1. Loss function
## -- (1) ReID Loss: Triplet, Cross Entropy, input: (score, feat, target)
## -- (2) Generator Loss: SmoothL1 , Discriminator
## -- (3) Embedding Adaptive: L1
###################################################################
def Loss_stage1_reid(cfg, num_classes):
    triplet = TripletLoss(cfg.SOLVER.MARGIN)
    if cfg.MODEL.LABEL_SMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, num_classes:", num_classes)
    else:
        xent = F.cross_entropy
    def loss_func(score, feat, target):
        return xent(score, target) + triplet(feat, target)[0]
    return loss_func

def Loss_stage1_generator(cfg):
    SMOOTHL1 = SmoothL1loss(use_gpu=True)
    def loss_func(gx, gt):
        return SMOOTHL1(gx, gt)
    return loss_func

def Loss_stage1_ea(cfg):  # embedding adaptive
    L1 = L1loss(use_gpu=True)
    def loss_func(clear_feat, foggy_feat):
        return L1(clear_feat, foggy_feat)
    return loss_func

########################################################################
# Stage 2. Loss function
# -- (1) Auxiliary Loss: Triplet, Cross Entropy, input: (score, feat, target) for syn auxiliary learning
# -- (2) Embedding Adaptive: L1, input(clear_feat, foggy_feat), for unsupervised reid learning.
# -- (3) Prior Loss: MIDC, CR, input(inC, Gf), for unsupervised domain transformation.
# -- (4) Cycle Consistency: SmoothL1, input(GAN, GT), for supervised domain transformation 
########################################################################
def Lstage2_reid_sup(cfg, num_classes):
    triplet = TripletLoss(cfg.SOLVER.MARGIN)
    if cfg.MODEL.LABEL_SMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, num_classes:", num_classes)
    else:
        xent = F.cross_entropy
    def loss_func(score, feat, target):
        return xent(score, target) + triplet(feat, target)[0]
    return loss_func

def Lstage2_reid_ea(cfg):
    L1 = L1loss(use_gpu=True)
    def loss_func(clear_feat, foggy_feat):
        return L1(clear_feat, foggy_feat)
    return loss_func

def Lstage2_gan_uns(cfg):
    MIDC = MIDCLoss()
    CR = CRLoss(3)
    def loss_func(inC, Gf):
        Lmidc = MIDC(inC, Gf)
        Lcr = CR(inC, Gf)
        return Lmidc + Lcr  # maybe we can change the hyperparam.
    return loss_func

def Lstage2_gan_sup(cfg):  # cycle consistency
    SMOOTHL1 = SmoothL1loss(use_gpu=True)
    def loss_func(gan, gt):
        return SMOOTHL1(gan, gt)
    return loss_func

########################################################################
# Stage 3. Loss function
# -- (1) Auxiliary Loss: Triplet, Cross Entropy, input: (score, feat, target) for syn auxiliary learning
# -- (2) Embedding Adaptive: L1, input(clear_feat, foggy_feat), for unsupervised reid learning.
# -- (3) Prior Loss: Dark Channel, Total Variation, CR, input(inF, Gc), for unsupervised domain transformation.
# -- (4) Cycle Consistency: SmoothL1, input(GAN, GT), for supervised domain transformation 
########################################################################
def Lstage3_reid_sup(cfg, num_classes):
    triplet = TripletLoss(cfg.SOLVER.MARGIN)
    if cfg.MODEL.LABEL_SMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, num_classes:", num_classes)
    else:
        xent = F.cross_entropy
    def loss_func(score, feat, target):
        return xent(score, target) + triplet(feat, target)[0]
    return loss_func

def Lstage3_reid_ea(cfg):
    L1 = L1loss(use_gpu=True)
    def loss_func(clear_feat, foggy_feat):
        return L1(clear_feat, foggy_feat)
    return loss_func

def Lstage3_gan_uns(cfg):
    '''
    Ldc = minimize. 1 -> 0  -- 2/13(mean) 
    Ltv = minimize. 2 -> 0  -- 2/13(mean)
    Lundcp = 0.5 -> 0 
    '''
    DC = DCLoss()
    TV = TVLoss()
    CR = CRLoss(3)
    def loss_func(inf, Gc):
        Wdc, Wtv, Wcr = 0.5, 2, 0.05
        foggy = inf[0].unsqueeze(0)
        clear = Gc[0].unsqueeze(0)
        Ldc = DC((clear+1)/2)
        Ltv = TV((clear+1)/2)
        Lcr = CR(clear, foggy)
        Ltotal = Wdc*Ldc + Wtv*Ltv + Wcr*Lcr
        return Ltotal, [Ldc, Ltv, Lcr]
    return loss_func

def Lstage3_gan_sup(cfg):  # cycle consistency
    SMOOTHL1 = SmoothL1loss(use_gpu=True)
    def loss_func(gan, gt):
        return SMOOTHL1(gan, gt)
    return loss_func

####################################################################
## Discriminator Loss : Discr, optimizer_D, lr_scheduler_D
####################################################################
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def Loss_discriminator(cfg):
    # input: imgs, label='valid'/'fake'
    # return loss.
    Discr = Discriminator([3, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]], use_gpu=True)
    Discr.apply(weights_init_normal)
    Discr.cuda()
    optimizer_D = torch.optim.Adam(Discr.parameters(), lr=0.0002, betas=(0.5, 0.999))
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D, lr_lambda=LambdaLR(cfg.SOLVER.MAX_EPOCHS, 0, 100).step
    )
    return Discr, optimizer_D, lr_scheduler_D