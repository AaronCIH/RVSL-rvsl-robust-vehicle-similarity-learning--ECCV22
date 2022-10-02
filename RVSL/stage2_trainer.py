#######################################################
# Stage 2. for real clear training...
# Creator: Cihsiang
# Email: f09921058@g.ntu.edu.tw  
#######################################################
import os
import os.path as osp

import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import shutil
import random

from config import cfg
import argparse
from tqdm import tqdm
import logging 

from dataset import make_FVRID_dataloader_REAL 
from models import build_Stage2_Model, convert_model 
from losses import Lstage2_reid_sup, Lstage2_reid_ea, Lstage2_gan_uns, Lstage2_gan_sup, Loss_discriminator, MSEloss
from utils import Render_plot, setup_logger, AvgerageMeter
from evaluate import eval_func, euclidean_dist, eval_dh  
from optim import make_optimizer, WarmupMultiStepLR

from torch.autograd import Variable

def train():
    # Step1. config setting
    parser = argparse.ArgumentParser(description="ReID training")
    parser.add_argument('-c', '--config_file', type=str, help='the path to the training config')
    parser.add_argument('-p', '--pre_train', action='store_true', default=False, help='Model test')
    parser.add_argument('-t', '--test', action='store_true', default=False, help='Model test')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('opts', help='overwriting the training config' 
                        'from commandline', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    seed = cfg.MODE.SEED
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Step2. output setting
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shutil.copy(args.config_file, cfg.OUTPUT_DIR)
    if not os.path.exists(os.path.join(output_dir, "checkpoint/")):
        os.makedirs(os.path.join(output_dir, "checkpoint/"))
    if cfg.TEST.VIS:  # sample for validation
        if not os.path.exists(os.path.join(output_dir, "sample_result_for_REAL/")):
            os.makedirs(os.path.join(output_dir, "sample_result_for_REAL/"))
        if not os.path.exists(os.path.join(output_dir, "sample_result_for_SYN/")):
            os.makedirs(os.path.join(output_dir, "sample_result_for_SYN/"))
        if not os.path.exists(os.path.join(output_dir, "sample_result_for_val/")):
            os.makedirs(os.path.join(output_dir, "sample_result_for_val/"))
    
    # Step3. logging
    num_gpus = torch.cuda.device_count()
    logger = setup_logger('RVSL_stage2', output_dir, 0)  # check
    logger.info('Using {} GPUS'.format(num_gpus))
    logger.info(args)
    logger.info('Running with config:\n{}'.format(cfg))
    logger.info("##############################################################")
    logger.info("# % TRAIN .....  Stage 2 !!!!!") # !!!!
    logger.info("# Backbone Model is {%s}" %(cfg.MODEL.NAME))
    logger.info("# Dataset is {%s}" %(cfg.DATASETS.NAMES))
    logger.info("# pretrain_model = {%s} #" %(cfg.MODEL.PRETRAIN_PATH))
    logger.info("# data_path = {%s} #" %(cfg.DATASETS.DATA_PATH))
    logger.info("# SYN_TRAIN_CLEAR_PATH = {%s} #" %(cfg.DATASETS.SYN_TRAIN_CLEAR_PATH))
    logger.info("# SYN_TRAIN_FOGGY_PATH = {%s} #" %(cfg.DATASETS.SYN_TRAIN_FOGGY_PATH))
    logger.info("# REAL_TRAIN_CLEAR_PATH = {%s} #" %(cfg.DATASETS.REAL_TRAIN_CLEAR_PATH))
    logger.info("# REAL_QUERY_CLEAR_PATH = {%s} #" %(cfg.DATASETS.REAL_QUERY_CLEAR_PATH))
    logger.info("# REAL_GALLERY_CLEAR_PATH = {%s} #" %(cfg.DATASETS.REAL_GALLERY_CLEAR_PATH))
    logger.info("# OUTPUT_dir = {%s} #" %(cfg.OUTPUT_DIR))
    logger.info("# RVSL_REID_W = {%s} #" %(cfg.MODEL.RVSL_BASE_W))
    logger.info("# RVSL_REST_W = {%s} #" %(cfg.MODEL.RVSL_REST_W))
    logger.info("##############################################################")

    # Step4. Create FVRID dataloader
    ####################################################################
    # SYN_train_loader, REAL_train_loader, REAL_val_loader, len(dataset.Real_query), SYN_num_classes, REAL_num_classes
    # - syn_train_dl: Getitem: {"Foggy", "Clear", "Foggy_paths", "Clear_paths", "pids", "camids"}
    # - real_train_dl:Getitem: {"imgs", "paths", "pids", "camids"}
    # - real_val_dl:  Getitem: {"imgs", "paths", "pids", "camids"}
    ####################################################################
    print("#In[0]: make_dataloader--------------------------------")
    SYN_train_dl, REAL_train_dl, REAL_val_dl, num_query, SYN_num_classes, _ = make_FVRID_dataloader_REAL(cfg, datatype='clear', num_gpus=num_gpus)

    # Step5. Build RVSL Stage2 Model    
    ####################################################################
    # Stage 2. structure, for Real clear training stage. use syn paris for auxiliary learning.
    # Input: (datas, mode='real'/'syn')
    # Output:
    #     -Traing['syn']: {'Class_clear', 'Class_foggy', 'GlobalFeat_clear', 'GlobalFeat_foggy', 'GAN_clear', 'GAN_foggy'}
    #     -Traing['real']: {'GlobalFeat_clear', 'GlobalFeat_foggy', 'GAN_clear', 'GAN_foggy'}
    #     -Eval: {'Feat_clear', 'Feat_foggy', 'GAN_clear', 'GAN_foggy'}
    ####################################################################
    print("#In[1]: build_model--------------------------------")
    print("auxiliary_num_classes = %d" %(SYN_num_classes))
    model = build_Stage2_Model(cfg, SYN_num_classes)
    if cfg.MODEL.PRETRAIN_PATH != '':
        param = torch.load(cfg.MODEL.PRETRAIN_PATH)
        for i in param:
            if 'fc' in i: continue
            if i not in model.state_dict().keys(): continue
            if param[i].shape != model.state_dict()[i].shape: continue
            model.state_dict()[i].copy_(param[i])
    
    # Step6. Define Loss function
    ####################################################################
    # sampler == 'RVSL',       
    # -- (1) Auxiliary Loss: Triplet, Cross Entropy, input: (score, feat, target) for syn auxiliary learning
    # -- (2) Embedding Adaptive: L1, input(clear_feat, foggy_feat), for unsupervised reid learning.
    # -- (3) Prior Loss: MIDC, CR, input(inC, Gf), for unsupervised domain transformation.
    # -- (4) Cycle Consistency: SmoothL1, input(GAN, GT), for supervised domain transformation 
    ########################################################################
    print("#In[2]: make_loss--------------------------------")
    Lreid_sup = Lstage2_reid_sup(cfg, SYN_num_classes)
    Lreid_ea = Lstage2_reid_ea(cfg)
    Lgan_undcp = Lstage2_gan_uns(cfg)
    Lgan_p2p = Lstage2_gan_sup(cfg)
    Ldisc, optimizer_D, lr_scheduler_D = Loss_discriminator(cfg)

    # Step7. Training
    print("#In[3]: Start traing--------------------------------")
    trainer = STAGE2_Trainer(cfg, model, SYN_train_dl, REAL_train_dl, REAL_val_dl, 
                             Lreid_sup, Lreid_ea, Lgan_undcp, Lgan_p2p, 
                             Ldisc, optimizer_D, lr_scheduler_D,
                             num_query, num_gpus)
    from itertools import cycle
    for epoch in range(trainer.epochs):
        print("----------- Epoch [%d/%d] ----------- "%(epoch,trainer.epochs))
        for ct, train_batch in enumerate(zip(trainer.SYN_train_dl, cycle(trainer.REAL_train_dl))):
            print('step: %d/%d'%(ct, len(trainer.SYN_train_dl)), end='\r')
            trainer.step(train_batch[0], train_batch[1])
            trainer.handle_new_batch()                   
        trainer.handle_new_epoch()

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class STAGE2_Trainer(object):
    def __init__(self, cfg, model, SYN_train_dl, REAL_train_dl, REAL_val_dl, 
                       Lreid_sup, Lreid_ea, Lgan_undcp, Lgan_p2p, 
                       Ldisc, optimizer_D, lr_scheduler_D,
                       num_query, num_gpus):
        self.cfg = cfg
        self.std = self.cfg.INPUT.PIXEL_STD
        self.mean = self.cfg.INPUT.PIXEL_MEAN
        self.model = model
        self.SYN_train_dl = SYN_train_dl
        self.REAL_train_dl = REAL_train_dl
        self.REAL_val_dl = REAL_val_dl
        self.len_batchs = np.max([len(SYN_train_dl), len(REAL_train_dl)])

        self.Lreid_sup = Lreid_sup
        self.Lreid_ea = Lreid_ea
        self.Lgan_undcp = Lgan_undcp
        self.Lgan_p2p = Lgan_p2p

        self.Ldisc = Ldisc
        self.optim_D = optimizer_D
        self.lr_scheduler_D = lr_scheduler_D
        self.fack_buffer = ReplayBuffer()

        self.mse_loss = MSEloss()
        self.num_query = num_query
        self.best_mAP_c = 0
        self.best_mAP_f = 0
        self.best_SSIM_c = 0
        self.best_SSIM_f = 0
        self.best_epoch = []

        self.train_epoch = 1
        self.batch_cnt = 0

        self.logger = logging.getLogger('RVSL_stage2.train') # !!! notice !!!
        self.log_period = cfg.SOLVER.LOG_PERIOD
        self.checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        self.eval_period = cfg.SOLVER.EVAL_PERIOD
        self.output_dir = cfg.OUTPUT_DIR
        self.device = cfg.MODEL.DEVICE
        self.epochs = cfg.SOLVER.MAX_EPOCHS

        if self.cfg.MODEL.TENSORBOARDX:
            print("############## create tensorboardx ##################")
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'Log_RVSL_Stage2'))
            self.writer_graph = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'Log_RVSL_Stage2/model'))

        # Single GPU model
        self.model.cuda()
        self.optim = make_optimizer(cfg, self.model, num_gpus)
        self.scheduler = WarmupMultiStepLR(self.optim, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                            cfg.SOLVER.WARMUP_FACTOR,
                                            cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)                    
        self.scheduler.step()
        self.mix_precision = False
        if cfg.SOLVER.FP16:
            # Single model using FP16
            from apex import amp
            self.model, self.optim = amp.initialize(self.model, self.optim,opt_level='O1')                  
            self.mix_precision = True
            self.logger.info('Using fp16 training')
        if self.mix_precision:
            from apex import amp
        self.logger.info('Trainer Built')
        return

    def step(self, SYN_batch, REAL_batch):
        self.model.train()
        self.optim.zero_grad()
        # if self.batch_cnt % self.cfg.SOLVER.LOG_PERIOD == 0:
        #     vis = True
        # else:
        #     vis = False
        vis = False
        
        # % Stage 2-1 Auxiliary Forward
        ## % load data  % Model forward & gan & discriminator
        ####################################################################
        ## - syn_train_dl: Getitem: {"Foggy", "Clear", "Foggy_paths", "Clear_paths", "pids", "camids"}
        ## - Traing['syn']: {'Class_clear', 'Class_foggy', 'GlobalFeat_clear', 'GlobalFeat_foggy', 'GAN_clear', 'GAN_foggy'}
        #################################################################### 
        SYN_foggy, SYN_clear, SYN_pids, SYN_foggy_pth, SYN_clear_pth = \
            SYN_batch['Foggy'], SYN_batch['Clear'], SYN_batch['pids'], SYN_batch['Foggy_paths'], SYN_batch['Clear_paths']
        SYN_foggy, SYN_clear, SYN_pids = SYN_foggy.cuda(), SYN_clear.cuda(), SYN_pids.cuda()
        SYN_Output = self.model([SYN_foggy, SYN_clear], mode='syn')

        ## % Loss 
        ####################################################################
        # -- (1) Auxiliary Loss: Triplet, Cross Entropy, input: (score, feat, target) for syn auxiliary learning
        # -- (2) Embedding Adaptive: L1, input(clear_feat, foggy_feat), for unsupervised reid learning.
        # -- (3) Cycle Consistency: SmoothL1, input(GAN, GT), for supervised domain transformation 
        #################################################################### 
        ### reid (1)(2)
        SYN_score_F, SYN_score_C = SYN_Output['Class_foggy'], SYN_Output['Class_clear']
        SYN_feat_F, SYN_feat_C = SYN_Output['GlobalFeat_foggy'], SYN_Output['GlobalFeat_clear']
        Lreid_sup_synF, Lreid_sup_synC = self.Lreid_sup(SYN_score_F, SYN_feat_F, SYN_pids), self.Lreid_sup(SYN_score_C, SYN_feat_C, SYN_pids)
        Lreid_ea_syn = self.Lreid_ea(SYN_feat_C, SYN_feat_F)
        ### gan (3)
        SYN_gan_F, SYN_gan_C = SYN_Output['GAN_foggy'], SYN_Output['GAN_clear']
        Lgan_p2p_synF, Lgan_p2p_synC = self.Lgan_p2p(SYN_gan_F, SYN_foggy), self.Lgan_p2p(SYN_gan_C, SYN_clear)
        ### Weight Loss
        Wreid_syn, Wea_syn, Wp2p_syn = 10, 1, 100
        Lsyn = Wreid_syn*(Lreid_sup_synF+Lreid_sup_synC) + Wea_syn*Lreid_ea_syn + Wp2p_syn*(Lgan_p2p_synF+Lgan_p2p_synC)

        # % Stage 2-2 REAL Clear Forward
        ## % load data  % Model forward & gan & discriminator
        ####################################################################
        ## - real_train_dl:Getitem: {"imgs", "paths", "pids", "camids"}
        ## - Traing['real']: {'GlobalFeat_clear', 'GlobalFeat_foggy', 'GAN_clear', 'GAN_foggy'}
        #################################################################### 
        REAL_clear, REAL_clear_pth = REAL_batch['imgs'], REAL_batch['paths']
        REAL_clear = REAL_clear.cuda()
        REAL_Output = self.model([REAL_clear], mode='real')

        ## % Loss 
        ####################################################################
        # -- (1) Auxiliary Loss: Triplet, Cross Entropy, input: (score, feat, target) for syn auxiliary learning
        # -- (2) Embedding Adaptive: L1, input(clear_feat, foggy_feat), for unsupervised reid learning.
        # -- (3) Prior Loss: MIDC, CR, input(inC, Gf), for unsupervised domain transformation.
        # -- (4) Cycle Consistency: SmoothL1, input(GAN, GT), for supervised domain transformation 
        #################################################################### 
        ### reid (1)(2)
        REAL_feat_F, REAL_feat_C = REAL_Output['GlobalFeat_foggy'], REAL_Output['GlobalFeat_clear']
        Lreid_ea_real = self.Lreid_ea(REAL_feat_C, REAL_feat_F)
        ### gan (3)(4)
        REAL_gan_F, REAL_gan_C = REAL_Output['GAN_foggy'], REAL_Output['GAN_clear']
        Lgan_undcp_real = self.Lgan_undcp(REAL_clear, REAL_gan_F)
        Lgan_p2p_real = self.Lgan_p2p(REAL_gan_C, REAL_clear)
        Ldisc_real = self.Ldisc(REAL_gan_F, 'valid', vis)
        # Weight Loss
        Wea_real, Wundcp_real, Wp2p_real, Wdisc_real = 1, 0.01, 100, 0.5
        Lreal = Wea_real*Lreid_ea_real + Wundcp_real*Lgan_undcp_real + Wp2p_real*Lgan_p2p_real + Wdisc_real*Ldisc_real
        
        # % Stage 2-3 optimize + backward
        ####################################################################
        # Total Loss: Lsyn + Lreal
        ####################################################################  
        Ltotal = Lsyn + Lreal
        if self.mix_precision:
            with amp.scale_loss(Ltotal, self.optim) as scaled_loss:
                scaled_loss.backward()
        else:
            Ltotal.backward()
        self.optim.step()
 
        # % Stage 2-4 tune discriminator
        ####################################################################
        # optim_D : Disc(In_clear, 'real'), Disc(Gc, 'fake')
        ####################################################################         
        # % FOR DISCRIMINATOR foggy
        self.optim_D.zero_grad()
        Real_loss = self.Ldisc(REAL_clear, 'valid')
        temp = self.fack_buffer.push_and_pop(REAL_gan_C)
        Fake_loss = self.Ldisc(temp.detach(), 'fake')
        loss_d = (Real_loss + Fake_loss)/2
        loss_d.backward()
        self.optim_D.step()

        # % eval ....
        ####################################################################
        # ACC, SSIM(gx,gt): ssim, psnr
        ####################################################################     
        ACC_c, ACC_f = (SYN_score_C.max(1)[1] == SYN_pids).float().mean(), (SYN_score_F.max(1)[1] == SYN_pids).float().mean()
        SSIM_c, PSNR_c = eval_dh(SYN_gan_C, SYN_clear)
        SSIM_f, PSNR_f = eval_dh(SYN_gan_F, SYN_foggy)

        SSIM_c_real, PSNR_c_real = eval_dh(REAL_gan_C, REAL_clear)      
        SSIM_f_real, PSNR_f_real = eval_dh(REAL_gan_F, REAL_clear)      

        # % record ....
        ####################################################################
        # ReID, GAN, SSIM, ACC
        # Syn_Plot, Real_Clear_Plot
        ####################################################################
        if self.batch_cnt % self.cfg.SOLVER.LOG_PERIOD == 0:
            self.logger.info('% Record... Epoch[{}] Iteration[{}/{}] : Base Lr: {:.2e}, Ltotal:{:.4e} ------- ' 
                             .format(self.train_epoch, self.batch_cnt, len(self.SYN_train_dl), self.scheduler.get_lr()[0], Ltotal.cpu().item()))
            self.logger.info('-SYN: Lreid(c/f):{:.5f}/{:.5f}, Lea:{:.5f}, Lp2p(c/f):{:.5f}/{:.5f}'
                             .format(Lreid_sup_synC.cpu().item(), Lreid_sup_synF.cpu().item(), Lreid_ea_syn.cpu().item(), Lgan_p2p_synC.cpu().item(), Lgan_p2p_synF.cpu().item())) 
            self.logger.info('-SYN: SSIM(c/f):{:.3f}/{:.3f}, PSNR(c/f):{:.3f}/{:.3f}, ACC(c/f):{:.4f}/{:.4f}'
                             .format(SSIM_c, SSIM_f, PSNR_c, PSNR_f, ACC_c, ACC_f))
            self.logger.info('-REAL: Lea:{:.5f}, Lundcp:{:.5f}, Lp2p:{:.5f}, Ldisc:{:.5f}'
                             .format(Lreid_ea_real.cpu().item(), Lgan_undcp_real.cpu().item(), Lgan_p2p_real.cpu().item(), Ldisc_real.cpu().item()))
            self.logger.info('-REAL: SSIM(c/f):{:.3f}/{:.3f}, PSNR(c/f):{:.3f}/{:.3f}'
                             .format(SSIM_c_real, SSIM_f_real, PSNR_c_real, PSNR_f_real))
            self.logger.info('-Discriminator:  Ldisc(n/a):{:.4f}, Real_L(n):{:.4f}, Fake_L(n):{:.4f}'
                             .format(loss_d.cpu().item(), Real_loss.cpu().item(), Fake_loss.cpu().item()))
            self.logger.info('----------------------------------------------------------------------------------------')
        # plot gan result
        if self.cfg.TEST.VIS and (self.batch_cnt == self.train_epoch):
            # for real
            sample_idx = 0
            sample_result_path = os.path.join(self.output_dir, "sample_result_for_REAL/")
            name = "Epoch_" + str(self.train_epoch) + "_" + REAL_clear_pth[sample_idx].split("/")[-1]
            save_full_path = sample_result_path + name
            Sinc = REAL_clear.cpu().numpy()[sample_idx].transpose((1, 2, 0))
            Sganc = REAL_gan_C.detach().cpu().numpy()[sample_idx].transpose((1,2,0))
            Sganf = REAL_gan_F.detach().cpu().numpy()[sample_idx].transpose((1,2,0))
            Render_plot((Sinc, Sganc, Sganf), save_full_path)
            # for syn
            sample_idx = 0
            sample_result_path = os.path.join(self.output_dir, "sample_result_for_SYN/")
            name = "Epoch_" + str(self.train_epoch) + "_" + SYN_clear_pth[sample_idx].split("/")[-1]
            save_full_path = sample_result_path + name
            Sinc = SYN_clear.cpu().numpy()[sample_idx].transpose((1, 2, 0))
            Sinf = SYN_foggy.cpu().numpy()[sample_idx].transpose((1, 2, 0))
            Sganc = SYN_gan_C.detach().cpu().numpy()[sample_idx].transpose((1,2,0))
            Sganf = SYN_gan_F.detach().cpu().numpy()[sample_idx].transpose((1,2,0))
            Render_plot((Sinc, Sganc, Sinf, Sganf), save_full_path)
        return 0

    def handle_new_batch(self,):
        self.batch_cnt += 1

    def handle_new_epoch(self):
        self.batch_cnt = 1
        self.scheduler.step()
        self.lr_scheduler_D.step()
        self.logger.info('Epoch {} done'.format(self.train_epoch))
        self.logger.info('-' * 20)

        if self.train_epoch % self.eval_period == 0:
            mAP, SSIM = self.evaluate()   # [mAP_c, mAP_f], [ssim_c, ssim_f]
            if mAP[0] > self.best_mAP_c:  # save the mAP_clear parameters.
                self.save()
                if len(self.best_epoch) > 5:
                    self.remove(self.best_epoch[0])
                    self.best_epoch.remove(self.best_epoch[0])
                self.best_epoch.append(self.train_epoch)
                self.best_mAP_c, self.best_mAP_f = mAP[0], mAP[1]
                self.best_SSIM_c, self.best_SSIM_f = SSIM[0], SSIM[1]
        
        if self.train_epoch % 50 == 0:
            self.save()

        self.train_epoch += 1
        self.logger.info('Best_epoch {}, best_mAP_c {}, best_mAP_f {}'.format(self.best_epoch[-1], self.best_mAP_c, self.best_mAP_f))

    def evaluate(self):
        self.model.eval()
        ####################################################################
        # - real_val_dl:  Getitem: {"imgs", "paths", "pids", "camids"}    
        # - Eval: {'Feat_clear', 'Feat_foggy', 'GAN_clear', 'GAN_foggy'}
        ####################################################################
        num_query = self.num_query
        BNc, BNf, pids, camids = [], [], [], []
        ssim_c, psnr_c, mse_c = [], [], []
        ssim_f, psnr_f, mse_f = [], [], []

        vis_idx = self.train_epoch
        if vis_idx > len(self.REAL_val_dl):
            vis_idx = vis_idx % len(self.REAL_val_dl)

        with torch.no_grad():
            for ct, batch in enumerate(tqdm(self.REAL_val_dl, total=len(self.REAL_val_dl), leave=False)):
                in_clear, path_clear, pid, camid = batch['imgs'], batch['paths'], batch['pids'], batch['camids']
                in_clear = in_clear.cuda()

                Output = self.model([in_clear], mode='real')  

                ### % reid.....
                feat_c, feat_f = Output['Feat_clear'], Output['Feat_foggy'],
                feat_c, feat_f = feat_c.detach().cpu(), feat_f.detach().cpu()
                BNc.append(feat_c)
                BNf.append(feat_f)
                pids.append(pid)
                camids.append(camid)

                ### % restoration.....
                gan_c, gan_f =  Output['GAN_clear'], Output['GAN_foggy'] 
                Gc, Gf = gan_c.cuda(), gan_f.cuda()

                loss_c, loss_f = self.mse_loss(Gc, in_clear), self.mse_loss(Gf, in_clear)
                SSIM_c, PSNR_c = eval_dh(Gc, in_clear)
                SSIM_f, PSNR_f = eval_dh(Gf, in_clear)

                ssim_c.append(float(SSIM_c)), psnr_c.append(float(PSNR_c)), mse_c.append(float(loss_c))
                ssim_f.append(float(SSIM_f)), psnr_f.append(float(PSNR_f)), mse_f.append(float(loss_f))

                if ct == vis_idx:
                    if self.cfg.TEST.VIS:
                        sample_idx = 0
                        sample_result_path = os.path.join(self.output_dir, "sample_result_for_val/")
                        name = "Epoch_" + str(self.train_epoch) + "_" + path_clear[sample_idx].split("/")[-1]
                        save_full_path = sample_result_path + name 
                        Sinc = in_clear.cpu().numpy()[sample_idx].transpose((1, 2, 0))
                        Sganc = Gc.detach().cpu().numpy()[sample_idx].transpose((1,2,0))
                        Sganf = Gf.detach().cpu().numpy()[sample_idx].transpose((1,2,0))
                        Render_plot((Sinc, Sganc, Sganf), save_full_path)
                
            ### % reid.....
            BNc, BNf = torch.cat(BNc, dim=0), torch.cat(BNf, dim=0)
            pids, camids = torch.cat(pids, dim=0), torch.cat(camids, dim=0)

            query_feat_c = BNc[:num_query]
            query_feat_f = BNf[:num_query]
            query_pid = pids[:num_query]
            query_camid = camids[:num_query]

            gallery_feat_c = BNc[num_query:]
            gallery_feat_f = BNf[num_query:]
            gallery_pid = pids[num_query:]
            gallery_camid = camids[num_query:]

            distmat_c, distmat_f = euclidean_dist(query_feat_c, gallery_feat_c), euclidean_dist(query_feat_f, gallery_feat_f)

            cmc_c, mAP_c, _ = eval_func(distmat_c.numpy(), query_pid.numpy(), gallery_pid.numpy(), 
                                        query_camid.numpy(), gallery_camid.numpy())

            cmc_f, mAP_f, _ = eval_func(distmat_f.numpy(), query_pid.numpy(), gallery_pid.numpy(), 
                                        query_camid.numpy(), gallery_camid.numpy())

            ### % dehaze.....
            ssim_c, psnr_c, mse_c = np.array(ssim_c).mean(),  np.array(psnr_c).mean(),  np.array(mse_c).mean()
            ssim_f, psnr_f, mse_f = np.array(ssim_f).mean(),  np.array(psnr_f).mean(),  np.array(mse_f).mean()

            self.logger.info('Validation Result:')
            for r in self.cfg.TEST.CMC:
                self.logger.info('CMC Rank-{}: clear:{:.4%}, foggy:{:.4%}'.format(r, cmc_c[r-1], cmc_f[r-1]))
            self.logger.info('mAP_c: {:.4%}, mAP_f: {:.4%}'.format(mAP_c, mAP_f))
            self.logger.info('SSIM_c: {:.3f}, SSIM_f: {:.3f}'.format(ssim_c, ssim_f))
            self.logger.info('PSNR_c: {:.3f}, PSNR_f: {:.3f}'.format(psnr_c, psnr_f))
            self.logger.info('MSE_c: {:.4f}, MSE_f: {:.4f}'.format(mse_c, mse_f))
            self.logger.info('-' * 20)

            return [mAP_c, mAP_f], [ssim_c, ssim_f]

    def save(self):
        torch.save(self.model.state_dict(), osp.join(self.output_dir, "checkpoint/", 
                self.cfg.MODEL.NAME + '_epoch' + str(self.train_epoch) + '.pth'))
        torch.save(self.model.state_dict(), osp.join(self.output_dir, "checkpoint/", 
                'best.pth'))
        if self.train_epoch > 20:
            torch.save(self.optim.state_dict(), osp.join(self.output_dir, "checkpoint/", 
                    self.cfg.MODEL.NAME + '_epoch'+ str(self.train_epoch) + '_optim.pth'))

    def remove(self, epoch):
        pre_checkpoint = osp.join(self.output_dir, "checkpoint/", self.cfg.MODEL.NAME + '_epoch' + str(epoch) + '.pth')
        if os.path.isfile(pre_checkpoint):
            os.remove(pre_checkpoint)

        pre_checkpointopt = osp.join(self.output_dir, "checkpoint/", self.cfg.MODEL.NAME + '_epoch'+ str(epoch) + '_optim.pth')
        if os.path.isfile(pre_checkpointopt):
            os.remove(pre_checkpointopt)     


if __name__ == '__main__':
    train()

