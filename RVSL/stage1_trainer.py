#######################################################
# Stage 1. for syn dataset training...
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

from dataset import make_FVRID_dataloader_SYN   
from models import build_Stage1_Model, convert_model
from losses import Loss_stage1_reid, Loss_stage1_generator, Loss_stage1_ea, Loss_discriminator, MSEloss 
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
        if not os.path.exists(os.path.join(output_dir, "sample_result_for_train/")):
            os.makedirs(os.path.join(output_dir, "sample_result_for_train/"))
        if not os.path.exists(os.path.join(output_dir, "sample_result_for_val/")):
            os.makedirs(os.path.join(output_dir, "sample_result_for_val/"))

    # Step3. logging
    num_gpus = torch.cuda.device_count()
    logger = setup_logger('RVSL_stage1', output_dir, 0)  # check
    logger.info('Using {} GPUS'.format(num_gpus))
    logger.info(args)
    logger.info('Running with config:\n{}'.format(cfg))
    logger.info("##############################################################")
    logger.info("# % TRAIN .....  Stage 1 !!!!!") # !!!!
    logger.info("# Backbone Model is {%s}" %(cfg.MODEL.NAME))
    logger.info("# Dataset is {%s}" %(cfg.DATASETS.NAMES))
    logger.info("# pretrain_model = {%s} #" %(cfg.MODEL.PRETRAIN_PATH))
    logger.info("# pretrain_baseline = {%s} #" %(cfg.MODEL.PRETRAIN_PATH_reid))
    logger.info("# pretrain_restoration = {%s} #" %(cfg.MODEL.PRETRAIN_PATH_res))
    logger.info("# data_path = {%s} #" %(cfg.DATASETS.DATA_PATH))
    logger.info("# SYN_TRAIN_CLEAR_PATH = {%s} #" %(cfg.DATASETS.SYN_TRAIN_CLEAR_PATH))
    logger.info("# SYN_TRAIN_FOGGY_PATH = {%s} #" %(cfg.DATASETS.SYN_TRAIN_FOGGY_PATH))
    logger.info("# SYN_QUERY_CLEAR_PATH = {%s} #" %(cfg.DATASETS.SYN_QUERY_CLEAR_PATH))
    logger.info("# SYN_QUERY_FOGGY_PATH = {%s} #" %(cfg.DATASETS.SYN_QUERY_FOGGY_PATH))
    logger.info("# SYN_GALLERY_CLEAR_PATH = {%s} #" %(cfg.DATASETS.SYN_GALLERY_CLEAR_PATH))
    logger.info("# SYN_GALLERY_FOGGY_PATH = {%s} #" %(cfg.DATASETS.SYN_GALLERY_FOGGY_PATH))
    logger.info("# OUTPUT_dir = {%s} #" %(cfg.OUTPUT_DIR))
    logger.info("# RVSL_REID_W = {%s} #" %(cfg.MODEL.RVSL_BASE_W))
    logger.info("# RVSL_REST_W = {%s} #" %(cfg.MODEL.RVSL_REST_W))
    logger.info("##############################################################")

    # Step4. Create FVRID dataloader
    ####################################################################
    ## SYN_train_loader, SYN_val_loader, len(dataset.SYN_query), SYN_num_classes
    ## - syn_train_dl: Getitem: {"Foggy", "Clear", "Foggy_paths", "Clear_paths", "pids", "camids"}
    ## - syn_val_dl:  Getitem: {"Foggy", "Clear", "Foggy_paths", "Clear_paths", "pids", "camids"}
    ####################################################################
    print("#In[0]: make_dataloader--------------------------------")
    train_dl, val_dl, num_query, num_classes = make_FVRID_dataloader_SYN(cfg, num_gpus)

    # Step5. Build RVSL Stage1 Model    
    ####################################################################
    # Stage 1. structure, for Syn datas training stage.
    # Input: 
    #     (clear, foggy), syn hazy pairs data.
    # Output:
    #     -Traing: {'Class_clear', 'Class_foggy', 'GlobalFeat_clear', 'GlobalFeat_foggy', 'GAN_clear', 'GAN_foggy'} 
    #     -Eval: {'Feat_clear', 'Feat_foggy', 'GAN_clear', 'GAN_foggy'}
    ####################################################################
    print("#In[1]: build_model--------------------------------")
    print("num_classes = ", num_classes)
    model = build_Stage1_Model(cfg, num_classes)
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
    # -- (1) ReID Loss: Triplet, Cross Entropy, input: (score, feat, target)
    # -- (2) Generator Loss: SmoothL1 , Discriminator
    # -- (3) Embedding Adaptive: L1
    ####################################################################
    print("#In[2]: make_loss--------------------------------")
    loss_reid = Loss_stage1_reid(cfg, num_classes)
    loss_gan = Loss_stage1_generator(cfg)
    loss_ea = Loss_stage1_ea(cfg)
    loss_dc, optimizer_Dc, lr_scheduler_Dc = Loss_discriminator(cfg)
    loss_df, optimizer_Df, lr_scheduler_Df = Loss_discriminator(cfg)

    # Step7. Training
    print("#In[3]: Start traing--------------------------------")
    trainer = STAGE1_Trainer(cfg, model, train_dl, val_dl, 
                             loss_reid, loss_gan, loss_ea, 
                             loss_dc, optimizer_Dc, lr_scheduler_Dc,
                             loss_df, optimizer_Df, lr_scheduler_Df,
                             num_query, num_gpus)    
    for epoch in range(trainer.epochs):
        print("----------- Epoch [%d/%d] ----------- "%(epoch,trainer.epochs))
        ct = 0
        for train_batch in trainer.train_dl:
            ct += 1
            print('step: %d'%(ct), end='\r')
            trainer.step(train_batch)
            trainer.handle_new_batch()
        trainer.handle_new_epoch()
    print("\n######### RVSL Stage1 Training Finish!! #############")

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

class STAGE1_Trainer(object):
    def __init__(self, cfg, model, train_dl, val_dl, 
                       loss_reid, loss_gan, loss_ea, 
                       loss_Dc, optim_Dc, lr_scheduler_Dc,
                       loss_Df, optim_Df, lr_scheduler_Df,
                       num_query, num_gpus):
        self.cfg = cfg
        self.std = self.cfg.INPUT.PIXEL_STD
        self.mean = self.cfg.INPUT.PIXEL_MEAN
        self.model = model
        self.train_dl = train_dl
        self.len_batchs = len(train_dl)
        self.val_dl = val_dl

        self.loss_reid = loss_reid
        self.loss_gan = loss_gan
        self.loss_ea = loss_ea

        self.loss_Dc = loss_Dc
        self.optim_Dc = optim_Dc
        self.lr_scheduler_Dc = lr_scheduler_Dc
        self.fake_C_buffer = ReplayBuffer()

        self.loss_Df = loss_Df
        self.optim_Df = optim_Df
        self.lr_scheduler_Df = lr_scheduler_Df  
        self.fake_F_buffer = ReplayBuffer()    

        self.mse_loss = MSEloss()
        self.num_query = num_query
        self.best_mAP_c = 0
        self.best_mAP_f = 0
        self.best_SSIM_c = 0
        self.best_SSIM_f = 0
        self.best_epoch = []

        self.AVG_CLEAR_reid = AvgerageMeter()
        self.AVG_CLEAR_gan = AvgerageMeter()
        self.AVG_CLEAR_SSIM = AvgerageMeter()
        self.AVG_CLEAR_ACC = AvgerageMeter()
        self.AVG_FOGGY_reid = AvgerageMeter()
        self.AVG_FOGGY_gan = AvgerageMeter()
        self.AVG_FOGGY_SSIM = AvgerageMeter()
        self.AVG_FOGGY_ACC = AvgerageMeter()
        self.AVG_EA = AvgerageMeter()
        self.train_epoch = 1
        self.batch_cnt = 0

        self.logger = logging.getLogger('RVSL_stage1.train') # !!! notice !!!
        self.log_period = cfg.SOLVER.LOG_PERIOD
        self.checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        self.eval_period = cfg.SOLVER.EVAL_PERIOD
        self.output_dir = cfg.OUTPUT_DIR
        self.device = cfg.MODEL.DEVICE
        self.epochs = cfg.SOLVER.MAX_EPOCHS

        if self.cfg.MODEL.TENSORBOARDX:
            print("############## create tensorboardx ##################")
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'Log_RVSL_Stage1'))
            self.writer_graph = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'Log_RVSL_Stage1/model'))
            # self.writer_graph.add_graph(self.model, 
            #                             torch.FloatTensor(np.random.rand(8, 3, 256, 256)), 
            #                             torch.FloatTensor(np.random.rand(8, 3, 256, 256)))    #  !!!! notice !!!!!

        # Single GPU model
        self.model.cuda()
        self.optim = make_optimizer(cfg, self.model, num_gpus)
        self.scheduler = WarmupMultiStepLR(self.optim, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                            cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        self.scheduler.step()
        self.mix_precision = False
        if cfg.SOLVER.FP16:
            from apex import amp
            self.model, self.optim = amp.initialize(self.model, self.optim, opt_level='O1')
            self.mix_precision = True
            self.logger.info('Using fp16 training')
        self.logger.info('Trainer Built')
        return

    def step(self, train_batch):
        self.model.train()
        self.optim.zero_grad()

        # % Stage 1. supervised learning
        # % load data & forward..
        #################################################################################
        ## syn_train_dl: Getitem: {"Foggy", "Clear", "Foggy_paths", "Clear_paths", "pids", "camids"}
        ## model(clear, foggy): {'Class_clear', 'Class_foggy', 'GlobalFeat_clear', 'GlobalFeat_foggy', 'GAN_clear', 'GAN_foggy'} 
        #################################################################################
        in_foggy, in_clear, path_foggy, path_clear, pids = train_batch['Foggy'], train_batch['Clear'], train_batch['Foggy_paths'], train_batch['Clear_paths'], train_batch['pids']
        in_foggy, in_clear, pids = in_foggy.cuda(), in_clear.cuda(), pids.cuda()
        Output = self.model(in_clear, in_foggy)  

        # % ReID branch + Embedding Adaptive
        ####################################################################
        # ReID Loss: input(score, feat, target), output:(xent+triplet)
        # EA Loss : input(clear_feat, foggy_feat), output: L1
        ####################################################################        
        Sc, Sf = Output['Class_clear'], Output['Class_foggy']
        GFc, GFf = Output['GlobalFeat_clear'], Output['GlobalFeat_foggy']
        Lreid_c, Lreid_f = self.loss_reid(Sc, GFc, pids), self.loss_reid(Sf, GFf, pids)
        Lea = self.loss_ea(GFc, GFf)

        # % GAN branch
        ####################################################################
        # GAN Loss: input(gx, gt), output: Smoothl1
        # Discri Loss: input(imgs, label='valid'/'fake'), output: loss
        ####################################################################  
        Gc, Gf =  Output['GAN_clear'].cuda(), Output['GAN_foggy'].cuda()
        if self.cfg.DATALOADER.NORMALZIE:
            Gc, Gf = Gc*0.5 + 0.5, Gf*0.5 + 0.5
            in_clear, in_foggy = in_clear*0.5 + 0.5, in_foggy*0.5 + 0.5
        Lgan_c, Lgan_f = self.loss_gan(Gc, in_clear), self.loss_gan(Gf, in_foggy)

        if self.batch_cnt % self.cfg.SOLVER.LOG_PERIOD == 0:
            vis = True
        else:
            vis = False
        Ldisc_c = self.loss_Dc(Gc, 'valid', vis)
        Ldisc_f = self.loss_Df(Gf, 'valid', vis)

        # % optimize + backward
        ####################################################################
        # Total Loss: W_reid*Lreid + W_gan*Lgan + W_ea*Lea + W_disc*Ldisc
        ####################################################################          
        W_reid, W_gan, W_ea, W_disc = 1, 0.5, 0.5, 0.5 
        L_total = W_reid*(Lreid_c+Lreid_f) + W_gan*(Lgan_c+Lgan_f) + W_ea*Lea + W_disc*((Ldisc_c+Ldisc_f)/2)
        if self.mix_precision:
            with amp.scale_loss(L_total, self.optim) as scaled_loss:
                scaled_loss.backward()
        else:
            L_total.backward()
        self.optim.step()

        # % FOR DISCRIMINATOR Clear
        self.optim_Dc.zero_grad()
        RC_loss = self.loss_Dc(in_clear, 'valid', vis)
        Gc_ = self.fake_C_buffer.push_and_pop(Gc)
        FC_loss = self.loss_Dc(Gc_.detach(), 'fake', vis)
        loss_dc = (RC_loss + FC_loss)/2
        loss_dc.backward()
        self.optim_Dc.step()

        # % FOR DISCRIMINATOR Foggy        
        self.optim_Df.zero_grad()
        RF_loss = self.loss_Df(in_foggy, 'valid', vis)
        Gf_ = self.fake_F_buffer.push_and_pop(Gf)
        FF_loss = self.loss_Df(Gf_.detach(), 'fake', vis)
        loss_df = (RF_loss + FF_loss)/2
        loss_df.backward()
        self.optim_Df.step()

        loss_d = (loss_dc + loss_df)/2
        
        # % eval ....
        ####################################################################
        # ACC, SSIM(gx,gt): ssim, psnr
        ####################################################################     
        ACC_c, ACC_f = (Sc.max(1)[1] == pids).float().mean(), (Sf.max(1)[1] == pids).float().mean()
        SSIM_c, PSNR_c = eval_dh(Gc, in_clear)
        SSIM_f, PSNR_f = eval_dh(Gf, in_clear)

        self.AVG_CLEAR_reid.update(Lreid_c.cpu().item())
        self.AVG_CLEAR_gan.update(Lgan_c.cpu().item()) 
        self.AVG_CLEAR_SSIM.update(SSIM_c)
        self.AVG_CLEAR_ACC.update(ACC_c)
        self.AVG_FOGGY_reid.update(Lreid_f.cpu().item())
        self.AVG_FOGGY_gan.update(Lgan_f.cpu().item()) 
        self.AVG_FOGGY_SSIM.update(SSIM_f)
        self.AVG_FOGGY_ACC.update(ACC_f) 
        self.AVG_EA.update(Lea.cpu().item())

        # % record ....
        ####################################################################
        # ReID, GAN, SSIM, ACC, EA
        # Stage1_Plot
        ####################################################################
        if self.batch_cnt % self.cfg.SOLVER.LOG_PERIOD == 0:
            self.logger.info('% Record... Epoch[{}] Iteration[{}/{}] : Base Lr: {:.2e}, Total Loss: {:.4f} -------' 
                             .format(self.train_epoch, self.batch_cnt, len(self.train_dl), self.scheduler.get_lr()[0], L_total.cpu().item()))
            self.logger.info('-Clear Branch:  Lreid(n/a):{:.4f}/{:.4f}, Lgan(n/a):{:.4f}/{:.4f}, Lea(n/a):{:.4f}/{:.4f}, Ldisc_c:{:.4f}, SSIM:{:.3f}, ACC(n/a):{:.3f}/{:.3f}'
                             .format(Lreid_c.cpu().item(), self.AVG_CLEAR_reid.avg, Lgan_c.cpu().item(), self.AVG_CLEAR_gan.avg, 
                                    Lea.cpu().item(), self.AVG_EA.avg, Ldisc_c.cpu().item() ,self.AVG_CLEAR_SSIM.avg, ACC_c, self.AVG_CLEAR_ACC.avg))
            self.logger.info('-Foggy Branch:  Lreid(n/a):{:.4f}/{:.4f}, Lgan(n/a):{:.4f}/{:.4f}, Lea(n/a):{:.4f}/{:.4f}, Ldisc_c:{:.4f}, SSIM:{:.3f}, ACC(n/a):{:.3f}/{:.3f}'
                             .format(Lreid_f.cpu().item(), self.AVG_FOGGY_reid.avg, Lgan_f.cpu().item(), self.AVG_FOGGY_gan.avg, 
                                    Lea.cpu().item(), self.AVG_EA.avg, Ldisc_f.cpu().item(), self.AVG_FOGGY_SSIM.avg, ACC_f, self.AVG_FOGGY_ACC.avg))
            self.logger.info('-Discrimator Module: Lrc:{:.4f}, Lrf:{:.4f}, Lfc:{:.4f}, Lff:{:.4f}, Ltotal:{:.4f}'
                            .format(RC_loss.cpu().item(), RF_loss.cpu().item(), FC_loss.cpu().item(), FF_loss.cpu().item(), loss_d.cpu().item()))
            self.logger.info('----------------------------------------------------------------------------------------')
            if self.cfg.MODEL.TENSORBOARDX:
                self.writer.add_scalars("TRAIN/Loss",{"Loss":L_total}, (self.train_epoch* self.len_batchs) + self.batch_cnt)
                self.writer.add_scalars("TRAIN/CLEAR",{"Lreid":Lreid_c, "Lgan":Lgan_c, "Lea":Lea, "Ldisc":Ldisc_c})
                self.writer.add_scalars("TRAIN/FOGGY",{"Lreid":Lreid_f, "Lgan":Lgan_f, "Lea":Lea, "Ldisc":Ldisc_f})
                self.writer.add_scalars("TRAIN/SSIM",{"SSIM_c":SSIM_c, "SSIM_f":SSIM_f,}, (self.train_epoch* self.len_batchs) + self.batch_cnt)
                self.writer.add_scalars("TRAIN/PSNR",{"PSNR_c":PSNR_c, "PSNR_f":PSNR_f}, (self.train_epoch* self.len_batchs) + self.batch_cnt)
                self.writer.add_scalars("TRAIN/ACC",{"ACC_c":ACC_c, "ACC_f":ACC_f}, (self.train_epoch* self.len_batchs) + self.batch_cnt)
                self.writer.add_scalars("TRAIN/LR",{"LR":self.scheduler.get_lr()[0]}, (self.train_epoch* self.len_batchs) + self.batch_cnt)
        # plot gan result
        if self.cfg.TEST.VIS and (self.batch_cnt == self.train_epoch):
            sample_idx = 0
            sample_result_path = os.path.join(self.output_dir, "sample_result_for_train/")
            name = "Epoch_" + str(self.train_epoch) + "_" + path_foggy[sample_idx].split("/")[-1]
            save_full_path = sample_result_path + name
            Sinc = in_clear.cpu().numpy()[sample_idx].transpose((1, 2, 0))
            Sinf = in_foggy.cpu().numpy()[sample_idx].transpose((1, 2, 0))
            Sganc = Gc.detach().cpu().numpy()[sample_idx].transpose((1,2,0))
            Sganf = Gf.detach().cpu().numpy()[sample_idx].transpose((1,2,0))
            Render_plot((Sinc, Sganc, Sinf, Sganf), save_full_path)
        return 0

    def handle_new_batch(self,):
        self.batch_cnt += 1

    def handle_new_epoch(self):
        self.batch_cnt = 1
        self.scheduler.step()
        self.lr_scheduler_Dc.step()
        self.lr_scheduler_Df.step()
        self.logger.info('Epoch {} done'.format(self.train_epoch))
        self.logger.info('-' * 20)

        if self.train_epoch % self.eval_period == 0:
            mAP, SSIM = self.evaluate()   # [mAP_c, mAP_f], [ssim_c, ssim_f]
            if mAP[1] > self.best_mAP_f:  # save the mAP_foggy parameters.
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
        ## syn_val_dl:  Getitem: {"Foggy", "Clear", "Foggy_paths", "Clear_paths", "pids", "camids"}
        ## model(clear, foggy): {'Feat_clear', 'Feat_foggy', 'GAN_clear', 'GAN_foggy'}
        ####################################################################
        num_query = self.num_query
        BNc, BNf, pids, camids = [], [], [], []
        ssim_c, psnr_c, mse_c = [], [], []
        ssim_f, psnr_f, mse_f = [], [], []

        vis_idx = self.train_epoch
        if vis_idx > len(self.val_dl):
            vis_idx = vis_idx % len(self.val_dl)

        with torch.no_grad():
            for ct, batch in enumerate(tqdm(self.val_dl, total=len(self.val_dl), leave=False)):
                in_foggy, in_clear, path_foggy, path_clear, pid, camid = batch['Foggy'], batch['Clear'], batch['Foggy_paths'], batch['Clear_paths'], batch['pids'], batch['camids']
                in_foggy, in_clear = in_foggy.cuda(), in_clear.cuda()

                Output = self.model(in_clear, in_foggy)  

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
                if self.cfg.DATALOADER.NORMALZIE:
                    Gc, Gf = Gc*0.5 + 0.5, Gf*0.5 + 0.5
                    in_clear, in_foggy = in_clear*0.5 + 0.5, in_foggy*0.5 + 0.5

                loss_c, loss_f = self.mse_loss(Gc, in_clear), self.mse_loss(Gf, in_foggy)
                SSIM_c, PSNR_c = eval_dh(Gc, in_clear)
                SSIM_f, PSNR_f = eval_dh(Gf, in_clear)

                ssim_c.append(float(SSIM_c)), psnr_c.append(float(PSNR_c)), mse_c.append(float(loss_c))
                ssim_f.append(float(SSIM_f)), psnr_f.append(float(PSNR_f)), mse_f.append(float(loss_f))

                if ct == vis_idx:
                    if self.cfg.TEST.VIS:
                        sample_idx = 0
                        sample_result_path = os.path.join(self.output_dir, "sample_result_for_val/")
                        name = "Epoch_" + str(self.train_epoch) + "_" + path_foggy[sample_idx].split("/")[-1]
                        save_full_path = sample_result_path + name
                        Sinc = in_clear.cpu().numpy()[sample_idx].transpose((1, 2, 0))
                        Sinf = in_foggy.cpu().numpy()[sample_idx].transpose((1, 2, 0))
                        Sganc = Gc.detach().cpu().numpy()[sample_idx].transpose((1,2,0))
                        Sganf = Gf.detach().cpu().numpy()[sample_idx].transpose((1,2,0))
                        Render_plot((Sinc, Sganc, Sinf, Sganf), save_full_path)

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
                self.logger.info('CMC Rank-{}: clear:{:.2%}, foggy:{:.2%}'.format(r, cmc_c[r-1], cmc_f[r-1]))
            self.logger.info('mAP_c: {:.2%}, mAP_f: {:.2%}'.format(mAP_c, mAP_f))
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