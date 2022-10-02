#######################################################
# Inference..
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

from dataset import make_FVRID_dataloader_SYN, make_FVRID_dataloader_REAL
from models import build_Stage1_Model, convert_model
from losses import MSEloss
from utils import Render_plot, setup_logger
from evaluate import eval_func, euclidean_dist, eval_dh  

def test():
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
    if cfg.TEST.VIS:  # sample for validation
        if not os.path.exists(os.path.join(output_dir, "sample_result/")):
            os.makedirs(os.path.join(output_dir, "sample_result/"))
    
    # Step3. logging
    num_gpus = torch.cuda.device_count()
    logger = setup_logger('RVSL', output_dir, 0)  # check
    logger.info('Using {} GPUS'.format(num_gpus))
    logger.info(args)
    logger.info('Running with config:\n{}'.format(cfg))
    logger.info("##############################################################")
    logger.info("# % TESTING !!!!!") # !!!!
    logger.info("# TEST for {%s}" %(cfg.TEST.DATA))
    logger.info("# TEST_WEIGHT = {%s} #" %(cfg.TEST.WEIGHT))
    logger.info("# Backbone Model is {%s}" %(cfg.MODEL.NAME))
    logger.info("# Dataset is {%s}" %(cfg.DATASETS.NAMES))
    logger.info("# data_path = {%s} #" %(cfg.DATASETS.DATA_PATH))
    logger.info("# SYN_TRAIN_CLEAR_PATH = {%s} #" %(cfg.DATASETS.SYN_TRAIN_CLEAR_PATH))
    logger.info("# SYN_TRAIN_FOGGY_PATH = {%s} #" %(cfg.DATASETS.SYN_TRAIN_FOGGY_PATH))
    logger.info("# SYN_QUERY_CLEAR_PATH = {%s} #" %(cfg.DATASETS.SYN_QUERY_CLEAR_PATH))
    logger.info("# SYN_QUERY_FOGGY_PATH = {%s} #" %(cfg.DATASETS.SYN_QUERY_FOGGY_PATH))
    logger.info("# SYN_GALLERY_CLEAR_PATH = {%s} #" %(cfg.DATASETS.SYN_GALLERY_CLEAR_PATH))
    logger.info("# SYN_GALLERY_FOGGY_PATH = {%s} #" %(cfg.DATASETS.SYN_GALLERY_FOGGY_PATH))
    logger.info("# REAL_TRAIN_CLEAR_PATH = {%s} #" %(cfg.DATASETS.REAL_TRAIN_CLEAR_PATH))
    logger.info("# REAL_QUERY_CLEAR_PATH = {%s} #" %(cfg.DATASETS.REAL_QUERY_CLEAR_PATH))
    logger.info("# REAL_GALLERY_CLEAR_PATH = {%s} #" %(cfg.DATASETS.REAL_GALLERY_CLEAR_PATH))
    logger.info("# REAL_TRAIN_FOGGY_PATH = {%s} #" %(cfg.DATASETS.REAL_TRAIN_FOGGY_PATH))
    logger.info("# REAL_QUERY_FOGGY_PATH = {%s} #" %(cfg.DATASETS.REAL_QUERY_FOGGY_PATH))
    logger.info("# REAL_GALLERY_FOGGY_PATH = {%s} #" %(cfg.DATASETS.REAL_GALLERY_FOGGY_PATH))
    logger.info("# OUTPUT_dir = {%s} #" %(cfg.OUTPUT_DIR))
    logger.info("##############################################################")

    # Step4. Create FVRID dataloader
    print("#In[0]: make_dataloader--------------------------------")
    if cfg.TEST.DATA == 'SYN':
        SYN_train_dl, val_dl, num_query, SYN_num_classes = make_FVRID_dataloader_SYN(cfg, num_gpus)
    elif cfg.TEST.DATA == 'REAL_CLEAR':
        SYN_train_dl, REAL_train_dl, val_dl, num_query, SYN_num_classes, _ = make_FVRID_dataloader_REAL(cfg, datatype='clear', num_gpus=num_gpus)
    elif cfg.TEST.DATA == 'REAL_FOGGY':
        SYN_train_dl, REAL_train_dl, val_dl, num_query, SYN_num_classes, _ = make_FVRID_dataloader_REAL(cfg, datatype='foggy', num_gpus=num_gpus)

    # Step5. Build RVSL Model    
    print("#In[1]: build_model--------------------------------")
    model = build_Stage1_Model(cfg, SYN_num_classes)
    if cfg.TEST.MULTI_GPU:
        model = nn.DataParallel(model)
        model = convert_model(model)
        logger.info('Use multi gpu to inference')

    if cfg.TEST.WEIGHT != '':
        param = torch.load(cfg.TEST.WEIGHT)
        for i in param:
            if 'fc' in i: continue
            if i not in model.state_dict().keys(): continue
            if param[i].shape != model.state_dict()[i].shape: continue
            model.state_dict()[i].copy_(param[i])
    model.cuda()
    model.eval()

    # Step7. Testing
    print("#In[3]: Start testing--------------------------------")
    ####################################################################
    ## syn_val_dl:  Getitem: {"Foggy", "Clear", "Foggy_paths", "Clear_paths", "pids", "camids"}
    ## real_val_dl:  Getitem: {"imgs", "paths", "pids", "camids"}    
    ## model(clear, foggy): {'Feat_clear', 'Feat_foggy', 'GAN_clear', 'GAN_foggy'}
    ####################################################################
    BNc, BNf, pids, camids = [], [], [], []
    ssim_c, psnr_c, mse_c = [], [], []
    ssim_f, psnr_f, mse_f = [], [], []
    mse_loss = MSEloss()
    with torch.no_grad():
        for ct, batch in enumerate(tqdm(val_dl, total=len(val_dl), leave=False)):
            if cfg.TEST.DATA == 'SYN':
                in_foggy, in_clear, path_foggy, path_clear, pid, camid = batch['Foggy'], batch['Clear'], batch['Foggy_paths'], batch['Clear_paths'], batch['pids'], batch['camids']
            elif cfg.TEST.DATA == 'REAL_CLEAR':
                in_clear, path_clear, pid, camid = batch['imgs'], batch['paths'], batch['pids'], batch['camids']
                in_foggy, path_foggy = in_clear, path_clear  # temp
            elif cfg.TEST.DATA == 'REAL_FOGGY':
                in_foggy, path_foggy, pid, camid = batch['imgs'], batch['paths'], batch['pids'], batch['camids']
                in_clear, path_clear = in_foggy, path_foggy  # temp

            in_foggy, in_clear = in_foggy.cuda(), in_clear.cuda()
            Output = model(in_clear, in_foggy)  

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
            if cfg.DATALOADER.NORMALZIE:
                Gc, Gf = Gc*0.5 + 0.5, Gf*0.5 + 0.5
                in_clear, in_foggy = in_clear*0.5 + 0.5, in_foggy*0.5 + 0.5

            loss_c, loss_f = mse_loss(Gc, in_clear), mse_loss(Gf, in_foggy)
            SSIM_c, PSNR_c = eval_dh(Gc, in_clear)
            SSIM_f, PSNR_f = eval_dh(Gf, in_clear)

            ssim_c.append(float(SSIM_c)), psnr_c.append(float(PSNR_c)), mse_c.append(float(loss_c))
            ssim_f.append(float(SSIM_f)), psnr_f.append(float(PSNR_f)), mse_f.append(float(loss_f))

            if ct % 100 == 0:
                if cfg.TEST.VIS:
                    sample_idx = 0
                    sample_result_path = os.path.join(output_dir, "sample_result/")
                    name = "Test_%d_"%(ct) + path_foggy[sample_idx].split("/")[-1]
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

        logger.info('Validation Result:')
        for r in cfg.TEST.CMC:
            logger.info('CMC Rank-{}: clear:{:.2%}, foggy:{:.2%}'.format(r, cmc_c[r-1], cmc_f[r-1]))
        logger.info('mAP_c: {:.2%}, mAP_f: {:.2%}'.format(mAP_c, mAP_f))
        logger.info('SSIM_c: {:.3f}, SSIM_f: {:.3f}'.format(ssim_c, ssim_f))
        logger.info('PSNR_c: {:.3f}, PSNR_f: {:.3f}'.format(psnr_c, psnr_f))
        logger.info('MSE_c: {:.4f}, MSE_f: {:.4f}'.format(mse_c, mse_f))
        logger.info('-' * 20)

if __name__ == '__main__':
    test()

