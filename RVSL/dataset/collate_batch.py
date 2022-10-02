# encoding: utf-8
import torch

def syn_collate_FVRID_fn(batch):
    '''
    Getitem: {"Foggy", "Clear", "Foggy_paths", "Clear_paths", "pids", "camids"}
    '''
    foggy, clear, pids, camids, foggy_paths, clear_paths = zip(*batch) #     output[index]= [foggy, clear, pid, camid, foggy_img_path, clear_img_path]    
    foggy =  torch.stack(foggy, dim=0)
    clear =  torch.stack(clear, dim=0)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return {"Foggy":foggy, "Clear":clear,
            "Foggy_paths":foggy_paths, "Clear_paths":clear_paths,
            "pids":pids, "camids":camids}

def real_collate_FVRID_fn(batch):
    '''
    Getitem: {"imgs", "paths", "pids", "camids"}
    '''
    imgs, pids, camids, img_paths = zip(*batch) #   output[index]: [img, pid, camid, img_path]
    imgs =  torch.stack(imgs, dim=0)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return {"imgs":imgs, "paths":img_paths,
            "pids":pids, "camids":camids}