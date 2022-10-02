from .collate_batch import syn_collate_FVRID_fn, real_collate_FVRID_fn
from .triplet_sampler import RandomIdentitySampler_SYN, RandomIdentitySampler_REAL
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader
from .data import ImageDataset_FVRID, Dataset_Stage1, Dataset_Stage2, Dataset_Stage3

def make_FVRID_dataloader_SYN(cfg, num_gpus=1):
    """
    # return: SYN_train_loader, SYN_val_loader, len(dataset.SYN_query), SYN_num_classes
    # - syn_train_dl: Getitem: {"Foggy", "Clear", "Foggy_paths", "Clear_paths", "pids", "camids"}
    # - syn_val_dl:  Getitem: {"Foggy", "Clear", "Foggy_paths", "Clear_paths", "pids", "camids"}
    """
    print("###########################################################")
    print("### Make dataloader for FVRID !!!                       ###")
    print("###########################################################")
    num_workers = cfg.DATALOADER.NUM_WORKERS * num_gpus
    dataset = Dataset_Stage1(cfg)  # category: SYN_train, SYN_query, SYN_gallery

    SYN_num_classes = dataset.syn_train_pids

    # SYN DataLoader, output[index]= [foggy, clear, pid, camid, foggy_img_path, clear_img_path]
    syn_train_set = ImageDataset_FVRID(dataset.SYN_train, cfg, type='syn', is_train=True)
    SYN_train_loader = DataLoader(
        syn_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH * num_gpus,
        sampler=RandomIdentitySampler_SYN(dataset.SYN_train,
                                          cfg.SOLVER.IMS_PER_BATCH * num_gpus,
                                          cfg.DATALOADER.NUM_INSTANCE * num_gpus),
        num_workers=num_workers, collate_fn=syn_collate_FVRID_fn,
        pin_memory=True)

    SYN_val_set = ImageDataset_FVRID(dataset.SYN_query + dataset.SYN_gallery, cfg, type='syn', is_train=False) 
    SYN_val_loader = DataLoader(
            SYN_val_set, batch_size=cfg.TEST.IMS_PER_BATCH * num_gpus, shuffle=False,
            num_workers=num_workers,
            collate_fn=syn_collate_FVRID_fn
    )
    return SYN_train_loader, SYN_val_loader, len(dataset.SYN_query), SYN_num_classes

def make_FVRID_dataloader_REAL(cfg, datatype='clear', num_gpus=1):
    """
    # return: SYN_train_loader, REAL_train_loader, REAL_val_loader, len(dataset.Real_query), SYN_num_classes, REAL_num_classes
    # - syn_train_dl: Getitem: {"Foggy", "Clear", "Foggy_paths", "Clear_paths", "pids", "camids"}
    # - real_train_dl:Getitem: {"imgs", "paths", "pids", "camids"}
    # - real_val_dl:  Getitem: {"imgs", "paths", "pids", "camids"}
    """
    print("###########################################################")
    print("### Make dataloader for FVRID !!!                       ###")
    print("###########################################################")
    num_workers = cfg.DATALOADER.NUM_WORKERS * num_gpus
    if datatype=='clear':
        dataset = Dataset_Stage2(cfg)  # category: SYN_train, Real_train, Real_query, Real_gallery
    else:
        dataset = Dataset_Stage3(cfg)  # category: SYN_train, Real_train, Real_query, Real_gallery

    REAL_num_classes = dataset.real_train_pids
    SYN_num_classes = dataset.syn_train_pids

    # SYN DataLoader, output[index]= [foggy, clear, pid, camid, foggy_img_path, clear_img_path]
    syn_train_set = ImageDataset_FVRID(dataset.SYN_train, cfg, type='syn', is_train=True)
    SYN_train_loader = DataLoader(
        syn_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH * num_gpus,
        sampler=RandomIdentitySampler_SYN(dataset.SYN_train,
                                          cfg.SOLVER.IMS_PER_BATCH * num_gpus,
                                          cfg.DATALOADER.NUM_INSTANCE * num_gpus),
        num_workers=num_workers, collate_fn=syn_collate_FVRID_fn,
        pin_memory=True)

    # REAL DataLoader, output[index]: [img, pid, camid, img_path]
    real_train_set = ImageDataset_FVRID(dataset.Real_train, cfg, type='real', is_train=True) 
    REAL_train_loader = DataLoader(
        real_train_set, batch_size= cfg.DATALOADER.NUM_INSTANCE * 3 * num_gpus,
        sampler=RandomIdentitySampler_REAL(dataset.Real_train,
                                           cfg.DATALOADER.NUM_INSTANCE * 3 * num_gpus,
                                           cfg.DATALOADER.NUM_INSTANCE * num_gpus),
        num_workers=num_workers, collate_fn=real_collate_FVRID_fn,
        pin_memory=True
    )

    real_val_set = ImageDataset_FVRID(dataset.Real_query + dataset.Real_gallery, cfg, type='real', is_train=False) 
    REAL_val_loader = DataLoader(
            real_val_set, batch_size=cfg.TEST.IMS_PER_BATCH * num_gpus, shuffle=False,
            num_workers=num_workers,
            collate_fn=real_collate_FVRID_fn
    )
    return SYN_train_loader, REAL_train_loader, REAL_val_loader, len(dataset.Real_query), SYN_num_classes, REAL_num_classes