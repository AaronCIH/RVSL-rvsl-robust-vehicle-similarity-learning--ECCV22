import torch
import os.path as osp
import glob, re
from PIL import ImageFile, Image
import random, math
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_image(img_path):
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    try:
        img = Image.open(img_path).convert('RGB')
    except IOError:
        print("IOError incurred when reading '{}'. ".format(img_path))
        pass
    return img

#############################################################################
# unpairs real world data, paris syn data
#############################################################################
class ImageDataset_FVRID(Dataset):
    """
    For "syn", "real" Dataset.
    - SYN:  input=[foggy_img_path, clear_img_path,  pid, camid], output[index]: foggy, clear, pid, camid, foggy_img_path, clear_img_path   
    - REAL: input=[img_path,  pid, camid], output[index]: img, pid, camid, img_path
    % Augment: {Resize, CenterCrop, HFlip, Pad, Normal, RandomErase}
    """
    def __init__(self, dataset, cfg, type='real', is_train=True):
        self.dataset = dataset
        self.cfg = cfg
        self.type = type
        self.is_train = is_train
        # RandomErasing.. params
        self.mean = (0.4914, 0.4822, 0.4465)
        self.sl = 0.02
        self.sh = 0.4
        self.r1 = 0.3

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index): 
        if self.type=='real':
            img_path, pid, camid = self.dataset[index]  # [img_path,  pid, camid]
            img = read_image(img_path)
            if self.is_train:
                trm_param = self._get_trm_param()
                transform = self._get_trm(Flip=trm_param['Flip'], CropSize=trm_param['CropSize'], Erase=trm_param['Erase'], Erase_param=trm_param['Erase_param'], is_train=self.is_train)
            else:
                transform = self._get_trm(is_train=False)
            img = transform(img)
            return img, pid, camid, img_path
        else:
            foggy_img_path, clear_img_path, pid, camid = self.dataset[index]  # [foggy_img_path, clear_img_path,  pid, camid]
            foggy = read_image(foggy_img_path)
            clear = read_image(clear_img_path)
            if self.is_train:
                trm_param = self._get_trm_param()
                transform = self._get_trm(Flip=trm_param['Flip'], CropSize=trm_param['CropSize'], Erase=trm_param['Erase'], Erase_param=trm_param['Erase_param'], is_train=self.is_train)
            else:
                transform = self._get_trm(is_train=False)
            foggy = transform(foggy)
            clear = transform(clear)
            return foggy, clear, pid, camid, foggy_img_path, clear_img_path        

    def _get_trm_param(self):
        # RandomHorizontalFlip p
        flip = random.random() > 0.5
        # RandomCrop
        h, w = self.cfg.INPUT.SIZE_TRAIN[0], self.cfg.INPUT.SIZE_TRAIN[1]
        hr, wr = random.uniform(1, 1.2), random.uniform(1, 1.2)
        new_h, new_w = int(h*hr), int(w*wr)
        # RandomErasing
        Erase = random.random() > 0.5
        if Erase:
            for attempt in range(100):
                area = h * w
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)
                eh = int(round(math.sqrt(target_area * aspect_ratio)))
                ew = int(round(math.sqrt(target_area / aspect_ratio)))
                if ew < w and eh < h:
                    x1 = random.randint(0, h - eh)
                    y1 = random.randint(0, w - ew)
                    return {"Flip":flip, "CropSize":(new_h, new_w), "Erase":Erase, 'Erase_param':[eh, ew, x1, y1]}
        else:
            return {"Flip":flip, "CropSize":(new_h, new_w), "Erase":False, "Erase_param":[0,0,0,0]}  # if not match erasing

    def _get_trm(self, Flip=False, CropSize=(384,384), Erase=True, Erase_param=[0,0,0,0], is_train=True):  # Erase_param = [h,w,x1,y1]
        Trm_list = []
        if is_train:
            Trm_list.append(T.Resize(CropSize))
            if Flip: Trm_list.append(T.RandomHorizontalFlip(p=1.0))
            Trm_list.append(T.Pad(self.cfg.INPUT.PADDING))
            Trm_list.append(T.CenterCrop(self.cfg.INPUT.SIZE_TRAIN))
        else:
            Trm_list.append(T.Resize(self.cfg.INPUT.SIZE_TRAIN))    
        Trm_list.append(T.ToTensor())

        if self.cfg.DATALOADER.NORMALZIE:
            normalize_transform = T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
            Trm_list.append(normalize_transform)

        if is_train & Erase:
            Trm_list.append(T.Lambda(lambda img: self.__RandomErasing(img, Erase_param)))

        transform = T.Compose(Trm_list)
        return transform

    def __RandomErasing(self, img, params=[0,0,0,0]):
        # params = h, w, x1, y1
        if img.size()[0] == 3:
            img[0, params[2]:params[2] + params[0], params[3]:params[3] + params[1]] = self.mean[0]
            img[1, params[2]:params[2] + params[0], params[3]:params[3] + params[1]] = self.mean[1]
            img[2, params[2]:params[2] + params[0], params[3]:params[3] + params[1]] = self.mean[2]
        else:
            img[0, params[2]:params[2] + params[0], params[3]:params[3] + params[1]] = self.mean[0]
        return img

class FVRIDDataset:
    '''
    according stage, category: SYN_train, SYN_query, SYN_gallery, Real_train, Real_query, Real_gallery
    - SYN_train: [foggy_img_path, clear_img_path,  pid, camid]
    - REAL_train, REAL_query, REAL_gallery: [img_path,  pid, camid]
    '''
    def __init__(self, root="/", stage=1, dir_dict={},
                    verbose=True, **kwargs):
        self.dataset_dir = root
        self.stage = stage
        self.dir_dict = dir_dict
        # % For SYN Dataset
        self.syn_train_clear_dir = osp.join(self.dataset_dir, self.dir_dict['syn_train_clear_dir'])
        print("syn_train_clear_dir:", self.dir_dict['syn_train_clear_dir'])
        self.syn_train_foggy_dir = osp.join(self.dataset_dir, self.dir_dict['syn_train_foggy_dir'])
        print("syn_train_foggy_dir:", self.dir_dict['syn_train_foggy_dir'])  
        if self.stage == 1:
            self.syn_query_clear_dir = osp.join(self.dataset_dir, self.dir_dict['syn_query_clear_dir'])
            print("syn_query_clear_dir:", self.dir_dict['syn_query_clear_dir'])
            self.syn_query_foggy_dir = osp.join(self.dataset_dir, self.dir_dict['syn_query_foggy_dir'])
            print("syn_query_foggy_dir:", self.dir_dict['syn_query_foggy_dir'])
            self.syn_gallery_clear_dir = osp.join(self.dataset_dir, self.dir_dict['syn_gallery_clear_dir'])
            print("syn_gallery_clear_dir:", self.dir_dict['syn_gallery_clear_dir'])
            self.syn_gallery_foggy_dir = osp.join(self.dataset_dir, self.dir_dict['syn_gallery_foggy_dir'])
            print("syn_gallery_foggy_dir:", self.dir_dict['syn_gallery_foggy_dir'])

        # % For Real Dataset
        if self.stage != 1:
            self.real_train_dir = osp.join(self.dataset_dir, self.dir_dict['real_train_dir'])
            print("real_train_dir:", self.dir_dict['real_train_dir'])
            self.real_query_dir = osp.join(self.dataset_dir, self.dir_dict['real_query_dir'])
            print("real_query_dir:", self.dir_dict['real_query_dir'])
            self.real_gallery_dir = osp.join(self.dataset_dir, self.dir_dict['real_gallery_dir'])
            print("real_gallery_dir:", self.dir_dict['real_gallery_dir'])

        # 確認路徑存在 否則報錯
        self._check_before_run()

        # % SYN DATASET PROCESS
        # - Output: [foggy_img_path, clear_img_path,  pid, camid]
        self.SYN_train = self._process_dir_fvrid_syn(self.syn_train_clear_dir, self.syn_train_foggy_dir, relabel=True)
        if self.stage == 1:
            self.SYN_query = self._process_dir_fvrid_syn(self.syn_query_clear_dir, self.syn_query_foggy_dir, relabel=False)
            self.SYN_gallery = self._process_dir_fvrid_syn(self.syn_gallery_clear_dir, self.syn_gallery_foggy_dir, relabel=False)

        # % Real DATASET PROCESS
        # - Output: [img_path, pid, camid]
        if self.stage != 1:
            self.Real_train = self._process_dir_fvrid_real(self.real_train_dir, relabel=True)
            self.Real_query = self._process_dir_fvrid_real(self.real_query_dir, relabel=False)
            self.Real_gallery = self._process_dir_fvrid_real(self.real_gallery_dir, relabel=False)
   
        # % STASTIC
        self.syn_train_pids, self.syn_train_imgs, self.syn_train_cams = self.get_imagedata_info(self.SYN_train, type='syn')
        if self.stage == 1:
            self.syn_query_pids, self.syn_query_imgs, self.syn_query_cams = self.get_imagedata_info(self.SYN_query, type='syn')
            self.syn_gallery_pids, self.syn_gallery_imgs, self.syn_gallery_cams = self.get_imagedata_info(self.SYN_gallery, type='syn')
        else:
            self.real_train_pids, self.real_train_imgs, self.real_train_cams = self.get_imagedata_info(self.Real_train, type='real')
            self.real_query_pids, self.real_query_imgs, self.real_query_cams = self.get_imagedata_info(self.Real_query, type='real')
            self.real_gallery_pids, self.real_gallery_imgs, self.real_gallery_cams = self.get_imagedata_info(self.Real_gallery, type='real')
        if verbose:
            print("=> Data loaded")
            self.print_dataset_statistics()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.syn_train_clear_dir):
            raise RuntimeError("'{}' is not available".format(self.syn_train_clear_dir))
        if not osp.exists(self.syn_train_foggy_dir):
            raise RuntimeError("'{}' is not available".format(self.syn_train_foggy_dir))
        if self.stage == 1:
            if not osp.exists(self.syn_query_clear_dir):
                raise RuntimeError("'{}' is not available".format(self.syn_query_clear_dir))
            if not osp.exists(self.syn_query_foggy_dir):
                raise RuntimeError("'{}' is not available".format(self.syn_query_foggy_dir))
            if not osp.exists(self.syn_gallery_clear_dir):
                raise RuntimeError("'{}' is not available".format(self.syn_gallery_clear_dir))    
            if not osp.exists(self.syn_gallery_foggy_dir):
                raise RuntimeError("'{}' is not available".format(self.syn_gallery_foggy_dir))                
        else:
            if not osp.exists(self.real_train_dir):
                raise RuntimeError("'{}' is not available".format(self.real_train_dir))
            if not osp.exists(self.real_query_dir):
                raise RuntimeError("'{}' is not available".format(self.real_query_dir))
            if not osp.exists(self.real_gallery_dir):
                raise RuntimeError("'{}' is not available".format(self.real_gallery_dir))

    def _process_dir_fvrid_real(self, paths, relabel=False, num=None):
        if num == None:
            img_paths = glob.glob(osp.join(paths, '*.jpg'))
        else:
            img_paths = glob.glob(osp.join(paths, '*.jpg'))[:num]
        pattern = re.compile(r'([-\d]+)_c([\d]+)')
        # make relabel library for train classification 
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        # make basedataset
        dataset = []
        for img_path in img_paths:  
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))    # [img_path,  pid, camid]
        return dataset

    def _process_dir_fvrid_syn(self, clear_path, foggy_path, relabel=False, num=None):
        if num == None:
            foggy_img_paths = glob.glob(osp.join(foggy_path, '*.jpg'))
        else:
            foggy_img_paths = glob.glob(osp.join(foggy_path, '*.jpg'))[:num]
        pattern = re.compile(r'([-\d]+)_c([\d]+)')
        # make relabel library for train classification 
        pid_container = set()
        for img_path in foggy_img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        # make basedataset
        dataset = []
        for img_path in foggy_img_paths:  
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            clear_img_path = osp.join(clear_path, img_path.split("/")[-1])
            dataset.append((img_path, clear_img_path, pid, camid))    # [foggy_img_path, clear_img_path,  pid, camid]
        return dataset

    def get_imagedata_info(self, data, type='real'):
        pids, cams = [], []
        if type=='real':
            for _, pid, camid in data:
                pids += [pid]
                cams += [camid]
        else:
            for _, _, pid, camid in data:
                pids += [pid]
                cams += [camid]            
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self, ):
        print("Dataset statistics:")
        print("  -------------------------------------------")
        print("  subset       | # ids | # images | # cameras")
        print("  -------------------------------------------")
        print("  SYN_train    | {:5d} | {:8d} | {:9d}".format(self.syn_train_pids, self.syn_train_imgs, self.syn_train_cams))
        if self.stage == 1:
            print("  SYN_query    | {:5d} | {:8d} | {:9d}".format(self.syn_query_pids, self.syn_query_imgs, self.syn_query_cams))
            print("  SYN_gallery  | {:5d} | {:8d} | {:9d}".format(self.syn_gallery_pids, self.syn_gallery_imgs, self.syn_gallery_cams))
        else:
            print("  REAL_train   | {:5d} | {:8d} | {:9d}".format(self.real_train_pids, self.real_train_imgs, self.real_train_cams))
            print("  REAL_query   | {:5d} | {:8d} | {:9d}".format(self.real_query_pids, self.real_query_imgs, self.real_query_cams))
            print("  REAL_gallery | {:5d} | {:8d} | {:9d}".format(self.real_gallery_pids, self.real_gallery_imgs, self.real_gallery_cams))
        print("  -------------------------------------------")

def Dataset_Stage1(cfg):
    '''
    - category: 
        SYN_train, SYN_query, SYN_gallery, 
        syn_train_pids, syn_train_imgs, syn_train_cams,
        syn_query_pids, syn_query_imgs, syn_query_cams,
        syn_gallery_pids, syn_gallery_imgs, syn_gallery_cams
    - getitem:
        SYN_data: [foggy_img_path, clear_img_path,  pid, camid]
    '''
    return FVRIDDataset(root=cfg.DATASETS.DATA_PATH, stage=1, dir_dict={
                            'syn_train_clear_dir':cfg.DATASETS.SYN_TRAIN_CLEAR_PATH, 
                            'syn_train_foggy_dir':cfg.DATASETS.SYN_TRAIN_FOGGY_PATH,  
                            'syn_query_clear_dir':cfg.DATASETS.SYN_QUERY_CLEAR_PATH, 
                            'syn_query_foggy_dir':cfg.DATASETS.SYN_QUERY_FOGGY_PATH, 
                            'syn_gallery_clear_dir':cfg.DATASETS.SYN_GALLERY_CLEAR_PATH, 
                            'syn_gallery_foggy_dir':cfg.DATASETS.SYN_GALLERY_FOGGY_PATH})

def Dataset_Stage2(cfg):
    '''
    - category: 
        SYN_train, Real_train, Real_query, Real_gallery,
        syn_train_pids, syn_train_imgs, syn_train_cams,
        real_train_pids, real_train_imgs, real_train_cams,
        real_query_pids, real_query_imgs, real_query_cams,
        real_gallery_pids, real_gallery_imgs, real_gallery_cams
    - getitem:
        SYN_data: [foggy_img_path, clear_img_path,  pid, camid]
        REAL_data: [img_path,  pid, camid]        
    '''
    return FVRIDDataset(root=cfg.DATASETS.DATA_PATH, stage=2, dir_dict={
                            'syn_train_clear_dir':cfg.DATASETS.SYN_TRAIN_CLEAR_PATH, 
                            'syn_train_foggy_dir':cfg.DATASETS.SYN_TRAIN_FOGGY_PATH,  
                            'real_train_dir':cfg.DATASETS.REAL_TRAIN_CLEAR_PATH, 
                            'real_query_dir':cfg.DATASETS.REAL_QUERY_CLEAR_PATH, 
                            'real_gallery_dir':cfg.DATASETS.REAL_GALLERY_CLEAR_PATH})

def Dataset_Stage3(cfg):
    '''
    - category: 
        SYN_train, Real_train, Real_query, Real_gallery,
        syn_train_pids, syn_train_imgs, syn_train_cams,
        real_train_pids, real_train_imgs, real_train_cams,
        real_query_pids, real_query_imgs, real_query_cams,
        real_gallery_pids, real_gallery_imgs, real_gallery_cams
    - getitem:
        SYN_data: [foggy_img_path, clear_img_path,  pid, camid]
        REAL_data: [img_path,  pid, camid]        
    '''
    return FVRIDDataset(root=cfg.DATASETS.DATA_PATH, stage=3, dir_dict={
                            'syn_train_clear_dir':cfg.DATASETS.SYN_TRAIN_CLEAR_PATH, 
                            'syn_train_foggy_dir':cfg.DATASETS.SYN_TRAIN_FOGGY_PATH,  
                            'real_train_dir':cfg.DATASETS.REAL_TRAIN_FOGGY_PATH, 
                            'real_query_dir':cfg.DATASETS.REAL_QUERY_FOGGY_PATH, 
                            'real_gallery_dir':cfg.DATASETS.REAL_GALLERY_FOGGY_PATH})
