from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# Mode : 
# stage1. syn-data training stage,  
# stage2. real clean data training stage,
# stage3. real foggy data training stage.
# -----------------------------------------------------------------------------
_C.MODE = CN()
_C.MODE.STAGE = 'STAGE1'
_C.MODE.SEED = 1229
# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NAME = 'resnet50'   # resnet50, resnet101, resnet152
_C.MODEL.LAST_STRIDE = 1
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.PRETRAIN_PATH_reid = ''
_C.MODEL.PRETRAIN_PATH_res = ''
_C.MODEL.RVSL_BASE_W = 1.0
_C.MODEL.RVSL_REST_W = 1.0
_C.MODEL.LABEL_SMOOTH = False
_C.MODEL.TENSORBOARDX = True

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.0
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.0
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('FVRID')
# Root PATH to the dataset
_C.DATASETS.DATA_PATH = '/home/data/FVRID/'
# ------- use for opticalflow all syn data path
# PATH to train set
_C.DATASETS.SYN_TRAIN_CLEAR_PATH = 'syn_clear_input'
_C.DATASETS.SYN_TRAIN_FOGGY_PATH = 'syn_foggy_input'
_C.DATASETS.SYN_QUERY_CLEAR_PATH = 'syn_query'
_C.DATASETS.SYN_QUERY_FOGGY_PATH = 'syn_use_to_dehaze'
_C.DATASETS.SYN_GALLERY_CLEAR_PATH = 'syn_bounding_box_test'
_C.DATASETS.SYN_GALLERY_FOGGY_PATH = 'syn_use_to_dehaze'
# ------- use for Real world data path
# PATH to train set
_C.DATASETS.REAL_TRAIN_CLEAR_PATH = 'clear_input'
_C.DATASETS.REAL_TRAIN_FOGGY_PATH = 'foggy_input'
# PATH to query set
_C.DATASETS.REAL_QUERY_CLEAR_PATH = 'query'
_C.DATASETS.REAL_QUERY_FOGGY_PATH = 'use_to_dehaze'
# PATH to gallery set
_C.DATASETS.REAL_GALLERY_CLEAR_PATH = 'bounding_box_test'
_C.DATASETS.REAL_GALLERY_FOGGY_PATH = 'use_to_dehaze'
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax_triplet'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16
# Normalize
_C.DATALOADER.NORMALZIE = False

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.FP16 = False

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.MARGIN = 0.3

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30, 55)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 50
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.EVAL_PERIOD = 50
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64
# Whether or not use Cython to eval
_C.SOLVER.CYTHON = True

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.WEIGHT = ""
_C.TEST.DEBUG = False
_C.TEST.MULTI_GPU = False
_C.TEST.CMC = [1,5,10]
_C.TEST.VIS = False
_C.TEST.VIS_Q_NUM = 10
_C.TEST.VIS_G_NUM = 10
_C.TEST.RERANK = True
_C.TEST.DATA = 'REAL_FOGGY'

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""

# Alias for easy usage
cfg = _C
