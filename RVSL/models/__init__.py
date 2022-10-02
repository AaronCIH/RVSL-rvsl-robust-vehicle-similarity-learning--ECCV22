from .models import Stage1_Model, Stage2_Model, Stage3_Model
from .sync_bn import convert_model

def build_Stage1_Model(cfg, num_classes):
    '''
    Stage 1. structure, for Syn datas training stage.
    Input: 
        (clear, foggy), syn hazy pairs data.
    Output:
        -Traing: {'Class_clear', 'Class_foggy', 'GlobalFeat_clear', 'GlobalFeat_foggy', 'GAN_clear', 'GAN_foggy'} 
        -Eval: {'Feat_clear', 'Feat_foggy', 'GAN_clear', 'GAN_foggy'}
    '''
    img_shape = (1, 64, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])
    model = Stage1_Model(cfg.MODEL.NAME, num_classes, img_shape, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH_reid, cfg.MODEL.PRETRAIN_PATH_res)
    print(cfg.MODEL.NAME)
    return model

def build_Stage2_Model(cfg, syn_num_classes):
    '''
    Stage 2. structure, for Real clear training stage. use syn paris for auxiliary learning.
    Input: (datas, mode='real'/'syn')
    Output:
        -Traing['syn']: {'Class_clear', 'Class_foggy', 'GlobalFeat_clear', 'GlobalFeat_foggy', 'GAN_clear', 'GAN_foggy'}
        -Traing['real']: {'GlobalFeat_clear', 'GlobalFeat_foggy', 'GAN_clear', 'GAN_foggy'}
        -Eval: {'Feat_clear', 'Feat_foggy', 'GAN_clear', 'GAN_foggy'}
    '''
    img_shape = (1, 64, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])
    model = Stage2_Model(cfg.MODEL.NAME, syn_num_classes, img_shape, cfg.MODEL.LAST_STRIDE)
    print(cfg.MODEL.NAME)
    return model

def build_Stage3_Model(cfg, syn_num_classes):
    '''
    Stage 3. structure. for Real foggy training stage. use syn paris for auxiliary learning.
    Input: (datas, mode='real'/'syn')
    Output:
        -Traing['syn']: {'Class_clear', 'Class_foggy', 'GlobalFeat_clear', 'GlobalFeat_foggy', 'GAN_clear', 'GAN_foggy'}
        -Traing['real']: {'GlobalFeat_clear', 'GlobalFeat_foggy', 'GAN_clear', 'GAN_foggy'}
        -Eval: {'Feat_clear', 'Feat_foggy', 'GAN_clear', 'GAN_foggy'}
    '''
    img_shape = (1, 64, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])
    model = Stage3_Model(cfg.MODEL.NAME, syn_num_classes, img_shape, cfg.MODEL.LAST_STRIDE)
    print(cfg.MODEL.NAME)
    return model