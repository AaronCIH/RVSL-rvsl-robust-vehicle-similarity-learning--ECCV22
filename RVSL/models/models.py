# created by Cihsiang.
import torch
from torch import nn

from .backbones.resnet import create_Encoder_ResNet, create_Decoder_ResNet
from .backbones.restoration import Decoder_Generator

#################################################
## modules initialization.
#################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

###################################################################
## Three stage models, different inputs, different forward path
###################################################################
class Stage1_Model(nn.Module):
    '''
    Stage 1. structure, for Syn datas training stage.
    Input: 
        (clear, foggy), syn hazy pairs data.
    Output:
        -Traing: {'Class_clear', 'Class_foggy', 'GlobalFeat_clear', 'GlobalFeat_foggy', 'GAN_clear', 'GAN_foggy'} 
        -Eval: {'Feat_clear', 'Feat_foggy', 'GAN_clear', 'GAN_foggy'}
    '''
    in_planes = 2048
    common_feature_planes = 256
    def __init__(self, backbone, num_classes, input_shape, last_stride, model_reid_path="", model_rest_path=""):
        super(Stage1_Model, self).__init__()
        ########################## Encoder ###################################
        self.Ec = create_Encoder_ResNet(backbone, last_stride=1, pretrained=True)
        self.Ef = create_Encoder_ResNet(backbone, last_stride=1, pretrained=True)

        ########################## Generator ###################################
        self.Df = Decoder_Generator(self.common_feature_planes, 64, input_shape, bilinear=True)
        self.Dc = Decoder_Generator(self.common_feature_planes, 64, input_shape, bilinear=True)

        ########################## ReID Decoder ###################################
        self.Dreid = create_Decoder_ResNet(backbone, last_stride=1, pretrained=True)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        self.number_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.number_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        if model_reid_path != "": self.load_param(model_reid_path)
        if model_rest_path != "": self.load_param(model_rest_path)

    def forward(self, clear, foggy):
        ####################### ReID #################################
        # 1. Encoder for clear, foggy,  CF: common feature
        CFc, CFc_hidden = self.Ec(clear)    # r1(w/4,h/4,64), [f1(w/2,h/2,64), df1(w/4,h/4,64), r1(w/4,h/4,64)]
        CFf, CFf_hidden = self.Ef(foggy)    # r1(w/4,h/4,64), [f1(w/2,h/2,64), df1(w/4,h/4,64), r1(w/4,h/4,64)]
        
        # 2. Decoder for clear, foggy in the same time. EF:embedding feature, EFG:embedding global featrue, BNF:Bottle Neck feat
        EFc, EFc_hidden = self.Dreid(CFc)  # r4 (w/32, h/32, 2048), [r2 (w/8. h/8, 512), r3 (w/16, h/16, 1024), r4 (w/32, h/32, 2048)] 
        EFf, EFf_hidden = self.Dreid(CFf)  # r4 (w/32, h/32, 2048), [r2 (w/8. h/8, 512), r3 (w/16, h/16, 1024), r4 (w/32, h/32, 2048)] 
        EFGc, EFGf = self.gap(EFc), self.gap(EFf)  # (b, 2048, 1, 1)
        EFGc, EFGf = EFGc.view(EFGc.shape[0], -1), EFGf.view(EFGf.shape[0], -1) # flatten to (bs, 2048)
        BNFc, BNFf = self.bottleneck(EFGc), self.bottleneck(EFGf)  # normalize for angular softmax

        ####################### Reconstraction #################################
        Gc = self.Dc(CFf, CFf_hidden[0])  # generate_syn_clear
        Gf = self.Df(CFc, CFc_hidden[0])  # generate_syn_foggy

        ########################## Output #################################
        if self.training:
            cls_score_clear = self.classifier(BNFc)
            cls_score_foggy = self.classifier(BNFf)
            return {'Class_clear':cls_score_clear, 'Class_foggy':cls_score_foggy, 
                    'GlobalFeat_clear':EFGc, 'GlobalFeat_foggy':EFGf,
                    'GAN_clear':Gc, 'GAN_foggy':Gf}  
        else:
            return {'Feat_clear':BNFc, 'Feat_foggy':BNFf, 
                    'GAN_clear':Gc, 'GAN_foggy':Gf} 

    def load_param(self, model_path):
        param = torch.load(model_path)
        for i in param:
            if 'fc' in i: continue
            if i not in self.state_dict().keys(): continue
            if param[i].shape != self.state_dict()[i].shape: continue
            self.state_dict()[i].copy_(param[i])

class Stage2_Model(nn.Module):
    '''
    Stage 2. structure, for Real clear training stage. use syn paris for auxiliary learning.
    Input: (datas, mode='real'/'syn')
    Output:
        -Traing['syn']: {'Class_clear', 'Class_foggy', 'GlobalFeat_clear', 'GlobalFeat_foggy', 'GAN_clear', 'GAN_foggy'}
        -Traing['real']: {'GlobalFeat_clear', 'GlobalFeat_foggy', 'GAN_clear', 'GAN_foggy'}
        -Eval: {'Feat_clear', 'Feat_foggy', 'GAN_clear', 'GAN_foggy'}
    '''
    in_planes = 2048
    common_feature_planes = 256
    def __init__(self, backbone, syn_num_classes, input_shape, last_stride):
        super(Stage2_Model, self).__init__()
        ########################## Encoder ###################################
        self.Ec = create_Encoder_ResNet(backbone, last_stride=1, pretrained=True)
        self.Ef = create_Encoder_ResNet(backbone, last_stride=1, pretrained=True)

        ########################## Generator ###################################
        self.Df = Decoder_Generator(self.common_feature_planes, 64, input_shape, bilinear=True)
        self.Dc = Decoder_Generator(self.common_feature_planes, 64, input_shape, bilinear=True)      

        ########################## ReID Decoder ###################################
        self.Dreid = create_Decoder_ResNet(backbone, last_stride=1, pretrained=True)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        
        self.syn_num_classes = syn_num_classes
        self.syn_classifier = nn.Linear(self.in_planes, self.syn_num_classes, bias=False)
        self.syn_classifier.apply(weights_init_classifier) 

    def forward(self, datas, mode='real'):
        if mode=='real':
            clear = datas[0]
            ####################### Clear to Foggy' #################################
            CFc, CFc_hidden = self.Ec(clear)    # r1(w/4,h/4,64), [f1(w/2,h/2,64), df1(w/4,h/4,64), r1(w/4,h/4,64)]
            Gf = self.Df(CFc, CFc_hidden[0])  # generate_syn_foggy
            ####################### Foggy' to Clear' #################################
            CFf, CFf_hidden = self.Ef(Gf)    # r1(w/4,h/4,64), [f1(w/2,h/2,64), df1(w/4,h/4,64), r1(w/4,h/4,64)]
            Gc = self.Dc(CFf, CFf_hidden[0])  # generate_syn_clear
            
        elif mode=='syn':
            foggy, clear = datas
            ####################### Clear to Foggy' #################################
            CFc, CFc_hidden = self.Ec(clear)    # r1(w/4,h/4,64), [f1(w/2,h/2,64), df1(w/4,h/4,64), r1(w/4,h/4,64)]
            Gf = self.Df(CFc, CFc_hidden[0])  # generate_syn_foggy
            ####################### Foggy' to Clear' #################################
            CFf, CFf_hidden = self.Ef(foggy)    # r1(w/4,h/4,64), [f1(w/2,h/2,64), df1(w/4,h/4,64), r1(w/4,h/4,64)]
            Gc = self.Dc(CFf, CFf_hidden[0])  # generate_syn_clear            
        
        ########################## ReID encoder ##################################
        # Decoder for clear, foggy in the same time. EF:embedding feature, EFG:embedding global featrue, BNF:Bottle Neck feat
        EFc, EFc_hidden = self.Dreid(CFc)  # r4 (w/32, h/32, 2048), [r2 (w/8. h/8, 512), r3 (w/16, h/16, 1024), r4 (w/32, h/32, 2048)] 
        EFf, EFf_hidden = self.Dreid(CFf)  # r4 (w/32, h/32, 2048), [r2 (w/8. h/8, 512), r3 (w/16, h/16, 1024), r4 (w/32, h/32, 2048)]         
        EFGc, EFGf = self.gap(EFc), self.gap(EFf)  # (b, 2048, 1, 1)
        EFGc, EFGf = EFGc.view(EFGc.shape[0], -1), EFGf.view(EFGf.shape[0], -1) # flatten to (bs, 2048)
        BNFc, BNFf = self.bottleneck(EFGc), self.bottleneck(EFGf)  # normalize for angular softmax

        ########################## Output #################################
        if self.training:
            forward = {'GlobalFeat_clear':EFGc, 'GlobalFeat_foggy':EFGf,
                       'GAN_clear':Gc, 'GAN_foggy':Gf}
            if mode=='syn':
                cls_score_clear = self.syn_classifier(BNFc)
                cls_score_foggy = self.syn_classifier(BNFf)  
                forward['Class_clear'] = cls_score_clear
                forward['Class_foggy'] = cls_score_foggy
            return forward
        else:
            return {'Feat_clear':BNFc, 'Feat_foggy':BNFf, 
                    'GAN_clear':Gc, 'GAN_foggy':Gf} 

    def load_param(self, model_path):
        param = torch.load(model_path)
        for i in param:
            if 'fc' in i: continue
            if i not in self.state_dict().keys(): continue
            if param[i].shape != self.state_dict()[i].shape: continue
            self.state_dict()[i].copy_(param[i])

class Stage3_Model(nn.Module):
    '''
    Stage 3. structure. for Real foggy training stage. use syn paris for auxiliary learning.
    Input: (datas, mode='real'/'syn')
    Output:
        -Traing['syn']: {'Class_clear', 'Class_foggy', 'GlobalFeat_clear', 'GlobalFeat_foggy', 'GAN_clear', 'GAN_foggy'}
        -Traing['real']: {'GlobalFeat_clear', 'GlobalFeat_foggy', 'GAN_clear', 'GAN_foggy'}
        -Eval: {'Feat_clear', 'Feat_foggy', 'GAN_clear', 'GAN_foggy'}
    '''
    in_planes = 2048
    common_feature_planes = 256
    def __init__(self, backbone, syn_num_classes, input_shape, last_stride):
        super(Stage3_Model, self).__init__()
        ########################## Encoder ###################################
        self.Ec = create_Encoder_ResNet(backbone, last_stride=1, pretrained=True)
        self.Ef = create_Encoder_ResNet(backbone, last_stride=1, pretrained=True)

        ########################## Generator ###################################
        self.Df = Decoder_Generator(self.common_feature_planes, 64, input_shape, bilinear=True)
        self.Dc = Decoder_Generator(self.common_feature_planes, 64, input_shape, bilinear=True)      

        ########################## ReID Decoder ###################################
        self.Dreid = create_Decoder_ResNet(backbone, last_stride=1, pretrained=True)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        
        self.syn_num_classes = syn_num_classes
        self.syn_classifier = nn.Linear(self.in_planes, self.syn_num_classes, bias=False)
        self.syn_classifier.apply(weights_init_classifier) 

    def forward(self, datas, mode='real'):
        if mode=='real':
            foggy = datas[0]
            ####################### Foggy' to Clear' #################################
            CFf, CFf_hidden = self.Ef(foggy)    # r1(w/4,h/4,64), [f1(w/2,h/2,64), df1(w/4,h/4,64), r1(w/4,h/4,64)]
            Gc = self.Dc(CFf, CFf_hidden[0])  # generate_syn_clear
            ####################### Clear to Foggy' #################################
            CFc, CFc_hidden = self.Ec(Gc)    # r1(w/4,h/4,64), [f1(w/2,h/2,64), df1(w/4,h/4,64), r1(w/4,h/4,64)]
            Gf = self.Df(CFc, CFc_hidden[0])  # generate_syn_foggy
        elif mode=='syn':
            foggy, clear = datas
            ####################### Clear to Foggy' #################################
            CFc, CFc_hidden = self.Ec(clear)    # r1(w/4,h/4,64), [f1(w/2,h/2,64), df1(w/4,h/4,64), r1(w/4,h/4,64)]
            Gf = self.Df(CFc, CFc_hidden[0])  # generate_syn_foggy
            ####################### Foggy' to Clear' #################################
            CFf, CFf_hidden = self.Ef(foggy)    # r1(w/4,h/4,64), [f1(w/2,h/2,64), df1(w/4,h/4,64), r1(w/4,h/4,64)]
            Gc = self.Dc(CFf, CFf_hidden[0])  # generate_syn_clear            
        
        ########################## ReID encoder ##################################
        # Decoder for clear, foggy in the same time. EF:embedding feature, EFG:embedding global featrue, BNF:Bottle Neck feat
        EFc, EFc_hidden = self.Dreid(CFc)  # r4 (w/32, h/32, 2048), [r2 (w/8. h/8, 512), r3 (w/16, h/16, 1024), r4 (w/32, h/32, 2048)] 
        EFf, EFf_hidden = self.Dreid(CFf)  # r4 (w/32, h/32, 2048), [r2 (w/8. h/8, 512), r3 (w/16, h/16, 1024), r4 (w/32, h/32, 2048)]         
        EFGc, EFGf = self.gap(EFc), self.gap(EFf)  # (b, 2048, 1, 1)
        EFGc, EFGf = EFGc.view(EFGc.shape[0], -1), EFGf.view(EFGf.shape[0], -1) # flatten to (bs, 2048)
        BNFc, BNFf = self.bottleneck(EFGc), self.bottleneck(EFGf)  # normalize for angular softmax

        ########################## Output #################################
        if self.training:
            forward = {'GlobalFeat_clear':EFGc, 'GlobalFeat_foggy':EFGf,
                       'GAN_clear':Gc, 'GAN_foggy':Gf}
            if mode=='syn':
                cls_score_clear = self.syn_classifier(BNFc)
                cls_score_foggy = self.syn_classifier(BNFf)  
                forward['Class_clear'] = cls_score_clear
                forward['Class_foggy'] = cls_score_foggy
            return forward
        else:
            return {'Feat_clear':BNFc, 'Feat_foggy':BNFf, 
                    'GAN_clear':Gc, 'GAN_foggy':Gf} 

    def load_param(self, model_path):
        param = torch.load(model_path)
        for i in param:
            if 'fc' in i: continue
            if i not in self.state_dict().keys(): continue
            if param[i].shape != self.state_dict()[i].shape: continue
            self.state_dict()[i].copy_(param[i])

