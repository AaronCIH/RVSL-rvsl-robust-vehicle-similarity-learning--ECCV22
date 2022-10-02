import torch
import torch.nn as nn
import functools
import numpy as np
from torch.autograd import Variable

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
class Discriminator(nn.Module):
    def __init__(self, input_shape, use_gpu=True):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1))
        self.use_gpu = use_gpu
        self.Tensor = torch.cuda.FloatTensor if self.use_gpu  else torch.FloatTensor
        self.adversarial = torch.nn.MSELoss()

    def forward(self, imgs, label='valid', vis=False):
        if label=='valid':
            ll = Variable(self.Tensor(np.ones((imgs.size(0), *self.output_shape))), requires_grad=False)
        else:
            ll = Variable(self.Tensor(np.zeros((imgs.size(0), *self.output_shape))), requires_grad=False)

        pred = self.model(imgs)
        if vis:
            print(label, pred[0,0,0,:5])
        loss = self.adversarial(pred.float(), ll)  
        return loss