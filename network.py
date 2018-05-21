import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from custom_layers import *

# defined for code simplicity.
def deconv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False, wn=False, pixel=False, only=False):
    if wn:  layers.append(equalized_conv2d(c_in, c_out, k_size, stride, pad))
    else:   layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if not only:
        if leaky:   layers.append(nn.LeakyReLU(0.2))
        else:       layers.append(nn.ReLU())
        if bn:      layers.append(nn.BatchNorm2d(c_out))
        if pixel:   layers.append(pixelwise_norm_layer())
    return layers

def conv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False, wn=False, pixel=False, gdrop=True, only=False):
    if gdrop:       layers.append(generalized_drop_out(mode='prop', strength=0.0))
    if wn:          layers.append(equalized_conv2d(c_in, c_out, k_size, stride, pad, initializer='kaiming'))
    else:           layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if not only:
        if leaky:   layers.append(nn.LeakyReLU(0.2))
        else:       layers.append(nn.ReLU())
        if bn:      layers.append(nn.BatchNorm2d(c_out))
        if pixel:   layers.append(pixelwise_norm_layer())
    return layers

def linear(layers, c_in, c_out, sig=True, wn=False):
    layers.append(Flatten())
    if wn:      layers.append(equalized_linear(c_in, c_out))
    else:       layers.append(Linear(c_in, c_out))
    if sig:     layers.append(nn.Sigmoid())
    return layers