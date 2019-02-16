import torch.nn as nn
import operator
import numpy as np
from functools import reduce

def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    
def mask_size_helper(args):
    return reduce(operator.mul, args) 
 
def create_dropconnect_mask(dc_keep_prob, dimensions):
    mask_vector = np.random.binomial(1, dc_keep_prob, mask_size_helper(dimensions))
    mask_array = mask_vector.reshape(dimensions)
    return mask_array
 
def dropconnect(W, dc_keep_prob):
    dimensions = W.shape
    return W * create_dropconnect_mask(dc_keep_prob, dimensions)


class ProtoNetDropConnect(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim=3, hid_dim=84, z_dim=64):
        super(ProtoNetDropConnect, self).__init__()

        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim)
        )

        self.classification = nn.Sequential(
            nn.Linear(z_dim, z_dim)
        )

    def forward(self, x):
        x_t = self.encoder(x)
        x_t = dropconnect(x_t, 0.2)
        y = self.classification(x_t)
        return y.view(x.size(0), -1)
