import torch
import torch.nn as nn
from loss_funcs import *

class LMMD_Loss(nn.Module):
    def __init__(self, **kwargs):
        super(LMMD_Loss, self).__init__()
        self.loss_func = LMMDLoss(**kwargs)

    def forward(self, source, target, **kwargs):
        return self.loss_func(source, target, **kwargs)