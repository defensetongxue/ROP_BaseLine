import torch
from . import meta_model
from torch import nn
from timm.loss import LabelSmoothingCrossEntropy
class incetionV3_loss(nn.Module):
    def __init__(self,smoothing):
        super().__init__()
        if smoothing>0:
            self.loss_cls=LabelSmoothingCrossEntropy(smoothing)
            self.loss_avg=LabelSmoothingCrossEntropy(smoothing)
        else:
            self.loss_cls=nn.CrossEntropyLoss()
            self.loss_avg=nn.CrossEntropyLoss()
    def forward(self,inputs,target):
        if isinstance(inputs,tuple):
            out,avg=inputs
            return self.loss_cls(out,target)+self.loss_avg(avg,target)*0.4
        return self.loss_cls(inputs,target)
def build_model(configs):
    
    model= getattr(meta_model,f"build_{configs['name']}")(configs)
    return model