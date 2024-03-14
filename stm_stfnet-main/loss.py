import torch
import torch.nn as nn
from torch.nn import functional as F

from pytorch_ssim import ssim
from kornia.filters.sobel import Sobel
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.perceptual_cri = CharbonnierLoss(loss_weight=1.0)
        self.sobel=Sobel()


    def forward(self, prediction, target,ff2,fb2,is_ds=False):
        if not is_ds:

            loss =1 - ssim(prediction, target) +  self.perceptual_cri(ff2, target) + self.perceptual_cri(fb2, target)+self.perceptual_cri(self.sobel(prediction),self.sobel(target))

        else:
            loss = self.perceptual_cri(prediction, target)
        return loss


class CharbonnierLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CharbonnierLoss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = 1e-3

    def forward(self, pred, target):
        return self.loss_weight * torch.mean(torch.sqrt(torch.square(target - pred) + self.eps * self.eps))

