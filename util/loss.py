import torch
from torch import nn
from pytorch_msssim import MS_SSIM

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5,lambd1=1):
        super().__init__()
        self.lambd = lambd
        self.lambd1 = lambd1
        #
        self.ssim = MS_SSIM(channel=1)
    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        sigloss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))


        if len(pred.shape) == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
            valid_mask = valid_mask.unsqueeze(1)
        pred = pred*valid_mask
        target = target*valid_mask
        ms_ssim =  self.ssim(target, pred)
        MS_SSIM_LOSS =  1 - ms_ssim

        loss = sigloss+self.lambd1*MS_SSIM_LOSS



        return loss
