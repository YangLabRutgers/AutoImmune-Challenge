import torch
import torch.nn as nn
import torch.functional as F
import math

class CLIPContrastiveLoss(nn.Module):
    def __init__(self,tau:float):
        super().__init__()
        self.tau = tau
        
    def forward(self,H_v,H_x):
        target = F.softmax(H_v.T@H_v + H_x.T@H_x,dim=-1)
        target = target/(2 * self.tau)
        dot = H_v.T@H_x
        CE = nn.CrossEntropyLoss()
        dot_targ_CE = CE(dot,target)
        dot_targ_CE_T = CE(dot.T,target.T)
        loss = torch.mean(dot_targ_CE + dot_targ_CE_T)
        return loss

