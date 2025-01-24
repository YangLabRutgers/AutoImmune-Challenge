import torch
import torch.nn as nn
import segmentation_models_pytorch as sm

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = sm.Unet()
    
    def forward(x):
        return self.model(x)
    
    
