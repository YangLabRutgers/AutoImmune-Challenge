import torch 
import torch.nn as nn
import torch.functional as F
from unet_utils import *

class UNet(nn.Module):
    def __init__(
        self,image_size:int,ascend_layers:int, descend_layers:int,
        encoder_channels: list[int], decoder_channels:list[int], 
        device:torch.device):
        
        """MAY NEED TO ADD PADDING DURING FORWARD DECODER PASS"""
        
        super.__init__()
        
        self.encoder = []
        
        for i in range(ascend_layers-1):
            
            self.encoder.append(
                UNet_Encoder_Layer(in_channels=encoder_channels[i],
                                   out_channels=encoder_channels[i+1]
                                   ,device=device))
        
        for i in range (descend_layers-1):
            
            self.encoder.append(
                UNet_Decoder_Layer(in_channels=encoder_channels[i],
                                   out_channels=encoder_channels[i+1]
                                   ,device=device))
            
    def forward(self,x):
        
        concat_storage = []
        
        for i in self.encoder:
            
            x = i(x)
            
            concat_storage.append(x)
            
        for i in range(len(self.decoder)):
            
            x = self.decoder[i](torch.concat((concat_storage[-i],x)))
        
        return x
            
            
            
        
        
        