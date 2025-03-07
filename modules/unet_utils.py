import torch 
import torch.nn as nn
import torch.functional as F


class UNet_Encoder_Layer(nn.Module):
    def __init__(
        self, in_channels:int, out_channels:int, device:torch.device
        ):
        
        super.__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,out_channels=out_channels,
            kernel_size=3,device=device
            )
        
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,out_channels=out_channels,
            kernel_size=3,device=device
            )
        
        self.pool = nn.MaxPool2d(kernel_size=2,device=device)

class UNet_Decoder_Layer(nn.Module):
    def __init__(
        self, in_channels:int, out_channels:int,
        device:torch.device,up_mode="bilinear"
        ):
        
        super.__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,out_channels=out_channels,
            kernel_size=3,device=device
            )
        
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,out_channels=out_channels,
            kernel_size=3,device=device
            )
        
        self.upsample = nn.Upsample(scale_factor=2,mode=up_mode)
        