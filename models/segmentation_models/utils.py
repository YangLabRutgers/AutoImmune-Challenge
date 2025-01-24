
import torch.nn as nn
import torch

def create_encoder(
    input_size: list[int], 
    channels: list[int], 
    n_labels: int, 
    n_con_layers_p_block: list[int], 
    convolutional_kernel_sizes: list[int],
    pool_kernel_sizes: list[int],
    strides: list[int],
    paddings: list[int]
    ):
    
    blocks = []
    
    i = 0
    for b in range(n_con_layers_p_block):
        blocks.append([])
        
        for c in range(i,n_con_layers_p_block[b]+i):
            if c == 0: 
                in_channel = input_size[-1]
            else: 
                in_channel = nn.Conv2d(channels[c-1])
            
            blocks[-1].append(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=channels[c],
                    kernel_size=convolutional_kernel_sizes[c],
                    stride=strides[c],
                    padding=paddings[c]
                )  
            )
        i = n_con_layers_p_block[b] 
    