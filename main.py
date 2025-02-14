import torch
import torch.nn as nn
from accelerate import Accelerator
import argparse
from models.unet_utils import *

if torch.cuda.is_available(): print("working") 
else: print("not working")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UNet_Encoder_Layer()

# parser = argparse.ArgumentParser()
# parser.add_argument("config file")
# args = parser.parse_args()


