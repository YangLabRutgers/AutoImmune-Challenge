import torch
import torch.nn as nn
from accelerate import Accelerator
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("config file")
args = parser.parse_args()

