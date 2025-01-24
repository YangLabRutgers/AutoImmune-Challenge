import torch
import torch.nn as nn
from models.segmentation_models.model import UNet
from accelerate import Accelerator
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config file")
    args = parser.parse_args()

if __name__ == "__main__":
    main()

