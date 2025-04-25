import torch
import torch.nn as nn
# from accelerate import Accelerator
import argparse
from modules.unet_utils import *
from models import *

# Data is on /home/xh300/link/IBD


# assert torch.cuda.is_available()

# from modules.Attention import MultiHeadSelfAttention,DotAttention

# N = 10
# D = 20
# D_k = 10
# D_v = N

# X = torch.rand([N,D])
# WQ = torch.rand([D,D_k])
# WK = torch.rand([D,D_k])
# WV = torch.rand([D,D_v])

# Q = X@WQ
# K = X@WK
# V = X@WV
# attention = DotAttention()
# MHA = MultiHeadSelfAttention(8,X.size(),D_k,attention)
# MHA(X)