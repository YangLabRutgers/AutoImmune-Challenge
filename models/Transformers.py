import torch
import torch.nn as nn
import torch.functional as F
import math
from modules.Attention import *

class TransformerEncoderLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multihead_attention = MultiHeadSelfDotAttention()
        

class Transformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positional_encoding = None
        self.attetion_mechanism = None
        self.encoder = []
        self.decoder = []

model = nn.T