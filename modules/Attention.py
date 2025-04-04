import torch
import torch.nn as nn
import torch.functional as F
import math



class DotAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(Q,K,V):
        d_k = Q.size(-1)
        inner_fun = (Q@K.T/math.sqrt(d_k))
        return F.softmax(inner_fun,dim=-1)@V
        
class EfficientAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(Q,K,V):
        d_k = Q.size(-1)
        
        def rho_q(Y):
            Y= Y/math.sqrt(d_k)
            Y = F.softmax(Y,dim=-2)
            
        def rho_k(Y):
            Y = Y/math.sqrt(d_k)
            Y = F.softmax(Y,dim=-1)
            
        return rho_q(Q)@(rho_k(K).T@V)
    
class MultiHeadSelfDotAttention(nn.Module):
    def __init__(self, 
                 K_size:list[int],Q_size:list[int],
                 V_size:list[int],heads:int, dim:int
                 ):
        
        super().__init__()
        self.WQ = nn.Parameter(torch.rand(heads,Q_size[-1],dim))
        self.WK = nn.Parameter(torch.rand(heads,K_size[-1],dim))
        self.WV = nn.Parameter(torch.rand(heads,V_size[-1],dim))
        self.WO = nn.Parameter(torch.rand(heads,V_size[-1],V_size[-1]))
        
    def forward(self,x):
        x = x.unsqueeze(0)
        Q = torch.matmul(x,self.WQ,dim=0)
        K = torch.matmul(x,self.WK,dim=0)
        V = torch.matmul(x,self.WV,dim=0)
        attention = DotAttention()
        attention_out = attention(Q,K,V)
        attention_out = attention_out.squeeze(0)
        attention_out = torch.matmul(attention_out,self.WO)
        return attention_out

class MultiHeadSelfDotAttention(nn.Module):
    def __init__(self, 
                 K_size:list[int],Q_size:list[int],
                 V_size:list[int],heads:int, dim:int
                 ):
        
        super().__init__()
        self.WQ = nn.Parameter(torch.rand(heads,Q_size[-1],dim))
        self.WK = nn.Parameter(torch.rand(heads,K_size[-1],dim))
        self.WV = nn.Parameter(torch.rand(heads,V_size[-1],dim))
        self.WO = nn.Parameter(torch.rand(heads,V_size[-1],V_size[-1]))
        
    def forward(self,x):
        x = x.unsqueeze(0)
        Q = torch.matmul(x,self.WQ,dim=0)
        K = torch.matmul(x,self.WK,dim=0)
        V = torch.matmul(x,self.WV,dim=0)
        attention = EfficientAttention()
        attention_out = attention(Q,K,V)
        attention_out = attention_out.squeeze(0)
        attention_out = torch.matmul(attention_out,self.WO)
        return attention_out




        