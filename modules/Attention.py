import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class DotAttention(nn.Module):
    def __init__(self,masked:bool):
        super().__init__()
        self.masked = masked
    
    def forward(self,Q,K,V):
        d_k = Q.size(-1)
        inner_fun = (torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(d_k))
        if self.masked:
            mask = torch.triu(torch.ones(inner_fun.size[-2],inner_fun.size[-1]),diagonal=1).bool()
            if len(Q.size())==3:
                mask = mask.unsqueeze(0).expand(Q.size(-1),-1,-1)
                inner_fun = inner_fun.masked_fill(mask,float('inf'))
        inner_fun = (torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(d_k))
        return F.softmax(inner_fun,dim=-1)@V
        
class EfficientAttention(nn.Module): 
    # Paper Title:
    # Efficient Attention: Attention with Linear Complexities
    def __init__(self):
        super().__init__()
        
    def forward(self,Q,K,V):
        # define d_k value
        d_k = Q.size(-1)
        # define rho_q function for Q matrix
        def rho_q(Y):
            Y= Y/math.sqrt(d_k)
            Y = F.softmax(Y,dim=-2)
            return Y
        # define rho_k for k matrix 
        def rho_k(Y):
            Y = Y/math.sqrt(d_k)
            # print(Y)
            Y = F.softmax(Y,dim=-1)
            return Y
            
        return rho_q(Q)@(rho_k(K).transpose(-2,-1)@V)


class KernalAttention(nn.Module): 
    # Paper Title:
    # Transformers are RNNs:Fast Autoregressive Transformers with Linear Attention
    def __init__(self,alpha=1,inplace=False):
        super().__init__()
        # set parameterizable alpha
        self.alpha = alpha
        # set inplace boolean
        self.inplace = inplace
    
    def forward(self,Q,K,V):
        # define phi function
        phi = nn.ELU(
            alpha=self.alpha,
            inplace = self.inplace
            )
        # set denominator of attention mechanism
        denominator = phi(Q)@(phi(K).transpose(-2,-1))
        # set numerator of attention mechanism
        numerator = denominator@V
        #return numerator over the denominator
        return numerator/denominator
    

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

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, heads, input_size:list[int], d_k:int, attention):
        
        super().__init__()
        self.WQ = nn.Parameter(torch.rand(heads,input_size[-1],d_k))
        self.WK = nn.Parameter(torch.rand(heads,input_size[-1],d_k))
        self.WV = nn.Parameter(torch.rand(heads,input_size[-1],input_size[-2]))
        self.WO = nn.Parameter(torch.rand(heads,input_size[-2],input_size[-2]))
        self.attention = attention
        
    def forward(self,x):
        x = x.unsqueeze(0)
        Q = torch.matmul(x,self.WQ)
        print(Q.size())
        K = torch.matmul(x,self.WK)
        print(K.size())
        V = torch.matmul(x,self.WV)
        
        attention_out = self.attention(Q,K,V)
        attention_out = attention_out.squeeze(0)
        attention_out = torch.matmul(attention_out,self.WO)
        return attention_out




        