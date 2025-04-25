import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class DotAttention(nn.Module):
    def __init__(self,masked:bool):
        super().__init__()
        self.masked = masked
    
    def forward(self,Q,K,V):
        #define d_k
        d_k = Q.size(-1)
        # calculate the scores
        inner_fun = (torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(d_k))
        #if we're masking enter jump
        if self.masked:
            # create a diagonal mask
            mask = torch.triu(torch.ones(inner_fun.size[-2],inner_fun.size[-1]),diagonal=1).bool()
            #if we're in a multi-headed function, jump here
            if len(Q.size())==3:
                # duplicate mask for n number of heads
                mask = mask.unsqueeze(0).expand(Q.size(-1),-1,-1)
            # mask infinity to diagonal 
            inner_fun = inner_fun.masked_fill(mask,float('inf'))
            
        # return 
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
        # define phi kernal
        phi = nn.ELU(
            alpha=self.alpha,
            inplace = self.inplace
            ) + 1
        # set denominator of attention mechanism
        denominator = phi(Q)@(phi(K).transpose(-2,-1))
        # set numerator of attention mechanism
        numerator = denominator@V
        #return numerator over the denominator
        return numerator/denominator
    

# class MultiHeadSelfDotAttention(nn.Module):
#     def __init__(self, 
#                  K_size:list[int],Q_size:list[int],
#                  V_size:list[int],heads:int, dim:int
#                  ):
        
#         super().__init__()
#         self.WQ = nn.Parameter(torch.rand(heads,Q_size[-1],dim))
#         self.WK = nn.Parameter(torch.rand(heads,K_size[-1],dim))
#         self.WV = nn.Parameter(torch.rand(heads,V_size[-1],dim))
#         self.WO = nn.Parameter(torch.rand(heads,V_size[-1],V_size[-1]))
        
#     def forward(self,x):
#         x = x.unsqueeze(0)
#         Q = torch.matmul(x,self.WQ,dim=0)
#         K = torch.matmul(x,self.WK,dim=0)
#         V = torch.matmul(x,self.WV,dim=0)
#         attention = DotAttention()
#         attention_out = attention(Q,K,V)
#         attention_out = attention_out.squeeze(0)
#         attention_out = torch.matmul(attention_out,self.WO)
#         return attention_out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, heads, input_size:list[int], d_k:int, attention):
        # derive Q,K,V from X and define our attention class
        super().__init__()
        self.WQ = nn.Parameter(torch.rand(heads,input_size[-1],d_k))
        self.WK = nn.Parameter(torch.rand(heads,input_size[-1],d_k))
        self.WV = nn.Parameter(torch.rand(heads,input_size[-1],input_size[-2]))
        self.WO = nn.Parameter(torch.rand(heads,input_size[-2],input_size[-2]))
        self.attention = attention
        
    def forward(self,x):
        #unsqueeze x to 3 dims
        x = x.unsqueeze(0)
        #compute Q
        Q = torch.matmul(x,self.WQ)
        # compute K
        K = torch.matmul(x,self.WK)
        # compute V
        V = torch.matmul(x,self.WV)
        # compute attention
        attention_out = self.attention(Q,K,V)
        # reduce down to 2 dims via concatenation
        attention_out = attention_out.squeeze(0)
        # compute linear layer
        attention_out = torch.matmul(attention_out,self.WO)
        # return 
        return attention_out




        