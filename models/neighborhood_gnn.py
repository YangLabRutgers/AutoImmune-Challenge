import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv
from torch_geometric.nn import aggr
from torch_geometric.nn import models

"""
Default GNN class comprised of  v 
layer, mean aggregation across the 3 channels, one 
dense layer and an embedding layer.

n = nodes

e = edges

x in R^{n X n X 3}
edge_list in R^{2 X e}

ChebConv: R^{n X n X 3} X R^{2 X e} -> R^{n X n X 3}

Aggregate Layer: R^{n X n X 3} -> R^{n X n}

Dense Layer: R^{n X n} -> R^{n X n}

Embedding Layer: R^{n X n} -> R^{n X m} 

such that m is the dimension of each embedding, 
resulting in n nodes, each with m dims

"""
class default_gnn(nn.Module):
    def __init__(self, nodes: int,
                 edges:int, 
                 k_filter_size = 1,
                 in_channels = -1, 
                 conv_out_channels = 3,
                 embedding_dims = 10,**kwargs
                ):
        
        super().__init__()
        
        self.dense_layer_in = kwargs.get('dense_layer_in',nodes)
        
        self.dense_layer_out = kwargs.get('dense_layer_out',nodes)
        
        self.gcl = ChebConv(
            in_channels = in_channels, 
            out_channels = conv_out_channels,
            K = k_filter_size
            )

        self.dense_layer = nn.Linear(
            in_features = self.dense_layer_in, 
            out_features = self.dense_layer_out
            )
        
        self.final_node_embedding = nn.Linear(
            in_features=nodes,
            out_features=embedding_dims
        )
        
        self.edges = edges
        
    def forward(self,x,edge_index):
        x = self.gcl(x,edge_index)
        
        x = aggr.MeanAggregation()(x,dim=-1)
        
        x = x.squeeze() #[n X n X 1] -> [n X n]
        
        x = self.dense_layer(x) 
        
        x = x.T
        
        x = self.final_node_embedding(x)
        
        return x


#srun --time=02:00:00 python3 ./models/neighborhood_gnn.py

# def test(n:int,m:int):
    
#     x = torch.randint(0,1,(n,n,3)).to(torch.float)
    
#     edge_index = torch.randint(0,1,[2,n*n])
    
#     model = default_gnn(
#         nodes = n, edges = m
#         )
    
#     return model(x,edge_index).size()

# test(10,60)