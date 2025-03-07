import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv
from torch_geometric.nn import aggr


class default_gnn(nn.Module):
    
    def __init__(self, in_channels:int, 
                 conv_out_channels:int,
                 k_filter_size: int,
                 dense_layer_out: int,
                 num_embeddings:int,
                 embedding_dims:int
                ):
        
        super().__init__()
        
        self.gcl = ChebConv(
            in_channels = in_channels, 
            out_channels = conv_out_channels,
            K = k_filter_size
            )

        self.dense_layer = nn.Linear(
            in_features = conv_out_channels, 
            out_features = dense_layer_out
            )
        
        self.final_node_embedding = nn.Linear(
            in_features=num_embeddings,
            out_features=embedding_dims
        )
        
    def forward(self,x):
        x = self.gcl(x,torch.randint(low=0,high=1,size=(2,2)))
        x = aggr.MeanAggregation()(x)
        x = self.dense_layer(x)    
        x = self.final_node_embedding(x)
        print(f"x final embedding {x}")
        return x

def test():
    
    x = torch.randn(4,4)
    
    model = default_gnn(
        in_channels=4,
        conv_out_channels=1,
        k_filter_size=1,
        dense_layer_out = 1,
        num_embeddings=1,
        embedding_dims=1
        )
    
    return model(x)

test()