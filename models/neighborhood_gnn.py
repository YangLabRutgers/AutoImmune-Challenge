import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv
from torch_geometric.nn import aggr

class default_gnn(nn.Modules):
    def __init__(self, in_channels:int, 
                 conv_out_channels = 1,
                 dense_layer_out: int,
                 num_embeddings:int,
                 embedding_dims:int
                )
        
        super.__init__()
        
        self.gcl = ChebConv(
            in_channels = in_channels, 
            conv_out_channels = out_channels,
            )
        
        # self.agg= aggr()
        self.dense_layer = nn.Linear(
            in_features = conv_out_channels, 
            out_features = dense_layer_out
            )
        
        self.final_node_embedding = nn.Embedding(
            num_embeddings = num_embeddings, 
            embedding_dims = embedding_dims
            )
    def forward(x):
        x = self.gcl(x)
        x = aggr.MeanAggregation(x)
        x = dense_layer(x)
        x = final_node_embedding(x)
        return x

def test():
    x = torch.randn(2,2)
    model = default_gnn(
        in_channels=1,
        conv_out_channels=1,
        dense_layer_out = 1
        num_embeddings=2
        embedding_dims=1
        )
    return model(x)

test()