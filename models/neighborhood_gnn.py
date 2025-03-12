import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv
from torch_geometric.nn import aggr
from torch_geometric.nn import models



class default_gnn(nn.Module):
    
    def __init__(self, in_channels:int, 
                 conv_out_channels:int,
                 edges:int,
                 k_filter_size: int,
                 dense_layer_in:int,
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
            in_features = dense_layer_in, 
            out_features = dense_layer_out
            )
        
        # self.final_node_embedding = nn.Linear(
        #     in_features=num_embeddings,
        #     out_features=embedding_dims
        # )
        
        self.final_node_embedding = 
        
        self.edges = edges
        
    def forward(self,x):
        print(f"x  {x.size()}")
        x = self.gcl(x,torch.randint(low=0,high=1,size=(2,self.edges)))
        print(f"x  {x.size()}")
        x = aggr.MeanAggregation()(x,dim=-1)
        print(f"x  {x.size()}")
        x = x.squeeze()
        print(f"x  {x.size()}")
        x = self.dense_layer(x) 
        print(f"x  {x.size()}")   
        x = x.T
        print(f"x  {x.size()}")   
        x = self.final_node_embedding(x)
        print(f"x final embedding {x.size()}")
        return x


#srun --time=02:00:00 python3 ./models/neighborhood_gnn.py

def test(n:int,m:int):
    
    x = torch.randint(0,1,(n,n,3)).to(torch.float)
    
    
    model = default_gnn(
        in_channels=-1,
        conv_out_channels=3,
        edges=100,
        k_filter_size=1,
        dense_layer_in = n,
        dense_layer_out = 10,
        num_embeddings=n,
        embedding_dims=m
        )
    
    return model(x).size()

test(10,60)