import argparse
import numpy as np
import pandas as pd
import torch
from scipy.spatial import KDTree
from collections import defaultdict
from scipy.sparse import coo_matrix
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

#  Define NeighborGNN in the same file
class NeighborGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=768, out_dim=1536, dropout=0.1):
        super().__init__()
        self.dropout = dropout

        self.gcn_small_1 = GCNConv(input_dim, hidden_dim)
        self.gcn_small_2 = GCNConv(hidden_dim, out_dim)

        self.gcn_medium_1 = GCNConv(input_dim, hidden_dim)
        self.gcn_medium_2 = GCNConv(hidden_dim, out_dim)

        self.gcn_large_1 = GCNConv(input_dim, hidden_dim)
        self.gcn_large_2 = GCNConv(hidden_dim, out_dim)

        self.scale_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    def forward(self, x, edge_small, edge_medium, edge_large):
        h_small = F.relu(self.gcn_small_1(x, edge_small))
        h_small = F.dropout(h_small, p=self.dropout, training=self.training)
        h_small = self.gcn_small_2(h_small, edge_small)

        h_medium = F.relu(self.gcn_medium_1(x, edge_medium))
        h_medium = F.dropout(h_medium, p=self.dropout, training=self.training)
        h_medium = self.gcn_medium_2(h_medium, edge_medium)

        h_large = F.relu(self.gcn_large_1(x, edge_large))
        h_large = F.dropout(h_large, p=self.dropout, training=self.training)
        h_large = self.gcn_large_2(h_large, edge_large)

        w = F.softmax(self.scale_weights, dim=0)
        return w[0] * h_small + w[1] * h_medium + w[2] * h_large

# Utility functions
def load_spatial_and_coords(dataset_id):
    spatial_matrix = np.loadtxt(f"data/{dataset_id}_loc.tsv", delimiter='\t')
    rows, cols = spatial_matrix.shape
    cell_coordinates = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            cell_id = spatial_matrix[r, c]
            if cell_id != 0:
                cell_coordinates[cell_id].append((r, c))
    return spatial_matrix, cell_coordinates

def build_cell_centers(cell_coordinates):
    return {cell_id: np.mean(coords, axis=0) for cell_id, coords in cell_coordinates.items()}

def build_edge_index(cell_centers, valid_ids, radius):
    coords = [cell_centers[cid] for cid in valid_ids]
    tree = KDTree(coords)
    edge_list = []
    for i, pt in enumerate(coords):
        neighbors = tree.query_ball_point(pt, r=radius)
        for j in neighbors:
            if i != j:
                edge_list.append((i, j))
    return torch.tensor(edge_list).t().contiguous()

# Main processing pipeline
def main(dataset_id):
    print(f"Building graph + GNN for {dataset_id}")

    expression_tensor = torch.load(f"processed/{dataset_id}_vectors.pt")
    valid_cell_ids = np.load(f"processed/{dataset_id}_ids.npy", allow_pickle=True)

    spatial_matrix, cell_coordinates = load_spatial_and_coords(dataset_id)
    cell_centers = build_cell_centers(cell_coordinates)

    neighbor_offsets = np.array([(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)])
    all_steps = []

    for cell_id, coords in cell_coordinates.items():
        if cell_id not in valid_cell_ids:
            continue
        for x, y in coords:
            for dx, dy in neighbor_offsets:
                step = 0
                while True:
                    step += 1
                    nx, ny = x + dx*step, y + dy*step
                    if not (0 <= nx < spatial_matrix.shape[0] and 0 <= ny < spatial_matrix.shape[1]):
                        break
                    neighbor_id = spatial_matrix[nx, ny]
                    if neighbor_id == cell_id:
                        break
                    if neighbor_id != 0:
                        all_steps.append(step)
                        break

    all_steps = np.array(all_steps)
    step_small  = int(np.percentile(all_steps, 25))
    step_medium = int(np.percentile(all_steps, 50))
    step_large  = int(np.percentile(all_steps, 95))

    edge_small  = build_edge_index(cell_centers, valid_cell_ids, step_small)
    edge_medium = build_edge_index(cell_centers, valid_cell_ids, step_medium)
    edge_large  = build_edge_index(cell_centers, valid_cell_ids, step_large)

    gnn = NeighborGNN(input_dim=expression_tensor.shape[1]).cuda()
    gnn.eval()
    with torch.no_grad():
        emb = gnn(expression_tensor.cuda(), edge_small.cuda(), edge_medium.cuda(), edge_large.cuda())

    torch.save(emb.cpu(), f"processed/{dataset_id}_neighbor_features.pt")
    print(f" Saved: processed/{dataset_id}_neighbor_features.pt")

# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", required=True)
    args = parser.parse_args()
    main(args.dataset_id)
