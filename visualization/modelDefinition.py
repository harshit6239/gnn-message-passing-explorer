import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool

class SimpleGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, gnn_type):
        super().__init__()
        self.layers = nn.ModuleList()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type

        # Input layer
        if gnn_type == "GCN":
            self.layers.append(GCNConv(in_dim, hidden_dim))
        elif gnn_type == "GAT":
            self.layers.append(GATConv(in_dim, hidden_dim))
        elif gnn_type == "SAGE":
            self.layers.append(SAGEConv(in_dim, hidden_dim))
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

        # Hidden layers
        for _ in range(num_layers - 2):
            if gnn_type == "GCN":
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == "GAT":
                self.layers.append(GATConv(hidden_dim, hidden_dim))
            elif gnn_type == "SAGE":
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))

        # Output layer
        if gnn_type == "GCN":
            self.layers.append(GCNConv(hidden_dim, out_dim))
        elif gnn_type == "GAT":
            self.layers.append(GATConv(hidden_dim, out_dim))
        elif gnn_type == "SAGE":
            self.layers.append(SAGEConv(hidden_dim, out_dim))

    def forward(self, x, edge_index, batch=None):
        intermediate_outputs = []
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            intermediate_outputs.append(x)

        x = self.layers[-1](x, edge_index)
        intermediate_outputs.append(x)

        if batch is not None:
            pooled_x = global_mean_pool(x, batch)
            return pooled_x, intermediate_outputs
        else:
            return x, intermediate_outputs
        
