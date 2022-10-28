from typing import Literal

from torch import nn

from torch_geometric.nn import GAT, GCNConv

# See https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html to install the GNN libraries


class GNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        kind: Literal["gcn", "gat"],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.kind = kind
        
        if kind == "gcn":
            pass