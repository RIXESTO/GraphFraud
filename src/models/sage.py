import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv   # GraphSAGE convolution layer


class GraphSAGE(nn.Module):
    """
    GraphSAGE (Graph SAmple and aggreGatE) for node-level fraud classification.

    ── How GraphSAGE differs from GCN and GAT ───────────────────────────────
    GCN  : averages ALL neighbours — must recompute when new nodes are added
    GAT  : learns attention weights for ALL neighbours — same scalability issue
    SAGE : SAMPLES a fixed number of neighbours, then AGGREGATES them

    Why sampling matters for your dataset:
    Some Bitcoin wallets have thousands of connections. GCN/GAT process
    ALL of them — slow and memory-heavy. SAGE picks, say, 10 neighbours,
    making it much more scalable.

    ── Inductive learning ───────────────────────────────────────────────────
    GCN and GAT are "transductive" — they need all nodes at training time.
    SAGE is "inductive" — it learns a general aggregation FUNCTION that
    can be applied to any new node, even ones not seen during training.
    This makes SAGE the most production-ready GNN architecture.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # ── Input layer ───────────────────────────────────────────────────────
        # SAGEConv(in, out): transforms a node's own features AND its
        # aggregated neighbour features, then combines them.
        # Default aggregation: mean of sampled neighbours.
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # ── Hidden layers ─────────────────────────────────────────────────────
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # ── Output layer ──────────────────────────────────────────────────────
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        # No BatchNorm on the output — want raw logits for CrossEntropy loss

    def forward(self, x, edge_index):
        # ── Hidden layers: conv → batchnorm → relu → dropout ─────────────────
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            # SAGEConv step:
            #   1. Aggregate neighbours: mean of all sampled neighbour features
            #   2. Concatenate: [node's own features | aggregated neighbour features]
            #   3. Linear transform + bias: learned projection to hidden_channels
            # Result: each node's representation is enriched with neighbourhood info

            x = bn(x)
            # Normalise activations — prevents gradients from exploding/vanishing

            x = F.relu(x)
            # Replace negatives with 0 — adds non-linearity

            x = F.dropout(x, p=self.dropout, training=self.training)
            # Randomly zero out 30% of neurons — only active during training

        # ── Output layer ──────────────────────────────────────────────────────
        x = self.convs[-1](x, edge_index)
        # Final layer: hidden_channels → 2 (one score per class)
        # No activation — raw logits returned

        return x   # shape: [num_nodes, 2]


def build_sage(config, in_channels):
    """
    Factory function — builds GraphSAGE from config.yaml.

    Usage:
        model = build_sage(config, in_channels=165)
    """
    cfg = config["gnn"]
    return GraphSAGE(
        in_channels     = in_channels,
        hidden_channels = cfg["hidden_channels"],  # 64
        out_channels    = 2,                        # licit or illicit
        num_layers      = cfg["num_layers"],        # 3
        dropout         = cfg["dropout"],           # 0.3
    )
