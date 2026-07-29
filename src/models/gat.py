import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv   # Graph Attention layer — different from GCNConv


class GAT(nn.Module):
    """
    Graph Attention Network for node-level fraud classification.

    Key difference from GCN:
      GCN  — treats ALL neighbours equally (simple average)
      GAT  — LEARNS which neighbours matter more (attention weights)

    Example: a suspicious node connected to 10 licit nodes and 2 fraud nodes.
    GCN averages all 12 equally → fraud signal gets diluted.
    GAT learns to pay more attention to the 2 fraud neighbours → stronger signal.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads, dropout):
        super().__init__()
        # Same as GCN — must be first line.

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # ── What `heads` means ───────────────────────────────────────────────
        # Multi-head attention = run attention MULTIPLE TIMES in parallel,
        # each "head" learns a different way to weight neighbours.
        # Then concatenate all heads → richer representation.
        #
        # Example: heads=4, hidden_channels=64
        #   Each head outputs 64 dims → concat → 64 * 4 = 256 dims total
        #
        # This is the same idea as in the famous "Transformer" architecture.

        # ── Input layer ──────────────────────────────────────────────────────
        self.convs.append(
            GATConv(
                in_channels,     # input: 165 features
                hidden_channels, # each head outputs this many dims (64)
                heads=heads,     # number of parallel attention heads (4)
                dropout=dropout, # dropout applied INSIDE the attention mechanism
                concat=True,     # concatenate head outputs (default)
                                 # → output size = hidden_channels * heads = 256
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        # BatchNorm size = hidden_channels * heads (256) because concat=True

        # ── Hidden layers ─────────────────────────────────────────────────────
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads, # input: 256 (previous layer's output)
                    hidden_channels,         # each head outputs 64
                    heads=heads,             # 4 heads → output = 64 * 4 = 256
                    dropout=dropout,
                    concat=True,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # ── Output layer ──────────────────────────────────────────────────────
        self.convs.append(
            GATConv(
                hidden_channels * heads, # input: 256
                out_channels,            # output: 2 (licit or illicit score)
                heads=1,                 # only 1 head on the output layer
                dropout=dropout,
                concat=False,            # don't concat — just average
                                         # → output size = out_channels = 2
            )
        )
        # No BatchNorm on the output layer — same reason as GCN (raw logits needed)

    def forward(self, x, edge_index):
        # ── Same structure as GCN.forward ──────────────────────────────────
        # But uses ELU instead of ReLU — standard choice for GAT (from the paper)
        #
        # ELU vs ReLU:
        #   ReLU: max(0, x)      → hard zero for negatives
        #   ELU:  x if x>0, else (e^x - 1)  → smooth curve for negatives
        #   ELU tends to train slightly better with attention mechanisms.

        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            # GATConv step:
            #   1. Compute attention scores: how much should node A attend to neighbour B?
            #      score(A, B) = learnable function of A's and B's features
            #   2. Softmax: normalise scores so they sum to 1 across all neighbours
            #   3. Weighted sum: aggregate neighbour features using attention weights
            #   4. Linear transform: project to hidden_channels
            #   (all 4 steps repeated `heads` times in parallel)

            x = bn(x)
            # Normalise — same purpose as in GCN

            x = F.elu(x)
            # ELU activation — adds non-linearity, same reason as ReLU in GCN
            # But smoother for negatives → slightly better gradient flow

            x = F.dropout(x, p=self.dropout, training=self.training)
            # Randomly zero out neurons — same purpose as in GCN

        x = self.convs[-1](x, edge_index)
        # Final layer: 256 → 2
        # heads=1, concat=False → output is just 2 scores per node
        # No activation — raw logits for CrossEntropy loss

        return x
        # Shape: [num_nodes, 2]


def build_gat(config, in_channels):
    """
    Factory function — creates a GAT from config.yaml.

    Usage:
        model = build_gat(config, in_channels=165)
    """
    cfg = config["gnn"]
    return GAT(
        in_channels     = in_channels,
        hidden_channels = cfg["hidden_channels"],  # 64
        out_channels    = 2,                        # licit or illicit
        num_layers      = cfg["num_layers"],        # 3
        heads           = 4,                        # 4 attention heads
        dropout         = cfg["dropout"],           # 0.3
    )
