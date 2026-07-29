import torch                              # core PyTorch library
import torch.nn as nn                     # neural network building blocks
import torch.nn.functional as F           # functions like relu, dropout
from torch_geometric.nn import GCNConv   # the graph convolution layer


class GCN(nn.Module):
    """
    Graph Convolutional Network for node-level fraud classification.
    Predicts whether each transaction node is illicit (1) or licit (0).
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        # ── What __init__ does ──────────────────────────────────────────────
        # This runs ONCE when you create the model: GCN(165, 64, 2, 3, 0.3)
        # You define all the layers here — no data flows yet, just setup.

        super().__init__()
        # super().__init__() calls nn.Module's own setup code.
        # MUST be the first line — without it PyTorch can't track your weights.

        self.dropout = dropout
        # Store dropout rate so forward() can use it later.
        # dropout=0.3 means: randomly zero out 30% of neurons during training.
        # This prevents the model from memorising training data (overfitting).

        # ── Layer containers ────────────────────────────────────────────────
        self.convs = nn.ModuleList()
        # nn.ModuleList is like a Python list, but PyTorch-aware.
        # Using a regular Python list [] would mean PyTorch can't find the
        # weights inside — they wouldn't be trained. ModuleList fixes that.

        self.bns = nn.ModuleList()
        # Will hold BatchNorm layers — one per hidden layer.
        # BatchNorm normalises activations so training is more stable.

        # ── Build layers dynamically based on num_layers ─────────────────
        # Example: num_layers=3 → conv1, conv2, conv3

        # Layer 1 — input layer
        # GCNConv(in, out): takes node features of size `in`, outputs size `out`
        # In_channels = 165 (your feature count), hidden_channels = 64
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        # BatchNorm1d(64): normalises the 64 outputs of conv1

        # Layers 2 to (num_layers-1) — hidden layers
        # If num_layers=3, this loop runs once → adds 1 hidden layer
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            # Each hidden layer takes 64 → outputs 64

        # Final layer — output layer
        # Output size = out_channels = 2 (one score per class: licit, illicit)
        # No BatchNorm on the output — we want raw scores (logits), not normalised
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        # ── What forward does ───────────────────────────────────────────────
        # This is called every time you do: model(x, edge_index)
        # x          → node feature matrix  [203769, 165]
        # edge_index → graph edges in COO format [2, 234355]
        # Data flows through layers here, transforming x step by step.

        # ── Pass through all layers except the last ──────────────────────
        # zip(self.convs[:-1], self.bns) pairs each conv with its batchnorm
        # convs[:-1] = all layers except the final output layer
        for conv, bn in zip(self.convs[:-1], self.bns):

            x = conv(x, edge_index)
            # GCNConv does two things in one step:
            #   1. For each node, gather features from all its neighbours
            #   2. Apply a learnable linear transformation
            # Result: each node now "knows" about its neighbours
            # Shape stays [num_nodes, hidden_channels]

            x = bn(x)
            # BatchNorm: normalise the values across the batch
            # Prevents values from exploding or vanishing during training
            # Makes training faster and more stable

            x = F.relu(x)
            # ReLU: replace all negative values with 0
            # relu([-2, 3, -1, 5]) → [0, 3, 0, 5]
            # WHY: without non-linearity, stacking linear layers = one linear layer
            # Non-linearity lets the network learn complex patterns

            x = F.dropout(x, p=self.dropout, training=self.training)
            # Randomly zero out 30% of values — but ONLY during training
            # self.training is True when model.train() is called
            # self.training is False when model.eval() is called
            # This forces the network not to rely on any single neuron

        # ── Final layer — no relu, no dropout, no batchnorm ─────────────
        x = self.convs[-1](x, edge_index)
        # convs[-1] = the last GCNConv layer (hidden → 2)
        # Output shape: [num_nodes, 2]
        # These are RAW SCORES (logits) for each class:
        #   column 0 = score for "licit"
        #   column 1 = score for "illicit"
        #
        # WHY no relu here: relu kills negative values. If the fraud score
        # is negative, relu would zero it out — destroying the signal.
        # The training loss function (CrossEntropy) expects raw logits.

        return x
        # Shape: [num_nodes, 2]
        # To get probabilities: apply softmax(x, dim=1)
        # To get predicted class: x.argmax(dim=1)


def build_gcn(config, in_channels):
    """
    Factory function — creates a GCN using hyperparameters from config.yaml.
    Caller doesn't need to remember the argument order.

    Usage:
        model = build_gcn(config, in_channels=165)
    """
    cfg = config["gnn"]
    return GCN(
        in_channels     = in_channels,
        hidden_channels = cfg["hidden_channels"],  # 64
        out_channels    = 2,                        # licit or illicit
        num_layers      = cfg["num_layers"],        # 3
        dropout         = cfg["dropout"],           # 0.3
    )
