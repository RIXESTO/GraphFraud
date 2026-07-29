"""
tests/test_models.py
─────────────────────
Unit tests for GNN model definitions:
  - GCN, GAT, GraphSAGE: instantiation and forward pass
  - build_* factory functions

Run with: pytest tests/test_models.py -v  (from GraphFraud/ directory)

These tests are "smoke tests" — they verify the model can be created
and produces output of the correct shape, WITHOUT loading the full dataset.
This makes them fast (< 5 seconds each).
"""

import sys
import pytest
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ingest import load_config


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def config():
    return load_config(str(ROOT / "configs" / "config.yaml"))


@pytest.fixture(scope="module")
def dummy_data():
    """
    Create a tiny synthetic graph for fast testing.
    No CSV loading needed — just random tensors with the right shapes.

    x          : 50 nodes, 165 features each
    edge_index : 80 random directed edges
    y          : 50 node labels (0 or 1 only — no unknowns)
    """
    num_nodes    = 50
    num_features = 165
    num_edges    = 80

    x = torch.randn(num_nodes, num_features)
    # randn: random values from a standard normal distribution
    # Shape matches real data: [num_nodes, in_channels]

    row = torch.randint(0, num_nodes, (num_edges,))
    col = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([row, col], dim=0)
    # edge_index: [2, num_edges] — COO (coordinate) format
    # row[i] → col[i] means "there is an edge from node row[i] to col[i]"

    y = torch.randint(0, 2, (num_nodes,))
    # Random labels: 0 (licit) or 1 (illicit)

    return x, edge_index, y, num_features


# ════════════════════════════════════════════════════════════════════════════════
# GCN TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestGCN:

    def test_gcn_instantiation(self, config, dummy_data):
        """GCN should instantiate without errors."""
        from src.models.gcn import build_gcn
        _, _, _, in_channels = dummy_data
        model = build_gcn(config, in_channels)
        assert model is not None

    def test_gcn_forward_shape(self, config, dummy_data):
        """GCN forward pass should return [num_nodes, 2] tensor."""
        from src.models.gcn import build_gcn
        x, edge_index, _, in_channels = dummy_data
        model = build_gcn(config, in_channels)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        assert out.shape == (50, 2), f"Expected (50, 2), got {out.shape}"

    def test_gcn_output_no_nan(self, config, dummy_data):
        """GCN output should not contain NaN or Inf values."""
        from src.models.gcn import build_gcn
        x, edge_index, _, in_channels = dummy_data
        model = build_gcn(config, in_channels)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        assert not torch.isnan(out).any(), "GCN output contains NaN"
        assert not torch.isinf(out).any(), "GCN output contains Inf"

    def test_gcn_train_eval_difference(self, config, dummy_data):
        """Dropout should cause different outputs in train vs eval mode."""
        from src.models.gcn import build_gcn
        x, edge_index, _, in_channels = dummy_data
        model = build_gcn(config, in_channels)

        model.train()
        out_train = model(x, edge_index)

        model.eval()
        with torch.no_grad():
            out_eval = model(x, edge_index)

        # Outputs should differ because dropout is ON in train, OFF in eval
        assert not torch.equal(out_train, out_eval), \
            "Train and eval outputs are identical — dropout may not be working"

    def test_gcn_parameter_count(self, config, dummy_data):
        """GCN should have trainable parameters."""
        from src.models.gcn import build_gcn
        _, _, _, in_channels = dummy_data
        model = build_gcn(config, in_channels)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params > 0, "GCN has no trainable parameters"


# ════════════════════════════════════════════════════════════════════════════════
# GAT TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestGAT:

    def test_gat_instantiation(self, config, dummy_data):
        """GAT should instantiate without errors."""
        from src.models.gat import build_gat
        _, _, _, in_channels = dummy_data
        model = build_gat(config, in_channels)
        assert model is not None

    def test_gat_forward_shape(self, config, dummy_data):
        """GAT forward pass should return [num_nodes, 2] tensor."""
        from src.models.gat import build_gat
        x, edge_index, _, in_channels = dummy_data
        model = build_gat(config, in_channels)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        assert out.shape == (50, 2), f"Expected (50, 2), got {out.shape}"

    def test_gat_output_no_nan(self, config, dummy_data):
        """GAT output should not contain NaN or Inf values."""
        from src.models.gat import build_gat
        x, edge_index, _, in_channels = dummy_data
        model = build_gat(config, in_channels)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        assert not torch.isnan(out).any(), "GAT output contains NaN"
        assert not torch.isinf(out).any(), "GAT output contains Inf"

    def test_gat_parameter_count(self, config, dummy_data):
        """GAT has more parameters than GCN (attention + multi-head)."""
        from src.models.gcn import build_gcn
        from src.models.gat import build_gat
        _, _, _, in_channels = dummy_data

        gcn_params = sum(p.numel() for p in build_gcn(config, in_channels).parameters())
        gat_params = sum(p.numel() for p in build_gat(config, in_channels).parameters())

        assert gat_params > gcn_params, \
            f"GAT ({gat_params}) should have more params than GCN ({gcn_params})"


# ════════════════════════════════════════════════════════════════════════════════
# GraphSAGE TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestGraphSAGE:

    def test_sage_instantiation(self, config, dummy_data):
        """GraphSAGE should instantiate without errors."""
        from src.models.sage import build_sage
        _, _, _, in_channels = dummy_data
        model = build_sage(config, in_channels)
        assert model is not None

    def test_sage_forward_shape(self, config, dummy_data):
        """GraphSAGE forward pass should return [num_nodes, 2] tensor."""
        from src.models.sage import build_sage
        x, edge_index, _, in_channels = dummy_data
        model = build_sage(config, in_channels)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        assert out.shape == (50, 2), f"Expected (50, 2), got {out.shape}"

    def test_sage_output_no_nan(self, config, dummy_data):
        """GraphSAGE output should not contain NaN or Inf values."""
        from src.models.sage import build_sage
        x, edge_index, _, in_channels = dummy_data
        model = build_sage(config, in_channels)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        assert not torch.isnan(out).any(), "SAGE output contains NaN"
        assert not torch.isinf(out).any(), "SAGE output contains Inf"


# ════════════════════════════════════════════════════════════════════════════════
# CROSS-MODEL TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestAllModels:

    @pytest.mark.parametrize("model_name", ["gcn", "gat", "sage"])
    def test_all_models_forward(self, config, dummy_data, model_name):
        """All three models should produce [50, 2] output on the same input."""
        from src.training.train import build_model
        x, edge_index, _, in_channels = dummy_data
        model = build_model(model_name, in_channels, config)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        assert out.shape == (50, 2), \
            f"{model_name.upper()} output shape {out.shape} != (50, 2)"

    @pytest.mark.parametrize("model_name", ["gcn", "gat", "sage"])
    def test_softmax_sums_to_one(self, config, dummy_data, model_name):
        """Softmax of logits must sum to 1 per node."""
        from src.training.train import build_model
        x, edge_index, _, in_channels = dummy_data
        model = build_model(model_name, in_channels, config)
        model.eval()
        with torch.no_grad():
            out   = model(x, edge_index)
            probs = torch.softmax(out, dim=1)
        row_sums = probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(50), atol=1e-5), \
            f"{model_name.upper()} softmax does not sum to 1"
