"""
tests/test_data.py
───────────────────
Unit tests for the data pipeline:
  - ingest.py       : load_config, load_raw, build_graph_df
  - preprocess.py   : get_feature_cols, split_known_unknown, train_val_test_split
  - graph_builder.py: build_networkx_graph, build_pyg_data

Run with: pytest tests/test_data.py -v  (from GraphFraud/ directory)
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add GraphFraud/ to path so imports work
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ingest import load_config, load_raw, build_graph_df


# ── Fixtures ──────────────────────────────────────────────────────────────────
# A pytest "fixture" is a reusable piece of setup code.
# Any test function that takes `config` or `df` as a parameter
# automatically gets the return value of the fixture.

@pytest.fixture(scope="session")
def config():
    """Load config once for the entire test session."""
    return load_config(str(ROOT / "configs" / "config.yaml"))


@pytest.fixture(scope="session")
def raw_data(config):
    """Load raw CSVs once for the entire test session."""
    return load_raw(config)


@pytest.fixture(scope="session")
def df(config, raw_data):
    """Build the merged graph DataFrame once."""
    classes, edges, features = raw_data
    return build_graph_df(features, classes, config)


# ════════════════════════════════════════════════════════════════════════════════
# INGEST TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestIngest:

    def test_config_has_required_keys(self, config):
        """Config file must have all sections we depend on."""
        assert "paths"    in config, "Missing 'paths' section in config"
        assert "labels"   in config, "Missing 'labels' section in config"
        assert "gnn"      in config, "Missing 'gnn' section in config"
        assert "training" in config, "Missing 'training' section in config"
        assert "baseline" in config, "Missing 'baseline' section in config"

    def test_config_labels(self, config):
        """Label values must be exactly 1, 0, -1."""
        lm = config["labels"]
        assert lm["illicit"] == 1,  "illicit should map to 1"
        assert lm["licit"]   == 0,  "licit should map to 0"
        assert lm["unknown"] == -1, "unknown should map to -1"

    def test_load_raw_shapes(self, raw_data):
        """CSVs must have the expected number of rows and columns."""
        classes, edges, features = raw_data
        assert len(features) == 203769, f"Expected 203769 rows, got {len(features)}"
        assert len(classes)  == 203769, "Classes CSV should have same number of rows"
        assert len(edges)    == 234355, f"Expected 234355 edges, got {len(edges)}"

    def test_features_has_correct_columns(self, raw_data):
        """Features DataFrame must have txId, timestep, and feat_* columns."""
        _, _, features = raw_data
        assert "txId"     in features.columns
        assert "timestep" in features.columns
        feat_cols = [c for c in features.columns if c.startswith("feat_")]
        assert len(feat_cols) == 165, f"Expected 165 feature cols, got {len(feat_cols)}"

    def test_build_graph_df_shape(self, df):
        """Merged DataFrame must have correct shape."""
        assert len(df) == 203769, "Should have 203769 rows after merge"
        assert "label" in df.columns, "Should have 'label' column after mapping"
        assert "class" in df.columns, "Should have 'class' column from classes CSV"

    def test_label_values(self, df):
        """Labels must only be -1, 0, or 1."""
        valid_labels = {-1, 0, 1}
        actual_labels = set(df["label"].unique())
        assert actual_labels == valid_labels, f"Unexpected labels: {actual_labels}"

    def test_label_counts(self, df):
        """Label distribution must match the known dataset stats."""
        counts = df["label"].value_counts()
        assert counts[-1] == 157205, f"Expected 157205 unknowns, got {counts[-1]}"
        assert counts[0]  == 42019,  f"Expected 42019 licit, got {counts[0]}"
        assert counts[1]  == 4545,   f"Expected 4545 illicit, got {counts[1]}"

    def test_no_null_labels(self, df):
        """No NaN in the label column — all rows should be mapped."""
        assert df["label"].isnull().sum() == 0, "Found NaN labels after mapping"


# ════════════════════════════════════════════════════════════════════════════════
# PREPROCESS TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestPreprocess:

    def test_get_feature_cols_count(self, df):
        from src.data.preprocess import get_feature_cols
        cols = get_feature_cols(df)
        assert len(cols) == 165, f"Expected 165 feature cols, got {len(cols)}"

    def test_get_feature_cols_naming(self, df):
        from src.data.preprocess import get_feature_cols
        cols = get_feature_cols(df)
        assert all(c.startswith("feat_") for c in cols), "All feature cols should start with feat_"

    def test_split_known_unknown_totals(self, df):
        from src.data.preprocess import split_known_unknown
        known, unknown = split_known_unknown(df)
        assert len(known) + len(unknown) == len(df), "Split totals must equal full df"

    def test_split_known_no_unknown_labels(self, df):
        from src.data.preprocess import split_known_unknown
        known, _ = split_known_unknown(df)
        assert -1 not in known["label"].values, "known_df must not contain label=-1"

    def test_split_unknown_only_unknown_labels(self, df):
        from src.data.preprocess import split_known_unknown
        _, unknown = split_known_unknown(df)
        assert set(unknown["label"].unique()) == {-1}, "unknown_df must only have label=-1"

    def test_train_val_test_split_shapes(self, df, config):
        from src.data.preprocess import get_feature_cols, split_known_unknown, train_val_test_split
        feature_cols = get_feature_cols(df)
        known, _     = split_known_unknown(df)
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(known, feature_cols, config)

        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(known), "Split totals must equal known df size"
        assert X_train.shape[1] == 165, "X_train should have 165 features"

    def test_train_val_test_no_unknown_labels(self, df, config):
        from src.data.preprocess import get_feature_cols, split_known_unknown, train_val_test_split
        feature_cols = get_feature_cols(df)
        known, _     = split_known_unknown(df)
        _, _, _, y_train, y_val, y_test = train_val_test_split(known, feature_cols, config)

        for split_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
            assert -1 not in y, f"{split_name} labels contain -1 (unknown) — should be excluded"


# ════════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestGraphBuilder:

    def test_networkx_graph_node_count(self, df, raw_data):
        from src.data.graph_builder import build_networkx_graph
        _, edges, _ = raw_data
        G = build_networkx_graph(edges, df)
        # Graph may have fewer nodes than df if some txIds appear only in edges
        assert G.number_of_nodes() > 0

    def test_networkx_graph_edge_count(self, df, raw_data):
        from src.data.graph_builder import build_networkx_graph
        _, edges, _ = raw_data
        G = build_networkx_graph(edges, df)
        assert G.number_of_edges() == 234355

    def test_pyg_data_shapes(self, df, raw_data):
        from src.data.graph_builder import build_pyg_data
        _, edges, _ = raw_data
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        data = build_pyg_data(df, edges, feature_cols)

        assert data.x.shape          == (203769, 165),  f"x shape wrong: {data.x.shape}"
        assert data.y.shape          == (203769,),       f"y shape wrong: {data.y.shape}"
        assert data.edge_index.shape[0] == 2,           "edge_index must have 2 rows"
        assert data.edge_index.shape[1] >  0,           "edge_index must have edges"

    def test_pyg_data_dtypes(self, df, raw_data):
        import torch
        from src.data.graph_builder import build_pyg_data
        _, edges, _ = raw_data
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        data = build_pyg_data(df, edges, feature_cols)

        assert data.x.dtype          == torch.float32, "x must be float32"
        assert data.y.dtype          == torch.int64,   "y must be int64 (long)"
        assert data.edge_index.dtype == torch.int64,   "edge_index must be int64"
