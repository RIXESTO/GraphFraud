"""
Patch existing results JSON files with auc_pr by re-evaluating saved checkpoints.
Run once from the GraphFraud/ root:
    python scripts/patch_auc_pr.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.ingest import load_config, load_raw, build_graph_df
from src.data.graph_builder import build_pyg_data
from src.training.evaluate import evaluate_gnn, create_node_masks
from src.training.train import build_model, get_device
import torch

print("Loading config and data...")
config = load_config("configs/config.yaml")
classes, edges, features = load_raw(config)
df = build_graph_df(features, classes, config)
feature_cols = [c for c in df.columns if c.startswith("feat_")]

data = build_pyg_data(df, edges, feature_cols)
_, _, test_mask = create_node_masks(df, config)

device = get_device(config)
data = data.to(device)
test_mask = test_mask.to(device)

models_dir  = Path(config["paths"]["models_dir"])
reports_dir = Path(config["paths"]["reports_dir"])

for model_name in ["gcn", "gat", "sage"]:
    results_path   = reports_dir / f"{model_name}_results.json"
    checkpoint_path = models_dir / f"{model_name}_best.pt"

    if not results_path.exists():
        print(f"  [{model_name.upper()}] No results JSON — skipping.")
        continue
    if not checkpoint_path.exists():
        print(f"  [{model_name.upper()}] No checkpoint — skipping.")
        continue

    with open(results_path) as f:
        results = json.load(f)

    if "auc_pr" in results:
        print(f"  [{model_name.upper()}] auc_pr already present ({results['auc_pr']:.4f}) — skipping.")
        continue

    print(f"  [{model_name.upper()}] Computing AUC-PR...")
    model = build_model(model_name, len(feature_cols), config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    metrics = evaluate_gnn(model, data, test_mask, device)
    results["auc_pr"] = float(metrics["auc_pr"])

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  [{model_name.upper()}] auc_pr = {results['auc_pr']:.4f}  ✓  saved → {results_path}")

print("\nDone.")
