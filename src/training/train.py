import os
import json
import torch
import torch.nn as nn
from pathlib import Path


def get_device(config):
    """
    Select the best available compute device.

    ── Device priority ───────────────────────────────────────────────────────
    1. MPS  — Apple Silicon GPU (M1/M2/M3/M4). Your Mac Air M4 has this.
              Much faster than CPU for tensor operations.
    2. CUDA — NVIDIA GPU. Not available on Macs.
    3. CPU  — Fallback. Always available, but slowest.

    The config file says "cpu" by default — this function overrides it if
    a faster device is actually available.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def build_model(model_name, in_channels, config):
    """
    Instantiate a GNN model by name.

    Parameters
    ----------
    model_name : "gcn", "gat", or "sage"
    in_channels: number of input features (165)
    config     : config dict

    Returns
    -------
    An untrained GNN model (nn.Module)
    """
    if model_name == "gcn":
        from src.models.gcn import build_gcn
        return build_gcn(config, in_channels)
    elif model_name == "gat":
        from src.models.gat import build_gat
        return build_gat(config, in_channels)
    elif model_name == "sage":
        from src.models.sage import build_sage
        return build_sage(config, in_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'gcn', 'gat', or 'sage'.")


def train_one_epoch(model, data, optimizer, criterion, train_mask, device):
    """
    Run one complete training step (one epoch).

    ── What happens in one epoch ─────────────────────────────────────────────
    1. model.train()    — enable dropout and batch norm in training mode
    2. optimizer.zero_grad()  — clear gradients from the previous step
    3. out = model(...)       — forward pass: compute predictions
    4. loss = criterion(...)  — measure how wrong the predictions are
    5. loss.backward()        — backpropagation: compute gradients
    6. optimizer.step()       — update weights using gradients

    Parameters
    ----------
    model      : GNN model
    data       : PyG Data object (full graph, on device)
    optimizer  : Adam optimizer
    criterion  : weighted CrossEntropyLoss
    train_mask : boolean tensor — which nodes to train on
    device     : compute device

    Returns
    -------
    loss value (float) for this epoch
    """
    model.train()
    # Switches on dropout and batch norm training behaviour.
    # MUST call this before every training step.

    optimizer.zero_grad()
    # Clear gradients from the PREVIOUS step.
    # PyTorch ACCUMULATES gradients by default — if you forget this,
    # gradients from 10 epochs ago are still adding up → training diverges.

    out = model(data.x, data.edge_index)
    # Forward pass: compute predictions for ALL 203769 nodes
    # out shape: [203769, 2]

    loss = criterion(out[train_mask], data.y[train_mask])
    # out[train_mask]: predictions for ONLY the training nodes [~32594, 2]
    # data.y[train_mask]: true labels for training nodes [~32594]
    #
    # criterion = CrossEntropyLoss with class weights
    # CrossEntropyLoss internally does:
    #   1. softmax(out) → probabilities
    #   2. -log(probability of the correct class)
    #   3. mean across all training nodes
    # Higher loss = model is more wrong = more gradient signal to learn from

    loss.backward()
    # Backpropagation: walk backward through all the computations and compute
    # how much each weight contributed to the loss.
    # Stores the gradient in each parameter's .grad attribute.

    optimizer.step()
    # Update ALL weights using the computed gradients:
    # weight = weight - learning_rate * gradient
    # Adam is smarter than basic gradient descent — it uses momentum and
    # adapts the learning rate per parameter.

    return loss.item()
    # .item() converts a 1-element tensor to a plain Python float


def run_training(model_name, config):
    """
    Full training pipeline for a GNN model with early stopping.

    ── The complete flow ─────────────────────────────────────────────────────
    1.  Load raw data and build graph DataFrame
    2.  Build PyG Data object (graph + features + labels)
    3.  Create train/val/test masks
    4.  Create model, optimizer, loss function
    5.  Training loop:
        a. Train one epoch → compute train loss
        b. Evaluate on val nodes → compute val F1
        c. If val F1 improved → save model checkpoint
        d. If val F1 hasn't improved for `patience` epochs → stop early
    6.  Load best checkpoint and evaluate on test nodes
    7.  Save and return results

    Parameters
    ----------
    model_name : "gcn", "gat", or "sage"
    config     : config dict from config.yaml

    Returns
    -------
    dict with test metrics and training history
    """
    import re
    from src.data.ingest import load_raw, build_graph_df
    from src.data.graph_builder import build_pyg_data
    from src.training.evaluate import evaluate_gnn, create_node_masks, compute_class_weights

    # ── 1. Prepare data ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Training {model_name.upper()}")
    print(f"{'='*50}")

    classes, edges, features = load_raw(config)
    df = build_graph_df(features, classes, config)

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    # Get feature column names without depending on preprocess module

    # ── 2. Build PyG graph ────────────────────────────────────────────────────
    print("Building graph...")
    data = build_pyg_data(df, edges, feature_cols)
    # data.x          : [203769, 165] feature matrix
    # data.edge_index : [2, 234355]   directed edges
    # data.y          : [203769]      labels (0, 1, or -1)

    # ── 3. Create masks ───────────────────────────────────────────────────────
    train_mask, val_mask, test_mask = create_node_masks(df, config)
    print(f"Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()} nodes")

    # ── 4. Setup: device, model, optimizer, loss ──────────────────────────────
    device = get_device(config)
    print(f"Device: {device}")

    in_channels = len(feature_cols)         # 165
    model = build_model(model_name, in_channels, config).to(device)
    # .to(device) moves ALL model weights to GPU (MPS) if available

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["gnn"]["lr"],                     # 0.001
        weight_decay=config["gnn"]["weight_decay"],  # 1e-5 (L2 regularisation)
    )
    # Adam optimizer: adapts the learning rate for each weight individually.
    # weight_decay adds a penalty for large weights → prevents overfitting.

    class_weights = compute_class_weights(df, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # CrossEntropyLoss with weights: penalises fraud misclassification ~9x more
    # than licit misclassification — matches the class imbalance ratio

    # Move data to device
    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask   = val_mask.to(device)
    test_mask  = test_mask.to(device)

    # ── 5. Training loop with early stopping ──────────────────────────────────
    epochs   = config["gnn"]["epochs"]    # 200 max
    patience = config["gnn"]["patience"]  # stop if no improvement for 20 epochs

    best_val_f1      = 0.0
    patience_counter = 0
    train_losses     = []
    val_f1s          = []

    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"{model_name}_best.pt"

    print(f"\nEpoch | Train Loss | Val F1  | Val AUC")
    print("-" * 45)

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        train_loss = train_one_epoch(model, data, optimizer, criterion, train_mask, device)
        train_losses.append(train_loss)

        # ── Validate ───────────────────────────────────────────────────────
        val_metrics = evaluate_gnn(model, data, val_mask, device)
        val_f1  = val_metrics["f1"]
        val_auc = val_metrics["auc_roc"]
        val_f1s.append(val_f1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:3d}  |  {train_loss:.4f}     |  {val_f1:.4f}  |  {val_auc:.4f}")

        # ── Early stopping ─────────────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            # Save the model weights when val F1 improves.
            # state_dict() = a dict of all parameter tensors.
            # We save the BEST version, not the final one
            # (the final might have overfit on training data)
        else:
            patience_counter += 1
            # Val F1 didn't improve this epoch

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
            # Stop training when val F1 hasn't improved for `patience` epochs.
            # This prevents wasting time and overfitting.

    # ── 6. Load best model and evaluate on test set ───────────────────────────
    print(f"\nLoading best model (val F1: {best_val_f1:.4f})...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # load_state_dict restores the saved weights — undoes any overfitting
    # that happened in epochs after the best checkpoint

    test_metrics = evaluate_gnn(model, data, test_mask, device)

    print(f"\n── Test Results ({model_name.upper()}) ──────────────")
    print(f"  F1 (macro): {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC:    {test_metrics['auc_roc']:.4f}")
    print(f"\n{test_metrics['report']}")

    # ── 7. Save results ───────────────────────────────────────────────────────
    results = {
        "model":        model_name,
        "f1":           float(test_metrics["f1"]),
        "auc_roc":      float(test_metrics["auc_roc"]),
        "best_val_f1":  float(best_val_f1),
        "epochs_run":   epoch,
        "train_losses": train_losses,
        "val_f1s":      val_f1s,
        "report":       test_metrics["report"],
    }

    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_path = reports_dir / f"{model_name}_results.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    return results


if __name__ == "__main__":
    # ── Run all three GNN models sequentially ─────────────────────────────────
    from src.data.ingest import load_config

    config = load_config("configs/config.yaml")

    all_results = {}
    for model_name in ["gcn", "gat", "sage"]:
        results = run_training(model_name, config)
        all_results[model_name] = results

    # ── Final comparison ──────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  FINAL COMPARISON")
    print(f"{'='*50}")
    print(f"{'Model':<10} {'F1 (macro)':<14} {'AUC-ROC'}")
    print("-" * 35)

    baselines = {
        "XGBoost":   {"f1": 0.9643, "auc_roc": 0.9971},
        "LightGBM":  {"f1": 0.9794, "auc_roc": 0.9985},
    }
    for name, r in baselines.items():
        print(f"{name:<10} {r['f1']:.4f}         {r['auc_roc']:.4f}")

    for name, r in all_results.items():
        print(f"{name.upper():<10} {r['f1']:.4f}         {r['auc_roc']:.4f}")
