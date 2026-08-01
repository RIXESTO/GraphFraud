import torch
import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)


def evaluate_gnn(model, data, mask, device):
    """
    Evaluate a trained GNN model on a subset of nodes defined by `mask`.

    ── What this function does ───────────────────────────────────────────────
    1. Switches the model to evaluation mode (disables dropout)
    2. Runs a forward pass on the ENTIRE graph (all 203k nodes)
    3. Extracts predictions only for the nodes in `mask`
    4. Computes classification metrics

    Parameters
    ----------
    model  : trained GNN model (GCN, GAT, or GraphSAGE)
    data   : PyG Data object (the full graph)
    mask   : boolean tensor [num_nodes] — True for nodes to evaluate
    device : "cpu" or "mps"

    Returns
    -------
    dict with keys: f1, auc_roc, report, confusion_matrix, y_true, y_pred, y_prob
    """
    model.eval()
    # model.eval() switches off dropout.
    # During training, dropout randomly zeros neurons to prevent overfitting.
    # During evaluation, you want the FULL network — every neuron active.
    # Forgetting this is a very common bug that makes eval metrics look worse.

    with torch.no_grad():
        # torch.no_grad() tells PyTorch: "don't track gradients".
        # During training, PyTorch records all operations to compute gradients
        # for backpropagation. During evaluation, you don't need gradients —
        # turning this off saves memory and makes inference 2–3x faster.

        out = model(data.x.to(device), data.edge_index.to(device))
        # Forward pass: model processes ALL nodes in the graph.
        # out shape: [203769, 2] — two logit scores per node (licit, illicit)

    # ── Extract only the nodes we care about ─────────────────────────────────
    out_masked = out[mask]
    # Boolean indexing: keep only rows where mask is True
    # If mask = val_mask, this gives predictions for validation nodes only

    y_true = data.y[mask].cpu().numpy()
    # True labels for masked nodes — move from GPU to CPU, convert to numpy
    # These are 0 (licit) or 1 (illicit) — no -1 unknowns in val/test masks

    # ── Convert logits to class predictions ───────────────────────────────────
    y_pred = out_masked.argmax(dim=1).cpu().numpy()
    # argmax(dim=1): for each node, pick the class with the higher score
    # dim=1 means "across the 2 class scores" (not across nodes)
    # Example: logits = [0.3, 0.8] → argmax = 1 → predicted as illicit

    # ── Convert logits to fraud probabilities ─────────────────────────────────
    y_prob = torch.softmax(out_masked, dim=1)[:, 1].cpu().numpy()
    # softmax converts raw logits to probabilities that sum to 1
    # [:, 1] takes the fraud (class 1) probability column
    # Used for AUC-ROC which needs a continuous score, not a hard 0/1 label

    # ── Compute metrics ───────────────────────────────────────────────────────
    return {
        "f1":               f1_score(y_true, y_pred, average="macro"),
        # macro F1: compute F1 for each class separately, then average
        # "macro" treats both classes equally regardless of size
        # Best for imbalanced data where you care equally about fraud AND licit

        "auc_roc":          roc_auc_score(y_true, y_prob),
        # AUC-ROC: Area Under the ROC Curve
        # Measures how well the model RANKS fraud above licit (order matters)
        # 1.0 = perfect, 0.5 = random guessing

        "auc_pr":           average_precision_score(y_true, y_prob),
        # AUC-PR: Area Under the Precision-Recall Curve
        # The PRIMARY metric for severely imbalanced data (9.8% fraud).
        # AUC-ROC can look artificially high because it rewards the model for
        # correctly classifying the easy 90% licit majority. AUC-PR only cares
        # about how well the model finds the rare positive (fraud) class.
        # A random classifier scores ~0.098 (the fraud base rate) here.
        # A perfect classifier scores 1.0.

        "report":           classification_report(y_true, y_pred, digits=4),
        # Precision, recall, F1 per class — detailed breakdown

        "confusion_matrix": confusion_matrix(y_true, y_pred),
        # 2x2 matrix: [[TN, FP], [FN, TP]]
        # TN: correctly called licit   FP: wrongly called fraud
        # FN: missed fraud (costly!)   TP: correctly caught fraud

        "y_true":           y_true,
        "y_pred":           y_pred,
        "y_prob":           y_prob,
        # Raw arrays — useful for plotting ROC curves and confusion matrices
    }


def create_node_masks(df, config):
    """
    Create boolean train/val/test masks for node classification.

    ── Why masks instead of separate arrays? ────────────────────────────────
    In graph ML, the model processes ALL nodes simultaneously (the full graph
    must be in memory for message passing). Masks let us:
      - Do a single forward pass over all 203k nodes
      - Then select only the relevant nodes for loss / evaluation

    The mask is a boolean tensor of size [num_nodes]:
      True  = this node belongs to this split
      False = ignore this node

    Parameters
    ----------
    df     : full merged DataFrame (203769 rows, index 0..N-1)
    config : config dict from config.yaml

    Returns
    -------
    train_mask, val_mask, test_mask : torch.BoolTensor [num_nodes]
    """
    from sklearn.model_selection import train_test_split

    seed      = config["training"]["random_seed"]
    test_size = config["training"]["test_size"]
    val_size  = config["training"]["val_size"]

    # ── Get the known (labelled) node indices ─────────────────────────────────
    known_mask    = df["label"] != -1
    known_indices = df.index[known_mask].tolist()
    # df.index values ARE the PyG node indices (0..N-1) because
    # the features CSV has a default RangeIndex preserved through the merge
    known_labels  = df.loc[known_mask, "label"].values

    # ── Split known indices into train / val / test ───────────────────────────
    train_val_idx, test_idx = train_test_split(
        known_indices,
        test_size=test_size,
        stratify=known_labels,
        random_state=seed,
    )
    train_val_labels = df.loc[train_val_idx, "label"].values
    val_ratio = val_size / (1 - test_size)

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        stratify=train_val_labels,
        random_state=seed,
    )

    # ── Build boolean masks ───────────────────────────────────────────────────
    num_nodes  = len(df)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    return train_mask, val_mask, test_mask


def compute_class_weights(df, device):
    """
    Compute class weights to handle the 9:1 class imbalance.

    ── Why class weights? ────────────────────────────────────────────────────
    Without this: the model learns "always predict licit" → 90% accuracy but
    catches ZERO fraud. Class weights penalise misclassifying fraud more.

    Formula: weight[class] = total_known / (2 * count[class])
    This gives higher weight to the minority (fraud) class.

    Returns a tensor [weight_licit, weight_illicit] for CrossEntropyLoss.
    """
    known = df[df["label"] != -1]
    n_licit   = (known["label"] == 0).sum()
    n_illicit = (known["label"] == 1).sum()
    n_total   = len(known)

    w_licit   = n_total / (2 * n_licit)
    w_illicit = n_total / (2 * n_illicit)

    return torch.tensor([w_licit, w_illicit], dtype=torch.float).to(device)
