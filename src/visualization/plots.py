"""
src/visualization/plots.py
───────────────────────────
Reusable plot functions for the GraphFraud project.
All functions return matplotlib Figure objects so they can be used both in
notebooks and in the Streamlit dashboard.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc


# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#1a1a2e",   # dark navy background
    "axes.facecolor":    "#16213e",   # slightly lighter axes
    "axes.edgecolor":    "#4a4a8a",
    "axes.labelcolor":   "#e0e0ff",
    "xtick.color":       "#e0e0ff",
    "ytick.color":       "#e0e0ff",
    "text.color":        "#e0e0ff",
    "grid.color":        "#2a2a5a",
    "grid.alpha":        0.5,
    "font.family":       "sans-serif",
})

PALETTE = ["#7b68ee", "#00d4aa", "#ff6b6b", "#ffd93d", "#6bcb77"]
# Purple, Teal, Red, Yellow, Green — high contrast on dark background


def fraud_over_time(df):
    """
    Line chart: fraud ratio per timestep.

    ── What this shows ───────────────────────────────────────────────────────
    The Elliptic dataset has 49 timesteps — two-week intervals.
    Each timestep is a snapshot of the Bitcoin transaction graph.
    This plot shows how the FRACTION of illicit transactions changes over time.

    Spikes indicate periods of higher fraudulent activity.
    The dataset was collected around 2011–2019 when Bitcoin was growing rapidly.
    """
    known = df[df["label"] != -1]
    # Only compute ratio for labelled transactions (excludes unknown=-1)

    ratio = known.groupby("timestep")["label"].apply(lambda x: (x == 1).mean())
    # group by timestep → for each timestep, compute fraction of illicit nodes
    # (x == 1) creates a boolean Series → .mean() gives the fraction

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ratio.index, ratio.values, color=PALETTE[2], linewidth=2.5, marker="o", markersize=4)
    ax.fill_between(ratio.index, ratio.values, alpha=0.2, color=PALETTE[2])
    # fill_between shades the area under the curve — makes trends more visible

    ax.set_xlabel("Timestep (biweekly snapshot)", fontsize=12)
    ax.set_ylabel("Fraud ratio", fontsize=12)
    ax.set_title("Illicit Transaction Rate Over Time", fontsize=14, fontweight="bold", pad=15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    # Format y-axis as percentages (0.1 → 10%)
    ax.grid(True)
    fig.tight_layout()
    return fig


def label_distribution(df):
    """
    Bar chart: how many nodes are illicit / licit / unknown.

    ── Why this matters ─────────────────────────────────────────────────────
    77% of nodes are unlabelled (unknown). This is a semi-supervised setting:
    the model can leverage GRAPH STRUCTURE of unknown nodes even without labels.
    This is a core advantage of GNN over tabular models.
    """
    counts = df["label"].value_counts().sort_index()
    labels = {-1: "Unknown", 0: "Licit", 1: "Illicit"}
    names  = [labels[k] for k in counts.index]
    colors = [PALETTE[3], PALETTE[1], PALETTE[2]]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, counts.values, color=colors, edgecolor="#0a0a1a", linewidth=1.5)

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1500,
            f"{val:,}",
            ha="center", fontsize=11, fontweight="bold"
        )
    # Annotate each bar with its exact count

    ax.set_title("Node Label Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Transactions", fontsize=12)
    ax.grid(True, axis="y")
    fig.tight_layout()
    return fig


def training_curves(train_losses, val_f1s, model_name):
    """
    Two-panel plot: training loss and validation F1 over epochs.

    ── How to read this ─────────────────────────────────────────────────────
    Left panel (Loss): should decrease smoothly. Spikes = unstable training.
    Right panel (Val F1): should increase and plateau. Drop after peak = overfitting.
    Early stopping triggers when val F1 stops improving.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss curve ─────────────────────────────────────────────────────────
    ax1.plot(epochs, train_losses, color=PALETTE[0], linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax1.set_title(f"{model_name.upper()} — Training Loss", fontsize=13, fontweight="bold")
    ax1.grid(True)

    # ── Val F1 curve ───────────────────────────────────────────────────────
    ax2.plot(epochs, val_f1s, color=PALETTE[1], linewidth=2)
    best_epoch = int(np.argmax(val_f1s)) + 1
    ax2.axvline(best_epoch, color=PALETTE[2], linestyle="--", linewidth=1.5, label=f"Best (ep {best_epoch})")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("F1 Score (macro)", fontsize=12)
    ax2.set_title(f"{model_name.upper()} — Validation F1", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(f"Training History — {model_name.upper()}", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def roc_curve_comparison(results_dict):
    """
    Overlay ROC curves for all models on one plot.

    ── What ROC AUC means ───────────────────────────────────────────────────
    For each possible decision threshold, the ROC curve plots:
      x-axis: False Positive Rate (licit wrongly flagged as fraud)
      y-axis: True Positive Rate (fraud correctly caught)
    A perfect model hugs the top-left corner → AUC = 1.0
    Random guessing = diagonal line → AUC = 0.5

    ── Why compare models? ──────────────────────────────────────────────────
    The curve shows the TRADEOFF at every threshold, not just the default 0.5.
    In fraud detection, you might want to catch 99% of fraud even if you flag
    some licit transactions — the ROC curve lets you pick that operating point.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (name, result) in enumerate(results_dict.items()):
        if "y_true" not in result or "y_prob" not in result:
            continue
        fpr, tpr, _ = roc_curve(result["y_true"], result["y_prob"])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], linewidth=2.5,
                label=f"{name.upper()} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "w--", linewidth=1, label="Random (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate (Licit flagged as Fraud)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Fraud Caught)", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.grid(True)
    fig.tight_layout()
    return fig


def confusion_matrix_plot(cm, model_name):
    """
    Heatmap of the confusion matrix.

    ── Reading a confusion matrix ───────────────────────────────────────────
    Rows = Actual class, Columns = Predicted class

         Predicted Licit | Predicted Fraud
    Actual Licit  [  TN  |  FP  ]   FP = licit wrongly flagged (annoying)
    Actual Fraud  [  FN  |  TP  ]   FN = fraud missed (costly!)

    In fraud detection: minimising FN is the priority (catching fraud matters
    more than avoiding false alarms). The class weights in training reflect this.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Licit", "Illicit"],
        yticklabels=["Licit", "Illicit"],
        ax=ax, linewidths=0.5, cbar=False,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"{model_name.upper()} — Confusion Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def model_comparison_bar(results_dict):
    """
    Side-by-side bar chart comparing F1 and AUC-ROC across all models.
    Includes baseline results for easy comparison.
    """
    baselines = {
        "XGBoost":  {"f1": 0.9643, "auc_roc": 0.9971},
        "LightGBM": {"f1": 0.9794, "auc_roc": 0.9985},
    }
    all_results = {**baselines, **results_dict}

    names    = list(all_results.keys())
    f1s      = [r["f1"]      for r in all_results.values()]
    aucs     = [r["auc_roc"] for r in all_results.values()]
    x        = np.arange(len(names))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, f1s,  width, label="F1 (macro)", color=PALETTE[0], alpha=0.85)
    bars2 = ax.bar(x + width/2, aucs, width, label="AUC-ROC",   color=PALETTE[1], alpha=0.85)

    for bar in bars1 + bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{bar.get_height():.4f}",
            ha="center", fontsize=8.5, fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in names], fontsize=11)
    ax.set_ylim(0.92, 1.005)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Baselines vs GNNs", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y")

    # Draw a vertical line between baselines and GNNs
    ax.axvline(x=1.5, color="white", linestyle="--", alpha=0.3, linewidth=1)
    ax.text(0.5, 0.935, "← Baselines", ha="center", fontsize=9, color="gray",
            transform=ax.get_xaxis_transform())
    ax.text(3.0, 0.935, "GNNs →", ha="center", fontsize=9, color="gray",
            transform=ax.get_xaxis_transform())

    fig.tight_layout()
    return fig


def save_figure(fig, filename, config):
    """Save a figure to the outputs/figures/ directory."""
    figures_dir = Path(config["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Figure saved → {path}")
    return path


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]   # GraphFraud/
    sys.path.insert(0, str(ROOT))

    from src.data.ingest import load_config, load_raw, build_graph_df

    print("Loading config and data...")
    config  = load_config(str(ROOT / "configs" / "config.yaml"))
    classes, edges, features = load_raw(config)
    df = build_graph_df(features, classes, config)
    print(f"  Loaded {len(df):,} transactions")

    # ── 1. Fraud over time ────────────────────────────────────────────────────
    print("\nGenerating: fraud_over_time...")
    fig = fraud_over_time(df)
    save_figure(fig, "fraud_over_time.png", config)
    plt.close(fig)

    # ── 2. Label distribution ─────────────────────────────────────────────────
    print("Generating: label_distribution...")
    fig = label_distribution(df)
    save_figure(fig, "label_distribution.png", config)
    plt.close(fig)

    # ── 3. Training curves (per model) ────────────────────────────────────────
    reports_dir = Path(config["paths"]["reports_dir"])
    for model_name in ["gcn", "gat", "sage"]:
        results_path = reports_dir / f"{model_name}_results.json"
        if results_path.exists():
            print(f"Generating: training_curves_{model_name}...")
            with open(results_path) as f:
                result = json.load(f)
            fig = training_curves(result["train_losses"], result["val_f1s"], model_name)
            save_figure(fig, f"training_curves_{model_name}.png", config)
            plt.close(fig)
        else:
            print(f"  Skipping training_curves_{model_name} (run train.py first)")

    # ── 4. Model comparison bar chart ─────────────────────────────────────────
    print("Generating: model_comparison...")
    gnn_results = {}
    for model_name in ["gcn", "gat", "sage"]:
        p = reports_dir / f"{model_name}_results.json"
        if p.exists():
            with open(p) as f:
                gnn_results[model_name] = json.load(f)

    if gnn_results:
        fig = model_comparison_bar(gnn_results)
        save_figure(fig, "model_comparison.png", config)
        plt.close(fig)

    # ── 5. ROC curves + confusion matrices (requires inference) ───────────────
    if gnn_results:
        print("Generating ROC curves and confusion matrices (running inference)...")
        import torch
        from src.data.graph_builder import build_pyg_data
        from src.training.evaluate  import create_node_masks, evaluate_gnn
        from src.training.train     import get_device, build_model

        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        data         = build_pyg_data(df, edges, feature_cols)
        _, _, test_mask = create_node_masks(df, config)
        device       = get_device(config)
        data         = data.to(device)
        test_mask    = test_mask.to(device)

        roc_data = {}
        for model_name in gnn_results:
            checkpoint = Path(config["paths"]["models_dir"]) / f"{model_name}_best.pt"
            if checkpoint.exists():
                print(f"  Inference: {model_name.upper()}...")
                model = build_model(model_name, len(feature_cols), config).to(device)
                model.load_state_dict(torch.load(checkpoint, map_location=device))
                metrics = evaluate_gnn(model, data, test_mask, device)
                roc_data[model_name] = {
                    "y_true": metrics["y_true"],
                    "y_prob": metrics["y_prob"],
                }
                fig = confusion_matrix_plot(metrics["confusion_matrix"], model_name)
                save_figure(fig, f"confusion_matrix_{model_name}.png", config)
                plt.close(fig)

        if roc_data:
            fig = roc_curve_comparison(roc_data)
            save_figure(fig, "roc_curve_comparison.png", config)
            plt.close(fig)

    # ── Summary ───────────────────────────────────────────────────────────────
    figures_dir = Path(config["paths"]["figures_dir"])
    saved = list(figures_dir.glob("*.png"))
    print(f"\n{'='*50}")
    print(f"  All done! {len(saved)} figures saved to {figures_dir}")
    print(f"{'='*50}")
    for p in sorted(saved):
        print(f"  · {p.name}")
