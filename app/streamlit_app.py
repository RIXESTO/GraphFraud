"""
app/streamlit_app.py
─────────────────────
Interactive dashboard for the GraphFraud project.

How to run:
    cd GraphFraud
    streamlit run app/streamlit_app.py

Tabs:
  1. Dataset Overview  — stats and fraud-over-time chart
  2. Train GNN         — train GCN / GAT / SAGE and see results
  3. Compare Models    — side-by-side comparison of all models
"""

import sys
import json
import time
import torch
import numpy as np
import streamlit as st
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Streamlit runs from the app/ directory. Add GraphFraud/ to sys.path
# so that 'src.*' imports resolve correctly.
ROOT = Path(__file__).resolve().parents[1]   # GraphFraud/
sys.path.insert(0, str(ROOT))

from src.data.ingest    import load_config, load_raw, build_graph_df
from src.data.download  import ensure_dataset
from src.visualization  import plots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "GraphFraud — Bitcoin Fraud Detection",
    page_icon  = "🔍",
    layout     = "wide",
)

# ── Custom CSS: dark theme ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main            { background-color: #0f0f23; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a3e, #2a2a5e);
        border-radius: 12px; padding: 1.2rem 1.5rem;
        border: 1px solid #3a3a7e;
    }
    h1 { color: #7b68ee !important; }
    h2 { color: #00d4aa !important; }
    .stTabs [data-baseweb="tab"] { color: #a0a0cc; }
    .stTabs [aria-selected="true"] { color: #7b68ee; border-bottom: 2px solid #7b68ee; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔍 GraphFraud — Bitcoin Fraud Detection")
st.markdown("**Graph Neural Networks on the Elliptic Bitcoin Dataset**")
st.divider()


# ── Load config once (cached) ──────────────────────────────────────────────────
@st.cache_resource
def get_config():
    """
    @st.cache_resource: runs once, then caches the result.
    Without caching, config would reload on every user interaction.
    """
    return load_config(str(ROOT / "configs" / "config.yaml"), root=ROOT)


@st.cache_data(show_spinner="Loading dataset (downloads from Hugging Face on first run, ~665 MB)...")
def get_data(_config):
    """
    @st.cache_data: caches data across reruns.
    Loading the Elliptic CSVs takes ~10s — caching avoids reloading
    every time the user clicks a button.
    The leading underscore in `_config` tells Streamlit not to hash this argument.
    """
    if not ensure_dataset(_config, root=ROOT):
        return None, None
    try:
        classes, edges, features = load_raw(_config)
        df = build_graph_df(features, classes, _config)
        return df, edges
    except FileNotFoundError:
        return None, None


config = get_config()
df, edges = get_data(config)
DATA_AVAILABLE = df is not None

if DATA_AVAILABLE:
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
else:
    feature_cols = ["feat_X"] * 165


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "🧠 Train GNN", "📈 Compare Models"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Elliptic Bitcoin Dataset")

    if not DATA_AVAILABLE:
        repo_id = config.get("huggingface", {}).get("repo_id", "RIXESTO/elliptic-bitcoin")
        st.warning(
            f"⚠️ **Dataset unavailable.** Could not load local files or download from "
            f"[Hugging Face](https://huggingface.co/datasets/{repo_id}). "
            "Displaying cached metrics and figures."
        )

    # ── Key stats row ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    
    if DATA_AVAILABLE:
        known = df[df["label"] != -1]
        n_total = len(df)
        n_edges = "234,355"
        n_fraud = (df["label"] == 1).sum()
        pct_fraud = f"{n_fraud/len(known)*100:.1f}%"
        n_unknown = (df["label"] == -1).sum()
    else:
        n_total = 203769
        n_edges = "234,355"
        n_fraud = 4545
        pct_fraud = "9.8%"
        n_unknown = 157205

    with col1:
        st.metric("Total Transactions", f"{n_total:,}", help="Nodes in the graph")
    with col2:
        st.metric("Total Edges", n_edges, help="Directed transaction links")
    with col3:
        st.metric("Illicit (Fraud)", f"{n_fraud:,}",
                  delta=f"{pct_fraud} of labelled",
                  delta_color="inverse")
    with col4:
        st.metric("Unlabelled", f"{n_unknown:,}",
                  help="Nodes with no ground truth label")

    st.divider()

    # ── Charts ─────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Fraud Rate Over Time")
        st.caption("Each timestep = 2-week snapshot of the Bitcoin network (49 total)")
        if DATA_AVAILABLE:
            fig_time = plots.fraud_over_time(df)
            st.pyplot(fig_time)
        else:
            st.image(str(ROOT / "outputs" / "figures" / "fraud_over_time.png"))

    with col_right:
        st.subheader("Label Distribution")
        if DATA_AVAILABLE:
            fig_dist = plots.label_distribution(df)
            st.pyplot(fig_dist)
        else:
            st.image(str(ROOT / "outputs" / "figures" / "label_distribution.png"))

    # ── Feature info ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Feature Information")
    st.markdown(f"""
    | Property | Value |
    |---|---|
    | Feature columns | **{len(feature_cols)}** (feat_1 to feat_165) |
    | Local features | 94 (transaction-level stats) |
    | Aggregated features | 71 (neighbourhood aggregations) |
    | Timesteps | 49 (biweekly snapshots, 2011–2019) |
    """)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN GNN
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Train a Graph Neural Network")

    col_config, col_info = st.columns([1, 2])

    with col_config:
        model_choice = st.selectbox(
            "Select GNN Architecture",
            options=["gcn", "gat", "sage"],
            format_func=lambda x: {
                "gcn":  "GCN — Graph Convolutional Network",
                "gat":  "GAT — Graph Attention Network",
                "sage": "GraphSAGE — Scalable + Inductive",
            }[x]
        )

        st.markdown("**Hyperparameters** (from config.yaml)")
        st.code(f"""
hidden_channels : {config['gnn']['hidden_channels']}
num_layers      : {config['gnn']['num_layers']}
dropout         : {config['gnn']['dropout']}
lr              : {config['gnn']['lr']}
epochs (max)    : {config['gnn']['epochs']}
patience        : {config['gnn']['patience']}
        """)

    with col_info:
        model_info = {
            "gcn": {
                "name": "GCN — Graph Convolutional Network",
                "desc": "Aggregates neighbour features using equal weights (simple mean). "
                        "Fast and effective. Best starting point for node classification.",
                "paper": "Kipf & Welling (2017)"
            },
            "gat": {
                "name": "GAT — Graph Attention Network",
                "desc": "Learns which neighbours to pay attention to using attention scores. "
                        "More expressive than GCN — can focus on suspicious connections.",
                "paper": "Veličković et al. (2018)"
            },
            "sage": {
                "name": "GraphSAGE — Inductive Representation Learning",
                "desc": "Samples a fixed number of neighbours (scalable). Inductive — "
                        "can generalise to new nodes not seen during training.",
                "paper": "Hamilton et al. (2017)"
            },
        }
        info = model_info[model_choice]
        st.info(f"**{info['name']}**\n\n{info['desc']}\n\n*Paper: {info['paper']}*")

    st.divider()

    # ── Check for saved results ────────────────────────────────────────────────
    results_path = ROOT / config["paths"]["reports_dir"] / f"{model_choice}_results.json"

    if results_path.exists():
        with open(results_path) as f:
            saved = json.load(f)

        st.success(f"✅ Pre-trained results found for {model_choice.upper()}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("F1 (macro)", f"{saved['f1']:.4f}")
        m2.metric("AUC-ROC",    f"{saved['auc_roc']:.4f}")
        m3.metric("AUC-PR",     f"{saved['auc_pr']:.4f}" if "auc_pr" in saved else "—",
                  help="Area Under Precision-Recall Curve — the primary metric for imbalanced data. "
                       "Random baseline ≈ 0.098 (the fraud rate).")
        m4.metric("Epochs run", saved["epochs_run"])

        st.subheader("Training Curves")
        fig_curves = plots.training_curves(saved["train_losses"], saved["val_f1s"], model_choice)
        st.pyplot(fig_curves)

        # ── Confusion matrix ────────────────────────────────────────────────────
        cm_path = ROOT / "outputs" / "figures" / f"confusion_matrix_{model_choice}.png"
        if cm_path.exists():
            st.subheader("Confusion Matrix")
            st.caption(
                "Rows = actual class, Columns = predicted class. "
                "Off-diagonal entries are errors: top-right = false alarms (licit flagged as fraud), "
                "bottom-left = missed fraud (most costly)."
            )
            st.image(str(cm_path))

        st.subheader("Classification Report")
        st.text(saved["report"])

        st.markdown("---")
        if not DATA_AVAILABLE:
            st.warning("⚠️ Live training is disabled because the dataset could not be loaded.")
            st.button(f"🔁 Retrain {model_choice.upper()}", disabled=True)
        else:
            if st.button(f"🔁 Retrain {model_choice.upper()}"):
                results_path.unlink()
                st.rerun()

    else:
        st.warning(f"No saved results for {model_choice.upper()}. Click below to train.")

        if not DATA_AVAILABLE:
            st.error("⚠️ Cannot train: dataset not available.")
            st.button(f"🚀 Train {model_choice.upper()} Now", type="primary", disabled=True)
        else:
            if st.button(f"🚀 Train {model_choice.upper()} Now", type="primary"):
                from src.training.train import run_training

                progress_bar = st.progress(0)
                status       = st.empty()
                log_area     = st.empty()

                status.info("⏳ Training started — this may take a few minutes...")
                log_area.code("Initialising...")

                with st.spinner("Training in progress..."):
                    results = run_training(model_choice, config)

                progress_bar.progress(100)
                status.success("✅ Training complete!")

                st.metric("F1 (macro)", f"{results['f1']:.4f}")
                st.metric("AUC-ROC",    f"{results['auc_roc']:.4f}")
                st.metric("AUC-PR",     f"{results['auc_pr']:.4f}" if "auc_pr" in results else "—")

                fig_curves = plots.training_curves(
                    results["train_losses"], results["val_f1s"], model_choice
                )
                st.pyplot(fig_curves)
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARE MODELS
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Model Comparison")

    reports_dir = ROOT / config["paths"]["reports_dir"]
    gnn_results = {}

    for m in ["gcn", "gat", "sage"]:
        p = reports_dir / f"{m}_results.json"
        if p.exists():
            with open(p) as f:
                gnn_results[m] = json.load(f)

    if not gnn_results:
        st.warning("No GNN results found yet. Train at least one model in the 'Train GNN' tab first.")
    else:
        # ── Comparison table ────────────────────────────────────────────────────
        st.subheader("Results Table")

        baselines = {
            "XGBoost":  {"f1": 0.9643, "auc_roc": 0.9971, "auc_pr": None, "type": "Baseline"},
            "LightGBM": {"f1": 0.9794, "auc_roc": 0.9985, "auc_pr": None, "type": "Baseline"},
        }
        gnn_rows = {k: {**v, "type": "GNN"} for k, v in gnn_results.items()}
        all_results = {**baselines, **gnn_rows}

        import pandas as pd
        table_data = []
        for name, r in all_results.items():
            auc_pr_val = r.get("auc_pr")
            table_data.append({
                "Model":      name.upper(),
                "Type":       r["type"],
                "F1 (macro)": f"{r['f1']:.4f}",
                "AUC-ROC":    f"{r['auc_roc']:.4f}",
                "AUC-PR ↑":   f"{auc_pr_val:.4f}" if auc_pr_val is not None else "—",
            })
        st.dataframe(pd.DataFrame(table_data), width='stretch', hide_index=True)
        st.caption(
            "**AUC-PR** (Area Under Precision-Recall Curve) is the primary metric for this "
            "9.8%-fraud dataset. AUC-ROC is inflated by the easy licit majority; "
            "AUC-PR reveals true discrimination power on the rare fraud class. "
            "Random baseline ≈ 0.098."
        )

        # ── Comparison bar chart ────────────────────────────────────────────────
        st.subheader("Visual Comparison")
        plot_data = {k: {"f1": v["f1"], "auc_roc": v["auc_roc"]} for k, v in all_results.items()}
        fig_cmp = plots.model_comparison_bar(gnn_results)
        st.pyplot(fig_cmp)

        # ── Per-model details ───────────────────────────────────────────────────
        if gnn_results:
            st.subheader("GNN Training Details")
            cols = st.columns(len(gnn_results))
            for i, (name, result) in enumerate(gnn_results.items()):
                with cols[i]:
                    st.markdown(f"**{name.upper()}**")
                    st.metric("F1",       f"{result['f1']:.4f}")
                    st.metric("AUC-ROC",  f"{result['auc_roc']:.4f}")
                    auc_pr_val = result.get("auc_pr")
                    st.metric("AUC-PR",   f"{auc_pr_val:.4f}" if auc_pr_val else "—")
                    st.metric("Epochs",   result["epochs_run"])
                    beats_lgbm_f1  = result["f1"]      > 0.9794
                    beats_lgbm_auc = result["auc_roc"] > 0.9985
                    if beats_lgbm_f1 and beats_lgbm_auc:
                        st.success("Beats LightGBM on both! 🎉")
                    elif beats_lgbm_f1 or beats_lgbm_auc:
                        st.info("Beats LightGBM on one metric")
                    else:
                        st.error("Below LightGBM baseline")

            # ── Confusion matrices side by side ─────────────────────────────────
            st.subheader("Confusion Matrices")
            st.caption("Rows = actual, Columns = predicted. Bottom-left = missed fraud (false negatives = most costly).")
            cm_cols = st.columns(len(gnn_results))
            for i, (name, _) in enumerate(gnn_results.items()):
                cm_path = ROOT / "outputs" / "figures" / f"confusion_matrix_{name}.png"
                with cm_cols[i]:
                    st.markdown(f"**{name.upper()}**")
                    if cm_path.exists():
                        st.image(str(cm_path))
                    else:
                        st.info("Image not found")
