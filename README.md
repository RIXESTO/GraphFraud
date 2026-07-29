# GraphFraud — Bitcoin Transaction Fraud Detection using Graph Neural Networks

> **A complete end-to-end machine learning system that applies Graph Neural Networks (GCN, GAT, GraphSAGE) to detect illicit (fraudulent) transactions in the Bitcoin blockchain, benchmarked against classical tabular baselines (XGBoost, LightGBM).**

---

## Table of Contents

1. [Project Motivation](#1-project-motivation)
   - 1.1 [Problem Statement](#11-problem-statement)
2. [The Problem: Why Is Fraud Detection Hard?](#2-the-problem-why-is-fraud-detection-hard)
3. [Why Graphs?](#3-why-graphs)
4. [The Dataset: Elliptic Bitcoin Dataset](#4-the-dataset-elliptic-bitcoin-dataset)
5. [System Architecture](#5-system-architecture)
6. [Data Pipeline](#6-data-pipeline)
7. [Baseline Models: XGBoost and LightGBM](#7-baseline-models-xgboost-and-lightgbm)
8. [Graph Neural Networks: Theory](#8-graph-neural-networks-theory)
9. [GCN — Graph Convolutional Network](#9-gcn--graph-convolutional-network)
10. [GAT — Graph Attention Network](#10-gat--graph-attention-network)
11. [GraphSAGE — Scalable Inductive Learning](#11-graphsage--scalable-inductive-learning)
12. [Training Methodology](#12-training-methodology)
13. [Results and Analysis](#13-results-and-analysis)
14. [Why GNNs Don't Always Beat Tabular Models](#14-why-gnns-dont-always-beat-tabular-models)
15. [Hyperparameter Tuning](#15-hyperparameter-tuning)
16. [How to Run](#16-how-to-run)
17. [Project Structure](#17-project-structure)
18. [Key Design Decisions](#18-key-design-decisions)
19. [Limitations and Future Work](#19-limitations-and-future-work)
20. [References](#20-references)

---

## 1. Project Motivation

Bitcoin and other cryptocurrencies operate on public, immutable blockchains. Every transaction is permanently recorded — yet fraud, money laundering, and ransomware payments remain widespread because blockchain addresses are pseudonymous (not directly tied to real identities).

**The challenge:** identify which transactions are illicit *before* the funds are moved or mixed through "tumbler" services that obfuscate the trail.

Traditional fraud detection in banking relies on centralised databases, user identity, and transaction history tied to verified accounts. Cryptocurrency transactions have **none of this**. What they do have is a rich **graph structure**: every transaction sends funds from one wallet to another, creating a web of financial relationships.

This project tests the hypothesis:
> *"The graph structure of cryptocurrency transactions — who sends to whom — provides additional signal for detecting fraud beyond the raw transaction features alone."*

---

## 1.1 Problem Statement

### Formal Definition

Given a directed graph **G = (V, E, X, Y)** where:

| Symbol | Meaning |
|---|---|
| **V** | Set of 203,769 Bitcoin transaction nodes |
| **E** | Set of 234,355 directed edges (transaction flows between wallets) |
| **X ∈ ℝ^{\|V\| × 165}** | Node feature matrix — 165 numerical attributes per transaction |
| **Y ∈ {0, 1, −1}^{\|V\|}** | Node labels — 0 (licit), 1 (illicit), −1 (unknown) |

**Task:** Learn a function **f: V → {0, 1}** that predicts whether each *labelled* transaction node is illicit (fraudulent) or licit (legitimate), using both the node features **X** and the graph structure **E**.

This is a **semi-supervised node classification** problem:
- Only 46,564 of 203,769 nodes are labelled (22.8%)
- The remaining 157,205 nodes are unlabelled but present in the graph
- Unlabelled nodes must be leveraged for structural context during training

---

### Research Question

> **Does modelling Bitcoin transaction data as a graph and applying Graph Neural Networks yield measurably better fraud detection performance compared to classical tabular machine learning methods that use only per-transaction features?**

Specifically:
1. Can GNNs capture fraud patterns in the *transaction network topology* that are invisible to tabular models?
2. Which GNN architecture (GCN, GAT, GraphSAGE) performs best on this semi-supervised, class-imbalanced graph?
3. What is the performance gap between graph-based and feature-only approaches, and what explains it?

---

### Objectives

| # | Objective | Type |
|---|---|---|
| O1 | Build a complete, reproducible data pipeline from raw CSV files to a PyG graph object | Engineering |
| O2 | Establish strong tabular baselines using XGBoost and LightGBM | Empirical |
| O3 | Implement and train three GNN architectures: GCN, GAT, GraphSAGE | Engineering + Research |
| O4 | Evaluate all models using F1 (macro) and AUC-ROC on a held-out test set | Empirical |
| O5 | Analyse the impact of hyperparameter choices (layers, hidden size, learning rate) on GNN performance | Research |
| O6 | Build an interactive dashboard to visualise results and training dynamics | Engineering |

---

### Constraints and Scope

- **Hardware:** Apple M4 MacBook Air — no NVIDIA GPU. PyTorch MPS backend used.
- **Dataset:** Fixed to the Elliptic Bitcoin Dataset (publicly available, industry-standard benchmark).
- **Time scope:** Static graph — all 49 timesteps merged; temporal dynamics not modelled.
- **Labelling:** Only the 46,564 labelled nodes are used for training, validation, and test evaluation. Unlabelled nodes participate in message passing only.
- **Task type:** Binary node classification (illicit vs. licit). Multi-class extension (fraud subcategories) is out of scope.

---

### Success Criteria

A model is considered **successful** if it meets the following thresholds on the held-out test set:

| Metric | Target | Rationale |
|---|---|---|
| AUC-ROC | ≥ 0.95 | Strong discriminative ability |
| F1 (macro) | ≥ 0.80 | Balanced precision and recall across classes |
| Fraud Recall | ≥ 0.80 | Catch at least 80% of actual fraud |

The **primary success criterion** for GNNs is: *does the graph-based model learn something that the tabular baseline cannot?* Even if GNNs don't exceed baselines in absolute F1, they are considered valuable if they achieve higher fraud recall or capture a different subset of fraud cases.

---



### 2.1 Class Imbalance

In the Elliptic dataset (and in real life), fraud is rare:

| Class | Count | Percentage of labelled |
|---|---|---|
| Licit (legitimate) | 42,019 | 90.2% |
| Illicit (fraud) | 4,545 | 9.8% |
| Unknown (unlabelled) | 157,205 | — |

A model that predicts "licit" for every single transaction would achieve **90.2% accuracy** — and catch **zero fraud**. This is the class imbalance problem. Accuracy is a misleading metric; we use **F1 (macro)** and **AUC-ROC** instead.

**How we handle it:**
- `scale_pos_weight=9` in XGBoost — tells the model fraud is 9× more costly to miss
- `class_weight="balanced"` in LightGBM — auto-computes weights from data
- `CrossEntropyLoss(weight=[w_licit, w_illicit])` in GNNs — weighted loss function

### 2.2 Semi-Supervised Setting

77% of nodes have **no ground truth label**. We know they exist and we know who they transact with, but we don't know if they are fraudulent. This is called a **semi-supervised** setting.

Tabular models simply discard these nodes — they can only train on labelled examples.

Graph Neural Networks can still **use the unlabelled nodes** during message passing — even without labels, their feature vectors and their position in the graph provide structural information that influences how neighbouring labelled nodes are represented.

### 2.3 Adversarial Evolution

Fraudsters adapt. They change wallet addresses, create shell transactions, and use mixing services. A model trained on past fraud patterns may fail to detect new, evolved fraud. This motivates the use of **inductive learning** (GraphSAGE), which learns a generalisable aggregation function rather than memorising specific node embeddings.

---

## 3. Why Graphs?

### 3.1 What is a Graph?

A graph G = (V, E) consists of:
- **V** — a set of nodes (vertices). Here: 203,769 Bitcoin transactions
- **E** — a set of edges. Here: 234,355 directed transaction flows between wallets

```
Transaction A ──funds──► Transaction B ──funds──► Transaction C
                                │
                                └──funds──► Transaction D (flagged: ILLICIT)
```

If D is illicit, B (which sent funds to D) is suspicious. C (which received from B) may also be implicated. A graph lets us reason about this chain.

### 3.2 What Graphs Capture That Tables Cannot

| Approach | What it sees |
|---|---|
| Tabular ML | 165 features of transaction X in isolation |
| GNN (1 layer) | 165 features of X + features of all direct neighbours |
| GNN (2 layers) | Everything above + features of neighbours' neighbours |

A transaction might look completely normal in isolation — standard amounts, normal timing — but if its money comes from 10 known fraud wallets and goes to 3 known fraud wallets, the graph context reveals the truth.

### 3.3 The Graph in This Project

- **Nodes** = individual Bitcoin transactions (not wallets)
- **Edges** = directed: edge (A→B) means A sent funds that B received
- **Node features** = 165-dimensional feature vector per transaction
- **Task** = node classification: label each node as illicit (1) or licit (0)

---

## 4. The Dataset: Elliptic Bitcoin Dataset

**Source:** Collected by Elliptic, a blockchain analytics company. Published in 2019.

**Citation:** Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics," KDD 2019.

### 4.1 Structure

| Property | Value |
|---|---|
| Nodes (transactions) | 203,769 |
| Edges (transaction flows) | 234,355 |
| Timesteps | 49 (biweekly snapshots, ~2011–2019) |
| Feature dimensions | 165 per node |
| Labelled nodes | 46,564 |
| Illicit nodes | 4,545 |
| Licit nodes | 42,019 |

### 4.2 The 165 Features

The features are split into two groups:

**Local features (feat_1 to feat_94) — 94 features:**
Properties of the transaction itself:
- Transaction amount (total BTC input/output)
- Number of inputs and outputs
- Fee paid to miners
- Time since first appearance in the mempool
- Transaction size in bytes
- Statistical properties of input/output values (mean, median, variance)

**Aggregated features (feat_95 to feat_165) — 71 features:**
Statistical summaries of the **immediate neighbourhood**:
- Mean, median, std of all incoming transaction amounts
- Mean, median, std of all outgoing transaction amounts
- Number of transactions in neighbourhood per timestep
- Aggregated input/output counts from neighbours

> **Critical insight:** The 71 aggregated features already encode 1-hop neighbourhood information. This is why GNNs on this dataset gain less from graph structure than on datasets with purely local node features — the "graph information" is partially pre-baked into the feature set.

### 4.3 Timesteps

The 49 timesteps represent biweekly windows. The fraud rate varies significantly over time:

- Early timesteps: lower fraud rate (Bitcoin was less adopted)
- Peaks around timesteps 25–35: known periods of ransomware and darknet market activity
- Later timesteps: varying patterns as authorities began prosecuting actors

The temporal nature of the data is important — in a production system, you would always train on past timesteps and evaluate on future ones to avoid data leakage.

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GraphFraud System                         │
│                                                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │  Raw Data    │──►│  Data        │──►│  Graph Builder      │  │
│  │  (3 CSVs)    │   │  Pipeline    │   │  (NetworkX + PyG)   │  │
│  └──────────────┘   └──────────────┘   └─────────────────────┘  │
│                             │                       │             │
│                             ▼                       ▼             │
│                   ┌──────────────────┐   ┌──────────────────┐   │
│                   │ Tabular Baseline │   │  GNN Training    │   │
│                   │ XGBoost/LightGBM │   │  GCN/GAT/SAGE    │   │
│                   └──────────────────┘   └──────────────────┘   │
│                             │                       │             │
│                             └───────────┬───────────┘            │
│                                         ▼                         │
│                              ┌──────────────────┐                │
│                              │   Evaluation     │                │
│                              │   F1, AUC-ROC    │                │
│                              └──────────────────┘                │
│                                         │                         │
│                                         ▼                         │
│                              ┌──────────────────┐                │
│                              │  Streamlit App   │                │
│                              │  (Dashboard)     │                │
│                              └──────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Pipeline

### 6.1 `src/data/ingest.py` — Loading Raw Data

**What:** Reads the three Elliptic CSV files and merges them into a single DataFrame.

**How:**

```
elliptic_txs_features.csv  →  203,769 rows × 167 cols (txId, timestep, feat_1..feat_165)
elliptic_txs_classes.csv   →  203,769 rows × 2 cols  (txId, class: "1"/"2"/"unknown")
elliptic_txs_edgelist.csv  →  234,355 rows × 2 cols  (txId1, txId2)
```

The class column uses string codes:
- `"1"` = illicit → mapped to integer `1`
- `"2"` = licit → mapped to integer `0`
- `"unknown"` → mapped to integer `-1`

**Why a left join (not inner join):**
An inner join would discard all 157,205 unknown nodes — we would lose the graph structure around those nodes. A left join keeps all transaction rows and assigns `-1` to unlabelled nodes, preserving the full graph for GNN message passing.

### 6.2 `src/data/preprocess.py` — Feature Engineering and Splitting

**Three functions:**

**`get_feature_cols(df)`** — returns the 165 column names starting with `feat_`.
Why: provides a clean, reusable way to identify feature columns without hardcoding indices.

**`split_known_unknown(df)`** — splits into:
- `known_df` (46,564 rows): labelled nodes → used for supervised training
- `unknown_df` (157,205 rows): unlabelled nodes → used in graph but not for training loss

**`train_val_test_split(known_df, feature_cols, config)`** — stratified split:
- Test: 20% of known (9,313 nodes)
- Validation: 10% of known (4,657 nodes)
- Train: 70% of known (32,594 nodes)

`stratify=y` ensures the ~9.8% fraud ratio is preserved in every split. Without this, random chance could put all fraud cases in training and none in the test set.

### 6.3 `src/data/graph_builder.py` — Building Graph Objects

**`build_networkx_graph(edgelist, df)`** — creates a `nx.DiGraph`:
- Nodes: all 203,769 transaction IDs
- Edges: all 234,355 directed flows
- Node attributes: `timestep`, `label` attached to each node
- Used for: graph analysis, visualisation, degree statistics

**`build_pyg_data(edgelist, df, feature_cols)`** — creates a PyTorch Geometric `Data` object:

```python
Data(
    x          = [203769, 165],  # float32 feature matrix
    edge_index = [2, 234355],    # int64 COO-format edge list
    y          = [203769],       # int64 labels (0, 1, or -1)
)
```

**The node index mapping problem:**
PyG requires node indices to be consecutive integers (0, 1, 2, ...). But Bitcoin txIds are large 64-bit numbers (e.g., `156317549`). We build a mapping:
```python
node_map = {txId: i for i, txId in enumerate(df["txId"].values)}
```
Then edges are remapped from txId space to index space. Edges where either endpoint doesn't exist in `df` (e.g., the edgelist references nodes not in our feature set) are dropped via `.notna()` filtering.

**COO Format:**
```
edge_index = [[src_0, src_1, src_2, ...],
              [dst_0, dst_1, dst_2, ...]]
```
Shape `[2, E]`. This is a standard sparse graph representation — row 0 contains all source nodes, row 1 contains all destination nodes.

---

## 7. Baseline Models: XGBoost and LightGBM

### 7.1 Why Baselines Matter

Before building a GNN, you must establish a **baseline**: the best you can do *without* using the graph structure. If your GNN can't beat the baseline, you haven't demonstrated that graph structure adds value.

Baselines answer: *"How much of the fraud signal is already in the raw features?"*

### 7.2 Gradient Boosting

Both XGBoost and LightGBM are **gradient boosting** algorithms. They build an ensemble of decision trees where each new tree focuses on correcting the errors of the previous trees.

```
Tree 1: initial prediction       → errors E1
Tree 2: trained on E1            → errors E2 (smaller)
Tree 3: trained on E2            → errors E3 (smaller)
...
Final prediction = weighted sum of all trees
```

This is called **boosting** (as opposed to bagging, used by Random Forest). Boosting is sequential — each tree depends on all previous trees.

### 7.3 XGBoost vs LightGBM

| Property | XGBoost | LightGBM |
|---|---|---|
| Tree growth | Level-wise (depth-first) | Leaf-wise (best-first) |
| Complexity control | `max_depth` | `num_leaves` |
| Imbalance handling | `scale_pos_weight` | `class_weight="balanced"` |
| Speed | Slower | 10–100× faster |
| Result here | F1: 0.9643, AUC: 0.9971 | F1: 0.9794, AUC: 0.9985 |

**`scale_pos_weight=9`** (XGBoost): Tells the model that a missed fraud (false negative) is 9× more costly than a missed licit transaction. This corresponds to the 9:1 class ratio in the labelled data.

**`class_weight="balanced"`** (LightGBM): Automatically computes `n_samples / (n_classes * count_per_class)` as the weight for each class. Equivalent to `scale_pos_weight` but computed automatically.

### 7.4 Hyperparameters Used

```yaml
xgboost:
  n_estimators: 200      # number of trees
  max_depth: 6           # max depth of each tree
  learning_rate: 0.05    # shrinkage factor per tree
  scale_pos_weight: 9    # class imbalance correction

lightgbm:
  n_estimators: 300      # more trees (faster to train each one)
  num_leaves: 63         # 2^6 - 1 = 63 ≈ max_depth 6 equivalent
  learning_rate: 0.05
  class_weight: "balanced"
```

---

## 8. Graph Neural Networks: Theory

### 8.1 The Core Idea: Message Passing

Every GNN layer performs **message passing**:

1. **Aggregate**: Each node collects feature vectors from all its neighbours
2. **Transform**: Apply a learned linear transformation to the aggregated result
3. **Activate**: Apply a non-linear activation (ReLU or ELU)

After `k` layers, each node has "seen" information from all nodes within `k` hops.

```
Layer 0 (input): node sees only its own 165 features
Layer 1:         node sees its own features + neighbours' features
Layer 2:         node sees above + neighbours' neighbours' features
```

### 8.2 Why Non-Linearity (ReLU/ELU)?

Without activation functions, stacking N linear layers is mathematically equivalent to a single linear layer:
```
W3 × W2 × W1 × x  =  W_combined × x
```
Non-linearities (ReLU, ELU) break this equivalence, allowing the network to learn non-linear, complex decision boundaries — essential for separating fraud from licit behaviour.

### 8.3 Over-Smoothing

A known failure mode of deep GNNs (3+ layers): as more aggregation steps are applied, all node representations converge to the same value — they become indistinguishable. This is called **over-smoothing**.

In the Elliptic dataset, with 77% unlabelled nodes, over-smoothing is severe: fraud nodes' representations get "washed out" by many unknown neighbours over multiple hops.

**Our fix:** Use 2 layers instead of 3. This dramatically improved performance:
```
3 layers: GCN F1 = 0.6594
2 layers: GCN F1 = 0.8533   (+29%)
```

### 8.4 Batch Normalisation

After each graph convolution, we apply `BatchNorm1d`:

```python
x = BatchNorm1d(hidden_channels)(x)
```

**What it does:** Normalises the activations to have zero mean and unit variance across the batch.

**Why it matters:**
- Prevents gradient explosion/vanishing during training
- Allows higher learning rates
- Acts as mild regularisation
- Makes the model less sensitive to weight initialisation

### 8.5 Dropout

```python
x = F.dropout(x, p=0.3, training=self.training)
```

During training, randomly zero out 30% of neuron activations. This forces the network to not rely on any single feature — a form of regularisation that prevents overfitting.

`self.training` is `True` during `model.train()` and `False` during `model.eval()`. Dropout is **only active during training** — during evaluation, all neurons are used.

---

## 9. GCN — Graph Convolutional Network

**Paper:** Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks," ICLR 2017.

### 9.1 What GCN Does

GCN uses a **symmetric normalised adjacency matrix** to aggregate neighbour features. For each node v in each layer:

```
h_v^(l+1) = σ( Σ_{u ∈ N(v) ∪ {v}} (1/√d_v × 1/√d_u) × W^(l) × h_u^(l) )
```

Where:
- `h_v^(l)` = feature vector of node v at layer l
- `N(v)` = set of neighbours of v
- `d_v` = degree (number of connections) of node v
- `W^(l)` = learned weight matrix at layer l
- `σ` = activation function (ReLU)

In plain English: **take a weighted average of all neighbour features, where the weight of each neighbour is inversely proportional to both nodes' degrees** (so high-degree hub nodes contribute less).

### 9.2 Architecture Used

```
Input:  [203769, 165]
GCNConv(165 → 128) + BatchNorm + ReLU + Dropout(0.3)
GCNConv(128 → 2)                          ← raw logits, no activation
Output: [203769, 2]   ← one score per class per node
```

### 9.3 Results

```
F1 (macro): 0.8533   AUC-ROC: 0.9730
Fraud precision: 63.8%   Fraud recall: 88.3%
```

GCN catches 88.3% of all fraud. The relatively lower precision (63.8%) means some licit transactions are flagged as fraud — acceptable in fraud detection where missing fraud is costlier than false alarms.

---

## 10. GAT — Graph Attention Network

**Paper:** Veličković et al., "Graph Attention Networks," ICLR 2018.

### 10.1 The Problem with GCN

GCN assigns weights based only on node degree — a structural property. It treats all neighbours of similar degree equally, regardless of their features. This can be suboptimal: for a fraud node, a neighbour that is also a fraud node is far more informative than a licit neighbour.

### 10.2 What GAT Does

GAT learns **attention coefficients** — scores indicating how much node v should attend to each specific neighbour u:

```
α_vu = softmax( LeakyReLU( a^T [W h_v || W h_u] ) )
```

Where:
- `a` = learnable attention vector
- `W` = learnable weight matrix
- `||` = concatenation
- `softmax` = normalise so all neighbour weights sum to 1

Then: `h_v^(l+1) = σ( Σ_{u ∈ N(v)} α_vu × W × h_u )`

The network **learns** which neighbours are more relevant based on feature compatibility — not just graph structure.

### 10.3 Multi-Head Attention

GAT uses **H independent attention mechanisms** (heads) in parallel, each learning different "what to attend to" patterns. Their outputs are concatenated:

```
Output_v = ||_{k=1}^{H} σ( Σ α_vu^k × W^k × h_u )
```

With H=4 heads and hidden_channels=128:
- Each head outputs 128 dimensions
- Concatenated → 512 dimensions per intermediate layer
- Final layer uses 1 head with `concat=False` → 2 output dimensions (logits)

### 10.4 Architecture Used

```
Input:    [203769, 165]
GATConv(165 → 128, heads=4, concat=True)  + BatchNorm(512) + ELU + Dropout(0.3)
GATConv(512 → 2,   heads=1, concat=False)                  ← raw logits
Output:   [203769, 2]
```

**Why ELU instead of ReLU?** ELU (Exponential Linear Unit) has a smooth curve for negative values instead of a hard zero. This provides better gradient flow through the attention mechanism and was shown to work better in the original GAT paper.

### 10.5 Results

```
F1 (macro): 0.8306   AUC-ROC: 0.9797
Fraud precision: 56.8%   Fraud recall: 92.9%   ← highest recall of all models
```

GAT catches **92.9% of all fraud** — the best recall of any model in this project, including the tabular baselines. The trade-off is lower precision (more false alarms). In a real fraud system, you might prefer GAT at the first screening stage, then use a more precise model for final flagging.

---

## 11. GraphSAGE — Scalable Inductive Learning

**Paper:** Hamilton et al., "Inductive Representation Learning on Large Graphs," NeurIPS 2017.

### 11.1 The Transductive vs. Inductive Problem

GCN and GAT are **transductive**: they learn fixed embeddings for each specific node in the training graph. If a new node appears (a new Bitcoin wallet), you must retrain from scratch.

GraphSAGE is **inductive**: it learns an **aggregation function** — a general recipe for combining a node's features with its neighbours' features. This function can be applied to any node, even ones not seen during training.

**Why this matters for Bitcoin:**
New wallets appear constantly. An inductive model can immediately compute a fraud score for a brand-new wallet without retraining — essential for real-time fraud detection.

### 11.2 What GraphSAGE Does

For each training step, GraphSAGE:
1. **Samples** a fixed-size subset of neighbours (not all of them)
2. **Aggregates**: computes mean of sampled neighbour features
3. **Concatenates**: combines [node's own features | aggregated neighbour features]
4. **Transforms**: applies a learned linear projection

```
h_v^(l+1) = σ( W^(l) × CONCAT( h_v^(l), MEAN({h_u^(l) : u ∈ Sample(N(v))}) ) )
```

**Why sampling?** Some Bitcoin wallets have thousands of connections. Processing all neighbours is expensive and memory-intensive. Sampling a fixed number (e.g., 10 per hop) makes training scalable to massive graphs.

### 11.3 Architecture Used

```
Input:    [203769, 165]
SAGEConv(165 → 128)  + BatchNorm(128) + ReLU + Dropout(0.3)
SAGEConv(128 → 2)                              ← raw logits
Output:   [203769, 2]
```

### 11.4 Results — Best GNN

```
F1 (macro): 0.9329   AUC-ROC: 0.9860
Fraud precision: 86.7%   Fraud recall: 89.1%
Accuracy: 97.6%
```

GraphSAGE is the **strongest GNN** in this project. It achieves:
- AUC 0.9860 — within 0.013 of LightGBM (0.9985)
- F1 0.9329 — within 0.046 of LightGBM (0.9794)
- Best balance of precision and recall

The val F1 was still climbing at epoch 300 — further training would likely improve results.

---

## 12. Training Methodology

### 12.1 Node Masking

The full graph (all 203,769 nodes) is loaded into memory. We define boolean **masks** that select which nodes to use for training, validation, or testing:

```
train_mask: [False, True, False, True, ...]  ← 32,594 True values
val_mask:   [False, False, True, False, ...]  ← 4,657 True values
test_mask:  [True, False, False, False, ...]  ← 9,313 True values
```

Forward pass: predictions computed for ALL 203,769 nodes simultaneously.
Loss: computed only for `train_mask` nodes.
Metrics: computed only for `val_mask` or `test_mask` nodes.

Unknown nodes (`label=-1`) are never in any mask — they participate in message passing (so GNNs benefit from their graph position) but never contribute to the loss.

### 12.2 Loss Function

**CrossEntropyLoss with class weights:**

```python
weight = [w_licit, w_illicit]
w_licit   = n_total / (2 × n_licit)    ≈ 0.55
w_illicit = n_total / (2 × n_illicit)  ≈ 5.12

loss = CrossEntropyLoss(weight=weight)
```

Internally, CrossEntropyLoss does:
1. Softmax: converts raw logits to probabilities
2. Negative log-likelihood: `-log(p_correct_class)`
3. Weighted mean across all training nodes

### 12.3 Optimiser: Adam

```python
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
```

**Adam** (Adaptive Moment Estimation) improves on basic gradient descent by:
- Maintaining a **momentum** term (exponential moving average of past gradients)
- Maintaining a **velocity** term (exponential moving average of squared gradients)
- Adapting the learning rate **per parameter** — parameters with small historical gradients get larger updates

`weight_decay=1e-5` adds L2 regularisation: penalises large weight values, preventing overfitting.

### 12.4 Early Stopping

```python
patience = 40
if val_f1 > best_val_f1:
    best_val_f1 = val_f1
    save_checkpoint()
else:
    patience_counter += 1
    if patience_counter >= patience:
        stop training
```

**Why early stopping?**
Neural networks can **overfit** — they memorise the training data and perform poorly on new data. We monitor validation F1 (not training loss) and stop when it stops improving.

We save the **best checkpoint** (not the final weights). The model at the best validation F1 is the one we evaluate on the test set — this prevents using a model that has overfit in later epochs.

### 12.5 Device Detection

```python
if torch.backends.mps.is_available():
    device = "mps"    # Apple Silicon GPU (M1/M2/M3/M4)
elif torch.cuda.is_available():
    device = "cuda"   # NVIDIA GPU
else:
    device = "cpu"    # fallback
```

This project runs on an **Apple M4 MacBook Air** using MPS (Metal Performance Shaders) — Apple's GPU compute framework. All tensors and model weights are moved to MPS via `.to(device)`.

---

## 13. Results and Analysis

### 13.1 Full Results Table

| Model | Type | F1 (macro) | AUC-ROC | Fraud Recall | Fraud Precision |
|---|---|---|---|---|---|
| LightGBM | Baseline | **0.9794** | **0.9985** | — | — |
| XGBoost | Baseline | 0.9643 | 0.9971 | — | — |
| GraphSAGE | GNN | **0.9329** | **0.9860** | 89.1% | 86.7% |
| GCN | GNN | 0.8533 | 0.9730 | 88.3% | 63.8% |
| GAT | GNN | 0.8306 | 0.9797 | **92.9%** | 56.8% |

### 13.2 Metric Explanations

**F1 (macro):**
- Computes F1 separately for each class, then averages
- F1 = 2 × (Precision × Recall) / (Precision + Recall)
- "macro" means both classes are weighted equally regardless of size
- Best metric for imbalanced classification

**AUC-ROC (Area Under the Receiver Operating Characteristic Curve):**
- Measures how well the model **ranks** fraud above licit across all possible decision thresholds
- 1.0 = perfect separation, 0.5 = random guessing
- Independent of the decision threshold — useful for comparing models
- Higher AUC = model can catch more fraud with fewer false alarms at any threshold

**Confusion Matrix (GraphSAGE on test set):**
```
                 Predicted
              Licit    Illicit
Actual Licit  [8,273  |   131]   ← 131 licit flagged as fraud (false positives)
Actual Illicit[  100  |   809]   ← 100 fraud missed (false negatives)
```
- 100 missed fraud cases out of 909 (11% miss rate)
- 131 false alarms out of 8,404 licit (1.6% false alarm rate)

### 13.3 Impact of Hyperparameter Tuning

| Parameter | Original | Tuned | Effect |
|---|---|---|---|
| `num_layers` | 3 | **2** | Primary driver of improvement — reduced over-smoothing |
| `hidden_channels` | 64 | **128** | More model capacity |
| `lr` | 0.001 | **0.01** | Faster convergence |
| `patience` | 20 | **40** | Models trained longer before stopping |
| `epochs` | 200 | **300** | More budget for training |

**Result:** GraphSAGE improved from F1 0.7367 → 0.9329 (+26.6%).

---

## 14. Why GNNs Don't Always Beat Tabular Models

This is a critical question for any GNN paper on the Elliptic dataset.

### 14.1 The Pre-Aggregated Features Problem

71 of the 165 features are **already neighbourhood aggregations** — mean, median, and std of neighbour transaction amounts, counts, etc. These were hand-engineered by Elliptic's data scientists.

When a GNN aggregates neighbour features, it is essentially doing the same operation again — on data that already contains neighbourhood information. This "double aggregation" means the graph structure provides diminishing marginal returns.

**Analogy:** If you've already averaged your neighbours' test scores and added that as a feature, having the model average over the original scores again adds little new information.

### 14.2 Label Sparsity

77% of nodes have no label. During GNN message passing:

```
Fraud node → aggregates from: [licit, unknown, unknown, unknown, illicit]
                                 known  77% of nodes are unknown
```

The unknown nodes have real features (transaction amounts, etc.) but their label is hidden. Their features may carry useful structural signal, but statistically, the fraud signal gets diluted across many uncertain neighbours.

### 14.3 Dataset Scale vs. Model Complexity

GNNs shine on datasets where:
1. Node features are sparse or weak (so neighbourhood context matters more)
2. The fraction of unlabelled nodes is low (so signal propagates cleanly)
3. The graph is very large and heterogeneous

The Elliptic dataset has rich features, many unknowns, and a relatively small graph for deep learning. LightGBM (a highly optimised tabular model) exploits the rich features more efficiently.

### 14.4 When GNNs Win

On datasets like DGraph-Fin (3.7M nodes, purely local features) or citation networks where node features are bag-of-words representations, GNNs consistently outperform tabular baselines by large margins.

In a real Bitcoin fraud system, you would likely use **both**: LightGBM for initial screening (fast, high accuracy) + GNN for contextual analysis of flagged transactions (slower, catches patterns LightGBM misses).

---

## 15. Hyperparameter Tuning

### 15.1 What Was Tuned and Why

**`num_layers: 3 → 2`** — Most impactful change.
Fewer layers = smaller receptive field = less over-smoothing. With 77% unknown nodes, every additional hop introduces more noisy (unknown) signal into a node's representation.

**`hidden_channels: 64 → 128`** — Model capacity.
More neurons per layer = ability to learn more complex feature interactions. With 165 input features, 64 hidden units is a bottleneck (165 → 64 is aggressive compression). 128 is a more balanced reduction.

**`lr: 0.001 → 0.01`** — Learning rate.
The original models stopped at 28–35 epochs with lr=0.001, indicating they were learning very slowly. A 10× higher learning rate causes faster convergence, allowing the model to find better solutions within the epoch budget.

**`patience: 20 → 40`** — Early stopping patience.
With lr=0.001, the model was making tiny improvements each epoch. A patience of 20 was too short — the model would stop just as it was beginning to improve meaningfully. With lr=0.01 and patience=40, models trained for 255–300 epochs and found much better solutions.

### 15.2 What Was NOT Tuned (Future Work)

- **Architecture search**: different GNN types (GraphTransformer, GIN)
- **Layer normalisation**: instead of or in addition to BatchNorm
- **Learning rate scheduling**: cosine annealing, warmup
- **Graph augmentation**: dropping edges, adding virtual nodes
- **Neighbour sampling** in GraphSAGE: the number of neighbours sampled per hop

---

## 16. How to Run

### Prerequisites

```bash
# Python 3.10+ required
pip install -r requirements.txt
```

### Step-by-Step

```bash
cd GraphFraud

# 1. Test the data pipeline
python -m src.data.ingest         # loads CSVs, prints label distribution
python -m src.data.preprocess     # splits data, prints split sizes
python -m src.data.graph_builder  # builds graph, prints node/edge count

# 2. Run baseline models
python -m src.models.baseline     # trains XGBoost + LightGBM, prints results

# 3. Train all GNNs (takes 10–15 min on M4)
python -m src.training.train

# 4. Run all tests
python -m pytest tests/ -v

# 5. Launch the dashboard
streamlit run app/streamlit_app.py
# Open http://localhost:8501 in your browser
```

---

## 17. Project Structure

```
GraphFraud/
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── configs/
│   └── config.yaml                    ← All hyperparameters and paths
├── data/
│   └── raw/                           ← Elliptic Bitcoin Dataset CSVs
│       ├── elliptic_txs_features.csv
│       ├── elliptic_txs_classes.csv
│       └── elliptic_txs_edgelist.csv
├── src/
│   ├── data/
│   │   ├── ingest.py                  ← CSV loading, label mapping
│   │   ├── preprocess.py              ← Feature cols, splits, scaling
│   │   └── graph_builder.py           ← NetworkX + PyG graph objects
│   ├── models/
│   │   ├── baseline.py                ← XGBoost + LightGBM training
│   │   ├── gcn.py                     ← Graph Convolutional Network
│   │   ├── gat.py                     ← Graph Attention Network
│   │   └── sage.py                    ← GraphSAGE
│   ├── training/
│   │   ├── train.py                   ← Full GNN training pipeline
│   │   └── evaluate.py                ← Metrics, masks, class weights
│   └── visualization/
│       └── plots.py                   ← Matplotlib visualisation functions
├── app/
│   └── streamlit_app.py               ← Interactive dashboard
├── tests/
│   ├── test_data.py                   ← 13 unit tests for data pipeline
│   └── test_models.py                 ← 18 unit tests for GNN models
├── outputs/
│   ├── models/                        ← Saved GNN checkpoints (.pt files)
│   ├── reports/                       ← JSON results files
│   └── figures/                       ← Saved plot images
└── notebooks/                         ← Exploratory notebooks
```

---

## 18. Key Design Decisions

### Why functional programming (functions instead of scripts)?

Each module defines functions that accept data as arguments — no global state, no module-level side effects. This enables:
- **Unit testing**: call individual functions with controlled inputs
- **Reusability**: import `build_pyg_data` without loading CSVs
- **Composability**: combine functions in different orders for different experiments

### Why separate `evaluate.py` from `train.py`?

Evaluation logic (metrics computation, mask creation, class weights) is reused by:
- The training loop (validation after each epoch)
- The test evaluation (final results)
- The Streamlit dashboard (displaying results)

Separating it prevents code duplication and makes each function independently testable.

### Why PyTorch Geometric instead of DGL or raw PyTorch?

PyG provides:
- `GCNConv`, `GATConv`, `SAGEConv` — well-tested, optimised implementations
- The `Data` class — standard interface for graph + features + labels
- Efficient sparse message passing on GPU (including MPS)

### Why save the best checkpoint (not the last)?

Neural networks often overfit in later training epochs. The best validation F1 model — not necessarily the final epoch — generalises best to the test set. `torch.save(model.state_dict(), path)` saves weights; `model.load_state_dict(torch.load(path))` restores them.

### Why `@st.cache_data` in Streamlit?

Loading 700MB of CSV files takes ~30 seconds. Without caching, Streamlit reruns the entire script on every user interaction (button click, tab switch). `@st.cache_data` stores the DataFrame in memory after the first load, making all subsequent interactions instant.

---

## 19. Limitations and Future Work

### Current Limitations

1. **Transductive GCN/GAT**: Cannot generalise to new wallets without retraining.
2. **Temporal structure ignored**: All 49 timesteps are merged into one static graph. A temporal GNN (e.g., TGN — Temporal Graph Networks) would model the evolution of fraud patterns over time.
3. **No edge features**: Each edge just says "A paid B". In reality, the amount, timing, and fee of the transaction are informative edge features not yet used.
4. **GraphSAGE still below LightGBM**: The aggregated features in the dataset partially explain this gap.

### Future Improvements

| Improvement | Expected Impact | Difficulty |
|---|---|---|
| Temporal GNN (T-GNN) | Capture fraud pattern evolution | High |
| Edge features in message passing | Richer graph signal | Medium |
| DGraph-Fin dataset (18× larger) | More training data | Low |
| Label propagation for unknown nodes | Use semi-supervised structure | Medium |
| GraphTransformer / GIN | More expressive architectures | Medium |
| Ensemble: GNN + LightGBM | Best of both worlds | Low |

---

## 20. References

1. **Elliptic Dataset Paper:** M. Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics," KDD 2019 Workshop. [arXiv:1908.02591](https://arxiv.org/abs/1908.02591)

2. **GCN:** T.N. Kipf & M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," ICLR 2017. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)

3. **GAT:** P. Veličković et al., "Graph Attention Networks," ICLR 2018. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)

4. **GraphSAGE:** W.L. Hamilton et al., "Inductive Representation Learning on Large Graphs," NeurIPS 2017. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)

5. **XGBoost:** T. Chen & C. Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD 2016. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)

6. **LightGBM:** G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," NeurIPS 2017.

7. **PyTorch Geometric:** M. Fey & J.E. Lenssen, "Fast Graph Representation Learning with PyTorch Geometric," ICLR 2019. [arXiv:1903.02428](https://arxiv.org/abs/1903.02428)

8. **Over-smoothing in GNNs:** Q. Li et al., "Deeper Insights into Graph Convolutional Networks for Semi-Supervised Classification," AAAI 2018.

---

*Built with PyTorch, PyTorch Geometric, XGBoost, LightGBM, NetworkX, and Streamlit.*
*Trained on Apple M4 (MPS backend). Python 3.12.*
