"""
src/detector.py — Abuse-Ring Sentinel

Consumes the graph + candidate rings from graph_build.py, trains a GraphSAGE
node classifier over standardized account features, aggregates per-account fraud
scores into per-ring risk scores, and ranks the candidate rings.

Three requirements, wired in from the start (not retrofitted):
  1. IP edges are DOWN-WEIGHTED. IP dominates the graph (~87% of edges) but is
     the weakest fraud signal — shared wifi/offices/hostels put unrelated people
     on one IP. Per-type edge weights feed a weighted-mean neighbour aggregation,
     so the message passing leans on device/funding, not IP. (EDGE_TYPE_WEIGHTS)
  2. CLASS IMBALANCE (~12% fraud) is handled with balanced-class-weighted
     cross-entropy, and the decision threshold is TUNED on validation (max F1),
     not left at 0.5.
  3. ACCOUNT -> RING. GraphSAGE scores accounts; the product's unit is the ring,
     so member scores are aggregated (mean risk, with flagged-fraction reported)
     and rings are ranked.

Public API:
    result = run_detector(G, candidate_rings, accounts_df)
    result.account_scores   # {account_id: fraud_probability}
    result.ranked_rings     # list of ring dicts, sorted by risk desc
    result.threshold        # tuned decision threshold
    result.node_metrics     # held-out precision/recall/F1/AUC
    result.ring_metrics     # ranking quality vs legit-dense groups

Run standalone (from the PROJECT ROOT, not a parent dir):
    python src/detector.py
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.nn import MessagePassing

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
# Per-type edge weights. device/funding are strong coordination signals; ip is
# the weakest and most abundant, so it is heavily down-weighted; phone sits low.
EDGE_TYPE_WEIGHTS = {"device": 1.0, "funding": 1.0, "ip": 0.15, "phone": 0.35}
DEFAULT_EDGE_WEIGHT = 0.5  # any unexpected edge type
KNOWN_EDGE_TYPES = set(EDGE_TYPE_WEIGHTS)

FEATURE_COLS = ["txn_count", "total_amount", "avg_amount", "account_age_days"]
LABEL_COL = "fraud"
ID_COL = "account_id"
GROUP_COL = "group_type"   # optional; used only for the diagnostic breakdown

HIDDEN_DIM = 64
DROPOUT = 0.3
EPOCHS = 300
LR = 0.01
WEIGHT_DECAY = 5e-4
PATIENCE = 60  # early-stop on val AUC
SEED = 42


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Weighted GraphSAGE
# --------------------------------------------------------------------------- #
class WeightedSAGEConv(MessagePassing):
    """GraphSAGE-mean layer with per-edge weighting.

    Standard SAGEConv ignores edge weights; we need them so IP edges contribute
    less to a node's aggregated neighbourhood. This is the mean aggregator with
    a weighted mean (sum of w*x_j divided by sum of w) plus a separate root
    ('self') transform — i.e. GraphSAGE, not a different architecture.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin_neigh = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_neigh.reset_parameters()
        self.lin_self.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        # weighted sum of neighbour features, aggregated at the target node
        agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        # Normalize by UNWEIGHTED degree (neighbour count), NOT the weight-sum.
        # Dividing by the weight-sum cancels the down-weighting for any node whose
        # edges are all one type (a pure-IP node would get Σ(0.15·x)/Σ(0.15) =
        # mean(x) — full strength). Normalizing by count instead makes a pure-IP
        # neighbourhood aggregate at ~0.15x a device neighbourhood, so IP edges
        # genuinely contribute less to the representation. (requirement #1)
        count = torch.zeros(x.size(0), device=x.device)
        count.scatter_add_(0, edge_index[1], torch.ones_like(edge_weight))
        agg = agg / count.clamp(min=1.0).unsqueeze(-1)
        return self.lin_neigh(agg) + self.lin_self(x)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN_DIM, n_classes: int = 2,
                 dropout: float = DROPOUT):
        super().__init__()
        self.conv1 = WeightedSAGEConv(in_dim, hidden)
        self.conv2 = WeightedSAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index, edge_weight))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h)


# --------------------------------------------------------------------------- #
# Graph -> PyG tensors
# --------------------------------------------------------------------------- #
def _detect_edge_type_key(G, sample: int = 1000) -> str | None:
    """Find which edge attribute holds the shared-attribute type. Values may be
    a single type ('ip') or a comma-joined composite ('device,ip'), so we split
    and check whether any component is a known type. Avoids hard-coding the
    attribute name graph_build.py happened to use."""
    hits: Counter = Counter()
    for i, (_, _, data) in enumerate(G.edges(data=True)):
        for k, v in data.items():
            if isinstance(v, str) and any(
                    p.strip() in KNOWN_EDGE_TYPES for p in v.split(",")):
                hits[k] += 1
        if i + 1 >= sample:
            break
    return hits.most_common(1)[0][0] if hits else None


def _edge_weight_for(etype) -> float:
    """Resolve an edge's weight from its (possibly composite) shared-type string.
    An edge that shares a device/funding is a strong edge even if it ALSO shares
    an IP, so we take the MAX weight over the edge's component types. This is
    requirement #1: pure-IP edges are down-weighted; any device/funding edge
    keeps full weight."""
    if etype is None:
        return DEFAULT_EDGE_WEIGHT
    parts = [p.strip() for p in str(etype).split(",") if p.strip()]
    weights = [EDGE_TYPE_WEIGHTS.get(p, DEFAULT_EDGE_WEIGHT) for p in parts]
    return max(weights) if weights else DEFAULT_EDGE_WEIGHT


def _align_features(nodes: list, accounts_df: pd.DataFrame):
    """Return (feature_matrix, labels) aligned to `nodes`. Coerces id dtype to
    str if a direct join misses (node ids vs account_id dtype mismatch)."""
    df = accounts_df.set_index(ID_COL)
    keys: list = nodes
    if not pd.Index(nodes).isin(df.index).all():
        df.index = df.index.map(str)
        keys = [str(n) for n in nodes]
        if not pd.Index(keys).isin(df.index).all():
            missing = [n for n, k in zip(nodes, keys) if k not in df.index][:5]
            raise KeyError(f"Graph nodes not found in accounts_df, e.g. {missing}")
    feats = df.loc[keys, FEATURE_COLS].to_numpy(dtype=float)
    labels = df.loc[keys, LABEL_COL].to_numpy(dtype=int)
    return feats, labels


def _build_pyg(G, accounts_df):
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    feats, labels = _align_features(nodes, accounts_df)

    type_key = _detect_edge_type_key(G)
    src, dst, wts = [], [], []
    weight_counter: Counter = Counter()   # resolved weight -> #edges
    for u, v, data in G.edges(data=True):
        etype = data.get(type_key) if type_key else None
        w = _edge_weight_for(etype)
        weight_counter[w] += 1
        a, b = node_idx[u], node_idx[v]
        # symmetric: undirected graph -> both directions
        src += [a, b]
        dst += [b, a]
        wts += [w, w]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(wts, dtype=torch.float)
    return nodes, feats, labels, edge_index, edge_weight, type_key, weight_counter


def _aligned_column(nodes: list, accounts_df: pd.DataFrame, col: str):
    """Return `col` aligned to `nodes` (str-coercion fallback), or None if the
    column isn't present."""
    if col not in accounts_df.columns:
        return None
    df = accounts_df.set_index(ID_COL)
    keys = nodes
    if not pd.Index(nodes).isin(df.index).all():
        df.index = df.index.map(str)
        keys = [str(n) for n in nodes]
    return df.loc[keys, col].to_numpy()


def _group_type_breakdown(nodes, prob, holdout_idx, accounts_df, threshold):
    """Held-out mean predicted fraud-prob per group_type. This is the diagnostic
    that answers 'is the model separating fraud rings from legit-dense groups, or
    flagging every dense cluster?' — legit_* rows sitting near the fraud_ring row
    mean the model is confusing them."""
    gt = _aligned_column(nodes, accounts_df, GROUP_COL)
    if gt is None:
        return None
    rows = []
    for g in sorted(set(gt[holdout_idx].tolist())):
        sel = holdout_idx[gt[holdout_idx] == g]
        rows.append((str(g), len(sel), float(prob[sel].mean()),
                     float((prob[sel] >= threshold).mean())))
    return sorted(rows, key=lambda r: r[2], reverse=True)


def _make_masks(labels: np.ndarray):
    idx = np.arange(len(labels))
    train, tmp = train_test_split(idx, test_size=0.40, stratify=labels,
                                  random_state=SEED)
    val, test = train_test_split(tmp, test_size=0.50, stratify=labels[tmp],
                                 random_state=SEED)
    return train, val, test


# --------------------------------------------------------------------------- #
# Ring aggregation
# --------------------------------------------------------------------------- #
def _ring_members(ring: Any) -> list:
    """Extract member account ids from a candidate ring, tolerating a few shapes:
    a set/list/tuple of ids, a dict with a members-like key, or a comma string."""
    if isinstance(ring, dict):
        for key in ("members", "member_account_ids", "member_accounts",
                    "accounts", "nodes"):
            if key in ring:
                ring = ring[key]
                break
        else:
            raise KeyError(f"candidate ring dict has no members key: {list(ring)}")
    if isinstance(ring, str):
        return [m.strip() for m in ring.split(",") if m.strip()]
    return list(ring)


def _best_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class DetectorResult:
    account_scores: dict          # {account_id: fraud probability}
    ranked_rings: list            # ring dicts sorted by risk desc
    threshold: float
    node_metrics: dict
    ring_metrics: dict
    model: nn.Module = field(repr=False)
    scaler: StandardScaler = field(repr=False)
    node_order: list = field(repr=False, default_factory=list)
    edge_weight_counts: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_detector(G, candidate_rings, accounts_df, *, epochs: int = EPOCHS,
                 class_weight_power: float = 1.0, device: str | None = None,
                 verbose: bool = True) -> DetectorResult:
    """class_weight_power scales how hard the minority (fraud) class is up-weighted:
    1.0 = full balanced weighting (aggressive, high recall / more false positives);
    <1.0 (e.g. 0.5) softens it, trading recall for precision — the lever for the
    'model flags whole legit-dense groups' problem."""
    _set_seed()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    nodes, feats, labels, edge_index, edge_weight, type_key, weight_counter = \
        _build_pyg(G, accounts_df)
    train_i, val_i, test_i = _make_masks(labels)

    # standardize on TRAIN nodes only (no leakage from val/test into scaling)
    scaler = StandardScaler().fit(feats[train_i])
    X = torch.tensor(scaler.transform(feats), dtype=torch.float, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    train_mask = torch.zeros(len(nodes), dtype=torch.bool, device=device)
    val_mask = torch.zeros_like(train_mask)
    train_mask[train_i] = True
    val_mask[val_i] = True

    # balanced class weights -> weighted cross-entropy (imbalance guard)
    cw = compute_class_weight("balanced", classes=np.array([0, 1]),
                              y=labels[train_i])
    cw = cw ** class_weight_power   # soften (<1) or sharpen (>1) the imbalance handling
    class_weight = torch.tensor(cw, dtype=torch.float, device=device)

    model = GraphSAGE(in_dim=X.size(1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)

    def val_auc() -> float:
        model.eval()
        with torch.no_grad():
            p = F.softmax(model(X, edge_index, edge_weight), dim=1)[:, 1]
        pv = p[val_mask].cpu().numpy()
        yv = labels[val_i]
        return roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else 0.0

    best_auc, best_state, wait = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(X, edge_index, edge_weight)
        loss = loss_fn(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        auc = val_auc()
        if auc > best_auc:
            best_auc, wait = auc, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
        if verbose and (epoch % 25 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d} | loss {loss.item():.4f} | val AUC {auc:.4f}")
        if wait >= PATIENCE:
            if verbose:
                print(f"  early stop @ epoch {epoch} (best val AUC {best_auc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # full-graph probabilities
    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(X, edge_index, edge_weight), dim=1)[:, 1].cpu().numpy()

    # tune threshold on validation, evaluate on test
    threshold, val_f1 = _best_threshold(labels[val_i], prob[val_i])
    test_pred = (prob[test_i] >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        labels[test_i], test_pred, average="binary", zero_division=0)
    test_auc = (roc_auc_score(labels[test_i], prob[test_i])
                if len(np.unique(labels[test_i])) > 1 else float("nan"))
    node_metrics = {"precision": float(p), "recall": float(r), "f1": float(f1),
                    "auc": float(test_auc), "threshold": threshold,
                    "val_f1": float(val_f1)}

    # ---- account -> ring aggregation + ranking ----------------------------- #
    account_scores = {nid: float(prob[i]) for i, nid in enumerate(nodes)}
    fraud_of = {nid: int(labels[i]) for i, nid in enumerate(nodes)}

    ranked = []
    for ring in candidate_rings:
        members = _ring_members(ring)
        m_scores = [account_scores[m] for m in members if m in account_scores]
        if not m_scores:
            continue
        m_true = [fraud_of[m] for m in members if m in fraud_of]
        ranked.append({
            "members": members,
            "n_members": len(members),
            "ring_risk": float(np.mean(m_scores)),          # ranking key
            "max_risk": float(np.max(m_scores)),
            "frac_flagged": float(np.mean(np.array(m_scores) >= threshold)),
            "true_fraud": int(np.mean(m_true) > 0.5) if m_true else 0,
        })
    ranked.sort(key=lambda d: d["ring_risk"], reverse=True)

    # ring-level ranking quality: do fraud rings float to the top?
    ring_metrics = _ring_ranking_metrics(ranked)

    # diagnostic: where does predicted risk concentrate, by group type?
    holdout = np.concatenate([val_i, test_i])
    group_diag = _group_type_breakdown(nodes, prob, holdout, accounts_df, threshold)

    if verbose:
        _print_report(node_metrics, ring_metrics, ranked, type_key,
                      weight_counter, group_diag)

    return DetectorResult(
        account_scores=account_scores, ranked_rings=ranked, threshold=threshold,
        node_metrics=node_metrics, ring_metrics=ring_metrics, model=model,
        scaler=scaler, node_order=nodes, edge_weight_counts=dict(weight_counter))


def _ring_ranking_metrics(ranked: list) -> dict:
    n_fraud = sum(r["true_fraud"] for r in ranked)
    n_total = len(ranked)
    if n_fraud == 0 or n_total == 0:
        return {"n_fraud_rings": n_fraud, "n_candidate_rings": n_total}
    y = np.array([r["true_fraud"] for r in ranked])           # already sorted desc
    scores = np.array([r["ring_risk"] for r in ranked])
    prec_at_k = float(np.mean(y[:n_fraud]))                   # precision@(#fraud)
    ranks = [i + 1 for i, r in enumerate(ranked) if r["true_fraud"]]
    return {
        "n_candidate_rings": n_total,
        "n_fraud_rings": int(n_fraud),
        "precision_at_k": prec_at_k,
        "avg_precision": float(average_precision_score(y, scores)),
        "fraud_ring_ranks": ranks,
        "worst_fraud_rank": max(ranks),
    }


def _print_report(node_metrics, ring_metrics, ranked, type_key, weight_counter,
                  group_diag):
    print("\n" + "=" * 62)
    print("DETECTOR REPORT")
    print("=" * 62)
    print(f"edge-type attribute detected: {type_key!r}")
    print("resolved edge weights (IP down-weighting applied, max over "
          "composite types):")
    w_label = {1.0: "device/funding-bearing (strong)", 0.35: "phone-only",
               0.15: "ip-only (down-weighted)", DEFAULT_EDGE_WEIGHT: "unknown type"}
    total = sum(weight_counter.values()) or 1
    for w in sorted(weight_counter, reverse=True):
        c = weight_counter[w]
        print(f"    w={w:<4} : {c:5d} edges ({100 * c / total:4.0f}%)  "
              f"{w_label.get(w, '')}")
    print("-" * 62)
    print("NODE-LEVEL (held-out test):")
    print(f"  precision {node_metrics['precision']:.3f} | "
          f"recall {node_metrics['recall']:.3f} | "
          f"F1 {node_metrics['f1']:.3f} | AUC {node_metrics['auc']:.3f}")
    print(f"  tuned threshold = {node_metrics['threshold']:.2f} "
          f"(val F1 {node_metrics['val_f1']:.3f})")
    if group_diag:
        print("-" * 62)
        print("SCORE BY GROUP TYPE (held-out) — the fraud-vs-legit-dense test:")
        print("  group_type        n   mean_risk  %flagged")
        for g, n, mean_p, frac in group_diag:
            print(f"  {g:<16s} {n:4d}    {mean_p:.3f}     {frac:.2f}")
        print("  (legit_* mean_risk close to fraud_ring = model is flagging "
              "density, not fraud)")
    print("-" * 62)
    print("RING-LEVEL (all candidate rings, ranked by risk):")
    if "avg_precision" in ring_metrics:
        print(f"  {ring_metrics['n_fraud_rings']} fraud rings among "
              f"{ring_metrics['n_candidate_rings']} candidate groups")
        print(f"  precision@{ring_metrics['n_fraud_rings']} = "
              f"{ring_metrics['precision_at_k']:.3f} | "
              f"avg precision = {ring_metrics['avg_precision']:.3f}")
        print(f"  worst fraud-ring rank = {ring_metrics['worst_fraud_rank']} "
              f"(1 = top). The honest question: are legit-dense groups slipping "
              f"above fraud rings?")
    else:
        print(f"  {ring_metrics}")
    print("-" * 62)
    print("TOP 15 RANKED RINGS  (rank  risk  flagged  members  truth):")
    for i, r in enumerate(ranked[:15], 1):
        tag = "FRAUD" if r["true_fraud"] else "legit"
        print(f"  {i:2d}  {r['ring_risk']:.3f}  {r['frac_flagged']:.2f}   "
              f"n={r['n_members']:<3d}  {tag}")
    print("=" * 62 + "\n")


# --------------------------------------------------------------------------- #
# Standalone run
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # run from the PROJECT ROOT so data/ resolves
    accounts = pd.read_csv("data/accounts.csv")

    try:
        from graph_build import build_graph          # python src/detector.py
    except ImportError:
        from src.graph_build import build_graph      # python -m src.detector

    # NOTE: adjust this call if build_graph()'s signature differs.
    G, candidate_rings = build_graph()

    print(f"loaded graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges, {len(candidate_rings)} candidate rings")
    run_detector(G, candidate_rings, accounts)