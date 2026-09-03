"""
detect.py — Abuse-Ring Sentinel, Day 4 entry point (the safety net).

One command: a batch of accounts + shared-attribute records in
             -> ranked flagged fraud rings (risk score + evidence) out.

Wires together the Day-2 graph builder and the Day-3 GraphSAGE detector.
No modelling here — this is plumbing over already-working pieces.

Run from the project root:
    python detect.py
    python detect.py --data-dir path/to/batch --results-dir results
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# --- project root on path so `src` imports work regardless of CWD ----------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.graph_build import build_graph        # Day 2
    from src.detector import run_detector          # Day 3
except ModuleNotFoundError as e:
    sys.exit(
        f"Could not import src modules ({e}).\n"
        f"Run this from the project root (the folder that contains src/): {ROOT}"
    )

_MISSING = object()


# ===========================================================================
# ADAPTERS over the Day-3 return contract.
# This is the ONE integration seam. If run_detector returns a shape different
# from what's assumed here, adjust only these helpers — the rest is stable.
# ===========================================================================
def _pick(obj, *names, default=_MISSING):
    """First present key (dict) or attribute (object / namedtuple)."""
    for n in names:
        if isinstance(obj, dict):
            if n in obj:
                return obj[n]
        elif hasattr(obj, n):
            return getattr(obj, n)
    if default is _MISSING:
        raise KeyError(f"none of {names} found on {type(obj).__name__}")
    return default


def unpack_detector_result(result):
    """
    Normalise run_detector(...) ->
        (account_scores, ranked_rings, threshold, node_metrics, ring_metrics).

    Handles a dict, a dataclass/namedtuple, or a plain 5-tuple in the documented
    order: (account_scores, ranked_rings, threshold, node_metrics, ring_metrics).
    """
    if (
        isinstance(result, (tuple, list))
        and not hasattr(result, "_fields")   # exclude namedtuples
        and len(result) == 5
    ):
        account_scores, ranked_rings, threshold, node_metrics, ring_metrics = result
        return account_scores, ranked_rings, threshold, node_metrics, ring_metrics

    account_scores = _pick(result, "account_scores", "node_scores", "scores", default={})
    ranked_rings = _pick(result, "ranked_rings", "rings")
    threshold = _pick(result, "threshold", default=None)
    node_metrics = _pick(result, "node_metrics", default={})
    ring_metrics = _pick(result, "ring_metrics", default={})
    return account_scores, ranked_rings, threshold, node_metrics, ring_metrics


def ring_members(ring):
    """Extract the member account-id list from one ranked ring."""
    m = _pick(ring, "members", "member_account_ids", "accounts", "account_ids",
              "nodes", default=None)
    if m is None and isinstance(ring, (tuple, list)):
        # e.g. (members, score) or (id, members, score)
        m = next((x for x in ring
                  if isinstance(x, (list, set, tuple)) and not isinstance(x, str)), None)
    if m is None:
        raise KeyError(f"could not find members on ranked ring: {ring!r}")
    return list(m)


def ring_score(ring):
    """Extract the risk score from one ranked ring."""
    s = _pick(ring, "risk_score", "risk", "score", "ring_risk", "ring_score",
              default=None)
    if s is None and isinstance(ring, (tuple, list)):
        s = next((x for x in ring
                  if isinstance(x, float) or (isinstance(x, int) and not isinstance(x, bool))),
                 None)
    if s is None:
        raise KeyError(f"could not find risk score on ranked ring: {ring!r}")
    return float(s)


def ring_id(ring, fallback):
    return _pick(ring, "ring_id", "id", "community_id", default=fallback)


def headline_metrics(node_metrics, ring_metrics):
    """Pull the README numbers out of the Day-3 metric dicts (best-effort keys)."""
    def g(m, *names):
        return _pick(m, *names, default=None)

    node = {
        "precision": g(node_metrics, "precision"),
        "recall": g(node_metrics, "recall"),
        "f1": g(node_metrics, "f1", "f1_score"),
        "auc": g(node_metrics, "auc", "roc_auc", "auroc"),
    }
    ring = {
        "precision_at_k": g(ring_metrics, "precision_at_k", "ring_precision_at_k",
                            "precision@k"),
        "k": g(ring_metrics, "k", "at_k"),
        "average_precision": g(ring_metrics, "average_precision", "ap", "map"),
    }
    return node, ring


# ===========================================================================
# EVIDENCE — pulled straight from the raw records, consistent with rings.csv.
# Independent of graph internals, so it holds regardless of edge encoding.
# ===========================================================================
ATTR_TYPES = ("device", "funding", "ip", "phone")


def _coerce_ids(members, ref_series):
    """Match member ids to the dtype of the accounts.account_id column."""
    if pd.api.types.is_integer_dtype(ref_series):
        return [int(m) for m in members]
    if pd.api.types.is_string_dtype(ref_series) or ref_series.dtype == object:
        return [str(m) for m in members]
    return list(members)


def shared_attribute_counts(members, shared_df):
    """Distinct attribute_ids of each type shared by >=2 of these members."""
    sub = shared_df[shared_df["account_id"].isin(members)]
    counts = {}
    for atype in ATTR_TYPES:
        t = sub[sub["attribute_type"] == atype]
        if t.empty:
            counts[atype] = 0
            continue
        per_attr = t.groupby("attribute_id")["account_id"].nunique()
        counts[atype] = int((per_attr >= 2).sum())
    return counts


def exposure(members, accounts_df):
    """(total txn count, ₹ exposed) across the ring's members."""
    sub = accounts_df[accounts_df["account_id"].isin(members)]
    return int(sub["txn_count"].sum()), float(sub["total_amount"].sum())


def build_ring_records(ranked_rings, threshold, accounts_df, shared_df):
    """One evidence record per ring, ranked by risk (defensively re-sorted)."""
    ref = accounts_df["account_id"]
    raw = [(_coerce_ids(ring_members(r), ref), ring_score(r), ring_id(r, None))
           for r in ranked_rings]
    raw.sort(key=lambda t: t[1], reverse=True)  # ensure ranked by risk desc

    records = []
    for i, (members, score, rid) in enumerate(raw, start=1):
        members = sorted(members)
        counts = shared_attribute_counts(members, shared_df)
        txn_count, amount_exposed = exposure(members, accounts_df)
        records.append({
            "rank": i,
            "ring_id": rid if rid is not None else f"R{i:02d}",
            "risk_score": round(score, 4),
            "flagged": (threshold is None) or (score >= threshold),
            "n_members": len(members),
            "n_shared_devices": counts["device"],
            "n_shared_funding": counts["funding"],
            "n_shared_ip": counts["ip"],
            "n_shared_phone": counts["phone"],
            "total_txn_count": txn_count,
            "total_amount_exposed": round(amount_exposed, 2),
            "member_account_ids": members,
        })
    return records


# ===========================================================================
# OUTPUT
# ===========================================================================
def _json_default(o):
    """Make numpy scalars / arrays JSON-serialisable."""
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def save_outputs(records, node_metrics, ring_metrics, threshold, results_dir,
                 account_scores=None, meta=None):
    results_dir.mkdir(parents=True, exist_ok=True)
    flagged = [r for r in records if r["flagged"]]
    node_head, ring_head = headline_metrics(node_metrics, ring_metrics)

    # Per-account scores for the members of flagged rings only (keeps the file
    # lean). The app reads these under top-level "account_scores" to size nodes
    # by real per-account risk — and to stop reporting them as "absent".
    flagged_members = {str(m) for r in flagged for m in r["member_account_ids"]}
    scores_out = {
        str(a): round(float(s), 4)
        for a, s in (account_scores or {}).items()
        if str(a) in flagged_members
    }

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "threshold": threshold,
        "n_ranked": len(records),
        "n_flagged": len(flagged),
        "meta": meta or {},                 # accounts_scanned, candidate_rings, graph size
        "metrics": {"node": node_head, "ring": ring_head},
        "account_scores": scores_out,       # {account_id: risk} for flagged members
        "rings": flagged,  # app loads this — one record per flagged ring
    }
    json_path = results_dir / "detected_rings.json"
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))

    csv_rows = [
        {**r, "member_account_ids": json.dumps(r["member_account_ids"], default=_json_default)}
        for r in flagged
    ]
    csv_path = results_dir / "detected_rings.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    return json_path, csv_path, flagged


def write_run_log(results_dir, run_meta, threshold, n_ranked, n_flagged, started):
    """Append one operational line per run: timestamp, graph size, counts, threshold."""
    results_dir.mkdir(parents=True, exist_ok=True)
    thr = f"{threshold:.3f}" if isinstance(threshold, float) else str(threshold)
    line = (
        f"{started.isoformat(timespec='seconds')}  "
        f"nodes={run_meta.get('graph_nodes')} edges={run_meta.get('graph_edges')} "
        f"candidates={run_meta.get('candidate_rings')} "
        f"ranked={n_ranked} flagged={n_flagged} threshold={thr}"
    )
    with open(results_dir / "run_log.txt", "a") as f:
        f.write(line + "\n")
    return line


def print_summary(records, flagged, node_metrics, ring_metrics, threshold,
                  json_path, csv_path):
    node, ring = headline_metrics(node_metrics, ring_metrics)
    thr = f"{threshold:.3f}" if isinstance(threshold, float) else str(threshold)

    print("\n" + "=" * 72)
    print("ABUSE-RING SENTINEL — detection run")
    print("=" * 72)
    if threshold is None:
        print("(!) run_detector returned no threshold — flagging all ranked rings.")
    print(f"Ranked rings: {len(records)}   Flagged (risk >= {thr}): {len(flagged)}\n")

    header = (f"{'#':>3}  {'ring':<8} {'risk':>6}  {'mem':>4} {'dev':>4} "
              f"{'fund':>5}  {'txns':>7}  {'INR exposed':>16}")
    print(header)
    print("-" * len(header))
    show = records[:30]
    for r in show:
        mark = "" if r["flagged"] else "   (below threshold)"
        print(f"{r['rank']:>3}  {str(r['ring_id']):<8} {r['risk_score']:>6.3f}  "
              f"{r['n_members']:>4} {r['n_shared_devices']:>4} {r['n_shared_funding']:>5}  "
              f"{r['total_txn_count']:>7}  {r['total_amount_exposed']:>16,.2f}{mark}")
    if len(records) > len(show):
        print(f"... and {len(records) - len(show)} more")

    print("\n--- README numbers (held-out, reused from Day-3 eval) ---")
    printed = False
    node_parts = []
    if node["precision"] is not None: node_parts.append(f"precision {node['precision']:.3f}")
    if node["recall"] is not None:    node_parts.append(f"recall {node['recall']:.3f}")
    if node["f1"] is not None:        node_parts.append(f"F1 {node['f1']:.3f}")
    if node["auc"] is not None:       node_parts.append(f"AUC {node['auc']:.3f}")
    if node_parts:
        print("node-level:  " + " / ".join(node_parts)); printed = True
    if ring["precision_at_k"] is not None:
        ktxt = f"@{ring['k']}" if ring["k"] is not None else "@k"
        print(f"ring-level:  precision{ktxt} = {ring['precision_at_k']:.3f}"); printed = True
    if ring["average_precision"] is not None:
        print(f"ring-level:  average precision = {ring['average_precision']:.3f}"); printed = True
    if not printed:
        # keys didn't match — surface the raw dicts so the number is still visible
        print("(metric keys not recognised — raw dicts below; adjust headline_metrics())")
        print(f"  node_metrics = {node_metrics}")
        print(f"  ring_metrics = {ring_metrics}")

    print(f"\nSaved:\n  {json_path}\n  {csv_path}")
    print("=" * 72 + "\n")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Abuse-Ring Sentinel detector — batch in, ranked rings out.")
    ap.add_argument("--data-dir", default=str(ROOT / "data"),
                    help="folder with accounts.csv + shared_attributes.csv (default: ./data)")
    ap.add_argument("--results-dir", default=str(ROOT / "results"),
                    help="where to write detected_rings.{json,csv} (default: ./results)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    started = datetime.now()

    accounts_df = pd.read_csv(data_dir / "accounts.csv")
    shared_df = pd.read_csv(data_dir / "shared_attributes.csv")
    print(f"Loaded {len(accounts_df)} accounts, {len(shared_df)} shared-attribute "
          f"records from {data_dir}")

    G, candidate_rings = build_graph(str(data_dir))
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(candidate_rings)} candidate rings")

    result = run_detector(G, candidate_rings, accounts_df)
    account_scores, ranked_rings, threshold, node_metrics, ring_metrics = \
        unpack_detector_result(result)

    if not ranked_rings:
        print("No candidate rings returned by the detector — nothing to flag.")
        return

    run_meta = {
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "candidate_rings": len(candidate_rings),
        "accounts_scanned": len(accounts_df),
    }

    records = build_ring_records(ranked_rings, threshold, accounts_df, shared_df)
    json_path, csv_path, flagged = save_outputs(
        records, node_metrics, ring_metrics, threshold, results_dir,
        account_scores=account_scores, meta=run_meta)
    print_summary(records, flagged, node_metrics, ring_metrics, threshold,
                  json_path, csv_path)

    log_line = write_run_log(results_dir, run_meta, threshold,
                             len(records), len(flagged), started)
    print(f"[run] {log_line}")
    print(f"[run] elapsed {(datetime.now() - started).total_seconds():.1f}s  "
          f"log: {results_dir / 'run_log.txt'}")


if __name__ == "__main__":
    main()