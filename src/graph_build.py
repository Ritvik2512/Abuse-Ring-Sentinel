"""
graph_build.py — Day 2 of Abuse-Ring Sentinel.

Turn the generated CSVs into a NetworkX account graph and extract discrete
candidate rings (groups of accounts) that the Day-3 detector will score and the
Day-5 app will display. No ML here — just structure.

Public API:
    build_graph(data_dir, ...) -> (G, candidate_rings)
        G                : networkx.Graph, nodes = accounts (with features + labels),
                           edges = shared-attribute links (weighted, typed).
        candidate_rings  : list[list[account_id]], the discrete groups to score later.

Run as a script to build from data/, print a sanity summary, and save outputs
to results/:
    python src/graph_build.py                 # uses <project>/data and <project>/results
    python src/graph_build.py --method components
    python src/graph_build.py --data-dir path/to/data --out-dir path/to/results
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities

# Resolve paths relative to THIS file, not the current working directory, so the
# module works no matter where it's launched from (avoids the run-from-root gotcha).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results"

# Sharing a device or funding source is a stronger coordination signal than a
# shared IP (public wifi / carrier NAT) or phone; edge weight reflects that.
# These are defaults — Day 3/4 can re-weight without touching graph topology.
DEFAULT_TYPE_WEIGHTS = {"device": 2.0, "funding": 2.0, "ip": 1.0, "phone": 1.0}

# Account columns copied onto each node (everything the detector/app might want).
NODE_FEATURE_COLS = [
    "txn_count",
    "total_amount",
    "avg_amount",
    "account_age_days",
    "signup_timestamp",
    "fraud",
    "group_id",
    "group_type",
]

SEED = 42


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load_csvs(data_dir: Path):
    """Load the three CSVs. account_id is coerced to str everywhere for a
    consistent node-id type across files."""
    data_dir = Path(data_dir)
    accounts = pd.read_csv(data_dir / "accounts.csv")
    shared = pd.read_csv(data_dir / "shared_attributes.csv")
    rings_path = data_dir / "rings.csv"
    rings = pd.read_csv(rings_path) if rings_path.exists() else None

    accounts["account_id"] = accounts["account_id"].astype(str)
    shared["account_id"] = shared["account_id"].astype(str)
    # attribute_id may be numeric per type; keep as str and namespace by type below.
    shared["attribute_id"] = shared["attribute_id"].astype(str)
    shared["attribute_type"] = shared["attribute_type"].astype(str)
    return accounts, shared, rings


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def _add_nodes(G: nx.Graph, accounts: pd.DataFrame) -> None:
    for row in accounts.itertuples(index=False):
        attrs = {c: getattr(row, c) for c in NODE_FEATURE_COLS if hasattr(row, c)}
        G.add_node(str(row.account_id), **attrs)


def _add_edges(
    G: nx.Graph,
    shared: pd.DataFrame,
    type_weights: dict[str, float],
) -> int:
    """Connect accounts that share an attribute.

    Two accounts are linked iff they share the same (attribute_type, attribute_id)
    — namespacing by type so a device '5' and an ip '5' never collide. Each shared
    attribute contributes its type weight to the edge; the edge also records the
    set of shared types and the raw count of shared attributes.

    Returns the number of (account, attribute) rows skipped because the account
    wasn't in accounts.csv.
    """
    node_set = set(G.nodes)
    skipped = 0

    # Accumulate per unordered pair so multiple shared attributes fold into one edge.
    pair = defaultdict(lambda: {"types": set(), "n_shared": 0, "weight": 0.0})

    # Group by the full (type, id) key. Dedupe accounts within a group so a
    # duplicated row can't create a self-pair or double-count.
    grouped = shared.groupby(["attribute_type", "attribute_id"], sort=False)
    for (atype, _attr_id), g in grouped:
        accts = sorted({a for a in g["account_id"] if a in node_set})
        dropped = len(set(g["account_id"])) - len(accts)
        skipped += dropped
        if len(accts) < 2:
            continue  # unique / unmatched attribute -> no link
        w = type_weights.get(atype, 1.0)
        for i in range(len(accts)):
            a = accts[i]
            for j in range(i + 1, len(accts)):
                b = accts[j]
                rec = pair[(a, b)]
                rec["types"].add(atype)
                rec["n_shared"] += 1
                rec["weight"] += w

    for (a, b), rec in pair.items():
        # shared_types stored as a sorted, comma-joined string so the graph stays
        # serialisable (GraphML can't hold Python sets); split on ',' to recover.
        G.add_edge(
            a,
            b,
            weight=float(rec["weight"]),
            n_shared=int(rec["n_shared"]),
            shared_types=",".join(sorted(rec["types"])),
        )
    return skipped


# --------------------------------------------------------------------------- #
# Candidate-ring extraction
# --------------------------------------------------------------------------- #
def _connected_components(G: nx.Graph, min_size: int) -> list[list[str]]:
    comps = [sorted(c) for c in nx.connected_components(G)]
    return [c for c in comps if len(c) >= min_size]


def _louvain(G: nx.Graph, min_size: int, seed: int) -> list[list[str]]:
    # Louvain on the whole graph; isolated nodes fall out as singletons and are
    # filtered by min_size. Weighted so device/funding ties bind more tightly.
    comms = louvain_communities(G, weight="weight", seed=seed)
    comms = [sorted(c) for c in comms]
    return [c for c in comms if len(c) >= min_size]


def _tag_nodes(G: nx.Graph, rings: list[list[str]], attr: str) -> None:
    """Annotate each node with the id of the candidate ring it belongs to
    (-1 if it's in none), for convenient downstream lookups."""
    lookup = {}
    for i, r in enumerate(rings):
        for n in r:
            lookup[n] = i
    for n in G.nodes:
        G.nodes[n][attr] = lookup.get(n, -1)


# --------------------------------------------------------------------------- #
# Sanity check: do candidate rings recover the seeded fraud rings?
# --------------------------------------------------------------------------- #
def _seeded_groups(accounts: pd.DataFrame, group_type: str) -> dict[str, list[str]]:
    """Membership of each seeded group of a given group_type, from accounts.csv
    (the single source of truth for the node set). Background rows have no
    group_id and are excluded."""
    df = accounts[
        (accounts["group_type"] == group_type) & accounts["group_id"].notna()
    ]
    return {
        str(gid): [str(a) for a in sub["account_id"]]
        for gid, sub in df.groupby("group_id")
    }


def _recovery(seeded: dict[str, list[str]], rings: list[list[str]]) -> dict:
    """For each seeded group, the fraction of its members that land in a single
    candidate ring (best case). Reports the mean fraction and how many groups are
    'mostly' (>=0.8) recovered vs 'shattered' (<0.5)."""
    membership = {}
    for i, r in enumerate(rings):
        for n in r:
            membership[n] = i

    fracs = []
    for members in seeded.values():
        if not members:
            continue
        counts = defaultdict(int)
        for m in members:
            rid = membership.get(m, -1)  # -1 => isolated / not in any candidate ring
            counts[rid] += 1
        best = max(counts.values())
        fracs.append(best / len(members))

    if not fracs:
        return {"n": 0, "mean_frac": 0.0, "mostly": 0, "shattered": 0}
    return {
        "n": len(fracs),
        "mean_frac": sum(fracs) / len(fracs),
        "mostly": sum(f >= 0.8 for f in fracs),
        "shattered": sum(f < 0.5 for f in fracs),
    }


def _ring_composition(G: nx.Graph, rings: list[list[str]]) -> dict:
    """How clean are the candidate rings? A ring is 'fraud-majority' if >50% of
    its members are fraud. Reports how many candidate rings are fraud-majority and
    the mean fraud purity of those rings — a first read on whether legit-dense
    groups stay separate from fraud rings."""
    fraud_major = 0
    purities = []
    for r in rings:
        labels = [int(G.nodes[n].get("fraud", 0)) for n in r]
        frac = sum(labels) / len(labels)
        if frac > 0.5:
            fraud_major += 1
            purities.append(frac)
    mean_purity = sum(purities) / len(purities) if purities else 0.0
    return {"n_rings": len(rings), "fraud_majority": fraud_major, "mean_purity": mean_purity}


def _edge_type_breakdown(G: nx.Graph) -> dict[str, int]:
    counts = defaultdict(int)
    for _, _, d in G.edges(data=True):
        for t in d.get("shared_types", "").split(","):
            if t:
                counts[t] += 1
    return dict(counts)


def print_sanity_summary(
    G: nx.Graph,
    accounts: pd.DataFrame,
    cc_rings: list[list[str]],
    louvain_rings: list[list[str]],
    chosen_rings: list[list[str]],
    method: str,
    skipped_rows: int,
) -> dict:
    """Print the Day-2 sanity summary and return it as a dict."""
    n_iso = sum(1 for _, d in G.degree() if d == 0)
    comps_all = list(nx.connected_components(G))
    largest_cc = max((len(c) for c in comps_all), default=0)

    fraud_seeded = _seeded_groups(accounts, "fraud_ring")
    cc_rec = _recovery(fraud_seeded, cc_rings)
    lv_rec = _recovery(fraud_seeded, louvain_rings)

    # Recovery of legit-dense groups, for context (we WANT these to also cohere
    # structurally — that's what makes them hard negatives for the detector).
    legit_types = ["legit_family", "legit_hostel", "legit_office"]
    legit_seeded = {}
    for lt in legit_types:
        legit_seeded.update(_seeded_groups(accounts, lt))
    legit_rec_chosen = _recovery(legit_seeded, chosen_rings)

    comp = _ring_composition(G, chosen_rings)
    edge_types = _edge_type_breakdown(G)

    line = "-" * 64
    print(line)
    print("GRAPH BUILD — Day 2 sanity summary")
    print(line)
    print(f"nodes ................. {G.number_of_nodes()}")
    print(f"edges ................. {G.number_of_edges()}")
    print(f"isolated nodes ........ {n_iso}")
    if skipped_rows:
        print(f"skipped attr rows ..... {skipped_rows}  (account not in accounts.csv)")
    print(f"edges by shared type .. {edge_types}")
    print(f"connected components .. {len(comps_all)}  (largest = {largest_cc} nodes)")
    print(f"  candidate rings (cc, size>=2) ...... {len(cc_rings)}")
    print(f"  candidate rings (louvain, size>=2) . {len(louvain_rings)}")
    print(f"CHOSEN candidate rings ({method}) ....... {len(chosen_rings)}")
    print(line)
    print("Seeded FRAUD-ring recovery (best single candidate ring per seeded ring):")
    print(
        f"  connected components : mean {cc_rec['mean_frac']:.2f} | "
        f"mostly>=0.8: {cc_rec['mostly']}/{cc_rec['n']} | "
        f"shattered<0.5: {cc_rec['shattered']}"
    )
    print(
        f"  louvain communities  : mean {lv_rec['mean_frac']:.2f} | "
        f"mostly>=0.8: {lv_rec['mostly']}/{lv_rec['n']} | "
        f"shattered<0.5: {lv_rec['shattered']}"
    )
    print(
        f"Legit-dense group recovery ({method}) : mean {legit_rec_chosen['mean_frac']:.2f} "
        f"over {legit_rec_chosen['n']} groups"
    )
    print(
        f"Candidate-ring composition ({method}) : "
        f"{comp['fraud_majority']}/{comp['n_rings']} are fraud-majority, "
        f"mean purity {comp['mean_purity']:.2f}"
    )
    print(line)

    chosen_rec = lv_rec if method == "louvain" else cc_rec
    if chosen_rec["shattered"] > 0:
        print(
            f"WARNING: {chosen_rec['shattered']} seeded fraud ring(s) shattered "
            f"(<50% in one candidate ring) under '{method}'. "
            "Inspect before Day 3 — the detector can only score groups it can see."
        )
    else:
        print(f"OK: no seeded fraud ring is shattered under '{method}'.")
    print(line)

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "isolated": n_iso,
        "n_components": len(comps_all),
        "largest_cc": largest_cc,
        "n_cc_rings": len(cc_rings),
        "n_louvain_rings": len(louvain_rings),
        "method": method,
        "fraud_recovery_cc": cc_rec,
        "fraud_recovery_louvain": lv_rec,
        "legit_recovery_chosen": legit_rec_chosen,
        "composition_chosen": comp,
        "edge_types": edge_types,
    }


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #
def save_outputs(G: nx.Graph, candidate_rings: list[list[str]], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pickle preserves all attribute types exactly (write_gpickle was removed in nx 3.x).
    with open(out_dir / "graph.gpickle", "wb") as f:
        pickle.dump(G, f)

    # GraphML for interoperability; all attrs are already str/num-friendly.
    nx.write_graphml(G, out_dir / "graph.graphml")

    with open(out_dir / "candidate_rings.json", "w") as f:
        json.dump(candidate_rings, f, indent=2)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def build_graph(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    method: str = "louvain",
    min_ring_size: int = 2,
    type_weights: dict[str, float] | None = None,
    seed: int = SEED,
    verbose: bool = True,
):
    """Build the account graph and extract candidate rings.

    Parameters
    ----------
    data_dir : path to the folder holding accounts.csv / shared_attributes.csv / rings.csv
    method   : 'louvain' (tighter communities, default) or 'components'
               (connected components, looser but never splits a linked group)
    min_ring_size : candidate rings smaller than this are dropped (default 2)
    type_weights  : per-attribute-type edge weights (default device/funding=2, ip/phone=1)
    seed     : RNG seed for Louvain (reproducibility)
    verbose  : print the sanity summary

    Returns
    -------
    (G, candidate_rings)
        G : networkx.Graph
        candidate_rings : list[list[account_id]]
    """
    if method not in {"louvain", "components"}:
        raise ValueError("method must be 'louvain' or 'components'")
    type_weights = type_weights or DEFAULT_TYPE_WEIGHTS

    accounts, shared, _rings = _load_csvs(data_dir)

    G = nx.Graph()
    _add_nodes(G, accounts)
    skipped = _add_edges(G, shared, type_weights)

    cc_rings = _connected_components(G, min_ring_size)
    louvain_rings = _louvain(G, min_ring_size, seed)
    chosen = louvain_rings if method == "louvain" else cc_rings

    _tag_nodes(G, cc_rings, "cc_id")
    _tag_nodes(G, louvain_rings, "louvain_id")
    _tag_nodes(G, chosen, "candidate_ring_id")

    if verbose:
        print_sanity_summary(
            G, accounts, cc_rings, louvain_rings, chosen, method, skipped
        )

    return G, chosen


def _cli():
    p = argparse.ArgumentParser(description="Build account graph + candidate rings.")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--method", choices=["louvain", "components"], default="louvain")
    p.add_argument("--min-ring-size", type=int, default=2)
    p.add_argument("--no-save", action="store_true", help="skip writing results/")
    args = p.parse_args()

    G, rings = build_graph(
        data_dir=args.data_dir,
        method=args.method,
        min_ring_size=args.min_ring_size,
    )
    if not args.no_save:
        save_outputs(G, rings, args.out_dir)
        print(f"Saved graph + candidate rings to {args.out_dir}")


if __name__ == "__main__":
    _cli()