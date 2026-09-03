"""
Abuse-Ring Sentinel — Day 5 review app
======================================
A Streamlit + pyvis interface over the *already-computed* detector output.

This app is a DISPLAY LAYER. It does not build the graph, train, tune, or
re-run the GNN. It loads three things produced by earlier days:

  1. results/<detection results>   (JSON or CSV)  <- source of truth for
     which rings are flagged and their risk / evidence  (Day 4, detect.py)
  2. data/accounts.csv             <- per-account attributes & role
  3. data/shared_attributes.csv    <- account<->attribute sharing records

The per-ring graph shown on click is reconstructed from shared_attributes.csv
restricted to that ring's members (raw data, not detection logic), so the app
stays fully decoupled from the model.

--------------------------------------------------------------------------
IF THE APP CAN'T FIND A FIELD: the only thing that varies between builds is the
key names in your results file. Adjust them in ONE place -> `RESULTS_KEYS`
and `RING_KEYS` below. Nothing else should need to change.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_THRESHOLD = 0.51  # Day 4 operating point

# The app loads results/detected_rings.json by default (or an uploaded JSON).

# --- Schema tolerance -------------------------------------------------------
# Where the flagged-ring list lives inside a JSON results file.
RESULTS_LIST_KEYS = ["rings", "flagged_rings", "results", "candidates", "data"]

# Per-ring field name -> list of accepted aliases (first match wins).
RING_KEYS = {
    "ring_id":       ["ring_id", "id", "community_id", "community", "cluster_id"],
    "risk":          ["risk_score", "risk", "score", "ring_risk", "mean_risk"],
    "members":       ["member_account_ids", "members", "member_accounts",
                      "accounts", "account_ids", "nodes"],
    "member_count":  ["n_members", "member_count", "size", "num_members"],
    "shared_devices":["n_shared_devices", "shared_devices", "n_devices",
                      "devices", "device_count"],
    "shared_funding":["n_shared_funding", "shared_funding", "n_funding",
                      "funding", "funding_count"],
    "txn_count":     ["total_txn_count", "txn_count", "total_txns",
                      "n_txns", "transactions"],
    "amount":        ["total_amount_exposed", "amount_exposed", "rupees_exposed",
                      "exposure", "total_amount", "amount", "rupees"],
    "flagged":       ["flagged", "above_threshold", "is_flagged"],
}

# Optional per-account risk (for node sizing). Accepted shapes:
#   results["account_scores"] = {account_id: score}   OR a scores CSV.
ACCOUNT_SCORE_KEYS = ["account_scores", "node_scores", "scores"]
ACCOUNT_SCORE_FILE_CANDIDATES = ["account_scores.csv", "node_scores.csv"]

# --- Visual language --------------------------------------------------------
ROLE_COLORS = {
    "fraud":      "#e5484d",  # flagged fraud
    "legit":      "#4593c9",  # legit-but-dense (family / hostel / office)
    "background": "#8b8d98",  # sparse background
}
ATTR_STYLE = {  # shared-attribute hub nodes + their edges
    "device":  {"color": "#f5a623", "shape": "square"},
    "funding": {"color": "#30a46c", "shape": "diamond"},
    "ip":      {"color": "#8e6fd6", "shape": "triangle"},
    "phone":   {"color": "#12a5b8", "shape": "star"},
}
GROUP_TYPE_TO_ROLE = {
    "fraud_ring":   "fraud",
    "legit_family": "legit",
    "legit_hostel": "legit",
    "legit_office": "legit",
    "background":   "background",
}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def _first(d: dict, aliases: list[str], default=None):
    """Return d[k] for the first alias present, else default."""
    for k in aliases:
        if k in d and d[k] is not None:
            return d[k]
    return default


def fmt_inr(x) -> str:
    """Compact Indian-currency string: 1234567 -> '₹12.35 L'."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "₹0"
    if x >= 1e7:
        return f"₹{x/1e7:.2f} Cr"
    if x >= 1e5:
        return f"₹{x/1e5:.2f} L"
    if x >= 1e3:
        return f"₹{x/1e3:.1f} K"
    return f"₹{x:,.0f}"


def risk_band(score: float, threshold: float) -> tuple[str, str]:
    """(label, hex-color) for a risk score."""
    if score >= max(0.75, threshold):
        return "High", "#e5484d"
    if score >= threshold:
        return "Elevated", "#f5a623"
    return "Low", "#30a46c"


def risk_chip(score: float, threshold: float) -> str:
    label, color = risk_band(score, threshold)
    return (
        f"<span style='background:{color};color:#fff;padding:2px 10px;"
        f"border-radius:999px;font-weight:600;font-size:0.85rem;white-space:nowrap;'>"
        f"{label} · {score:.2f}</span>"
    )


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def _rows_from_results(obj) -> list[dict]:
    """Pull the list of ring rows out of whatever top-level shape we got."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in RESULTS_LIST_KEYS:
            if isinstance(obj.get(k), list):
                return obj[k]
        # dict-of-rings keyed by id
        vals = list(obj.values())
        if vals and all(isinstance(v, dict) for v in vals):
            return vals
    return []


def normalize_ring(raw: dict, idx: int, threshold: float) -> dict:
    members = _first(raw, RING_KEYS["members"], []) or []
    if isinstance(members, str):
        members = [m.strip() for m in members.split(",") if m.strip()]
    members = [str(m) for m in members]

    risk = float(_first(raw, RING_KEYS["risk"], 0.0) or 0.0)
    mc = _first(raw, RING_KEYS["member_count"])
    member_count = int(mc) if mc is not None else len(members)

    flagged = _first(raw, RING_KEYS["flagged"])
    if flagged is None:
        flagged = risk >= threshold

    return {
        "ring_id": str(_first(raw, RING_KEYS["ring_id"], idx)),
        "risk": risk,
        "members": members,
        "member_count": member_count,
        "shared_devices": _first(raw, RING_KEYS["shared_devices"]),
        "shared_funding": _first(raw, RING_KEYS["shared_funding"]),
        "txn_count": _first(raw, RING_KEYS["txn_count"]),
        "amount": float(_first(raw, RING_KEYS["amount"], 0.0) or 0.0),
        "flagged": bool(flagged),
        "_raw": raw,
    }


@st.cache_data(show_spinner=False)
def load_results(path_str: str, threshold: float) -> tuple[list[dict], dict, dict]:
    """Returns (rings, meta, account_scores)."""
    path = Path(path_str)
    account_scores: dict[str, float] = {}
    meta: dict = {}

    if path.suffix == ".json":
        obj = json.loads(path.read_text())
        rows = _rows_from_results(obj)
        if isinstance(obj, dict):
            meta = obj.get("meta", {}) if isinstance(obj.get("meta"), dict) else {}
            for k in ACCOUNT_SCORE_KEYS:
                if isinstance(obj.get(k), dict):
                    account_scores = {str(a): float(s) for a, s in obj[k].items()}
                    break
    else:  # CSV: one row per ring
        df = pd.read_csv(path)
        rows = df.to_dict(orient="records")

    rings = [normalize_ring(r, i, threshold) for i, r in enumerate(rows)]
    rings.sort(key=lambda r: r["risk"], reverse=True)

    # optional external per-account score file
    if not account_scores:
        for name in ACCOUNT_SCORE_FILE_CANDIDATES:
            p = RESULTS_DIR / name
            if p.exists():
                sdf = pd.read_csv(p)
                idc = sdf.columns[0]
                scol = next((c for c in sdf.columns[1:]
                             if sdf[c].dtype != object), sdf.columns[-1])
                account_scores = {str(a): float(s)
                                  for a, s in zip(sdf[idc], sdf[scol])}
                break
    return rings, meta, account_scores


@st.cache_data(show_spinner=False)
def load_accounts() -> pd.DataFrame:
    p = DATA_DIR / "accounts.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"account_id": str})
    df["account_id"] = df["account_id"].astype(str)
    if "group_type" in df.columns:
        df["role"] = df["group_type"].map(GROUP_TYPE_TO_ROLE).fillna("background")
    else:
        df["role"] = "background"
    return df.set_index("account_id", drop=False)


@st.cache_data(show_spinner=False)
def load_shared_attributes() -> pd.DataFrame:
    p = DATA_DIR / "shared_attributes.csv"
    if not p.exists():
        return pd.DataFrame(columns=["account_id", "attribute_type", "attribute_id"])
    df = pd.read_csv(p, dtype={"account_id": str, "attribute_id": str})
    df["account_id"] = df["account_id"].astype(str)
    return df


# ----------------------------------------------------------------------------
# Per-ring graph (reconstructed from raw sharing data, restricted to members)
# ----------------------------------------------------------------------------

def ring_shared_hubs(members: list[str], attrs_df: pd.DataFrame) -> pd.DataFrame:
    """Attributes shared by >=2 ring members -> hub rows to render."""
    if attrs_df.empty or not members:
        return pd.DataFrame(columns=["attribute_type", "attribute_id", "accounts"])
    sub = attrs_df[attrs_df["account_id"].isin(members)]
    grp = (sub.groupby(["attribute_type", "attribute_id"])["account_id"]
              .apply(lambda s: sorted(set(s)))
              .reset_index(name="accounts"))
    return grp[grp["accounts"].apply(len) >= 2].reset_index(drop=True)


# Physics + styling for the investigation graph. One options blob (set_options
# replaces anything barnes_hut() would set, so everything lives here): strong
# barnesHut repulsion + avoidOverlap spreads nodes; small node font with a white
# halo keeps labels legible over edges even when nodes are close.
PYVIS_OPTIONS = {
    "nodes": {
        "font": {"size": 11, "face": "Inter, Arial, sans-serif",
                 "color": "#2a2a33", "strokeWidth": 3, "strokeColor": "#ffffff"},
        "borderWidth": 1,
        "borderWidthSelected": 2,
    },
    "edges": {
        "width": 1,
        "smooth": {"enabled": True, "type": "continuous"},
        "selectionWidth": 1.5,
    },
    "physics": {
        "solver": "barnesHut",
        "barnesHut": {
            "gravitationalConstant": -28000,  # was ~-8000 (and silently ignored)
            "centralGravity": 0.12,
            "springLength": 200,
            "springConstant": 0.02,
            "damping": 0.09,
            "avoidOverlap": 0.6,              # keep nodes from stacking
        },
        "minVelocity": 0.75,
        "stabilization": {"enabled": True, "iterations": 220},
    },
    "interaction": {"hover": True, "tooltipDelay": 80},
}


def _short_label(acc: str) -> str:
    """Trim the 'ACC_' prefix so labels are short and don't collide (full id
    stays in the hover tooltip)."""
    s = str(acc)
    return s.split("_")[-1] if "_" in s else s


def _legend_overlay_html() -> str:
    """Small self-contained legend, baked into the graph HTML so the saved
    sample_output.html and README screenshots explain themselves."""
    def item(glyph, color, text):
        return (f"<span style='white-space:nowrap;margin-right:9px;'>"
                f"<span style='color:{color};font-size:13px;'>{glyph}</span> {text}</span>")
    nodes = (item("●", ROLE_COLORS["fraud"], "fraud acct")
             + item("●", ROLE_COLORS["legit"], "legit-dense")
             + item("●", ROLE_COLORS["background"], "background"))
    glyphs = {"device": "■", "funding": "◆", "ip": "▲", "phone": "★"}
    shared = "".join(item(glyphs[a], ATTR_STYLE[a]["color"], a) for a in glyphs)
    return (
        "<div style=\"position:absolute;top:10px;left:10px;z-index:999;"
        "background:rgba(255,255,255,0.94);border:1px solid #e4e6eb;border-radius:8px;"
        "padding:7px 10px;font-family:Inter,Arial,sans-serif;font-size:11px;color:#3a3a44;"
        "line-height:1.8;box-shadow:0 1px 4px rgba(0,0,0,0.08);max-width:95%;\">"
        f"<div>{nodes}</div>"
        f"<div><span style='color:#8a8a93;'>shared&nbsp;via:</span> {shared}</div>"
        "</div>"
    )


def _inject_legend(html: str) -> str:
    legend = _legend_overlay_html()
    return (html.replace("</body>", legend + "</body>", 1)
            if "</body>" in html else html + legend)


def build_ring_html(ring: dict, accounts_df: pd.DataFrame,
                    attrs_df: pd.DataFrame, account_scores: dict[str, float],
                    height: int = 560) -> tuple[str, pd.DataFrame]:
    """Bipartite pyvis graph: account nodes + shared-attribute hub nodes."""
    members = ring["members"]
    net = Network(height=f"{height}px", width="100%", bgcolor="#ffffff",
                  font_color="#1a1a1a", directed=False, cdn_resources="in_line")

    # account nodes
    for acc in members:
        role = "background"
        title_bits = [f"account {acc}"]
        if acc in accounts_df.index:
            row = accounts_df.loc[acc]
            role = row.get("role", "background")
            if "group_type" in row:
                title_bits.append(f"type: {row['group_type']}")
            if "txn_count" in row:
                title_bits.append(f"txns: {row['txn_count']}")
            if "total_amount" in row:
                title_bits.append(f"amount: {fmt_inr(row['total_amount'])}")
        score = float(account_scores.get(acc, ring["risk"]))
        size = 9 + 7 * max(0.0, min(1.0, score))   # small range: 9–16px
        title_bits.append(f"risk: {score:.2f}")
        net.add_node(acc, label=_short_label(acc),
                     color=ROLE_COLORS.get(role, ROLE_COLORS["background"]),
                     shape="dot", size=size, title="\n".join(title_bits))

    # shared-attribute hubs + edges
    hubs = ring_shared_hubs(members, attrs_df)
    for _, h in hubs.iterrows():
        atype = h["attribute_type"]
        style = ATTR_STYLE.get(atype, {"color": "#8b8d98", "shape": "square"})
        hub_id = f"{atype}:{h['attribute_id']}"
        shared_by = len(h["accounts"])
        net.add_node(hub_id, label=atype[:3].upper(),
                     color=style["color"], shape=style["shape"],
                     size=min(9 + 1.5 * shared_by, 20),   # capped: a 20-way hub isn't huge
                     title=f"{atype}: {h['attribute_id']}\nshared by {shared_by} accounts")
        for acc in h["accounts"]:
            if acc in members:
                net.add_edge(acc, hub_id, color=style["color"], width=1,
                             title=f"{atype}: {h['attribute_id']}")

    net.set_options(json.dumps(PYVIS_OPTIONS))

    # generate_html across pyvis versions
    try:
        html = net.generate_html(notebook=False)
    except TypeError:
        html = net.generate_html()
    except Exception:
        tmp = os.path.join(tempfile.gettempdir(), f"ring_{ring['ring_id']}.html")
        net.save_graph(tmp)
        html = Path(tmp).read_text()
    return _inject_legend(html), hubs


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

def render_queue(rings: list[dict], threshold: float):
    st.subheader("Flagged rings — triage queue")
    st.caption("Highest risk first. Click **Investigate** to open a ring.")
    for rank, ring in enumerate(rings, start=1):
        c = st.columns([0.5, 2.2, 1.3, 1.3, 1.3, 1.6, 1.2])
        c[0].markdown(f"**#{rank}**")
        c[1].markdown(risk_chip(ring["risk"], threshold), unsafe_allow_html=True)
        c[2].markdown(f"{ring['member_count']} members")
        dev = ring["shared_devices"]
        fun = ring["shared_funding"]
        c[3].markdown(f"🖥 {dev if dev is not None else '—'} dev")
        c[4].markdown(f"💳 {fun if fun is not None else '—'} fund")
        c[5].markdown(f"**{fmt_inr(ring['amount'])}**")
        if c[6].button("Investigate →", key=f"inv_{ring['ring_id']}_{rank}"):
            st.session_state.selected_ring = ring["ring_id"]
            st.rerun()
        st.divider()


def render_investigation(ring: dict, accounts_df, attrs_df, account_scores, threshold):
    top = st.columns([3, 1])
    with top[0]:
        st.subheader(f"Ring {ring['ring_id']}")
    with top[1]:
        if st.button("← Back to queue"):
            st.session_state.selected_ring = None
            st.rerun()

    if ring["member_count"] <= 1:
        st.warning("This ring has a single member — nothing to link. "
                   "Single-account flags are reviewed individually, not as a ring.")
        st.markdown(f"**Risk:** {risk_chip(ring['risk'], threshold)}",
                    unsafe_allow_html=True)
        return

    gcol, ecol = st.columns([2.3, 1])
    with gcol:
        # legend is baked into the graph HTML (build_ring_html), so it travels
        # with the saved sample_output.html and any screenshot.
        html, hubs = build_ring_html(ring, accounts_df, attrs_df, account_scores)
        components.html(html, height=580, scrolling=False)

    with ecol:
        st.markdown("#### Evidence")
        st.markdown(risk_chip(ring["risk"], threshold), unsafe_allow_html=True)
        st.metric("₹ exposed", fmt_inr(ring["amount"]))
        m1, m2 = st.columns(2)
        m1.metric("Members", ring["member_count"])
        m2.metric("Txns", ring["txn_count"] if ring["txn_count"] is not None else "—")

        # shared-attribute breakdown from reconstructed hubs
        try:
            hubs = ring_shared_hubs(ring["members"], attrs_df)
        except Exception:
            hubs = pd.DataFrame()
        if not hubs.empty:
            st.markdown("**Shared attributes**")
            counts = hubs.groupby("attribute_type").size().to_dict()
            for atype, style in ATTR_STYLE.items():
                if atype in counts:
                    st.markdown(
                        f"<span style='color:{style['color']};'>●</span> "
                        f"{atype}: {counts[atype]} shared",
                        unsafe_allow_html=True)

        with st.expander("Member accounts"):
            st.write(", ".join(ring["members"]))


def _candidate_ring_count(meta) -> int | None:
    """Candidate-ring count: from the detector's meta block if present, else the
    graph_build artifact (results/candidate_rings.json) if it's there. Returns
    None when genuinely unavailable — the caller then hides the metric rather
    than rendering a bare '—'."""
    n = meta.get("candidate_rings")
    if isinstance(n, int):
        return n
    art = RESULTS_DIR / "candidate_rings.json"
    if art.exists():
        try:
            data = json.loads(art.read_text())
            if isinstance(data, list):
                return len(data)
        except Exception:
            pass
    return None


def render_summary(rings, meta, accounts_df, threshold):
    scanned = meta.get("accounts_scanned") or (len(accounts_df) if not accounts_df.empty else None)
    candidates = _candidate_ring_count(meta)
    flagged = [r for r in rings if r["flagged"]]
    exposure = sum(r["amount"] for r in flagged)

    # Only render metrics we actually have a value for — never a bare "—".
    cells = []
    if isinstance(scanned, int):
        cells.append(("Accounts scanned", f"{scanned:,}"))
    if candidates is not None:
        cells.append(("Candidate rings", f"{candidates:,}"))
    cells.append(("Flagged rings", str(len(flagged))))
    cells.append(("₹ exposed (flagged)", fmt_inr(exposure)))

    cols = st.columns(len(cells))
    for col, (label, val) in zip(cols, cells):
        col.metric(label, val)


def main():
    st.set_page_config(page_title="Abuse-Ring Sentinel", layout="wide",
                       page_icon="🕸")
    st.session_state.setdefault("selected_ring", None)

    st.title("🕸 Abuse-Ring Sentinel")
    st.caption("Coordinated fraud-ring detection · analyst review console")

    # sidebar: threshold + optional upload (default source is the detector output)
    with st.sidebar:
        st.header("Batch")
        threshold = st.slider("Flag threshold (risk ≥)", 0.0, 1.0,
                              DEFAULT_THRESHOLD, 0.01)
        uploaded = st.file_uploader(
            "Upload results (optional)", type=["json"],
            help="Point at a different batch's detect.py output. By default the "
                 "app loads results/detected_rings.json.")
        st.markdown("---")

    # results source: uploaded JSON if provided, else the default detector output
    if uploaded is not None:
        tmp = Path(tempfile.gettempdir()) / uploaded.name
        tmp.write_bytes(uploaded.getvalue())
        results_path = tmp
    else:
        results_path = RESULTS_DIR / "detected_rings.json"

    if not results_path.exists():
        st.info("No results found. Run `python detect.py` to produce "
                "`results/detected_rings.json`, or upload a JSON from the sidebar.")
        st.stop()

    rings, meta, account_scores = load_results(str(results_path), threshold)
    accounts_df = load_accounts()
    attrs_df = load_shared_attributes()

    n_scored = sum(1 for r in rings if r["risk"] > 0)
    with st.sidebar:
        st.success(f"Loaded `{results_path.name}`")
        st.write(f"{len(rings)} rings in file")
        if account_scores:
            st.caption("Per-account risk scores: loaded")
        # else: node size falls back to ring risk — no caption (don't advertise a gap)

    if n_scored == 0:
        st.warning(
            f"`{results_path.name}` has no risk scores on any ring — this looks "
            "like the pre-scoring **candidate** file, not the `detect.py` output. "
            "Pick the scored results file in the sidebar (the one `detect.py` "
            "wrote with risk + ₹ exposed).")

    flagged = [r for r in rings if r["flagged"]]

    render_summary(rings, meta, accounts_df, threshold)
    st.divider()

    if not flagged:
        st.success("No rings above the current threshold. "
                   "Nothing needs review in this batch. "
                   "Lower the threshold in the sidebar to inspect borderline groups.")
        st.stop()

    # route: queue vs investigation
    sel = st.session_state.selected_ring
    selected = next((r for r in rings if r["ring_id"] == sel), None)
    if selected is not None:
        render_investigation(selected, accounts_df, attrs_df, account_scores, threshold)
    else:
        render_queue(flagged, threshold)


if __name__ == "__main__":
    main()