# Abuse-Ring Sentinel

**Coordinated fraud-ring detection with graph machine learning.**
Razorpay AI Buildathon · Track 02 (AI Risk Manager)

Point it at a batch of accounts. It finds the ones secretly working together — sharing devices, cards, or funding sources — groups them into rings, scores each ring's risk, and lays out the evidence in an interactive graph you can click into.

The problem: fraud is usually a *group* activity. A ring of accounts coordinates, but each account looks perfectly ordinary on its own. Score accounts one at a time and you miss it. The signal lives in the *relationships between accounts* — what they share — and that only shows up when you model the structure connecting them.

<!-- TODO: drop the demo GIF here -->
![Abuse-Ring Sentinel — ranked queue and an expanded investigation graph](results/demo.gif)

---

## What it does

1. Builds a graph from accounts and their shared-attribute records (device / card / IP / phone).
2. Pulls out candidate rings via community detection.
3. Scores every account's fraud risk with a GraphSAGE GNN, then aggregates account scores into a per-ring risk score and ranks the rings.
4. Presents the flagged rings as a ranked triage queue in a Streamlit app — click any ring to see its investigation graph and evidence panel (member accounts, what they share, and ₹ exposed).

An analyst works the queue top-down: highest-risk rings first, weaker evidence naturally surfacing lower.

---

## Results

Held-out node-level detection:

| Metric | Value |
|---|---|
| Precision | 0.776 |
| Recall | 0.937 |
| F1 | 0.849 |
| AUC | 0.983 |

Ring-level: **precision@k ≈ 0.962** — 25 of the top 26 candidate rings are genuine fraud. The worst-ranked real fraud ring still lands at position 28, so a short triage queue catches nearly everything.

The detector cleanly separates fraud rings from legitimate dense groups that *also* share attributes (mean held-out risk): fraud rings **0.81**, vs. hostels **0.40**, offices **0.36**, and background accounts **0.12**. That separation is the whole point — see limitations below.

**Baseline:** a transaction-only model (XGBoost, no graph structure) is the reference point. Per-account features alone were built to be non-separating between fraud and legitimate-dense groups, so a transaction-only model catches materially fewer rings — the lift here comes from graph topology, not from the features.

---

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python data/generate.py          # synthetic accounts + seeded rings + legit-dense groups
python detect.py                 # batch in → ranked flagged rings out (results/detected_rings.json)
streamlit run app.py             # the app
```

`detect.py` runs in under 20 seconds. It writes `results/detected_rings.json`, which the app auto-loads (you can also upload a JSON batch in the app as a fallback).

> **torch-geometric note:** install a CUDA `torch` build first and confirm `torch.cuda.is_available()`, *then* the matching PyG wheels — they're version-pinned to your torch/CUDA. `requirements.txt` deliberately omits `torch-scatter`/`sparse`/`cluster`; install those from the PyG wheel index for your torch version if your setup needs them.

Everything runs locally (developed on an RTX 3050, 6 GB). No cloud.

---

## What's under the hood

```
accounts + shared-attribute records
  → graph build (NetworkX)
  → community detection → candidate rings
  → GraphSAGE scores accounts → aggregate to ring risk → rank
  → Streamlit + pyvis: ranked rings, click-in, evidence, ₹ exposed
```

Detail on why it's modelled as a graph, why GraphSAGE, and why the edges are weighted the way they are is in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

Sample scan: 2,625 accounts scanned · 61 candidate rings · 31 flagged (risk ≥ 0.51) · ₹65.82 Cr exposed.

---

## Limitations

- **Synthetic data.** Accounts, links, and seeded rings are generated locally (`data/generate.py`). The generator deliberately closes the per-account feature gap so the model *has* to rely on graph structure — but it's still synthetic, not production traffic.
- **The hard residual is fraud vs. legit-dense.** Families, hostels, and offices legitimately share devices, cards, and IPs, so they structurally resemble fraud rings. The detector handles the clear cases well; the residual error is a small set of deliberately near-indistinguishable weak rings. That's error-awareness by design, not a defect being hidden.
- **A note on the numbers:** the legit-family separation figure comes from a small sample (n=16) — treat it as directional. The hostel/office gaps rest on firmer samples.

---

## Defense only

This is a **detection** tool. Nothing in this repo aids evasion. It surfaces coordinated abuse for a human reviewer; it does not help anyone avoid detection.