# Architecture

How Abuse-Ring Sentinel goes from a batch of accounts to a ranked queue of fraud rings, and why each piece is built the way it is.

```
accounts.csv + shared_attributes.csv
        │
        ▼
  graph build (NetworkX)                    src/graph_build.py
   accounts → nodes
   shared device/card/IP/phone → weighted edges
        │
        ▼
  community detection → candidate rings     src/graph_build.py
   connected communities become the discrete
   groups we score and display
        │
        ▼
  GraphSAGE → per-account risk              src/detector.py
   node classifier over the graph
        │
        ▼
  aggregate → ring risk → rank              src/detector.py
   account scores rolled up per ring
        │
        ▼
  ranked flagged rings + evidence           detect.py → results/detected_rings.json
        │
        ▼
  Streamlit + pyvis app                     app.py
   triage queue · click-in graph · evidence · ₹ exposed
```

The sample scan: 2,625 accounts → a graph of 2,625 nodes / 2,916 edges (1,975 background accounts sit isolated with no shared-attribute links) → 61 candidate rings → 31 flagged above the risk threshold.

---

## Why a graph

Fraud here is coordinated. A ring is a set of accounts that individually look normal — ordinary transaction counts, ordinary amounts, ordinary ages — but that are quietly tied together by shared infrastructure: the same device fingerprint, the same funding card, the same IP.

A per-account model never sees that tie. It has one account's features in front of it at a time, and by construction those features don't separate a fraud-ring member from a legitimate account. The discriminating signal isn't a property of any account; it's a property of the *relationships* between accounts. To use that signal you have to represent the relationships explicitly — which is a graph: accounts as nodes, shared attributes as edges.

This is also why the transaction-only XGBoost baseline is the honest comparison point. The synthetic data is generated so that per-account features alone can't tell fraud from legit-dense. Any lift the graph model shows over that baseline is lift that comes specifically from topology — exactly the claim the project is making.

---

## Why GraphSAGE

Once accounts are nodes and shared attributes are edges, the detector needs a model that learns from a node's *neighbourhood*, not just its own features. GraphSAGE does exactly this: it computes each account's representation by aggregating features from the accounts it's connected to, so an account sitting inside a tightly shared cluster gets a representation shaped by that cluster.

It fits the constraints well:

- **Inductive and local.** It aggregates over a node's local neighbourhood rather than requiring the whole graph to be fixed and known up front, which keeps it fast and keeps training tractable on a 6 GB laptop GPU.
- **Small graph, quick training.** ~2.6k nodes and ~2.9k edges train locally in minutes on an RTX 3050 — no cloud needed.
- **The right inductive bias.** "You are shaped by your neighbours" is precisely the assumption fraud-ring detection wants: a normal-looking account surrounded by shared-device links to other flagged accounts should itself look risky.

The model uses a weighted SAGE convolution so edges can carry different importance (see below), class-weighted training to handle the fraud/non-fraud imbalance, validation-based early stopping, and an F1-maximizing threshold on the held-out slice.

---

## Why the edges are weighted — IP down-weighting

Not all shared attributes mean the same thing.

Sharing a **device** or a **funding source (card)** is a strong signal of coordination — people don't usually share a phone's device fingerprint or a payment card by accident. Sharing an **IP** is weak: offices, hostels, cafés, and home Wi-Fi all put unrelated people behind one address. Shared **phone** sits in between.

If every edge counted equally, IP edges would dominate the message-passing and drag in huge numbers of innocent co-located accounts — the graph would light up every office and hostel as a ring. So device and funding edges are weighted **above** IP and phone edges. The GraphSAGE aggregation respects those weights, so a shared card pulls two accounts together far harder than a shared IP does.

This is the single most important tuning lever in the model. IP edges are the large majority of all edges but the weakest evidence; down-weighting them is what lets the detector separate a genuine device+funding ring from a hostel full of people on one router.

**A correctness note that matters here:** the weighted aggregation normalizes by the *sum of edge weights*, not by node degree. Normalizing by degree would silently cancel the down-weighting for any node whose edges are all the same type — the weights would divide back out. Normalizing by summed weight preserves the intended down-weighting in every case.

---

## Why softened class weighting

Fraud accounts are the minority, so the training loss is class-weighted to stop the model from collapsing to "predict everything legitimate." But a *hard* inverse-frequency weight over-corrects: it pushes the model to flag aggressively, drowning the queue in false positives.

The class weighting is therefore **softened** — enough to make the model take the minority class seriously, not so much that precision falls apart. The result is the operating point we actually want for a triage tool: recall 0.937 (we catch almost every real ring) at precision 0.776 (a tolerable false-positive rate, since ranking — not filtering — handles the false positives). The threshold of 0.51 is tuned on held-out data to maximize F1 at this balance.

---

## How account scores become ring scores

GraphSAGE produces a fraud-risk score per account. Rings are the unit an analyst acts on, so those account scores are aggregated up to a single risk score per candidate ring, and the rings are ranked by it.

Because a ring's score summarizes its members, a ring of uniformly high-risk accounts ranks above a ring with one suspicious account and several ambiguous ones — which is the right ordering for triage. It also means the ranking degrades gracefully: as you move down the queue, the evidence thins out and the false positives concentrate at the bottom, where an analyst can stop reading.

---

## What the metrics mean

- **Node-level (precision 0.776 / recall 0.937 / F1 0.849 / AUC 0.983):** how well the GNN scores individual accounts on the held-out slice. High recall is the priority for a detector — missing a ring is worse than a review of a false one. AUC 0.983 says the score ordering is strong almost regardless of where you set the threshold.
- **Ring-level precision@k ≈ 0.962:** of the top 26 candidate rings by risk, 25 are genuine fraud. This is the number that matters operationally — it's what an analyst experiences working the queue top-down. The worst real fraud ring still ranks 28th, so a short queue misses very little.
- **Fraud vs. legit-dense separation (mean held-out risk — fraud 0.81, hostel 0.40, office 0.36, background 0.12):** this is the hard part of the problem stated as a number. Legit-dense groups share attributes too, so the meaningful test isn't "fraud vs. random account" (easy) but "fraud vs. a hostel" (hard). A ~0.4-point gap on that comparison is the detector doing its actual job. (The legit-family figure of 0.46 comes from a small sample, n=16 — read it as directional, not precise.)

The residual difficulty is a small set of deliberately near-indistinguishable weak rings: rings whose only shared attribute is a weak one, engineered to sit right on the fraud/legit-dense boundary. The detector's misses concentrate there. That's designed-in, and surfacing it honestly is part of the point.

---

## Defense-only by design

Every component detects and explains coordinated abuse for a human reviewer. Nothing scores or suggests how to *avoid* detection. The tool's only output is a ranked, evidenced queue for triage.