"""
Abuse-Ring Sentinel — sample data generator (Day 1)

Produces the believable sample dataset the whole product runs on:
  - accounts.csv          node table (features + fraud label + group provenance)
  - shared_attributes.csv account -> device/funding/ip/phone (edges come from co-sharing)
  - rings.csv             ground-truth groups, with the fields a ring card renders

Three populations:
  1. Fraud rings          (fraud=1) accounts coordinating via shared devices/funding
  2. Legit-but-dense      (fraud=0) families / hostels / offices that ALSO share attrs
  3. Background            (fraud=0) sparse, mostly unconnected accounts

The legit-dense groups exist to create genuine ambiguity: some of them (hostels on a
single WiFi IP; families on a joint card) look structurally ring-like. Fraud is NOT
made cartoonishly separable from them.

Usage:
    python data/generate.py --seed 42
    python data/generate.py --seed 42 --out data --n-fraud-rings 25 --n-legit-groups 30 --n-background 2000

Reproducible: a single seeded numpy Generator drives every draw, and "now" is a fixed
reference date, so identical seeds produce byte-identical CSVs.
"""

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Fixed reference "now" so signup_timestamp is deterministic across runs/days.
REFERENCE_NOW = datetime(2026, 1, 1, 0, 0, 0)

ATTR_TYPES = ["device", "funding", "ip", "phone"]


# --------------------------------------------------------------------------- #
# ID generation
# --------------------------------------------------------------------------- #
class IdGen:
    """Globally-unique, human-readable ids per kind."""

    PREFIX = {
        "device": "DEV", "funding": "CARD", "ip": "IP", "phone": "PHN",
        "account": "ACC", "ring": "GRP",
    }
    WIDTH = {"account": 6}  # everything else = 5

    def __init__(self):
        self._n = {k: 0 for k in self.PREFIX}

    def new(self, kind):
        self._n[kind] += 1
        w = self.WIDTH.get(kind, 5)
        return f"{self.PREFIX[kind]}_{self._n[kind]:0{w}d}"

    def pool(self, kind, n):
        return [self.new(kind) for _ in range(max(1, int(n)))]


# --------------------------------------------------------------------------- #
# Feature draws
# --------------------------------------------------------------------------- #
def draw_features(rng, n, age_lo, age_hi, txn_lo, txn_hi, amt_lo, amt_hi,
                  coordinated=False):
    """Return per-account (age_days, txn_count, avg_amount, total_amount).

    coordinated=True clusters ages in a tight window (shared onboarding burst),
    which is the fraud-ring signal; legit groups leave ages spread out.
    """
    if coordinated:
        base = rng.integers(age_lo, age_hi + 1)
        window = max(1, (age_hi - age_lo) // 6)
        age = np.clip(base + rng.integers(0, window + 1, size=n), 1, None)
    else:
        age = rng.integers(age_lo, age_hi + 1, size=n)

    txn_count = rng.integers(txn_lo, txn_hi + 1, size=n)
    # log-uniform amounts so the tail is heavy but bounded
    avg_amount = np.round(np.exp(rng.uniform(np.log(amt_lo), np.log(amt_hi), size=n)), 2)
    total_amount = np.round(txn_count * avg_amount, 2)
    return age.astype(int), txn_count.astype(int), avg_amount, total_amount


def signup_ts(rng, age_days):
    dt = (REFERENCE_NOW
          - timedelta(days=int(age_days))
          - timedelta(hours=int(rng.integers(0, 24)),
                      minutes=int(rng.integers(0, 60))))
    return dt.isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Attribute assignment
# --------------------------------------------------------------------------- #
def assign_type(rng, members, kind, pool, coverage, acct_attrs, guarantee=False):
    """Point some members at a shared pool of `kind` ids.

    Members not covered keep their pre-assigned unique id. `guarantee` forces at
    least two members onto pool[0] so the group has a genuinely shared attribute.
    """
    for m in members:
        if rng.random() < coverage:
            acct_attrs[m][kind] = str(rng.choice(pool))
    if guarantee and len(members) >= 2:
        acct_attrs[members[0]][kind] = pool[0]
        acct_attrs[members[1]][kind] = pool[0]


def count_shared(members, kind, acct_attrs):
    """Number of distinct `kind` ids linked to >= 2 of these members."""
    counts = {}
    for m in members:
        aid = acct_attrs[m][kind]
        counts[aid] = counts.get(aid, 0) + 1
    return sum(1 for c in counts.values() if c >= 2)


# --------------------------------------------------------------------------- #
# Population builders
# --------------------------------------------------------------------------- #
def new_account(rng, idg, accounts, acct_attrs, feats, fraud, group_id, group_type):
    aid = idg.new("account")
    age, txn, avg, total = feats
    accounts.append({
        "account_id": aid,
        "txn_count": int(txn),
        "total_amount": float(total),
        "avg_amount": float(avg),
        "account_age_days": int(age),
        "signup_timestamp": signup_ts(rng, age),
        "fraud": int(fraud),
        "group_id": group_id if group_id else "",
        "group_type": group_type,
    })
    # every account owns a unique id of each type by default; groups override some
    acct_attrs[aid] = {k: idg.new(k) for k in ATTR_TYPES}
    return aid


def build_fraud_ring(rng, idg, accounts, acct_attrs):
    """One coordinated fraud ring. Intensity controls how tight the sharing and
    the feature separation are — weak rings are deliberately legit-looking."""
    n = int(rng.integers(4, 21))
    # Two independent axes:
    #   intensity     -> how tight the sharing is + transaction amounts
    #   age_archetype -> the ring's history (fresh burst vs seasoned mules)
    # Decoupling them is deliberate: a seasoned ring looks totally ordinary on node
    # features (old accounts, normal activity), so its ONLY tell is the graph
    # structure (shared devices/funding). That's what forces the GNN to earn its keep
    # instead of the age feature giving the answer away.
    intensity = rng.choice(["weak", "medium", "strong"], p=[0.30, 0.40, 0.30])
    age_archetype = rng.choice(["fresh", "seasoned", "mixed"], p=[0.30, 0.45, 0.25])
    gid = idg.new("ring")

    # amounts by intensity
    amt = {"strong": (2000, 40000), "medium": (800, 25000), "weak": (300, 12000)}[intensity]
    # age + history by archetype. fresh = newly created coordinated burst;
    # seasoned = aged/dormant accounts activated later (very common in real fraud, and
    # indistinguishable from legit on age); mixed = aged mules + a few fresh recruits.
    # seasoned/mixed accrue more history, so their txn_count is higher too.
    age_prof = {
        "fresh":    dict(age=(1, 220),    txn=(5, 160),  coord=True),
        "seasoned": dict(age=(300, 1400), txn=(40, 480), coord=False),
        "mixed":    dict(age=(20, 1400),  txn=(15, 420), coord=False),
    }[age_archetype]
    age, txn, avg, total = draw_features(
        rng, n, *age_prof["age"], *age_prof["txn"], *amt, coordinated=age_prof["coord"])

    members = [new_account(rng, idg, accounts, acct_attrs,
                           (age[i], txn[i], avg[i], total[i]),
                           fraud=1, group_id=gid, group_type="fraud_ring")
               for i in range(n)]

    # sharing density: strong = small pools (heavy reuse), weak = loose
    div = {"strong": (4, 5, 1, 2), "medium": (3, 4, 1, 1.5), "weak": (1.5, 2, 2, 1.3)}[intensity]
    dev_pool  = idg.pool("device",  int(np.ceil(n / div[0])))
    fund_pool = idg.pool("funding", int(np.ceil(n / div[1])))
    ip_pool   = idg.pool("ip",      int(div[2]))
    phn_pool  = idg.pool("phone",   int(np.ceil(n / div[3])))

    cov = {"strong": 1.0, "medium": 0.9, "weak": 0.65}[intensity]
    assign_type(rng, members, "device",  dev_pool,  cov,       acct_attrs, guarantee=True)
    assign_type(rng, members, "funding", fund_pool, cov,       acct_attrs, guarantee=True)
    assign_type(rng, members, "ip",      ip_pool,   cov * 0.8, acct_attrs)
    assign_type(rng, members, "phone",   phn_pool,  cov * 0.5, acct_attrs)

    return record_group(members, gid, "fraud_ring", intensity, fraud=1,
                        accounts=accounts, acct_attrs=acct_attrs,
                        age_archetype=age_archetype)


def build_legit_group(rng, idg, accounts, acct_attrs):
    """A dense-but-legit group. Shares infrastructure (home/office IP, a shared
    device, occasionally a joint card) without being fraud — the false-positive trap."""
    archetype = rng.choice(["family", "hostel", "office"], p=[0.4, 0.35, 0.25])
    size_rng = {"family": (3, 6), "hostel": (8, 20), "office": (6, 15)}[archetype]
    n = int(rng.integers(size_rng[0], size_rng[1] + 1))
    gid = idg.new("ring")

    # ~30% of legit groups are newly formed (new hostel intake, new office team,
    # newlyweds) so legit has a young cohort too — otherwise "young == fraud" and the
    # age feature leaks the label. Established groups skew old. Never coordinated.
    if rng.random() < 0.30:
        age_archetype, lage = "new", (5, 250)
    else:
        age_archetype, lage = "established", (200, 1400)
    age, txn, avg, total = draw_features(
        rng, n, *lage, 5, 400, 200, 30000, coordinated=False)
    members = [new_account(rng, idg, accounts, acct_attrs,
                           (age[i], txn[i], avg[i], total[i]),
                           fraud=0, group_id=gid, group_type=f"legit_{archetype}")
               for i in range(n)]

    if archetype == "family":
        assign_type(rng, members, "ip", idg.pool("ip", 1), 1.0, acct_attrs, guarantee=True)
        assign_type(rng, members, "device", idg.pool("device", 1), 0.4, acct_attrs)
        if rng.random() < 0.3:  # ~30% of families share a joint card -> ambiguity
            assign_type(rng, members, "funding", idg.pool("funding", 1), 0.5, acct_attrs)
    elif archetype == "hostel":
        # very dense on IP (shared WiFi) — structurally looks ring-like but is legit
        assign_type(rng, members, "ip", idg.pool("ip", int(rng.integers(1, 3))), 0.9,
                    acct_attrs, guarantee=True)
        assign_type(rng, members, "device", idg.pool("device", int(rng.integers(2, 4))),
                    0.4, acct_attrs)
    else:  # office
        assign_type(rng, members, "ip", idg.pool("ip", 1), 0.9, acct_attrs, guarantee=True)
        assign_type(rng, members, "device", idg.pool("device", 1), 0.3, acct_attrs)

    return record_group(members, gid, f"legit_{archetype}", archetype, fraud=0,
                        accounts=accounts, acct_attrs=acct_attrs,
                        age_archetype=age_archetype)


def build_background(rng, idg, accounts, acct_attrs, n):
    age, txn, avg, total = draw_features(
        rng, n, 30, 1500, 1, 500, 100, 40000, coordinated=False)
    for i in range(n):
        new_account(rng, idg, accounts, acct_attrs,
                    (age[i], txn[i], avg[i], total[i]),
                    fraud=0, group_id="", group_type="background")


def add_background_noise(rng, idg, accounts, acct_attrs, n_public_ips=4):
    """Realistic noise: a few public IPs (cafe/carrier NAT) shared by random
    strangers, so not every shared-attribute edge implies a ring."""
    bg_ids = [a["account_id"] for a in accounts if a["group_type"] == "background"]
    if not bg_ids:
        return
    for _ in range(n_public_ips):
        pub = idg.new("ip")
        k = int(rng.integers(5, 16))
        for m in rng.choice(bg_ids, size=min(k, len(bg_ids)), replace=False):
            acct_attrs[m]["ip"] = pub


# --------------------------------------------------------------------------- #
# Ring / group ground-truth record
# --------------------------------------------------------------------------- #
def record_group(members, gid, group_type, archetype, fraud, accounts, acct_attrs,
                 age_archetype=""):
    by_id = {a["account_id"]: a for a in accounts}
    total_txn = sum(by_id[m]["txn_count"] for m in members)
    total_amt = sum(by_id[m]["total_amount"] for m in members)
    return {
        "group_id": gid,
        "group_type": group_type,
        "archetype": archetype,
        "age_archetype": age_archetype,
        "fraud": int(fraud),
        "n_members": len(members),
        "member_account_ids": "|".join(members),
        "n_shared_devices": count_shared(members, "device", acct_attrs),
        "n_shared_funding": count_shared(members, "funding", acct_attrs),
        "n_shared_ip": count_shared(members, "ip", acct_attrs),
        "n_shared_phone": count_shared(members, "phone", acct_attrs),
        "total_txn_count": int(total_txn),
        "total_amount_exposed": round(float(total_amt), 2),
    }


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def fmt_inr(x):
    if x >= 1e7:
        return f"₹{x / 1e7:.2f}Cr"
    if x >= 1e5:
        return f"₹{x / 1e5:.2f}L"
    if x >= 1e3:
        return f"₹{x / 1e3:.1f}K"
    return f"₹{x:.0f}"


def attr_records(acct_attrs):
    rows = []
    for aid, attrs in acct_attrs.items():
        for kind, attr_id in attrs.items():
            rows.append({"account_id": aid, "attribute_type": kind, "attribute_id": attr_id})
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Abuse-Ring Sentinel sample data generator")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--n-fraud-rings", type=int, default=25)
    ap.add_argument("--n-legit-groups", type=int, default=30)
    ap.add_argument("--n-background", type=int, default=2000)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    idg = IdGen()

    accounts, acct_attrs, rings = [], {}, []

    for _ in range(args.n_fraud_rings):
        rings.append(build_fraud_ring(rng, idg, accounts, acct_attrs))
    for _ in range(args.n_legit_groups):
        rings.append(build_legit_group(rng, idg, accounts, acct_attrs))
    build_background(rng, idg, accounts, acct_attrs, args.n_background)
    add_background_noise(rng, idg, accounts, acct_attrs)

    accounts_df = pd.DataFrame(accounts).sort_values("account_id").reset_index(drop=True)
    attrs_df = pd.DataFrame(attr_records(acct_attrs)).sort_values(
        ["account_id", "attribute_type"]).reset_index(drop=True)
    rings_df = pd.DataFrame(rings).sort_values("group_id").reset_index(drop=True)

    os.makedirs(args.out, exist_ok=True)
    accounts_df.to_csv(os.path.join(args.out, "accounts.csv"), index=False)
    attrs_df.to_csv(os.path.join(args.out, "shared_attributes.csv"), index=False)
    rings_df.to_csv(os.path.join(args.out, "rings.csv"), index=False)

    # ---- summary ----
    n_acc = len(accounts_df)
    n_fraud_acc = int(accounts_df["fraud"].sum())
    fraud_rings = rings_df[rings_df["fraud"] == 1]
    legit_groups = rings_df[rings_df["fraud"] == 0]
    total_exposed = float(fraud_rings["total_amount_exposed"].sum())

    print("=" * 60)
    print("Abuse-Ring Sentinel — sample data generated")
    print("=" * 60)
    print(f"seed              : {args.seed}")
    print(f"output dir        : {os.path.abspath(args.out)}")
    print(f"accounts          : {n_acc}  "
          f"(fraud {n_fraud_acc} = {n_fraud_acc / n_acc:.1%}, "
          f"legit {n_acc - n_fraud_acc})")
    print(f"shared-attr rows  : {len(attrs_df)}")
    print(f"fraud rings       : {len(fraud_rings)}  "
          f"(sizes {int(fraud_rings['n_members'].min())}–{int(fraud_rings['n_members'].max())})")
    print(f"legit-dense groups: {len(legit_groups)}")
    print(f"total ₹ exposed   : {fmt_inr(total_exposed)}  (raw {total_exposed:,.0f})")
    print("-" * 60)
    print("fraud rings by intensity:")
    print(fraud_rings["archetype"].value_counts().to_string())
    print("legit groups by archetype:")
    print(legit_groups["archetype"].value_counts().to_string())
    print("-" * 60)
    print("feature overlap (median fraud vs legit — smaller Δ = less label leakage):")
    for col in ["account_age_days", "txn_count", "avg_amount"]:
        mf = accounts_df.loc[accounts_df.fraud == 1, col].median()
        ml = accounts_df.loc[accounts_df.fraud == 0, col].median()
        print(f"  {col:<17} fraud {mf:>9.0f}   legit {ml:>9.0f}   Δ {abs(mf - ml):>8.0f}")
    print("-" * 60)
    print("sample fraud ring (card fields):")
    top = fraud_rings.sort_values("total_amount_exposed", ascending=False).iloc[0]
    print(f"  {top['group_id']}: {top['n_members']} accounts | "
          f"{top['n_shared_devices']} shared devices | "
          f"{top['n_shared_funding']} shared funding | "
          f"{top['total_txn_count']} txns | "
          f"{fmt_inr(top['total_amount_exposed'])} exposed")
    print("=" * 60)


if __name__ == "__main__":
    main()