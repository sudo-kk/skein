"""
Synthetic fraud-ring injection for the IEEE-CIS Fraud Detection dataset.

Real fraud-ring data doesn't come pre-labeled - nobody hands you a column
that says "these five accounts are secretly the same person." So this
script manufactures ground truth: it clones real transactions into small
clusters that share a device/card fingerprint and fire in a tight time
window, the way a coordinated ring actually behaves. That gives you
labeled rings to train and evaluate a graph-based detector against.

Expected input columns (standard IEEE-CIS naming):
    TransactionID, TransactionDT, TransactionAmt, isFraud,
    card1, card2, addr1, addr2, DeviceType, DeviceInfo
If you pass --identity, it's left-joined onto the transaction table on
TransactionID before injection (DeviceType/DeviceInfo live there).

Usage:
    python inject_rings.py \
        --transaction train_transaction.csv \
        --identity train_identity.csv \
        --out augmented_with_rings.csv \
        --n-rings 40
"""

import argparse

import numpy as np
import pandas as pd

# Columns that actually tie a ring together - the "fingerprint" the ring
# shares across its supposedly-unrelated accounts.
DEFAULT_LINK_COLS = ["DeviceType", "DeviceInfo"]

# Columns that get perturbed per clone so each ring member still *looks*
# like a different account/card, unless they're used as a link column.
VARY_CANDIDATES = ["card1", "card2", "addr1", "addr2"]


def load_data(transaction_path, identity_path=None):
    tx = pd.read_csv(transaction_path)
    if identity_path:
        idn = pd.read_csv(identity_path)
        tx = tx.merge(idn, on="TransactionID", how="left")
    return tx


def pick_seed_rows(df, link_cols, n_rings, rng, rarity_max_count=50):
    """Seeds need a real (non-null) value in every link column, and ideally
    a *rare* one. Injecting a ring on top of a common value like
    DeviceInfo='Windows' is a weak, unrealistic signal - thousands of
    unrelated people share that. Rare combos make the ring's shared
    attribute actually distinctive, which is what a real ring looks like."""
    valid = df.dropna(subset=link_cols)
    if valid.empty:
        raise ValueError(
            f"No rows have values in all of {link_cols}. "
            "Did you merge the identity table with --identity?"
        )
    combo_counts = valid.groupby(link_cols)["TransactionID"].transform("count")
    candidates = valid[combo_counts <= rarity_max_count]
    if len(candidates) < n_rings:
        candidates = valid  # fall back if too few rare combos exist
    n = min(n_rings, len(candidates))
    return candidates.sample(n=n, random_state=int(rng.integers(1e9))).reset_index(drop=True)


def make_ring(seed_row, ring_id, ring_size, link_cols, dt_window_seconds,
              amt_jitter, next_txn_id, rng):
    """Clone one seed transaction into `ring_size` near-duplicates: same
    link_cols (the shared fingerprint), different vary-columns (so each
    looks like a separate account), all inside a tight time window
    (the velocity signal a spike detector should also catch)."""
    base_dt = seed_row["TransactionDT"]
    vary_cols = [c for c in VARY_CANDIDATES if c in seed_row.index and c not in link_cols]

    rows = []
    for i in range(ring_size):
        row = seed_row.copy()
        row["TransactionID"] = next_txn_id + i
        row["TransactionDT"] = base_dt + int(rng.integers(0, dt_window_seconds + 1))
        row["TransactionAmt"] = round(
            max(1.0, row["TransactionAmt"] * rng.uniform(1 - amt_jitter, 1 + amt_jitter)), 2
        )
        for col in vary_cols:
            if pd.notna(row[col]):
                row[col] = row[col] + rng.integers(1000, 9000)
        row["ring_id"] = ring_id
        row["is_injected_ring"] = True
        row["isFraud"] = 1
        rows.append(row)
    return pd.DataFrame(rows)


def inject_rings(df, n_rings=40, ring_size_range=(3, 7), dt_window_seconds=180,
                  amt_jitter=0.15, link_cols=None, seed=42):
    link_cols = [c for c in (link_cols or DEFAULT_LINK_COLS) if c in df.columns]
    if not link_cols:
        raise ValueError(
            "None of the requested link columns are in this dataframe - "
            "check the identity table got merged, or pass --link-cols explicitly."
        )

    rng = np.random.default_rng(seed)
    df = df.copy()
    df["ring_id"] = -1
    df["is_injected_ring"] = False

    seeds = pick_seed_rows(df, link_cols, n_rings, rng)
    next_id = int(df["TransactionID"].max()) + 1
    injected_chunks = []

    for ring_id, (_, seed_row) in enumerate(seeds.iterrows()):
        size = int(rng.integers(ring_size_range[0], ring_size_range[1] + 1))
        ring_df = make_ring(seed_row, ring_id, size, link_cols,
                             dt_window_seconds, amt_jitter, next_id, rng)
        injected_chunks.append(ring_df)
        next_id += size

    injected_df = pd.concat(injected_chunks, ignore_index=True)
    out = pd.concat([df, injected_df], ignore_index=True)
    return out, link_cols


def summarize(out, link_cols):
    ring_mask = out["is_injected_ring"]
    n_rings = out.loc[ring_mask, "ring_id"].nunique()
    n_txn = int(ring_mask.sum())
    sizes = out.loc[ring_mask].groupby("ring_id").size()
    print(f"Linked on: {link_cols}")
    print(f"Injected {n_rings} rings across {n_txn} synthetic transactions "
          f"(avg size {sizes.mean():.1f}, range {sizes.min()}-{sizes.max()})")
    print(f"Total rows: {len(out)}  |  isFraud=1 rows: {int(out['isFraud'].sum())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transaction", required=True, help="path to train_transaction.csv")
    ap.add_argument("--identity", default=None, help="path to train_identity.csv (optional)")
    ap.add_argument("--out", default="augmented_with_rings.csv")
    ap.add_argument("--n-rings", type=int, default=40)
    ap.add_argument("--min-ring-size", type=int, default=3)
    ap.add_argument("--max-ring-size", type=int, default=7)
    ap.add_argument("--dt-window-seconds", type=int, default=180,
                     help="how tightly clustered in time each ring's transactions are")
    ap.add_argument("--amt-jitter", type=float, default=0.15,
                     help="fractional random variation applied to cloned amounts")
    ap.add_argument("--link-cols", nargs="*", default=None,
                     help="columns that define the shared ring fingerprint "
                          "(default: DeviceType DeviceInfo)")
    ap.add_argument("--rarity-max-count", type=int, default=50,
                     help="only seed rings from link-col combos this common or rarer")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_data(args.transaction, args.identity)
    out, link_cols = inject_rings(
        df,
        n_rings=args.n_rings,
        ring_size_range=(args.min_ring_size, args.max_ring_size),
        dt_window_seconds=args.dt_window_seconds,
        amt_jitter=args.amt_jitter,
        link_cols=args.link_cols,
        seed=args.seed,
    )
    summarize(out, link_cols)
    out.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
