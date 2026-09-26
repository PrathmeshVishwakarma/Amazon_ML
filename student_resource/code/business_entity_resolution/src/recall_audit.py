"""Blocking recall audit at FULL pool scale.

Samples N train S1s, blocks them against the entire S2+S3 pool (per-country
shards, one index build each), and reports recall + avg pairs/S1 at several
top-k values. The max-k shortlist is truncated for smaller k (exact — same
ordering), so one query pass serves all k. Optionally unions the ZIP pass.

Usage (big box, ~1 hr for 20k S1s):
    python -m code.business_entity_resolution.src.recall_audit \
        --train-dir dataset/train --n 20000 --top-ks 20,30,50 --seed 42
    python -m code.business_entity_resolution.src.recall_audit \
        --train-dir dataset/train --n 20000 --top-ks 30 --use-zip-block --seed 42
"""
from __future__ import annotations

import argparse
import time
import numpy as np
import pandas as pd

from .blocking import blocking_recall, fit_index, query_topk, zip_block
from .common import add_block_text, load_ground_truth, load_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--top-ks", default="20,30,50")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-features", type=int, default=300_000)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--block-chunk", type=int, default=5000)
    ap.add_argument("--use-zip-block", action="store_true")
    ap.add_argument("--zip-cap", type=int, default=200)
    args = ap.parse_args()
    ks = sorted({int(x) for x in args.top_ks.split(",") if x})
    k_max = max(ks)

    print("loading full train pool...", flush=True)
    t0 = time.time()
    s1 = load_source(f"{args.train_dir}/train_source1.tsv")
    s2 = load_source(f"{args.train_dir}/train_source2.tsv")
    s3 = load_source(f"{args.train_dir}/train_source3.tsv")
    gt = load_ground_truth(f"{args.train_dir}/train_ground_truth.tsv")
    print(f"loaded in {time.time()-t0:.0f}s: S1={len(s1)} S2={len(s2)} S3={len(s3)}",
          flush=True)

    rng = np.random.RandomState(args.seed)
    ids = s1["entity_id"].to_numpy(copy=True)
    rng.shuffle(ids)
    sample = s1[s1["entity_id"].isin(set(ids[:args.n]))]
    print(f"audit S1s: {len(sample)}", flush=True)
    s23 = pd.concat([s2, s3], ignore_index=True)

    full: dict[str, list[tuple[str, float]]] = {}
    zip_all: dict[str, list[str]] = {}
    for country, g1 in sample.groupby("country", sort=False):
        pool = s23[s23["country"] == country]
        pool = pool if len(pool) else s23
        print(f"[audit] country={country!r}: {len(g1)} queries vs {len(pool)} docs",
              flush=True)
        pool_b = add_block_text(pool)
        vec, mat = fit_index(pool_b["_block"].tolist(),
                             args.max_features, args.min_df)
        print(f"[audit:{country}] index: {mat.shape[0]} docs, {mat.shape[1]} terms",
              flush=True)
        g1b = add_block_text(g1)
        res = query_topk(vec, mat, pool_b["entity_id"].to_numpy(),
                         g1b["entity_id"].to_numpy(), g1b["_block"].tolist(),
                         k_max, args.block_chunk, True, f"audit-{country}")
        full.update(res)
        if args.use_zip_block:
            # Kept SEPARATE from the TF-IDF list: union happens after
            # per-k truncation below (appending here would put ZIP picks
            # past the slice point and silently drop them).
            for sid, cids in zip_block(g1, pool, cap=args.zip_cap).items():
                zip_all[sid] = list(dict.fromkeys(cids))
        del vec, mat, res

    print(f"{'top_k':>6} {'recall':>8} {'avg_pairs':>10}", flush=True)
    for k in ks:
        trunc: dict[str, list[tuple[str, float]]] = {}
        for sid, pairs in full.items():
            base = list(pairs[:k])
            if args.use_zip_block:
                seen = {c for c, _ in base}
                for cid in zip_all.get(sid, ()):
                    if cid not in seen:
                        base.append((cid, 0.0))
                        seen.add(cid)
            trunc[sid] = base
        r = blocking_recall(trunc, gt)
        avg = sum(len(v) for v in trunc.values()) / max(len(trunc), 1)
        print(f"{k:>6} {r:>8.4f} {avg:>10.1f}", flush=True)


if __name__ == "__main__":
    main()
