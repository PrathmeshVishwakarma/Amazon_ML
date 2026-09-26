"""Inference: block test S1s -> score with LightGBM -> threshold -> write outputs.

Produces BOTH required files:
  output/candidate_pairs.tsv   (blocking set, pre-model)
  output/matching_results.tsv  (post-threshold finals, subset of candidates)

Usage:
    python -m code.business_entity_resolution.src.infer \
        --test-dir dataset/test --model models/lgbm.txt \
        --threshold 0.65 --out output --top-k 30
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from .blocking import fit_index, query_topk, zip_block
from .common import add_block_text, load_source
from .features import pair_features


def write_tsv(path: str, header: str, rows: dict[str, list[str]], all_ids: list[str]) -> None:
    with open(path, "w") as f:
        f.write(header + "\n")
        for sid in all_ids:
            ids = sorted(set(rows.get(sid, [])))
            f.write(f"{sid}\t{','.join(ids)}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--chunk-s1", type=int, default=100000,
                    help="S1 rows per blocking pass (RAM control).")
    ap.add_argument("--max-features", type=int, default=300_000,
                    help="TF-IDF vocab cap. Lower (e.g. 50000) to fit low-RAM boxes.")
    ap.add_argument("--min-df", type=int, default=2,
                    help="TF-IDF min document frequency. Raise (e.g. 5) on low-RAM boxes.")
    ap.add_argument("--block-chunk", type=int, default=2000,
                    help="Queries per sparse multiply. Lower (e.g. 1000) if a "
                         "full-vocab shard still OOMs.")
    ap.add_argument("--use-zip-block", action="store_true",
                    help="Union same-ZIP candidates with the TF-IDF shortlist.")
    ap.add_argument("--zip-cap", type=int, default=200,
                    help="Max candidates per ZIP in the ZIP-block pass.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("loading test...", flush=True)
    s1 = load_source(f"{args.test_dir}/test_source1.tsv")
    s2 = load_source(f"{args.test_dir}/test_source2.tsv")
    s3 = load_source(f"{args.test_dir}/test_source3.tsv")
    s23 = pd.concat([s2, s3], ignore_index=True)
    cmap = {r.entity_id: r for r in s23.itertuples()}
    model = lgb.Booster(model_file=args.model)

    all_ids = s1["entity_id"].tolist()
    cand_rows: dict[str, list[str]] = {}
    match_rows: dict[str, list[str]] = {}

    # One index build per country, then stream that country's S1s through it.
    # (Previously the index was rebuilt for every S1 chunk — same results,
    # ~10x slower on full-scale data.)
    for country, g1 in s1.groupby("country", sort=False):
        g23 = s23[s23["country"] == country]
        pool = g23 if len(g23) else s23
        print(f"[infer] country={country!r}: {len(g1)} queries vs "
              f"{len(pool)} candidates", flush=True)
        pool = add_block_text(pool)
        try:
            vec, mat = fit_index(pool["_block"].tolist(),
                                 args.max_features, args.min_df)
        except ValueError:
            for sid in g1["entity_id"]:
                cand_rows[str(sid)] = []
                match_rows[str(sid)] = []
            continue
        cand_ids = pool["entity_id"].to_numpy()
        g1 = add_block_text(g1)
        zb_all: dict[str, list[str]] = {}
        if args.use_zip_block:
            zb_all = zip_block(g1, pool, cap=args.zip_cap)
        for start in range(0, len(g1), args.chunk_s1):
            sub = g1.iloc[start:start + args.chunk_s1]
            print(f"[infer:{country}] block+score {start}/{len(g1)}...", flush=True)
            cand = query_topk(vec, mat, cand_ids, sub["entity_id"].to_numpy(),
                              sub["_block"].tolist(), args.top_k,
                              args.block_chunk, True, str(country))
            if args.use_zip_block:
                for sid, pairs in cand.items():
                    seen = {c for c, _ in pairs}
                    for cid in zb_all.get(sid, ()):
                        if cid not in seen:
                            pairs.append((cid, 0.0))
                            seen.add(cid)
            for sid in sub["entity_id"]:
                cand_rows[str(sid)] = [c for c, _ in cand.get(str(sid), [])]
            s1map = {r.entity_id: r for r in sub.itertuples()}
            Xs, keys = [], []
            for sid, pairs in cand.items():
                srow = s1map.get(sid)
                for cid, tscore in pairs:
                    crow = cmap.get(cid)
                    if crow is None:
                        continue
                    Xs.append(pair_features(
                        srow.business_name, srow.business_address, srow.country,
                        crow.business_name, crow.business_address, crow.country, tscore))
                    keys.append((sid, cid))
            if Xs:
                probs = model.predict(np.array(Xs, dtype=np.float32))
                for (sid, cid), p in zip(keys, probs):
                    if float(p) >= args.threshold:
                        match_rows.setdefault(sid, []).append(cid)
            for sid in sub["entity_id"]:
                match_rows.setdefault(str(sid), [])
                cand_rows.setdefault(str(sid), [])

    write_tsv(f"{args.out}/candidate_pairs.tsv",
              "source1_entity_id\tcandidate_entity_ids", cand_rows, all_ids)
    write_tsv(f"{args.out}/matching_results.tsv",
              "source1_entity_id\tmatched_entity_ids", match_rows, all_ids)
    n_match = sum(len(v) for v in match_rows.values())
    n_cand = sum(len(v) for v in cand_rows.values())
    print(f"done. S1={len(all_ids)} cand_pairs={n_cand} matches={n_match}", flush=True)


if __name__ == "__main__":
    main()
