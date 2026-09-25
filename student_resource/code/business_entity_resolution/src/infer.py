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

from .blocking import block_all
from .common import load_source
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

    for start in range(0, len(s1), args.chunk_s1):
        chunk = s1.iloc[start:start + args.chunk_s1]
        print(f"block+score {start}/{len(s1)}...", flush=True)
        cand = block_all(chunk, s2, s3, top_k=args.top_k)
        for sid in chunk["entity_id"]:
            pairs = cand.get(str(sid), [])
            cand_rows[str(sid)] = [c for c, _ in pairs]
        # Featurize + score this chunk's candidates.
        s1map = {r.entity_id: r for r in chunk.itertuples()}
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
        for sid in chunk["entity_id"]:
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
