"""Train LightGBM matcher on blocked pairs + tune F0.5 threshold.

Usage (AWS full train):
    python -m code.business_entity_resolution.src.train_matcher \
        --train-dir dataset/train --out models --top-k 30 --val-size 50000 --seed 42

Local smoke test (sampled S2/S3 pool, true matches preserved):
    python -m code.business_entity_resolution.src.train_matcher \
        --train-dir dataset/train --out /tmp/models --top-k 20 \
        --val-size 2000 --max-train-s1 8000 --max-s23 60000 --neg-per-pos 4
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from .blocking import block_all, blocking_recall
from .common import load_ground_truth, load_source
from .evaluate import macro_f05, sweep_threshold
from .features import FEATURE_NAMES, pair_features


def build_rows(s1, s2, s3, gt, candidates, s1_ids):
    s1_map = {r.entity_id: r for r in s1.itertuples()}
    c_map = {}
    for r in pd.concat([s2, s3]).itertuples():
        c_map[r.entity_id] = r
    X, y = [], []
    info = []  # (s1_id, cand_id, tfidf)
    for sid in s1_ids:
        true = gt.get(sid, set())
        srow = s1_map.get(sid)
        if srow is None:
            continue
        for cid, tscore in candidates.get(sid, []):
            crow = c_map.get(cid)
            if crow is None:
                continue
            X.append(pair_features(
                srow.business_name, srow.business_address, srow.country,
                crow.business_name, crow.business_address, crow.country, tscore))
            y.append(1 if cid in true else 0)
            info.append((sid, cid, tscore))
    return np.array(X, dtype=np.float32), np.array(y), info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--val-size", type=int, default=50000)
    ap.add_argument("--max-train-s1", type=int, default=0)
    ap.add_argument("--max-s23", type=int, default=0,
                    help="If >0, cap S2+S3 pool (true matches of sampled S1s "
                         "are always kept; rest is random distractors). "
                         "Local-only; use 0 on AWS for the full pool.")
    ap.add_argument("--neg-per-pos", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("loading...", flush=True)
    s1 = load_source(f"{args.train_dir}/train_source1.tsv")
    s2 = load_source(f"{args.train_dir}/train_source2.tsv")
    s3 = load_source(f"{args.train_dir}/train_source3.tsv")
    gt = load_ground_truth(f"{args.train_dir}/train_ground_truth.tsv")

    rng = np.random.RandomState(args.seed)
    # copy=True: to_numpy() may return a VIEW into the frame's block memory;
    # shuffling a view would permute s1's id column in place and detach every
    # id from its row (silent corruption of all downstream joins).
    all_ids = s1["entity_id"].to_numpy(copy=True)
    rng.shuffle(all_ids)
    val_ids = set(all_ids[:args.val_size])
    tr_ids = all_ids[args.val_size:]
    if args.max_train_s1:
        tr_ids = tr_ids[:args.max_train_s1]
    print(f"S1 total={len(all_ids)} train={len(tr_ids)} val={len(val_ids)}", flush=True)

    if args.max_s23:
        # Sampled pool for local iteration: keep every true match of the
        # sampled S1s, fill the rest with random distractors.
        keep_s1 = set(tr_ids) | val_ids
        need: set[str] = set()
        for sid in keep_s1:
            need.update(gt.get(str(sid), set()))
        s2_need = {i for i in need if i.startswith("S2-")}
        s3_need = {i for i in need if i.startswith("S3-")}
        half = args.max_s23 // 2
        s2_extra = s2[~s2["entity_id"].isin(s2_need)].sample(
            n=max(0, min(half - len(s2_need), len(s2))), random_state=args.seed)
        s3_extra = s3[~s3["entity_id"].isin(s3_need)].sample(
            n=max(0, min(half - len(s3_need), len(s3))), random_state=args.seed)
        s2 = pd.concat([s2[s2["entity_id"].isin(s2_need)], s2_extra], ignore_index=True)
        s3 = pd.concat([s3[s3["entity_id"].isin(s3_need)], s3_extra], ignore_index=True)
        print(f"sampled pool S2={len(s2)} S3={len(s3)}", flush=True)

    print("blocking train...", flush=True)
    s1tr = s1[s1["entity_id"].isin(set(tr_ids))]
    cand_tr = block_all(s1tr, s2, s3, top_k=args.top_k)
    print(f"train blocking recall={blocking_recall(cand_tr, gt):.4f}", flush=True)

    X, y, _ = build_rows(s1, s2, s3, gt, cand_tr, list(tr_ids))
    print(f"pairs={len(y)} pos_rate={y.mean() if len(y) else 0:.4f}", flush=True)

    # Subsample negatives for training speed/class balance.
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    keep_neg = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * args.neg_per_pos), replace=False)
    keep = np.concatenate([pos_idx, keep_neg])
    rng.shuffle(keep)
    Xtr, ytr = X[keep], y[keep]

    print("training LightGBM...", flush=True)
    ds = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_NAMES)
    params = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
                  num_leaves=127, feature_fraction=0.9, bagging_fraction=0.9,
                  bagging_freq=1, verbose=-1, num_threads=os.cpu_count() or 4)
    model = lgb.train(params, ds, num_boost_round=500)

    print("blocking val + scoring...", flush=True)
    s1v = s1[s1["entity_id"].isin(val_ids)]
    cand_v = block_all(s1v, s2, s3, top_k=args.top_k)
    print(f"val blocking recall={blocking_recall(cand_v, gt):.4f}", flush=True)
    Xv, yv, infov = build_rows(s1, s2, s3, gt, cand_v, list(val_ids))
    pv = model.predict(Xv)

    scored: dict[str, list[tuple[str, float]]] = {}
    for (sid, cid, _), p in zip(infov, pv):
        scored.setdefault(sid, []).append((cid, float(p)))
    for sid in val_ids:
        scored.setdefault(sid, [])

    best_t, best_f = sweep_threshold(scored, gt)
    # Baseline reference: tfidf-only (for the write-up).
    tfidf_only: dict[str, list[tuple[str, float]]] = {}
    for sid, pairs in cand_v.items():
        tfidf_only[sid] = list(pairs)
    _, tfidf_f = sweep_threshold(tfidf_only, gt, thresholds=[0.0, 0.05, 0.1, 0.15, 0.2])
    print(f"BEST thr={best_t} val macro-F0.5={best_f:.4f} (tfidf-only best={tfidf_f:.4f})", flush=True)

    model.save_model(os.path.join(args.out, "lgbm.txt"))
    with open(os.path.join(args.out, "threshold.txt"), "w") as f:
        f.write(str(best_t))
    with open(os.path.join(args.out, "config.txt"), "w") as f:
        f.write(f"top_k={args.top_k}\nfeatures={','.join(FEATURE_NAMES)}\n")
    print("saved.", flush=True)


if __name__ == "__main__":
    main()
