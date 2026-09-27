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
import json
import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

from .blocking import block_all, blocking_recall
from .common import load_ground_truth, load_source
from .evaluate import macro_f05, sweep_threshold
from .features import FEATURE_NAMES, pair_features


def ckpt_paths(out: str) -> dict[str, str]:
    return {
        "config": os.path.join(out, "ckpt_config.json"),
        "feat_train": os.path.join(out, "ckpt_feat_train.npz"),
        "feat_val": os.path.join(out, "ckpt_feat_val.npz"),
        "info_val": os.path.join(out, "ckpt_info_val.pkl"),
        "tr_ids": os.path.join(out, "ckpt_tr_ids.npy"),
        "val_ids": os.path.join(out, "ckpt_val_ids.npy"),
    }


def save_train_ckpt(out: str, cfg: dict, X: np.ndarray, y: np.ndarray,
                    tr_ids: np.ndarray, val_ids: set) -> None:
    """Phase-1 checkpoint: persisted BEFORE val blocking, so a death in the
    val stage never discards train blocking + featurizing."""
    p = ckpt_paths(out)
    with open(p["config"], "w") as f:
        json.dump(cfg, f, sort_keys=True)
    np.savez_compressed(p["feat_train"], X=X, y=y)
    np.save(p["tr_ids"], tr_ids)
    np.save(p["val_ids"], np.array(sorted(val_ids)))
    print("train-phase checkpoint saved.", flush=True)


def save_ckpt(out: str, cfg: dict, X: np.ndarray, y: np.ndarray,
              Xv: np.ndarray, yv: np.ndarray,
              infov: list, tr_ids: np.ndarray, val_ids: set) -> None:
    p = ckpt_paths(out)
    with open(p["config"], "w") as f:
        json.dump(cfg, f, sort_keys=True)
    if not os.path.exists(p["feat_train"]):
        np.savez_compressed(p["feat_train"], X=X, y=y)
        np.save(p["tr_ids"], tr_ids)
        np.save(p["val_ids"], np.array(sorted(val_ids)))
    np.savez_compressed(p["feat_val"], X=Xv, y=yv)
    with open(p["info_val"], "wb") as f:
        pickle.dump(infov, f)
    print("checkpoint saved.", flush=True)


def _cfg_ok(out: str, cfg: dict) -> bool:
    p = ckpt_paths(out)
    if not os.path.exists(p["config"]):
        return False
    with open(p["config"]) as f:
        ok = json.load(f) == cfg
    if not ok:
        print("checkpoint config differs from current flags; ignoring checkpoint.",
              flush=True)
    return ok


def load_ckpt(out: str, cfg: dict):
    """Full resume (train + val features) or None.

    Returns (X, y, Xv, yv, infov, tr_ids, val_ids, phase) where phase is
    'full', or ('train', X, y, tr_ids, val_ids) for a train-phase-only
    checkpoint (val blocking still to do).
    """
    p = ckpt_paths(out)
    if not _cfg_ok(out, cfg):
        return None
    try:
        if all(os.path.exists(p[k]) for k in ("feat_val", "info_val")):
            tr = np.load(p["feat_train"])
            va = np.load(p["feat_val"])
            with open(p["info_val"], "rb") as f:
                infov = pickle.load(f)
            tr_ids = np.load(p["tr_ids"], allow_pickle=True)
            val_ids = set(np.load(p["val_ids"], allow_pickle=True).tolist())
            return ("full", tr["X"], tr["y"], va["X"], va["y"],
                    infov, tr_ids, val_ids)
        if all(os.path.exists(p[k]) for k in ("feat_train", "tr_ids", "val_ids")):
            tr = np.load(p["feat_train"])
            tr_ids = np.load(p["tr_ids"], allow_pickle=True)
            val_ids = set(np.load(p["val_ids"], allow_pickle=True).tolist())
            return ("train", tr["X"], tr["y"], tr_ids, val_ids)
        return None
    except Exception as e:  # corrupt/incomplete write -> recompute cleanly
        print(f"checkpoint unreadable ({e}); recomputing.", flush=True)
        return None


def _sampled_pool(s2, s3, gt, keep_s1: set, max_s23: int, seed: int):
    """Cap S2+S3 pool keeping every true match of keep_s1 (local iteration)."""
    need: set[str] = set()
    for sid in keep_s1:
        need.update(gt.get(str(sid), set()))
    s2_need = {i for i in need if i.startswith("S2-")}
    s3_need = {i for i in need if i.startswith("S3-")}
    half = max_s23 // 2
    s2_extra = s2[~s2["entity_id"].isin(s2_need)].sample(
        n=max(0, min(half - len(s2_need), len(s2))), random_state=seed)
    s3_extra = s3[~s3["entity_id"].isin(s3_need)].sample(
        n=max(0, min(half - len(s3_need), len(s3))), random_state=seed)
    s2 = pd.concat([s2[s2["entity_id"].isin(s2_need)], s2_extra], ignore_index=True)
    s3 = pd.concat([s3[s3["entity_id"].isin(s3_need)], s3_extra], ignore_index=True)
    print(f"sampled pool S2={len(s2)} S3={len(s3)}", flush=True)
    return s2, s3


def do_val_stage(args, cfg, gt, s1, s2, s3, X, y, tr_ids, val_ids):
    print("blocking val + scoring...", flush=True)
    s1v = s1[s1["entity_id"].isin(set(val_ids))]
    cand_v = block_all(s1v, s2, s3, top_k=args.top_k,
                       max_features=args.max_features, min_df=args.min_df,
                       chunk_size=args.block_chunk,
                       use_zip_block=args.use_zip_block, zip_cap=args.zip_cap)
    print(f"val blocking recall={blocking_recall(cand_v, gt):.4f}", flush=True)
    Xv, yv, infov = build_rows(s1, s2, s3, gt, cand_v, list(val_ids))

    save_ckpt(args.out, cfg, X, y, Xv, yv, infov,
              np.array(tr_ids), set(val_ids))
    return train_and_score(args, gt, X, y, Xv, yv, infov, tr_ids, val_ids)


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
    ap.add_argument("--max-features", type=int, default=300_000,
                    help="TF-IDF vocab cap. Lower (e.g. 50000) to fit low-RAM boxes.")
    ap.add_argument("--min-df", type=int, default=2,
                    help="TF-IDF min document frequency. Raise (e.g. 5) to shrink "
                         "the index on low-RAM boxes.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from checkpoint in --out (blocking + features), "
                         "e.g. after a crash or shutdown. Flags must match.")
    ap.add_argument("--log-every", type=int, default=50,
                    help="Print training logloss every N boosting rounds.")
    ap.add_argument("--block-chunk", type=int, default=2000,
                    help="Queries per sparse multiply. Lower (e.g. 1000) if a "
                         "full-vocab shard still OOMs.")
    ap.add_argument("--use-zip-block", action="store_true",
                    help="Union same-ZIP candidates with the TF-IDF shortlist.")
    ap.add_argument("--zip-cap", type=int, default=200,
                    help="Max candidates per ZIP in the ZIP-block pass.")
    ap.add_argument("--train-countries", default="",
                    help="Comma list to restrict train S1s (LOCO, e.g. 'US'). "
                         "Empty = all.")
    ap.add_argument("--val-countries", default="",
                    help="Comma list to restrict val S1s (LOCO, e.g. 'India'). "
                         "Empty = all.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = {k: getattr(args, k) for k in
           ("top_k", "val_size", "max_train_s1", "max_s23",
            "max_features", "min_df", "seed", "use_zip_block", "zip_cap",
            "train_countries", "val_countries")}
    # Ground truth is small (IDs only) and needed for scoring even on resume.
    gt = load_ground_truth(f"{args.train_dir}/train_ground_truth.tsv")
    if args.resume:
        hit = load_ckpt(args.out, cfg)
        if hit is not None and hit[0] == "full":
            _, X, y, Xv, yv, infov, tr_ids, val_ids = hit
            print("resuming from checkpoint (blocking + features skipped).", flush=True)
            return train_and_score(args, gt, X, y, Xv, yv, infov, tr_ids, val_ids)
        if hit is not None and hit[0] == "train":
            _, X, y, tr_ids, val_ids = hit
            print("resuming from train-phase checkpoint (train work kept).", flush=True)
            print("loading...", flush=True)
            s1 = load_source(f"{args.train_dir}/train_source1.tsv")
            s2 = load_source(f"{args.train_dir}/train_source2.tsv")
            s3 = load_source(f"{args.train_dir}/train_source3.tsv")
            if args.max_s23:
                s2, s3 = _sampled_pool(s2, s3, gt, set(tr_ids) | set(val_ids),
                                       args.max_s23, args.seed)
            return do_val_stage(args, cfg, gt, s1, s2, s3, X, y, tr_ids, val_ids)
        print("no usable checkpoint; starting fresh.", flush=True)
    print("loading...", flush=True)
    s1 = load_source(f"{args.train_dir}/train_source1.tsv")
    s2 = load_source(f"{args.train_dir}/train_source2.tsv")
    s3 = load_source(f"{args.train_dir}/train_source3.tsv")

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
    # LOCO country filters (empty = all countries).
    cmap_s1 = dict(zip(s1["entity_id"], s1["country"]))
    if args.train_countries:
        keep_c = set(args.train_countries.split(","))
        tr_ids = np.array([i for i in tr_ids if cmap_s1.get(str(i)) in keep_c])
    if args.val_countries:
        keep_c = set(args.val_countries.split(","))
        val_ids = {i for i in val_ids if cmap_s1.get(str(i)) in keep_c}
    print(f"S1 total={len(all_ids)} train={len(tr_ids)} val={len(val_ids)}", flush=True)

    if args.max_s23:
        s2, s3 = _sampled_pool(s2, s3, gt, set(tr_ids) | val_ids,
                               args.max_s23, args.seed)

    print("blocking train...", flush=True)
    s1tr = s1[s1["entity_id"].isin(set(tr_ids))]
    cand_tr = block_all(s1tr, s2, s3, top_k=args.top_k,
                        max_features=args.max_features, min_df=args.min_df,
                        chunk_size=args.block_chunk,
                        use_zip_block=args.use_zip_block, zip_cap=args.zip_cap)
    print(f"train blocking recall={blocking_recall(cand_tr, gt):.4f}", flush=True)

    X, y, _ = build_rows(s1, s2, s3, gt, cand_tr, list(tr_ids))
    print(f"pairs={len(y)} pos_rate={y.mean() if len(y) else 0:.4f}", flush=True)
    save_train_ckpt(args.out, cfg, X, y, np.array(tr_ids), set(val_ids))

    return do_val_stage(args, cfg, gt, s1, s2, s3, X, y, tr_ids, val_ids)


def train_and_score(args, gt, X, y, Xv, yv, infov, tr_ids, val_ids) -> None:
    rng = np.random.RandomState(args.seed + 1)
    # Subsample negatives for training speed/class balance.
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    keep_neg = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * args.neg_per_pos), replace=False)
    keep = np.concatenate([pos_idx, keep_neg])
    rng.shuffle(keep)
    Xtr, ytr = X[keep], y[keep]

    print(f"training LightGBM on {len(ytr)} rows "
          f"({(ytr == 1).sum()} pos)...", flush=True)
    ds = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_NAMES)
    params = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
                  num_leaves=127, feature_fraction=0.9, bagging_fraction=0.9,
                  bagging_freq=1, verbose=-1, num_threads=os.cpu_count() or 4,
                  seed=args.seed, deterministic=True)
    model = lgb.train(params, ds, num_boost_round=500,
                      callbacks=[lgb.log_evaluation(period=args.log_every)])

    print("scoring val...", flush=True)
    pv = model.predict(Xv)

    scored: dict[str, list[tuple[str, float]]] = {}
    for (sid, cid, _), p in zip(infov, pv):
        scored.setdefault(sid, []).append((cid, float(p)))
    for sid in val_ids:
        scored.setdefault(sid, [])

    best_t, best_f = sweep_threshold(scored, gt)
    # Baseline reference: tfidf-only (for the write-up). Reuses the blocking
    # scores carried in infov, so no candidate dict is needed here.
    tfidf_only: dict[str, list[tuple[str, float]]] = {}
    for sid, cid, tscore in infov:
        tfidf_only.setdefault(sid, []).append((cid, float(tscore)))
    for sid in val_ids:
        tfidf_only.setdefault(sid, [])
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
