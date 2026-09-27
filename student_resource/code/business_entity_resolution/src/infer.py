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
import json
import multiprocessing as mp
import os
import numpy as np
import pandas as pd

from .blocking import fit_index, query_topk, zip_block
from .common import add_block_text, load_source
from .features import pair_features

# Worker state for --jobs > 1. Populated in the parent BEFORE forking, so
# children inherit the (read-only) index/model via copy-on-write with zero
# pickling. Linux-only path; the serial path below stays fork-free.
_W: dict = {}


def _run_parallel(country, g1, vec, mat, cand_ids, zb_all, cmap, model_path,
                  args, cand_rows, match_rows) -> None:
    """Split one country's S1s across fork workers sharing the index/model."""
    """Split one country's S1s across fork workers sharing the index/model."""
    import numpy as np

    ids = g1["entity_id"].tolist()
    blocks = g1["_block"].tolist()
    rows = [(r.entity_id, r.business_name, r.business_address, r.country)
            for r in g1.itertuples()]
    n = max(1, min(args.jobs, len(ids)))
    bounds = np.array_split(np.arange(len(ids)), n)
    payloads = []
    for b in bounds:
        b = b.tolist()
        payloads.append(([ids[i] for i in b], [blocks[i] for i in b],
                         [rows[i] for i in b],
                         {sid: zb_all[sid] for sid in (ids[i] for i in b)
                          if sid in zb_all} if args.use_zip_block else {}))
    # Publish shared state BEFORE forking: children inherit it copy-on-write,
    # no pickling of the GB-scale index. The model travels as a PATH and is
    # loaded inside workers (post-fork) to avoid OpenMP fork deadlock.
    _W.update(vec=vec, mat=mat, cand_ids=cand_ids, cmap=cmap, model=None,
              model_path=model_path, top_k=args.top_k, block_chunk=args.block_chunk,
              threshold=args.threshold, use_zip=args.use_zip_block,
              shard=str(country))
    ctx = mp.get_context("fork")
    with ctx.Pool(n) as pool:
        for i, (c_part, m_part) in enumerate(pool.imap(_w_run, payloads)):
            cand_rows.update(c_part)
            for k, v in m_part.items():
                match_rows.setdefault(k, []).extend(v)
            print(f"[infer:{country}] slice {i+1}/{len(payloads)} done "
                  f"({len(cand_rows)} S1s scored)", flush=True)
    for sid in ids:
        match_rows.setdefault(sid, [])
        cand_rows.setdefault(sid, [])


def write_tsv(path: str, header: str, rows: dict[str, list[str]], all_ids: list[str]) -> None:
    with open(path, "w") as f:
        f.write(header + "\n")
        for sid in all_ids:
            ids = sorted(set(rows.get(sid, [])))
            f.write(f"{sid}\t{','.join(ids)}\n")


def _safe_country(country: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(country))


def _parts_cfg(out: str) -> str:
    return os.path.join(out, ".parts_config.json")


def _part_paths(out: str, country: str) -> tuple[str, str]:
    s = _safe_country(country)
    return (os.path.join(out, f".part_{s}_cand.tsv"),
            os.path.join(out, f".part_{s}_match.tsv"))


def write_part(out: str, country: str, ids: list[str],
               cand_rows: dict[str, list[str]],
               match_rows: dict[str, list[str]]) -> None:
    """Flush one finished country's rows to part files (crash-safe)."""
    pc, pm = _part_paths(out, country)
    with open(pc, "w") as f:
        for sid in ids:
            f.write(f"{sid}\t{','.join(sorted(set(cand_rows.get(sid, []))))}\n")
    with open(pm, "w") as f:
        for sid in ids:
            f.write(f"{sid}\t{','.join(sorted(set(match_rows.get(sid, []))))}\n")
    print(f"[infer:{country}] part files flushed ({len(ids)} S1s)", flush=True)


def load_part(out: str, country: str,
              cand_rows: dict[str, list[str]],
              match_rows: dict[str, list[str]]) -> list[str]:
    """Load a finished country's part files. Returns its S1 ids ( [] if absent)."""
    pc, pm = _part_paths(out, country)
    if not (os.path.exists(pc) and os.path.exists(pm)):
        return []
    ids: list[str] = []
    with open(pc) as f:
        for line in f:
            line = line.rstrip("\n")
            sid, _, rest = line.partition("\t")
            cand_rows[sid] = rest.split(",") if rest else []
            ids.append(sid)
    with open(pm) as f:
        for line in f:
            line = line.rstrip("\n")
            sid, _, rest = line.partition("\t")
            match_rows[sid] = rest.split(",") if rest else []
    return ids


def _w_run(payload) -> tuple[dict, dict]:
    """Block + featurize + score one S1 slice. Runs in a forked worker."""
    import numpy as np  # local import: safe post-fork

    W = _W
    if W.get("model") is None:
        # Load LightGBM AFTER forking: its OpenMP runtime deadlocks when
        # inherited across fork. Also pin worker threads: parallelism comes
        # from processes, not threads.
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        import lightgbm as lgb

        W["model"] = lgb.Booster(model_file=W["model_path"])
    sub_ids, sub_blocks, sub_rows, zb_slice = payload
    W = _W
    cand = query_topk(W["vec"], W["mat"], W["cand_ids"], np.array(sub_ids),
                      sub_blocks, W["top_k"], W["block_chunk"], False, W["shard"])
    if W["use_zip"]:
        for sid, pairs in cand.items():
            seen = {c for c, _ in pairs}
            for cid in zb_slice.get(sid, ()):
                if cid not in seen:
                    pairs.append((cid, 0.0))
                    seen.add(cid)
    cmap, model = W["cmap"], W["model"]
    s1map = {r[0]: r for r in sub_rows}
    cand_rows: dict[str, list[str]] = {}
    match_rows: dict[str, list[str]] = {}
    Xs, keys = [], []
    for sid, pairs in cand.items():
        cand_rows[sid] = [c for c, _ in pairs]
        srow = s1map.get(sid)
        for cid, tscore in pairs:
            crow = cmap.get(cid)
            if crow is None:
                continue
            Xs.append(pair_features(
                srow[1], srow[2], srow[3],
                crow.business_name, crow.business_address, crow.country, tscore))
            keys.append((sid, cid))
    if Xs:
        probs = model.predict(np.array(Xs, dtype=np.float32))
        for (sid, cid), p in zip(keys, probs):
            if float(p) >= W["threshold"]:
                match_rows.setdefault(sid, []).append(cid)
    return cand_rows, match_rows


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
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel workers per country shard (Linux fork; the "
                         "index/model are shared copy-on-write). "
                         "--chunk-s1 is ignored when jobs > 1.")
    ap.add_argument("--use-zip-block", action="store_true",
                    help="Union same-ZIP candidates with the TF-IDF shortlist.")
    ap.add_argument("--zip-cap", type=int, default=200,
                    help="Max candidates per ZIP in the ZIP-block pass.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip countries with finished part files in --out "
                         "(crash-safe reruns). Flags must match.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = {k: getattr(args, k) for k in
           ("top_k", "threshold", "max_features", "min_df",
            "use_zip_block", "zip_cap", "model")}
    if args.resume and os.path.exists(_parts_cfg(args.out)):
        with open(_parts_cfg(args.out)) as f:
            if json.load(f) != cfg:
                print("part-file config differs from current flags; "
                      "ignoring parts and starting fresh.", flush=True)
                args.resume = False
    with open(_parts_cfg(args.out), "w") as f:
        json.dump(cfg, f, sort_keys=True)
    print("loading test...", flush=True)
    s1 = load_source(f"{args.test_dir}/test_source1.tsv")
    s2 = load_source(f"{args.test_dir}/test_source2.tsv")
    s3 = load_source(f"{args.test_dir}/test_source3.tsv")
    s23 = pd.concat([s2, s3], ignore_index=True)
    cmap = {r.entity_id: r for r in s23.itertuples()}
    if args.jobs > 1:
        # Parallel path: the model is loaded inside workers (post-fork), so
        # the parent never initializes OpenMP before forking. Pin parent-side
        # threads too; throughput comes from processes.
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        model = None
    else:
        import lightgbm as lgb

        model = lgb.Booster(model_file=args.model)

    all_ids = s1["entity_id"].tolist()
    cand_rows: dict[str, list[str]] = {}
    match_rows: dict[str, list[str]] = {}

    # One index build per country, then stream that country's S1s through it.
    # (Previously the index was rebuilt for every S1 chunk — same results,
    # ~10x slower on full-scale data.)
    for country, g1 in s1.groupby("country", sort=False):
        g1_ids = [str(sid) for sid in g1["entity_id"].tolist()]
        if args.resume:
            loaded = load_part(args.out, str(country), cand_rows, match_rows)
            if loaded:
                print(f"[infer:{country}] resumed {len(loaded)} S1s from part files",
                      flush=True)
                continue
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
            write_part(args.out, str(country), g1_ids, cand_rows, match_rows)
            continue
        cand_ids = pool["entity_id"].to_numpy()
        g1 = add_block_text(g1)
        zb_all: dict[str, list[str]] = {}
        if args.use_zip_block:
            zb_all = zip_block(g1, pool, cap=args.zip_cap)
        if args.jobs > 1:
            _run_parallel(country, g1, vec, mat, cand_ids, zb_all, cmap,
                          args.model, args, cand_rows, match_rows)
            write_part(args.out, str(country), g1_ids, cand_rows, match_rows)
            # Release the shard index promptly; peak RSS stays bounded.
            del vec, mat
            continue
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

        write_part(args.out, str(country), g1_ids, cand_rows, match_rows)
        del vec, mat

    write_tsv(f"{args.out}/candidate_pairs.tsv",
              "source1_entity_id\tcandidate_entity_ids", cand_rows, all_ids)
    write_tsv(f"{args.out}/matching_results.tsv",
              "source1_entity_id\tmatched_entity_ids", match_rows, all_ids)
    n_match = sum(len(v) for v in match_rows.values())
    n_cand = sum(len(v) for v in cand_rows.values())
    print(f"done. S1={len(all_ids)} cand_pairs={n_cand} matches={n_match}", flush=True)


if __name__ == "__main__":
    main()
