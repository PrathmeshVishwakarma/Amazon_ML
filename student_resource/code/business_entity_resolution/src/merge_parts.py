"""Merge per-country part files from split inference runs into final TSVs.

Each split run writes `.part_<country>_cand.tsv` / `.part_<country>_match.tsv`
(no headers, `id<TAB>comma-ids` per line). This script unions them across one
or more directories, checks every test S1 is covered exactly once, and writes
the two submission files in test_source1.tsv order.

Usage:
    python -m code.business_entity_resolution.src.merge_parts \
        --parts mybox/output friendbox/output_india \
        --test-dir dataset/test --out output_final
"""
from __future__ import annotations

import argparse
import glob
import os

from .common import load_source


def _read_part(path: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sid, _, rest = line.partition("\t")
            if sid in rows:
                raise ValueError(f"duplicate S1 {sid} in {path}")
            rows[sid] = rest.split(",") if rest else []
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True,
                    help="Output dirs holding .part_* files (own + friend's).")
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--empty-countries", default="",
                    help="DEADLINE FALLBACK only: S1s in these countries (comma "
                         "list, e.g. 'India') with no part coverage are emitted "
                         "as explicit abstentions (empty lists) instead of "
                         "erroring. Use when a shard cannot finish before a "
                         "hard cutoff; scores ~singleton-rate on those rows "
                         "instead of invalidating the whole submission.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    s1 = load_source(f"{args.test_dir}/test_source1.tsv")
    all_ids = s1["entity_id"].tolist()

    cand: dict[str, list[str]] = {}
    match: dict[str, list[str]] = {}
    for d in args.parts:
        for pc in sorted(glob.glob(os.path.join(d, ".part_*_cand.tsv"))):
            country = os.path.basename(pc)[len(".part_"):-len("_cand.tsv")]
            pm = os.path.join(d, f".part_{country}_match.tsv")
            if not os.path.exists(pm):
                raise ValueError(f"missing match part for {pc}")
            for sid, ids in _read_part(pc).items():
                if sid in cand:
                    raise ValueError(f"S1 {sid} covered twice ({pc})")
                cand[sid] = ids
            for sid, ids in _read_part(pm).items():
                match[sid] = ids
            print(f"merged {country}: {len(match)} S1s so far", flush=True)

    missing = [sid for sid in all_ids if sid not in cand]
    if missing:
        if args.empty_countries:
            cmap = dict(zip(s1["entity_id"], s1["country"]))
            fill_c = set(args.empty_countries.split(","))
            fillable = [sid for sid in missing if cmap.get(sid) in fill_c]
            rest = [sid for sid in missing if cmap.get(sid) not in fill_c]
            if rest:
                raise ValueError(
                    f"{len(rest)} S1s uncovered outside --empty-countries, "
                    f"e.g. {rest[:3]}")
            for sid in fillable:
                cand[sid] = []
                match[sid] = []
            print(f"abstentions emitted for {len(fillable)} S1s "
                  f"in {sorted(fill_c)} (deadline fallback)", flush=True)
        else:
            raise ValueError(f"{len(missing)} S1s uncovered, e.g. {missing[:3]}")
    extra = set(cand) - set(all_ids)
    if extra:
        raise ValueError(f"{len(extra)} part S1s not in test S1, e.g. {list(extra)[:3]}")

    with open(os.path.join(args.out, "candidate_pairs.tsv"), "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_ids:
            f.write(f"{sid}\t{','.join(sorted(set(cand.get(sid, []))))}\n")
    with open(os.path.join(args.out, "matching_results.tsv"), "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_ids:
            f.write(f"{sid}\t{','.join(sorted(set(match.get(sid, []))))}\n")
    n_match = sum(len(v) for v in match.values())
    print(f"merged finals: S1={len(all_ids)} matches={n_match}", flush=True)


if __name__ == "__main__":
    main()
