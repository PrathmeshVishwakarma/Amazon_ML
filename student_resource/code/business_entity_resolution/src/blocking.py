"""TF-IDF blocking, sharded by country (open set — France flows through).

For each distinct country value, a separate TF-IDF index is built over that
shard's S2+S3 records and only same-country S1 records query it. This gives a
~3x reduction with zero hard-coding. Records with an unseen/empty country
still work: they form their own shard.

Memory-safe: S1 queries are processed in chunks; top-K is extracted per row
with argpartition on the sparse score matrix (never densified).
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import add_block_text


def _topk_per_row(scores: csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of top-k per row. Zero-score rows give nothing."""
    n_rows = scores.shape[0]
    idx_out = np.full((n_rows, k), -1, dtype=np.int64)
    sco_out = np.zeros((n_rows, k), dtype=np.float32)
    for i in range(n_rows):
        lo, hi = scores.indptr[i], scores.indptr[i + 1]
        if lo == hi:
            continue
        cols = scores.indices[lo:hi]
        vals = scores.data[lo:hi].astype(np.float32)
        kk = min(k, len(vals))
        part = np.argpartition(-vals, kk - 1)[:kk]
        order = part[np.argsort(-vals[part], kind="stable")]
        idx_out[i, :kk] = cols[order]
        sco_out[i, :kk] = vals[order]
    return idx_out, sco_out


def block_shard(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    top_k: int = 30,
    chunk_size: int = 20000,
    max_features: int = 300_000,
    min_df: int = 2,
    verbose: bool = True,
    shard_name: str = "",
) -> dict[str, list[tuple[str, float]]]:
    """Block one country shard. Returns {s1_id: [(cand_id, tfidf_score)]}."""
    s1 = add_block_text(s1)
    s23 = add_block_text(s23)
    cand_ids = s23["entity_id"].to_numpy()
    out: dict[str, list[tuple[str, float]]] = {sid: [] for sid in s1["entity_id"].to_numpy()}

    if len(s23) == 0 or len(s1) == 0:
        return out

    t0 = time.time()
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=min_df,
        max_features=max_features, sublinear_tf=True,
    )
    try:
        s23_mat = vec.fit_transform(s23["_block"].tolist())
    except ValueError:
        return out  # empty vocabulary
    if s23_mat.shape[1] == 0:
        return out
    if verbose:
        print(f"[block:{shard_name}] index built: {len(s23)} docs, "
              f"{s23_mat.shape[1]} terms, {time.time()-t0:.0f}s", flush=True)

    s1_ids = s1["entity_id"].to_numpy()
    blocks = s1["_block"].tolist()
    n_chunks = (len(blocks) + chunk_size - 1) // chunk_size
    for ci, start in enumerate(range(0, len(blocks), chunk_size)):
        ct0 = time.time()
        chunk = vec.transform(blocks[start:start + chunk_size])
        scores = (chunk @ s23_mat.T).tocsr()
        idx, sco = _topk_per_row(scores, top_k)
        for j in range(idx.shape[0]):
            pairs: list[tuple[str, float]] = []
            for c, s in zip(idx[j], sco[j]):
                if c < 0 or s <= 0:
                    continue
                pairs.append((str(cand_ids[c]), float(s)))
            if pairs:
                out[str(s1_ids[start + j])] = pairs
        if verbose:
            done = min(start + chunk_size, len(blocks))
            el = time.time() - t0
            rate = done / el if el > 0 else 0
            print(f"[block:{shard_name}] chunk {ci+1}/{n_chunks}: "
                  f"{done}/{len(blocks)} queries, {time.time()-ct0:.0f}s "
                  f"({rate:.0f} q/s, {el:.0f}s elapsed)", flush=True)
    return out


def block_all(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    top_k: int = 30,
    chunk_size: int = 20000,
    max_features: int = 300_000,
    min_df: int = 2,
) -> dict[str, list[tuple[str, float]]]:
    """Country-sharded blocking over S2+S3. Covers every S1 id (empty if none)."""
    s23 = pd.concat([s2, s3], ignore_index=True)
    result: dict[str, list[tuple[str, float]]] = {}
    for country, g1 in s1.groupby("country", sort=False):
        g23 = s23[s23["country"] == country]
        # Fallback: if a shard has no same-country candidates (shouldn't happen),
        # back off to the global pool so we never silently drop entities.
        pool = g23 if len(g23) else s23
        print(f"[block] shard country={country!r}: {len(g1)} queries vs "
              f"{len(pool)} candidates", flush=True)
        shard = block_shard(g1, pool, top_k=top_k, chunk_size=chunk_size,
                            max_features=max_features, min_df=min_df,
                            shard_name=str(country))
        result.update(shard)
    return result


def blocking_recall(
    candidates: dict[str, list[tuple[str, float]]],
    gt: dict[str, set[str]],
) -> float:
    """Fraction of true matches present in candidate sets (over gt keys present)."""
    tot, hit = 0, 0
    for sid, true in gt.items():
        if sid not in candidates:
            continue
        cand = {c for c, _ in candidates[sid]}
        for t in true:
            tot += 1
            if t in cand:
                hit += 1
    return hit / tot if tot else 1.0
