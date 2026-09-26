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
from .text_norm import extract_zip

try:
    import resource

    def _peak_gb() -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux KB
except ImportError:  # macOS reports bytes; Windows lacks resource
    import os

    def _peak_gb() -> float:
        try:
            import resource as _r
            scale = 1e9 if os.uname().sysname == "Darwin" else 1e6
            return _r.getrusage(_r.RUSAGE_SELF).ru_maxrss / scale
        except Exception:
            return -1.0


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


def fit_index(texts: list[str], max_features: int, min_df: int):
    """Fit a TF-IDF index (float32). Raises ValueError on empty vocabulary."""
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=min_df,
        max_features=max_features, sublinear_tf=True,
    )
    mat = vec.fit_transform(texts).astype(np.float32)
    if mat.shape[1] == 0:
        raise ValueError("empty vocabulary")
    return vec, mat


def query_topk(vec, mat, cand_ids: np.ndarray, query_ids: np.ndarray,
               queries: list[str], top_k: int, chunk_size: int,
               verbose: bool = True, shard_name: str = "") -> dict[str, list[tuple[str, float]]]:
    """Stream queries against a fitted index in bounded-memory chunks."""
    out: dict[str, list[tuple[str, float]]] = {str(sid): [] for sid in query_ids}
    t0 = time.time()
    n_chunks = (len(queries) + chunk_size - 1) // chunk_size
    for ci, start in enumerate(range(0, len(queries), chunk_size)):
        ct0 = time.time()
        chunk = vec.transform(queries[start:start + chunk_size]).astype(np.float32)
        scores = (chunk @ mat.T).tocsr()
        idx, sco = _topk_per_row(scores, top_k)
        for j in range(idx.shape[0]):
            pairs: list[tuple[str, float]] = []
            for c, s in zip(idx[j], sco[j]):
                if c < 0 or s <= 0:
                    continue
                pairs.append((str(cand_ids[c]), float(s)))
            if pairs:
                out[str(query_ids[start + j])] = pairs
        if verbose:
            done = min(start + chunk_size, len(queries))
            el = time.time() - t0
            rate = done / el if el > 0 else 0
            print(f"[block:{shard_name}] chunk {ci+1}/{n_chunks}: "
                  f"{done}/{len(queries)} queries, {time.time()-ct0:.0f}s "
                  f"({rate:.0f} q/s, {el:.0f}s elapsed, peak {_peak_gb():.1f}GB)",
                  flush=True)
    return out


def zip_block(s1: pd.DataFrame, s23: pd.DataFrame,
              cap: int = 200) -> dict[str, list[str]]:
    """Same-ZIP candidate pass. Catches rebrands/typos TF-IDF misses.

    Scoped to the given frames — callers pass same-country shards so a US
    90210 never unions with a France 90210. ZIPs are exact strings, so US
    5-digit and India 6-digit never cross-match either. Empty-ZIP records
    are skipped. Per-ZIP cap bounds dense city ZIPs.
    """
    index: dict[str, list[str]] = {}
    for eid, addr in zip(s23["entity_id"], s23["business_address"]):
        z = extract_zip(addr)
        if not z:
            continue
        lst = index.setdefault(z, [])
        if len(lst) < cap:
            lst.append(str(eid))
    out: dict[str, list[str]] = {}
    for sid, addr in zip(s1["entity_id"], s1["business_address"]):
        z = extract_zip(addr)
        out[str(sid)] = list(index.get(z, ())) if z else []
    return out


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
    try:
        vec, s23_mat = fit_index(s23["_block"].tolist(), max_features, min_df)
    except ValueError:
        return out  # empty vocabulary
    if verbose:
        print(f"[block:{shard_name}] index built: {len(s23)} docs, "
              f"{s23_mat.shape[1]} terms, {time.time()-t0:.0f}s", flush=True)

    return query_topk(vec, s23_mat, cand_ids, s1["entity_id"].to_numpy(),
                      s1["_block"].tolist(), top_k, chunk_size, verbose, shard_name)


def block_all(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    top_k: int = 30,
    chunk_size: int = 20000,
    max_features: int = 300_000,
    min_df: int = 2,
    use_zip_block: bool = False,
    zip_cap: int = 200,
) -> dict[str, list[tuple[str, float]]]:
    """Country-sharded blocking over S2+S3. Covers every S1 id (empty if none).

    With use_zip_block, same-ZIP candidates (score 0.0 — the matcher re-scores
    everything) are unioned with the TF-IDF shortlist per country shard.
    """
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
        if use_zip_block:
            zb = zip_block(g1, pool, cap=zip_cap)
            n_extra = 0
            for sid, pairs in shard.items():
                seen = {c for c, _ in pairs}
                for cid in zb.get(sid, ()):
                    if cid not in seen:
                        pairs.append((cid, 0.0))
                        seen.add(cid)
                        n_extra += 1
            print(f"[block:{country}] zip-union added {n_extra} candidates",
                  flush=True)
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
