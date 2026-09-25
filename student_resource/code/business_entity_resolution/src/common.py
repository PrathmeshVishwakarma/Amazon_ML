"""Shared IO + normalisation helpers. Country-open: never hard-codes labels."""
from __future__ import annotations

import pandas as pd

from . import text_norm as tn

SRC_COLS = ["entity_id", "business_name", "business_address", "country"]


def load_source(path: str, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, nrows=nrows)
    for c in SRC_COLS:
        if c not in df.columns:
            raise ValueError(f"{path}: missing column {c}")
    return df.fillna("")


def load_ground_truth(path: str, nrows: int | None = None) -> dict[str, set[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, nrows=nrows).fillna("")
    gt: dict[str, set[str]] = {}
    for sid, m in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        m = (m or "").strip()
        gt[sid] = set(x for x in m.split(",") if x) if m else set()
    return gt


def block_text(name: str, addr: str) -> str:
    """Single TF-IDF document string. Name repeated 2x to weight it above address.

    Uses the same normalisation as features so blocking and matching agree.
    Appends extracted ZIP/PIN twice (strong disambiguator when present).
    """
    core = tn.name_tokens(name)
    full = tn.all_name_tokens(name)
    addr_toks = tn.expand_address(addr)
    z = tn.extract_zip(addr)
    parts: list[str] = []
    parts.extend(core)
    parts.extend(core)  # name core x2
    parts.extend(full)
    parts.extend(addr_toks)
    if z:
        parts.extend([z, z])
    return " ".join(parts) if parts else "emptyrecord"


def add_block_text(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_block"] = [block_text(n, a) for n, a in zip(df["business_name"], df["business_address"])]
    return out
