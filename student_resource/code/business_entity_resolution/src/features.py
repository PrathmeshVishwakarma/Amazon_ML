"""Pairwise match features. Deterministic, stdlib+RapidFuzz only, no external data."""
from __future__ import annotations

from rapidfuzz import fuzz

from . import text_norm as tn

FEATURE_NAMES = [
    "tfidf",
    "name_jac_core",
    "name_jac_full",
    "name_set",
    "name_ratio",
    "addr_jac",
    "addr_set",
    "addr_ratio",
    "zip_match",
    "zip_missing",
    "house_match",
    "same_country",
    "name_len_diff",
    "addr_len_diff",
]


def _jaccard(a: list[str] | set[str], b: list[str] | set[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def pair_features(
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    c_name: str,
    c_addr: str,
    c_country: str,
    tfidf: float = 0.0,
) -> list[float]:
    n1_core = tn.name_tokens(s1_name)
    n2_core = tn.name_tokens(c_name)
    n1_full = tn.all_name_tokens(s1_name)
    n2_full = tn.all_name_tokens(c_name)
    a1 = tn.expand_address(s1_addr)
    a2 = tn.expand_address(c_addr)

    s1n = tn.clean(s1_name)
    cn = tn.clean(c_name)
    s1a = tn.clean(s1_addr)
    ca = tn.clean(c_addr)

    z1, z2 = tn.extract_zip(s1_addr), tn.extract_zip(c_addr)
    h1, h2 = tn.extract_house_number(s1_addr), tn.extract_house_number(c_addr)

    zip_missing = 1.0 if (not z1 or not z2) else 0.0
    zip_match = 1.0 if (z1 and z1 == z2) else 0.0
    house_match = 1.0 if (h1 and h1 == h2) else 0.0

    def _lendiff(x: str, y: str) -> float:
        if not x and not y:
            return 0.0
        m = max(len(x), len(y), 1)
        return abs(len(x) - len(y)) / m

    return [
        float(tfidf),
        _jaccard(n1_core, n2_core),
        _jaccard(n1_full, n2_full),
        float(fuzz.token_set_ratio(s1n, cn)) / 100.0 if (s1n or cn) else 1.0,
        float(fuzz.ratio(s1n, cn)) / 100.0 if (s1n or cn) else 1.0,
        _jaccard(a1, a2),
        float(fuzz.token_set_ratio(s1a, ca)) / 100.0 if (s1a or ca) else 1.0,
        float(fuzz.ratio(s1a, ca)) / 100.0 if (s1a or ca) else 1.0,
        zip_match,
        zip_missing,
        house_match,
        1.0 if (s1_country and s1_country == c_country) else 0.0,
        _lendiff(s1n, cn),
        _lendiff(s1a, ca),
    ]
