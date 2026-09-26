"""Text normalisation for business names and addresses.

The pipeline applies the exact same normalisation to every record regardless
of source, so noisy variants collapse toward a common canonical form:

* Devanagari is transliterated to Latin (see ``transliterate``).
* Unicode is NFKD-folded and combining marks are stripped (``Café`` -> ``cafe``).
* Case is lowered and ``&`` becomes ``and``.
* Common legal suffixes and street-type abbreviations are expanded, so
  ``Corp``/``Corporation`` and ``Rd``/``Road`` become identical tokens.

Everything here is deterministic and depends only on the record text; no
external data is consulted.
"""
from __future__ import annotations

import re
import unicodedata

from .transliterate import has_devanagari, transliterate

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_DIGIT_RUN_RE = re.compile(r"\d{4,}")

# --- Legal-form tokens that carry no discriminative signal in a name --------
# They are removed from the "core" token set used for blocking and for the
# token-overlap features, but the full token set is kept for raw similarity.
LEGAL_TOKENS = frozenset({
    "inc", "incorporated", "llc", "llp", "lp", "lllp", "plc", "pllc", "pc",
    "pa", "ltd", "limited", "corp", "corporation", "co", "company", "cmpy",
    "pvt", "private", "pte", "opc", "the", "and", "of", "dba", "trading",
    "enterprise", "enterprises", "group", "holdings", "holding", "services",
    "service", "solutions", "international", "intl", "industries", "industry",
    "associates", "partners", "partnership", "ventures", "venture", "global",
    "technologies", "technology", "systems", "consulting", "consultants",
})

# --- Street-type / directional abbreviations -> canonical form ---------------
ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",    "hwy": "highway", "pkwy": "parkway", "pl": "place", "sq": "square",
    "ter": "terrace", "terr": "terrace", "cir": "circle", "apt": "apartment",
    "ste": "suite", "bldg": "building", "flr": "floor", "fl": "floor",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "mt": "mount", "ft": "fort", "po": "post", "p o": "post",
    "no": "number", "opp": "opposite", "nr": "near", "near": "near",
    "mkt": "market", "nagar": "nagar", "colony": "colony", "sector": "sector",
    "h no": "house", "hno": "house", "ho": "house", "shop": "shop",
    "complex": "complex", "tower": "tower", "towers": "tower",
    "rte": "route", "route": "route",
    "block": "block", "phase": "phase", "floor": "floor", "room": "room",
    "village": "village", "post": "post", "district": "district",
    "distt": "district", "dist": "district", "tal": "taluka",
    "tehsil": "tehsil", "state": "state", "country": "country",
    "unit": "unit", "suite": "suite", "building": "building",
    "cross": "cross", "main": "main", "old": "old", "new": "new",
}

# --- State / region abbreviations (kept short; both sides normalised) -------
US_STATES = frozenset({
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id",
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms",
    "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok",
    "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv",
    "wi", "wy", "dc",
})

_GENERIC_ADDR = frozenset({"post", "near", "opposite", "unit", "number"})


def _fold(text: str) -> str:
    """Transliterate Devanagari, NFKD-fold, strip accents, lowercase."""
    if has_devanagari(text):
        text = transliterate(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def clean(text: str) -> str:
    """Return a normalised, whitespace-collapsed string of alnum tokens."""
    if not text:
        return ""
    text = _fold(text).replace("&", " and ")
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Normalised alnum tokens of ``text``."""
    cleaned = clean(text)
    return cleaned.split() if cleaned else []


def name_tokens(name: str) -> list[str]:
    """Tokens of a business name with legal-form suffixes removed.

    Falls back to the full token list when removal would empty the name
    (e.g. a business literally called ``The Company``).
    """
    toks = tokenize(name)
    core = [t for t in toks if t not in LEGAL_TOKENS]
    return core or toks


def all_name_tokens(name: str) -> list[str]:
    """Every normalised token of a business name, stop-words included."""
    return tokenize(name)


def expand_address(addr: str) -> list[str]:
    """Tokens of an address with street-type abbreviations expanded."""
    out = []
    for t in tokenize(addr):
        if t in US_STATES:
            out.append(t)
        else:
            out.append(ADDR_ABBREV.get(t, t))
    return out


def address_tokens(addr: str) -> list[str]:
    """Tokens of an address, abbreviation-expanded, generic words dropped."""
    return [t for t in expand_address(addr) if t not in _GENERIC_ADDR]


def extract_zip(text: str) -> str:
    """Return the most likely postal code (US 5-digit or India 6-digit).

    We take the *last* 4+-digit run, which is almost always the ZIP/PIN even
    when the street number appears earlier in the address.
    """
    if not text:
        return ""
    runs = _DIGIT_RUN_RE.findall(text)
    if not runs:
        return ""
    # Prefer a 5- or 6-digit run; otherwise the longest.
    for r in reversed(runs):
        if len(r) in (5, 6):
            return r
    return max(runs, key=len)


def extract_house_number(text: str) -> str:
    """Return the leading street/house number of an address, if any."""
    if not text:
        return ""
    cleaned = clean(text)
    if not cleaned:
        return ""
    first = cleaned.split()[0]
    return first if first.isdigit() else ""
