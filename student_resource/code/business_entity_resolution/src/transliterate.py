"""Rule-based Indic (Devanagari) -> Latin transliteration.

No external data is used: the mappings below are standard phonetic
equivalences between Devanagari codepoints and Latin letters.  This lets us
compare Latin-script Source-1 names against Devanagari Source-2/3 names with
the same string-similarity machinery used everywhere else.
"""
from __future__ import annotations

# Independent vowels.
_INDEPENDENT = {
    "\u0905": "a", "\u0906": "aa", "\u0907": "i", "\u0908": "ii",
    "\u0909": "u", "\u090a": "uu", "\u090b": "ri", "\u090c": "li",
    "\u090d": "e", "\u090f": "e", "\u0910": "ai", "\u0911": "o",
    "\u0912": "o", "\u0913": "o", "\u0914": "au",
}

# Dependent vowel signs (matras).
_MATRA = {
    "\u093e": "a", "\u093f": "i", "\u0940": "ii", "\u0941": "u",
    "\u0942": "uu", "\u0943": "ri", "\u0944": "ri", "\u0945": "e",
    "\u0947": "e", "\u0948": "ai", "\u0949": "o", "\u094b": "o",
    "\u094c": "au", "\u094d": "", "\u0950": "om",
}

# Consonants (without the inherent vowel).
_CONSONANT = {
    "\u0915": "k", "\u0916": "kh", "\u0917": "g", "\u0918": "gh",
    "\u0919": "n", "\u091a": "ch", "\u091b": "chh", "\u091c": "j",
    "\u091d": "jh", "\u091e": "n", "\u091f": "t", "\u0920": "th",
    "\u0921": "d", "\u0922": "dh", "\u0923": "n", "\u0924": "t",
    "\u0925": "th", "\u0926": "d", "\u0927": "dh", "\u0928": "n",
    "\u092a": "p", "\u092b": "ph", "\u092c": "b", "\u092d": "bh",
    "\u092e": "m", "\u092f": "y", "\u0930": "r", "\u0932": "l",
    "\u0933": "l", "\u0934": "l", "\u0935": "v", "\u0936": "sh",
    "\u0937": "sh", "\u0938": "s", "\u0939": "h",
    "\u0958": "q", "\u0959": "kh", "\u095a": "g", "\u095b": "z",
    "\u095c": "r", "\u095d": "rh", "\u095e": "f", "\u095f": "y",
    "\u0929": "n", "\u0931": "r",
}

# Signs that nasalise / aspirate a preceding vowel.
_MODIFIER = {
    "\u0901": "n", "\u0902": "n", "\u0903": "h", "\u093c": "",
    "\u093d": "'", "\u0964": " ", "\u0965": " ",
}

# Digits.
_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}

_VIRAMA = "\u094d"
_MAP = {}
_MAP.update(_INDEPENDENT)
_MAP.update(_MATRA)
_MAP.update(_CONSONANT)
_MAP.update(_MODIFIER)
_MAP.update(_DIGITS)

# Devanagari unicode block plus danda / common punctuation.
_DEV_RANGE = (0x0900, 0x097F)


def _is_devanagari(ch: str) -> bool:
    return _DEV_RANGE[0] <= ord(ch) <= _DEV_RANGE[1]


def has_devanagari(text: str) -> bool:
    """True when ``text`` contains at least one Devanagari codepoint."""
    return any(_is_devanagari(c) for c in text)


def transliterate(text: str) -> str:
    """Transliterate every Devanagari codepoint in ``text`` to Latin letters.

    Non-Devanagari characters are passed through unchanged.  An inherent 'a'
    is emitted after a consonant unless the consonant is followed by a matra
    (dependent vowel) or the virama (vowel suppressor).  This yields
    ``raja`` for ``राज`` and ``kamala`` for ``कमल`` — readable, phonetic
    keys that fuzzy matching can compare against their Latin counterparts.
    """
    out = []
    n = len(text)
    for i, ch in enumerate(text):
        if not _is_devanagari(ch):
            out.append(ch)
            continue
        if ch in _CONSONANT:
            out.append(_CONSONANT[ch])
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt not in _MATRA and nxt != _VIRAMA:
                out.append("a")
        elif ch in _MAP:
            out.append(_MAP[ch])
        else:
            # Unmapped Devanagari (rare ligatures) -> drop silently.
            pass
    return "".join(out)
