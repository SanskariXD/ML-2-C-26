"""Label-free normalization. Keep native-script letters AND combining marks."""
from __future__ import annotations
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from .data import Record

NORMALIZER_VERSION = "unicode-preserving-v1"
SUFFIXES = frozenset("inc incorporated corp corporation ltd limited llc pvt private sarl sas gmbh".split())


def normalize(text: str) -> str:
    chars = []
    for c in unicodedata.normalize("NFKC", text).casefold():
        chars.append(c if unicodedata.category(c)[0] in "LNM" else " ")
    return " ".join("".join(chars).split())


def fold_latin(text: str) -> str:
    """Remove combining accents from Latin sequences, not Indic vowel marks."""
    out, latin = [], False
    for c in unicodedata.normalize("NFD", text):
        if unicodedata.category(c).startswith("M"):
            if not latin:
                out.append(c)
        else:
            latin = "LATIN" in unicodedata.name(c, "")
            out.append(c)
    return unicodedata.normalize("NFC", "".join(out))


def trigrams(text: str) -> set[str]:
    text = " ".join(text.split())
    if not text:
        return set()
    text = "^" + text + "$"
    return {text[i:i + 3] for i in range(max(1, len(text) - 2))}


def bottom_tokens(tokens: set[str], limit: int) -> list[str]:
    return sorted(tokens, key=lambda t: (hashlib.blake2b(t.encode(), digest_size=8).digest(), t))[:limit]


@dataclass(frozen=True, slots=True)
class TextView:
    name: str
    address: str
    root: str
    latin_name: str
    latin_address: str
    country: str
    name_tokens: frozenset[str]
    address_tokens: frozenset[str]
    numbers: frozenset[str]
    postals: frozenset[str]


def view(rec: Record) -> TextView:
    name, addr = normalize(rec.business_name), normalize(rec.business_address)
    nums = frozenset(re.findall(r"\b\d+\b", addr))
    # Deliberately a weak 'postal-like' feature, not a country-specific address parser.
    postal = frozenset(n for n in nums if len(n) in (5, 6))
    return TextView(name, addr, " ".join(t for t in name.split() if t not in SUFFIXES),
                    fold_latin(name), fold_latin(addr), normalize(rec.country),
                    frozenset(name.split()), frozenset(addr.split()), nums, postal)


CHANNELS = ("name", "root", "address", "postal_number", "name_word", "address_word", "gram")


def blocking_keys(v: TextView) -> dict[str, list[str]]:
    """Bound per-record posting growth. Probe benchmarks must audit truncation loss."""
    shortnums = sorted(n for n in v.numbers if len(n) < 5)[:4]
    return {
        "name": [v.latin_name] if v.latin_name else [],
        "root": [fold_latin(v.root)] if v.root else [],
        "address": [v.latin_address] if v.latin_address else [],
        "postal_number": [f"{p}|{n}" for p in sorted(v.postals)[:2] for n in shortnums],
        "name_word": bottom_tokens(set(v.latin_name.split()), 8),
        "address_word": bottom_tokens(set(v.latin_address.split()), 10),
        "gram": bottom_tokens(trigrams(v.latin_name[:256]), 12),
    }
