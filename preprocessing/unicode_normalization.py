"""Conservative Khmer Unicode normalisation.

The guiding rule is: **a valid Khmer string must come out byte-identical.**  Only
sequences that are invalid, deprecated or unambiguously equivalent are rewritten.
Every rule below is individually toggleable and individually counted, so a
pipeline run reports exactly what it changed (``NormalizationReport.changes``).

Rules
-----
1. ``NFC`` - canonical composition (safe for Khmer; the block has no canonical
   decompositions that would be lost).
2. **Deprecated code points** are replaced with their modern spelling:
   ``U+17A3`` -> ``U+17A2``, ``U+17A4`` -> ``U+17A2 U+17B6``,
   ``U+17B4``/``U+17B5`` (invisible inherent vowels) and ``U+17D3`` are removed.
3. **Vowel composition** - ``U+17C1 U+17B6`` (េ + ា, produced by keyboards that
   type the two visual halves) becomes the single code point ``U+17C4`` (ោ).
   The same applies to ``U+17C1 U+17B8`` -> ``U+17C5``.
4. **COENG hygiene** - a doubled ``U+17D2`` collapses to one; a ``U+17D2`` that
   is not followed by a consonant is dropped (it renders as a dotted circle).
5. **Canonical cluster ordering** - inside one orthographic cluster the parts are
   stably sorted to ``base < coeng-pairs < register shifter < vowel < signs``.
   Valid text already satisfies this, so the sort is a no-op for it; mistyped
   text such as ``U+17C6 U+17BB`` is repaired to ``U+17BB U+17C6``.
6. **Duplicate marks** - an identical vowel or sign repeated inside one cluster
   is collapsed to a single occurrence.
7. **Zero-width characters** - ZWNJ/ZWJ/BOM/word-joiner are removed; ZWSP is
   handled by ``zwsp_policy`` (see below) because in Khmer it carries word-break
   information rather than being pure noise.
8. **Space-like characters** - NBSP and friends become U+0020; C0/C1 control
   characters other than ``\\n`` and ``\\t`` are removed.
9. **Full-width Latin/digits** are folded to ASCII so ``ＱＮ－４５００Ａ``
   matches ``QN-4500A``.  Khmer digits are *not* converted - that would destroy
   information; ``preprocessing.khmer_detection`` reports their ratio instead.
10. **Whitespace** - runs collapse to a single space, and spaces before Khmer
    punctuation (``។ ៕ ៖``) are removed while a space is guaranteed after them.

``zwsp_policy``
    ``"collapse"`` (default) - runs collapse to one ZWSP, and a ZWSP touching
    whitespace, punctuation or a string boundary is dropped.  Keeps genuine
    word-break hints, removes decorative noise.
    ``"strip"`` - remove every ZWSP.  Used for LLM training corpora, where ZWSP
    inflates the token count without helping the model.
    ``"keep"`` - leave ZWSP untouched.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Final, Literal

from preprocessing.khmer_script import (
    COENG,
    ZWJ,
    ZWNJ,
    ZWSP,
    CharClass,
    classify_char,
    is_khmer_char,
    iter_clusters,
)

__all__ = [
    "NormalizationConfig",
    "NormalizationReport",
    "normalize_for_hashing",
    "normalize_khmer",
    "normalize_text",
]

ZwspPolicy = Literal["collapse", "strip", "keep"]

# --- rule tables ------------------------------------------------------------
_DEPRECATED_MAP: Final[dict[str, str]] = {
    "ឣ": "អ",  # U+17A3 KHMER INDEPENDENT VOWEL QAQ  -> U+17A2
    "ឤ": "អា",  # U+17A4 KHMER INDEPENDENT VOWEL QAA  -> U+17A2 U+17B6
    "឴": "",  # U+17B4 KHMER VOWEL INHERENT AQ (must not appear in text)
    "឵": "",  # U+17B5 KHMER VOWEL INHERENT AA
    "៓": "",  # U+17D3 KHMER SIGN BATHAMASAT (deprecated)
}

_VOWEL_COMPOSITION: Final[tuple[tuple[str, str], ...]] = (
    ("េា", "ោ"),  # U+17C1 U+17B6 -> U+17C4
    ("េី", "ៅ"),  # U+17C1 U+17B8 -> U+17C5
)

_SPACE_LIKE: Final = {
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    " ",
    "　",
}
_REMOVE_ZERO_WIDTH: Final = {ZWNJ, ZWJ, "﻿", "⁠", "᠎"}

_KHMER_TERMINATORS: Final = "។៕៖៚៙"
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_SPACE_BEFORE_KHMER_PUNCT = re.compile(rf"[ \t]+([{_KHMER_TERMINATORS}])")
_KHMER_PUNCT_NO_SPACE = re.compile(rf"([{_KHMER_TERMINATORS}])(?=[^\s{_KHMER_TERMINATORS}])")

_FULLWIDTH_START, _FULLWIDTH_END = 0xFF01, 0xFF5E
_FULLWIDTH_OFFSET = 0xFF01 - 0x21


@dataclass(slots=True)
class NormalizationConfig:
    """Per-rule switches.  Defaults are the conservative production settings."""

    nfc: bool = True
    fix_deprecated: bool = True
    compose_vowels: bool = True
    fix_coeng: bool = True
    canonical_cluster_order: bool = True
    collapse_duplicate_marks: bool = True
    zwsp_policy: ZwspPolicy = "collapse"
    remove_zero_width: bool = True
    normalize_spaces: bool = True
    fold_fullwidth: bool = True
    normalize_whitespace: bool = True
    strip_control_chars: bool = True

    @classmethod
    def for_training_corpus(cls) -> NormalizationConfig:
        """Slightly more aggressive: ZWSP removed to save tokens."""
        return cls(zwsp_policy="strip")

    @classmethod
    def minimal(cls) -> NormalizationConfig:
        """NFC + control-char stripping only - for auditing raw data."""
        return cls(
            fix_deprecated=False,
            compose_vowels=False,
            fix_coeng=False,
            canonical_cluster_order=False,
            collapse_duplicate_marks=False,
            zwsp_policy="keep",
            remove_zero_width=False,
            fold_fullwidth=False,
            normalize_whitespace=False,
        )


@dataclass(slots=True)
class NormalizationReport:
    """What normalisation did, so a corpus run can be audited."""

    text: str
    changes: dict[str, int] = field(default_factory=dict)
    original_length: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.changes)

    def _bump(self, rule: str, amount: int = 1) -> None:
        if amount:
            self.changes[rule] = self.changes.get(rule, 0) + amount


def _fold_fullwidth(text: str) -> tuple[str, int]:
    out: list[str] = []
    count = 0
    for ch in text:
        cp = ord(ch)
        if _FULLWIDTH_START <= cp <= _FULLWIDTH_END:
            out.append(chr(cp - _FULLWIDTH_OFFSET))
            count += 1
        else:
            out.append(ch)
    return "".join(out), count


def _strip_controls(text: str) -> tuple[str, int]:
    out: list[str] = []
    count = 0
    for ch in text:
        if ch in ("\n", "\t"):
            out.append(ch)
            continue
        if ch == "\r":
            out.append("\n")
            count += 1
            continue
        if unicodedata.category(ch) in ("Cc", "Cf") and ch not in (ZWSP, ZWNJ, ZWJ):
            count += 1
            continue
        out.append(ch)
    return "".join(out), count


def _apply_zwsp_policy(text: str, policy: ZwspPolicy) -> tuple[str, int]:
    if policy == "keep" or ZWSP not in text:
        return text, 0
    if policy == "strip":
        return text.replace(ZWSP, ""), text.count(ZWSP)

    before = text.count(ZWSP)
    # collapse runs, then drop ZWSP that carries no word-break information
    collapsed = re.sub(f"{ZWSP}+", ZWSP, text)
    collapsed = re.sub(rf"\s{ZWSP}|{ZWSP}\s", lambda m: m.group(0).replace(ZWSP, ""), collapsed)
    collapsed = re.sub(rf"^{ZWSP}+|{ZWSP}+$", "", collapsed)
    collapsed = re.sub(rf"{ZWSP}(?=[{_KHMER_TERMINATORS}])", "", collapsed)
    return collapsed, before - collapsed.count(ZWSP)


def _fix_coeng(text: str) -> tuple[str, int]:
    """Collapse doubled COENG and drop a COENG with no consonant after it."""
    fixed = 0
    doubled = re.sub(f"{COENG}{{2,}}", COENG, text)
    fixed += len(text) - len(doubled)

    out: list[str] = []
    i = 0
    n = len(doubled)
    while i < n:
        ch = doubled[i]
        if ch == COENG:
            nxt = doubled[i + 1] if i + 1 < n else ""
            if not nxt or classify_char(nxt) is not CharClass.BASE:
                fixed += 1
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out), fixed


def _reorder_cluster(cluster: str, *, collapse_duplicates: bool) -> tuple[str, int]:
    """Stable-sort one cluster into canonical order; collapse repeated marks."""
    if len(cluster) < 2:
        return cluster, 0

    # Decompose into (rank, piece) keeping COENG+consonant glued together.
    pieces: list[tuple[int, str]] = []
    i = 0
    n = len(cluster)
    while i < n:
        ch = cluster[i]
        cls = classify_char(ch)
        if cls is CharClass.COENG and i + 1 < n:
            pieces.append((int(CharClass.COENG), cluster[i : i + 2]))
            i += 2
            continue
        pieces.append((int(cls), ch))
        i += 1

    if collapse_duplicates:
        seen: set[str] = set()
        deduped: list[tuple[int, str]] = []
        for rank, piece in pieces:
            if rank in (int(CharClass.VOWEL), int(CharClass.SIGN), int(CharClass.SHIFTER)):
                if piece in seen:
                    continue
                seen.add(piece)
            deduped.append((rank, piece))
        pieces = deduped

    ordered = sorted(pieces, key=lambda item: item[0])  # list.sort is stable
    result = "".join(piece for _, piece in ordered)
    return result, 0 if result == cluster else 1


def _canonicalise_clusters(
    text: str, *, reorder: bool, collapse_duplicates: bool
) -> tuple[str, int, int]:
    if not reorder and not collapse_duplicates:
        return text, 0, 0
    reordered = 0
    original_len = len(text)
    out: list[str] = []
    for cluster in iter_clusters(text):
        if len(cluster) > 1 and is_khmer_char(cluster[0]):
            fixed, changed = _reorder_cluster(cluster, collapse_duplicates=collapse_duplicates)
            reordered += changed
            out.append(fixed)
        else:
            out.append(cluster)
    joined = "".join(out)
    return joined, reordered, original_len - len(joined)


def _normalize_whitespace(text: str) -> str:
    text = _MULTI_SPACE.sub(" ", text)
    text = _SPACE_BEFORE_KHMER_PUNCT.sub(r"\1", text)
    text = _KHMER_PUNCT_NO_SPACE.sub(r"\1 ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def normalize_khmer(text: str, config: NormalizationConfig | None = None) -> NormalizationReport:
    """Normalise ``text`` and report every rule that fired."""
    cfg = config or NormalizationConfig()
    report = NormalizationReport(text=text, original_length=len(text))
    if not text:
        return report

    work = text

    if cfg.strip_control_chars:
        work, removed = _strip_controls(work)
        report._bump("control_chars_removed", removed)

    if cfg.nfc:
        composed = unicodedata.normalize("NFC", work)
        if composed != work:
            report._bump("nfc_applied")
        work = composed

    if cfg.fix_deprecated:
        for src, dst in _DEPRECATED_MAP.items():
            if src in work:
                report._bump("deprecated_replaced", work.count(src))
                work = work.replace(src, dst)

    if cfg.compose_vowels:
        for src, dst in _VOWEL_COMPOSITION:
            if src in work:
                report._bump("vowels_composed", work.count(src))
                work = work.replace(src, dst)

    if cfg.fix_coeng:
        work, fixed = _fix_coeng(work)
        report._bump("coeng_fixed", fixed)

    work, reordered, dropped = _canonicalise_clusters(
        work,
        reorder=cfg.canonical_cluster_order,
        collapse_duplicates=cfg.collapse_duplicate_marks,
    )
    report._bump("clusters_reordered", reordered)
    report._bump("duplicate_marks_removed", dropped)

    if cfg.remove_zero_width:
        removed = sum(work.count(ch) for ch in _REMOVE_ZERO_WIDTH)
        if removed:
            for ch in _REMOVE_ZERO_WIDTH:
                work = work.replace(ch, "")
            report._bump("zero_width_removed", removed)

    work, zwsp_removed = _apply_zwsp_policy(work, cfg.zwsp_policy)
    report._bump("zwsp_normalised", zwsp_removed)

    if cfg.normalize_spaces:
        converted = 0
        for ch in _SPACE_LIKE:
            if ch in work:
                converted += work.count(ch)
                work = work.replace(ch, " ")
        report._bump("space_like_converted", converted)

    if cfg.fold_fullwidth:
        work, folded = _fold_fullwidth(work)
        report._bump("fullwidth_folded", folded)

    if cfg.normalize_whitespace:
        collapsed = _normalize_whitespace(work)
        if collapsed != work:
            report._bump("whitespace_normalised")
        work = collapsed

    report.text = work
    return report


def normalize_text(text: str, config: NormalizationConfig | None = None) -> str:
    """Convenience wrapper returning just the normalised string."""
    return normalize_khmer(text, config).text


_HASH_CONFIG = NormalizationConfig(zwsp_policy="strip")
_PUNCT_FOR_HASH = re.compile(rf"[\s{_KHMER_TERMINATORS}!-/:-@\[-`{{-~]+")


def normalize_for_hashing(text: str) -> str:
    """Aggressive normal form used only as a *deduplication key*.

    Lower-cases Latin, drops all whitespace and punctuation and strips ZWSP so
    that two records differing only in formatting hash identically.  Never write
    this form back into a dataset - it is lossy by design.
    """
    normalised = normalize_text(text, _HASH_CONFIG)
    return _PUNCT_FOR_HASH.sub("", normalised.lower())
