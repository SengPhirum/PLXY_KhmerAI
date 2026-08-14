"""HTML loader.

Reuses the crawl-hardened stripper from ``preprocessing.html_cleanup`` so the
same navigation/cookie/footer rules apply to company web exports as to public
corpora, and additionally lifts ``<meta>`` tags into loader metadata (many CMS
exports carry the effective date and the owner there).
"""

from __future__ import annotations

import os
import re
from html import unescape

from company_data.loaders.base import LoadedDocument, check_file
from company_data.loaders.text_loader import _read_text
from preprocessing.html_cleanup import clean_web_text

__all__ = ["load_html", "extract_meta"]

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_RE = re.compile(r"<meta\s+([^>]+?)/?>", re.IGNORECASE)
_ATTR_RE = re.compile(r"([\w:-]+)\s*=\s*(\"([^\"]*)\"|'([^']*)'|([^\s\"'>]+))")
_LANG_RE = re.compile(r"<html[^>]*\blang\s*=\s*[\"']?([\w-]+)", re.IGNORECASE)

# Meta-tag names that map directly onto CompanyDocument fields.
_CANONICAL_META_FIELDS = frozenset(
    {
        "document_title", "product_id", "product_name", "service_id", "service_name",
        "category", "subcategory", "version", "effective_date", "expiration_date",
        "language", "confidentiality", "access_level", "status", "owner", "source_url",
    }
)


def extract_meta(html: str) -> dict[str, str]:
    """Collect ``<meta name|property=... content=...>`` pairs."""
    out: dict[str, str] = {}
    for match in _META_RE.finditer(html):
        attrs: dict[str, str] = {}
        for attr in _ATTR_RE.finditer(match.group(1)):
            value = attr.group(3) or attr.group(4) or attr.group(5) or ""
            attrs[attr.group(1).lower()] = unescape(value)
        key = attrs.get("name") or attrs.get("property") or attrs.get("http-equiv")
        if key and "content" in attrs:
            out[key.lower()] = attrs["content"]
    return out


def load_html(path: str | os.PathLike[str]) -> list[LoadedDocument]:
    """Load an HTML file into clean text plus its meta tags."""
    target = check_file(path)
    raw = _read_text(str(target))

    title_match = _TITLE_RE.search(raw)
    title = unescape(title_match.group(1)).strip() if title_match else ""
    lang_match = _LANG_RE.search(raw)

    meta = extract_meta(raw)
    metadata: dict[str, object] = {f"html_meta_{k}": v for k, v in meta.items()}
    # Meta tags whose name matches a canonical schema field are also exposed
    # unprefixed, so a CMS export that declares `<meta name="status" ...>`
    # governs the record the same way YAML front matter would.
    for name, value in meta.items():
        canonical = name.split(":")[-1]
        if canonical in _CANONICAL_META_FIELDS:
            metadata.setdefault(canonical, value)
    if lang_match:
        metadata["html_lang"] = lang_match.group(1)
        metadata.setdefault("language", lang_match.group(1).split("-")[0])
    if "og:title" in meta and not title:
        title = meta["og:title"]

    return [
        LoadedDocument(
            text=clean_web_text(raw),
            source_path=str(target),
            title=title,
            metadata=metadata,
            loader="html",
        )
    ]
