"""Plain text and Markdown loader with front-matter support.

Company teams overwhelmingly prefer editing a Markdown file to filling in a
spreadsheet, so ``.md`` with a YAML front-matter block is the recommended source
format for policies.  The front matter maps straight onto the canonical schema::

    ---
    document_title: គោលការណ៍ធានា
    product_id: QN-4500A
    category: warranty
    version: "2.1"
    effective_date: 2026-01-01
    status: active
    confidentiality: public
    owner: after-sales
    ---
    ផលិតផលនេះមានការធានារយៈពេល ២ ឆ្នាំ។
"""

from __future__ import annotations

import os
import re

import yaml

from company_data.loaders.base import LoadedDocument, LoaderError, check_file

__all__ = ["load_text", "split_front_matter"]

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_ENCODINGS = ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1")


def split_front_matter(text: str) -> tuple[dict[str, object], str]:
    """Return ``(front_matter, body)``; an absent or invalid block yields ``({}, text)``."""
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}, text
    if not isinstance(data, dict):
        return {}, text
    return data, text[match.end() :]


def _read_text(path: str) -> str:
    """Decode with a small ladder of encodings.

    Khmer files exported from Windows tools are frequently UTF-16 or UTF-8 with a
    BOM; failing on those would silently lose whole document sets.
    """
    raw = open(path, "rb").read()  # noqa: SIM115, PTH123 - explicit binary read
    try:
        from charset_normalizer import from_bytes  # noqa: PLC0415 - optional

        best = from_bytes(raw).best()
        if best is not None:
            return str(best)
    except ImportError:
        pass
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise LoaderError(f"could not decode {path} with any of {_ENCODINGS}")


def load_text(path: str | os.PathLike[str]) -> list[LoadedDocument]:
    """Load a ``.txt`` or ``.md`` file, honouring YAML front matter."""
    target = check_file(path)
    text = _read_text(str(target))
    front_matter, body = split_front_matter(text)

    title = str(front_matter.get("document_title") or "")
    if not title and target.suffix.lower() in (".md", ".markdown"):
        heading = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        if heading:
            title = heading.group(1).strip()

    return [
        LoadedDocument(
            text=body.strip(),
            source_path=str(target),
            title=title,
            metadata=dict(front_matter),
            loader="text",
        )
    ]
