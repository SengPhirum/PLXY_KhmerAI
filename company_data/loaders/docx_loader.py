"""DOCX loader (python-docx).

Tables are preserved as pipe-delimited rows rather than being flattened into
prose: a warranty matrix loses all of its meaning when the column structure is
dropped, and the chunker in ``rag/chunking.py`` keeps a table row intact.
"""

from __future__ import annotations

import os

from company_data.loaders.base import LoadedDocument, LoaderError, check_file, require

__all__ = ["load_docx"]

_HEADING_PREFIX = {
    "Heading 1": "# ",
    "Heading 2": "## ",
    "Heading 3": "### ",
    "Heading 4": "#### ",
    "Title": "# ",
}


def _table_to_text(table: object) -> str:
    rows: list[str] = []
    for row in table.rows:  # type: ignore[attr-defined]
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    if not rows:
        return ""
    return "\n".join(rows)


def load_docx(path: str | os.PathLike[str]) -> list[LoadedDocument]:
    """Extract paragraphs, headings and tables from a .docx file, in order."""
    target = check_file(path)
    docx = require("docx", extra="rag")

    try:
        document = docx.Document(str(target))
    except Exception as exc:  # noqa: BLE001 - python-docx raises package-specific errors
        raise LoaderError(f"could not open DOCX {target}: {exc}") from exc

    blocks: list[str] = []
    heading_count = 0

    # Walk the body in document order so tables stay where the author put them.
    body = document.element.body
    tables = iter(document.tables)
    paragraphs = iter(document.paragraphs)
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            try:
                paragraph = next(paragraphs)
            except StopIteration:
                continue
            text = paragraph.text.strip()
            if not text:
                continue
            style = getattr(paragraph.style, "name", "") or ""
            prefix = _HEADING_PREFIX.get(style, "")
            if prefix:
                heading_count += 1
            blocks.append(f"{prefix}{text}")
        elif tag == "tbl":
            try:
                table = next(tables)
            except StopIteration:
                continue
            rendered = _table_to_text(table)
            if rendered:
                blocks.append(rendered)

    core = document.core_properties
    metadata = {
        "docx_author": core.author or "",
        "docx_created": core.created.isoformat() if core.created else "",
        "docx_modified": core.modified.isoformat() if core.modified else "",
        "docx_category": core.category or "",
        "docx_comments": core.comments or "",
        "docx_headings": heading_count,
        "docx_tables": len(document.tables),
    }

    return [
        LoadedDocument(
            text="\n\n".join(blocks).strip(),
            source_path=str(target),
            title=(core.title or "").strip(),
            metadata=metadata,
            loader="docx",
        )
    ]
