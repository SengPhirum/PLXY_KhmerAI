"""CSV/TSV loader - the format product catalogues actually arrive in.

A price list is not prose.  Flattening a 400-row CSV into one document destroys
retrieval: a question about one product would match a chunk containing fifty
unrelated products.  This loader therefore emits **one document per row** by
default, rendering the row as ``field: value`` lines in a stable column order.
That gives the chunker a self-contained, semantically complete unit.

Column names are matched case-insensitively against the canonical schema
(``product_id``, ``price``, ``warranty_months``, ...) plus common Khmer headers,
so a catalogue exported from a Khmer ERP maps without hand editing.
"""

from __future__ import annotations

import csv
import io
import os
import re
from typing import Any

from company_data.loaders.base import LoadedDocument, LoaderError, check_file
from company_data.loaders.text_loader import _read_text

__all__ = ["CANONICAL_COLUMNS", "canonical_column", "load_csv"]

# Canonical field -> accepted header spellings (lower-cased, punctuation-free).
CANONICAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "product_id": (
        "productid",
        "sku",
        "modelnumber",
        "model",
        "itemcode",
        "code",
        "លេខម៉ូដែល",
        "លេខកូដ",
    ),
    "product_name": ("productname", "name", "item", "title", "ឈ្មោះផលិតផល", "ផលិតផល"),
    "service_id": ("serviceid", "servicecode"),
    "service_name": ("servicename", "service", "សេវាកម្ម"),
    "category": ("category", "type", "ប្រភេទ"),
    "subcategory": ("subcategory", "subtype"),
    "price": ("price", "unitprice", "retailprice", "តម្លៃ"),
    "currency": ("currency", "ccy", "រូបិយប័ណ្ណ"),
    "warranty": ("warranty", "warrantymonths", "warrantyperiod", "ការធានា", "រយៈពេលធានា"),
    "availability": ("availability", "stock", "instock", "quantity", "ស្តុក", "ចំនួន"),
    "specification": ("specification", "specs", "spec", "លក្ខណៈបច្ចេកទេស"),
    "description": ("description", "details", "notes", "ការពិពណ៌នា"),
    "effective_date": ("effectivedate", "validfrom", "startdate", "ថ្ងៃចាប់ផ្តើម"),
    "expiration_date": ("expirationdate", "validto", "enddate", "expiry", "ថ្ងៃផុតកំណត់"),
    "version": ("version", "rev", "revision"),
    "status": ("status", "state", "ស្ថានភាព"),
    "language": ("language", "lang", "ភាសា"),
    "owner": ("owner", "responsible", "department"),
    # Governance columns. Without these a catalogue inherits the safe default
    # (`internal`) and is therefore never served to customers - which looks like
    # "retrieval returns nothing" rather than like a configuration problem, so
    # `company_data/validate.py` reports it explicitly.
    "confidentiality": ("confidentiality", "visibility", "sensitivity"),
    "access_level": ("accesslevel", "audience"),
    "source_url": ("sourceurl", "url", "link"),
}
_NORMALISE = re.compile(r"[^a-z0-9ក-៿]")


def canonical_column(header: str) -> str | None:
    """Map a raw header to a canonical field name, or ``None`` if unknown."""
    key = _NORMALISE.sub("", header.strip().lower())
    if not key:
        return None
    for canonical, spellings in CANONICAL_COLUMNS.items():
        if key == _NORMALISE.sub("", canonical) or key in spellings:
            return canonical
    return None


def _sniff_dialect(sample: str) -> type[csv.Dialect] | csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def _render_row(row: dict[str, str], mapping: dict[str, str | None]) -> str:
    """Render a row as readable ``label: value`` lines, canonical fields first."""
    canonical_lines: list[str] = []
    extra_lines: list[str] = []
    for header, value in row.items():
        if value is None or not str(value).strip():
            continue
        label = mapping.get(header) or header.strip()
        line = f"{label}: {str(value).strip()}"
        (canonical_lines if mapping.get(header) else extra_lines).append(line)
    return "\n".join(canonical_lines + extra_lines)


def load_csv(
    path: str | os.PathLike[str],
    *,
    row_per_document: bool = True,
    max_rows: int | None = None,
) -> list[LoadedDocument]:
    """Load a CSV/TSV catalogue.

    With ``row_per_document=False`` the whole file becomes one document, which is
    only appropriate for very small reference tables.
    """
    target = check_file(path)
    raw = _read_text(str(target))
    if not raw.strip():
        raise LoaderError(f"{target} contains no data")

    dialect = _sniff_dialect(raw[:8192])
    reader = csv.DictReader(io.StringIO(raw), dialect=dialect)
    if not reader.fieldnames:
        raise LoaderError(f"{target} has no header row")

    mapping = {header: canonical_column(header) for header in reader.fieldnames}
    recognised = {v for v in mapping.values() if v}

    documents: list[LoadedDocument] = []
    rows: list[dict[str, str]] = []
    for index, row in enumerate(reader, start=1):
        if max_rows is not None and index > max_rows:
            break
        cleaned = {k: (v or "") for k, v in row.items() if k is not None}
        if not any(str(v).strip() for v in cleaned.values()):
            continue
        rows.append(cleaned)

    if not rows:
        raise LoaderError(f"{target} has a header but no data rows")

    base_metadata: dict[str, Any] = {
        "csv_columns": list(reader.fieldnames),
        "csv_recognised_columns": sorted(recognised),
        "csv_rows": len(rows),
        "csv_delimiter": getattr(dialect, "delimiter", ","),
    }

    if not row_per_document:
        body = "\n\n".join(_render_row(row, mapping) for row in rows)
        return [
            LoadedDocument(
                text=body,
                source_path=str(target),
                metadata=base_metadata,
                loader="csv",
            )
        ]

    for index, row in enumerate(rows, start=1):
        structured = {
            canonical: row[header].strip()
            for header, canonical in mapping.items()
            if canonical and row.get(header, "").strip()
        }
        identifier = (
            structured.get("product_id")
            or structured.get("service_id")
            or structured.get("product_name")
            or f"row{index}"
        )
        title = structured.get("product_name") or structured.get("service_name") or identifier
        documents.append(
            LoadedDocument(
                text=_render_row(row, mapping),
                source_path=str(target),
                title=str(title),
                metadata={**base_metadata, **structured, "csv_row": index},
                part=f"row{index}",
                loader="csv",
            )
        )
    return documents
