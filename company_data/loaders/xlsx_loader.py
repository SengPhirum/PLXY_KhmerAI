"""XLSX loader (openpyxl).

Same philosophy as the CSV loader: one document per data row so retrieval stays
precise, with the sheet name carried through as ``part`` for citations.  Merged
header rows and leading blank rows - both endemic in hand-made price lists - are
handled by scanning for the first row that looks like a header.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any

from company_data.loaders.base import LoadedDocument, LoaderError, check_file, require
from company_data.loaders.csv_loader import canonical_column

__all__ = ["load_xlsx"]

_MAX_HEADER_SCAN = 10


def _cell_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return (
            value.date().isoformat() if value.time() == datetime.min.time() else value.isoformat()
        )
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _find_header_row(rows: list[list[Any]]) -> int:
    """Index of the first row that looks like a header, else 0."""
    best_index, best_score = 0, -1
    for index, row in enumerate(rows[:_MAX_HEADER_SCAN]):
        cells = [_cell_to_text(c) for c in row]
        filled = sum(1 for c in cells if c)
        if filled < 2:
            continue
        recognised = sum(1 for c in cells if c and canonical_column(c))
        score = recognised * 10 + filled
        if score > best_score:
            best_index, best_score = index, score
    return best_index


def load_xlsx(
    path: str | os.PathLike[str],
    *,
    row_per_document: bool = True,
    sheets: list[str] | None = None,
    max_rows: int | None = None,
) -> list[LoadedDocument]:
    """Load every (or selected) worksheet of an .xlsx workbook."""
    target = check_file(path)
    openpyxl = require("openpyxl", extra="rag")

    try:
        workbook = openpyxl.load_workbook(str(target), read_only=True, data_only=True)
    except Exception as exc:
        raise LoaderError(f"could not open XLSX {target}: {exc}") from exc

    documents: list[LoadedDocument] = []
    try:
        for sheet in workbook.worksheets:
            if sheets and sheet.title not in sheets:
                continue
            rows = [list(r) for r in sheet.iter_rows(values_only=True)]
            rows = [r for r in rows if any(_cell_to_text(c) for c in r)]
            if not rows:
                continue

            header_index = _find_header_row(rows)
            headers = [_cell_to_text(c) for c in rows[header_index]]
            headers = [h or f"column_{i + 1}" for i, h in enumerate(headers)]
            mapping = {h: canonical_column(h) for h in headers}
            data_rows = rows[header_index + 1 :]
            if max_rows is not None:
                data_rows = data_rows[:max_rows]

            base_metadata: dict[str, Any] = {
                "xlsx_sheet": sheet.title,
                "xlsx_columns": headers,
                "xlsx_recognised_columns": sorted({v for v in mapping.values() if v}),
                "xlsx_rows": len(data_rows),
            }

            rendered_rows: list[tuple[dict[str, str], str]] = []
            for row in data_rows:
                values = {
                    headers[i]: _cell_to_text(cell)
                    for i, cell in enumerate(row)
                    if i < len(headers) and _cell_to_text(cell)
                }
                if not values:
                    continue
                lines = [
                    f"{mapping.get(header) or header}: {value}" for header, value in values.items()
                ]
                # Only columns that map onto a canonical field become structured
                # metadata; the rest survive in the rendered body text.
                structured: dict[str, str] = {}
                for header, value in values.items():
                    canonical = mapping.get(header)
                    if canonical:
                        structured[canonical] = value
                rendered_rows.append((structured, "\n".join(lines)))

            if not rendered_rows:
                continue

            if not row_per_document:
                documents.append(
                    LoadedDocument(
                        text="\n\n".join(body for _, body in rendered_rows),
                        source_path=str(target),
                        title=sheet.title,
                        metadata=base_metadata,
                        part=sheet.title,
                        loader="xlsx",
                    )
                )
                continue

            for index, (structured, body) in enumerate(rendered_rows, start=1):
                identifier = (
                    structured.get("product_id")
                    or structured.get("service_id")
                    or structured.get("product_name")
                    or f"{sheet.title}-row{index}"
                )
                documents.append(
                    LoadedDocument(
                        text=body,
                        source_path=str(target),
                        title=str(
                            structured.get("product_name")
                            or structured.get("service_name")
                            or identifier
                        ),
                        metadata={**base_metadata, **structured, "xlsx_row": index},
                        part=f"{sheet.title}!row{index}",
                        loader="xlsx",
                    )
                )
    finally:
        workbook.close()

    if not documents:
        raise LoaderError(f"{target} contains no readable rows")
    return documents
