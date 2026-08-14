"""Document loaders.

Each loader turns one file into one or more :class:`LoadedDocument` values -
raw text plus whatever metadata the format itself carries (PDF outline, DOCX
core properties, HTML meta tags, CSV/XLSX column headers).  Interpreting that
metadata into the canonical schema is ``company_data.normalize``'s job.

Optional third-party parsers are imported lazily inside each loader so that
``import company_data`` never requires ``requirements/rag.txt``.  A missing
parser raises :class:`LoaderDependencyError` with the exact install command
rather than an opaque ``ImportError``.
"""

from __future__ import annotations

from company_data.loaders.base import (
    LoadedDocument,
    LoaderDependencyError,
    LoaderError,
    UnsupportedFormatError,
)
from company_data.loaders.csv_loader import load_csv
from company_data.loaders.docx_loader import load_docx
from company_data.loaders.html_loader import load_html
from company_data.loaders.pdf_loader import load_pdf
from company_data.loaders.registry import (
    SUPPORTED_EXTENSIONS,
    iter_documents,
    load_any,
    load_directory,
    loader_for,
)
from company_data.loaders.text_loader import load_text
from company_data.loaders.xlsx_loader import load_xlsx

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "LoadedDocument",
    "LoaderDependencyError",
    "LoaderError",
    "UnsupportedFormatError",
    "iter_documents",
    "load_any",
    "load_csv",
    "load_directory",
    "load_docx",
    "load_html",
    "load_pdf",
    "load_text",
    "load_xlsx",
    "loader_for",
]
