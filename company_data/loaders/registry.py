"""Extension -> loader dispatch, plus safe directory walking."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

from company_data.loaders.base import (
    LoadedDocument,
    LoaderError,
    UnsupportedFormatError,
)
from company_data.loaders.csv_loader import load_csv
from company_data.loaders.docx_loader import load_docx
from company_data.loaders.html_loader import load_html
from company_data.loaders.pdf_loader import load_pdf
from company_data.loaders.text_loader import load_text
from company_data.loaders.xlsx_loader import load_xlsx

__all__ = ["SUPPORTED_EXTENSIONS", "loader_for", "load_any", "load_directory", "iter_documents"]

Loader = Callable[..., list[LoadedDocument]]

_REGISTRY: dict[str, Loader] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".html": load_html,
    ".htm": load_html,
    ".xhtml": load_html,
    ".txt": load_text,
    ".md": load_text,
    ".markdown": load_text,
    ".csv": load_csv,
    ".tsv": load_csv,
    ".xlsx": load_xlsx,
    ".xlsm": load_xlsx,
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_REGISTRY)

# Anything not in the allow-list is refused outright.  This is the Phase 16
# "allowed file types" control: an ingestion directory is untrusted input, and a
# .exe or .lnk placed there must never reach a parser.
_DENY_NAMES = frozenset({".ds_store", "thumbs.db", "desktop.ini"})


def loader_for(path: str | os.PathLike[str]) -> Loader:
    suffix = Path(path).suffix.lower()
    loader = _REGISTRY.get(suffix)
    if loader is None:
        raise UnsupportedFormatError(
            f"no loader for {suffix or '(no extension)'}; supported: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return loader


def load_any(path: str | os.PathLike[str], **kwargs: object) -> list[LoadedDocument]:
    """Load one file with the loader registered for its extension."""
    return loader_for(path)(path, **kwargs)


def iter_documents(
    root: str | os.PathLike[str],
    *,
    recursive: bool = True,
    allowed_extensions: frozenset[str] | None = None,
) -> Iterator[Path]:
    """Yield ingestible files under ``root``, skipping hidden and denied names."""
    base = Path(root)
    if base.is_file():
        yield base
        return
    if not base.is_dir():
        raise LoaderError(f"not a directory: {base}")

    allowed = allowed_extensions or SUPPORTED_EXTENSIONS
    pattern = "**/*" if recursive else "*"
    for path in sorted(base.glob(pattern)):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(base).parts):
            continue
        if path.name.lower() in _DENY_NAMES:
            continue
        if path.suffix.lower() not in allowed:
            continue
        yield path


def load_directory(
    root: str | os.PathLike[str],
    *,
    recursive: bool = True,
    allowed_extensions: frozenset[str] | None = None,
    on_error: Callable[[Path, Exception], None] | None = None,
) -> list[LoadedDocument]:
    """Load every supported file under ``root``.

    A failure on one file is reported through ``on_error`` and skipped: one
    corrupt PDF must never abort a 500-document ingestion batch.
    """
    documents: list[LoadedDocument] = []
    for path in iter_documents(root, recursive=recursive, allowed_extensions=allowed_extensions):
        try:
            documents.extend(load_any(path))
        except Exception as exc:  # noqa: BLE001 - deliberately broad, reported not raised
            if on_error is None:
                raise
            on_error(path, exc)
    return documents
