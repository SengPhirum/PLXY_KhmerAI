"""PDF loader (pypdf).

Khmer PDFs are the hardest real input in this project: many are produced by
Khmer-legacy fonts (Limon, ABC-Zerk) that encode Khmer as *Latin code points*
with a custom glyph map.  Text extracted from those files is mojibake, not
Khmer, and silently indexing it poisons retrieval.

``load_pdf`` therefore does two things beyond extraction:

* reports per-page extraction success, so a scanned (image-only) PDF is
  detected rather than yielding an empty document;
* flags legacy-font mojibake by checking whether a page that *should* be Khmer
  contains no Khmer code points while using a known legacy font name.  Flagged
  documents are marked ``needs_review`` by ``company_data.validate`` instead of
  entering the index.
"""

from __future__ import annotations

import os
import re

from company_data.loaders.base import LoadedDocument, LoaderError, check_file, require

__all__ = ["LEGACY_KHMER_FONTS", "load_pdf"]

# Font families that encode Khmer in a non-Unicode private mapping.
LEGACY_KHMER_FONTS = (
    "limon",
    "abc-zerk",
    "abczerk",
    "khek",
    "kh-",
    "preahvihear-legacy",
    "truth",
    "sbbic-legacy",
)
_KHMER_RE = re.compile(r"[ក-៿]")
_MOJIBAKE_HINT = re.compile(r"[·¬¤¦§¨©ª«¯°±²³´µ¶¸¹º»¼½¾]{3,}")


def _looks_like_legacy_khmer(text: str, fonts: set[str]) -> bool:
    if _KHMER_RE.search(text):
        return False
    if _MOJIBAKE_HINT.search(text):
        return True
    lowered = {f.lower() for f in fonts}
    return any(any(marker in font for marker in LEGACY_KHMER_FONTS) for font in lowered)


def _page_fonts(page: object) -> set[str]:
    """Best-effort font-name extraction; never raises on a malformed resource dict."""
    fonts: set[str] = set()
    try:
        resources = page["/Resources"]  # type: ignore[index]
        font_dict = resources.get("/Font", {})
        for ref in font_dict.values():
            obj = ref.get_object() if hasattr(ref, "get_object") else ref
            name = obj.get("/BaseFont")
            if name:
                fonts.add(str(name).lstrip("/"))
    except Exception:
        return fonts
    return fonts


def load_pdf(
    path: str | os.PathLike[str], *, split_pages: bool = False, password: str | None = None
) -> list[LoadedDocument]:
    """Extract text from a PDF.

    ``split_pages=True`` yields one :class:`LoadedDocument` per page, which keeps
    page numbers available for citations in long policy documents.
    """
    target = check_file(path)
    pypdf = require("pypdf")

    try:
        reader = pypdf.PdfReader(str(target))
        if reader.is_encrypted and reader.decrypt(password or "") == 0:
            raise LoaderError(f"{target} is password protected")
    except LoaderError:
        raise
    except Exception as exc:
        raise LoaderError(f"could not open PDF {target}: {exc}") from exc

    doc_info = {}
    try:
        if reader.metadata:
            doc_info = {
                str(k).lstrip("/"): str(v) for k, v in reader.metadata.items() if v is not None
            }
    except Exception:
        doc_info = {}

    pages: list[tuple[int, str, set[str]]] = []
    failed_pages: list[int] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            failed_pages.append(number)
            continue
        pages.append((number, text, _page_fonts(page)))

    all_text = "\n".join(text for _, text, _ in pages)
    all_fonts: set[str] = set()
    for _, _, fonts in pages:
        all_fonts |= fonts

    empty_pages = sum(1 for _, text, _ in pages if not text.strip())
    base_metadata = {
        "pdf_pages": len(reader.pages),
        "pdf_empty_pages": empty_pages,
        "pdf_failed_pages": failed_pages,
        "pdf_fonts": sorted(all_fonts)[:20],
        "pdf_info": doc_info,
        "likely_scanned": bool(reader.pages) and empty_pages == len(reader.pages),
        "legacy_khmer_font_suspected": _looks_like_legacy_khmer(all_text, all_fonts),
    }
    title = str(doc_info.get("Title", "")).strip()

    if split_pages:
        return [
            LoadedDocument(
                text=text.strip(),
                source_path=str(target),
                title=title,
                metadata={**base_metadata, "page": number},
                part=f"p{number}",
                loader="pdf",
            )
            for number, text, _ in pages
            if text.strip()
        ]

    return [
        LoadedDocument(
            text=all_text.strip(),
            source_path=str(target),
            title=title,
            metadata=base_metadata,
            loader="pdf",
        )
    ]
