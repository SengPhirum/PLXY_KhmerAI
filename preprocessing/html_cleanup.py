"""HTML / boilerplate removal for crawled Khmer corpora.

Web-crawled Khmer (FineWeb2, CulturaX) arrives with navigation menus, cookie
banners, share widgets and footers.  Left in, they teach the model to emit
``ចែករំលែក | Facebook | Twitter`` in the middle of a support answer.

The stripper is written against the standard library ``html.parser`` so the
preprocessing pipeline installs from ``requirements/base.txt``.  When
``beautifulsoup4`` is available (``requirements/rag.txt``) the same public
functions transparently use it for malformed markup, which is more robust on
real crawl data.
"""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import Final

__all__ = [
    "strip_html",
    "remove_boilerplate",
    "clean_web_text",
    "BOILERPLATE_PATTERNS",
]

_DROP_ELEMENTS: Final = frozenset(
    {
        "script", "style", "noscript", "svg", "canvas", "iframe", "object",
        "embed", "template", "head", "nav", "footer", "aside", "form",
        "button", "select", "option", "input", "textarea", "video", "audio",
    }
)
_BLOCK_ELEMENTS: Final = frozenset(
    {
        "p", "div", "br", "hr", "li", "tr", "td", "th", "section", "article",
        "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "table",
        "ul", "ol", "dl", "dd", "dt", "figcaption",
    }
)
# Navigation-ish containers identified by attribute, not tag.
_DROP_ATTR_HINTS: Final = re.compile(
    r"(?:^|[\s_-])(?:nav|menu|breadcrumb|sidebar|footer|header|cookie|consent|"
    r"banner|advert|ads?|popup|modal|share|social|comment|related|pagination|"
    r"newsletter|subscribe|widget|skip-link)(?:[\s_-]|$)",
    re.IGNORECASE,
)


class _TextExtractor(HTMLParser):
    """Collect visible text, dropping non-content elements and their subtrees."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._skip_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip_depth:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        if tag in _DROP_ELEMENTS or self._looks_like_chrome(attrs):
            self._skip_tag = tag
            self._skip_depth = 1
            return
        if tag in _BLOCK_ELEMENTS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._skip_tag = None
            return
        if tag in _BLOCK_ELEMENTS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(data)

    def error(self, message: str) -> None:  # pragma: no cover - py<3.10 compat hook
        return

    @staticmethod
    def _looks_like_chrome(attrs: list[tuple[str, str | None]]) -> bool:
        for name, value in attrs:
            if name in ("class", "id", "role", "aria-label") and value:
                if _DROP_ATTR_HINTS.search(value):
                    return True
            if name == "role" and value in ("navigation", "banner", "complementary"):
                return True
        return False

    def text(self) -> str:
        return "".join(self.parts)


def _strip_with_bs4(html: str) -> str | None:
    """Use BeautifulSoup when installed - it recovers from broken markup."""
    try:
        from bs4 import BeautifulSoup  # noqa: PLC0415 - optional dependency
    except ImportError:
        return None

    soup = BeautifulSoup(html, "html.parser")
    for element in soup.find_all(list(_DROP_ELEMENTS)):
        element.decompose()
    for element in soup.find_all(attrs={"class": _DROP_ATTR_HINTS}):
        element.decompose()
    for element in soup.find_all(attrs={"id": _DROP_ATTR_HINTS}):
        element.decompose()
    return soup.get_text(separator="\n")


def strip_html(html: str) -> str:
    """Return the visible text of an HTML document."""
    if not html:
        return ""
    if "<" not in html:
        return html
    text = _strip_with_bs4(html)
    if text is None:
        parser = _TextExtractor()
        try:
            parser.feed(html)
            parser.close()
        except Exception:  # noqa: BLE001 - malformed markup must not kill a corpus run
            text = re.sub(r"<[^>]+>", " ", html)
        else:
            text = parser.text()
    return unescape(text)


# Boilerplate that survives tag stripping.  Khmer and English variants both
# occur on Cambodian sites, which are frequently bilingual.
BOILERPLATE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p, re.IGNORECASE | re.MULTILINE)
    for p in (
        r"^\s*(?:copyright\s*)?(?:©|\(c\))\s*\d{4}.*$",
        r"^\s*all rights reserved.*$",
        r"^\s*រក្សាសិទ្ធិ.*$",
        r"^\s*(?:read more|អានបន្ថែម|មើលបន្ថែម)\s*(?:»|>|\.\.\.)?\s*$",
        r"^\s*(?:share|ចែករំលែក)\s*(?:on|:)?\s*(?:facebook|twitter|telegram|line)?\s*$",
        r"^\s*(?:this (?:site|website) uses cookies|we use cookies)[^\n]*$",
        r"^\s*(?:accept|agree|យល់ព្រម)\s*(?:all)?\s*cookies?\s*$",
        r"^\s*(?:subscribe|newsletter|ជាវ)\s*[^\n]{0,60}$",
        r"^\s*(?:previous|next|មុន|បន្ទាប់)\s*(?:page|post|article)?\s*$",
        r"^\s*(?:home|about us|contact us|ទំព័រដើម|អំពីយើង|ទំនាក់ទំនង)\s*$",
        r"^\s*(?:posted|published|updated)\s+(?:on|by)\s+[^\n]{0,80}$",
        r"^\s*(?:tags?|categor(?:y|ies)|ស្លាក|ប្រភេទ)\s*:\s*[^\n]{0,120}$",
        r"^\s*\d+\s*(?:comments?|views?|likes?|shares?)\s*$",
        r"^\s*(?:log ?in|sign ?up|register|ចូល|ចុះឈ្មោះ)\s*$",
        r"^\s*javascript is (?:disabled|required)[^\n]*$",
        r"^\s*(?:loading|កំពុងផ្ទុក)\s*\.{0,3}\s*$",
    )
)

# A line that is mostly separators/pipes is a menu remnant.
_MENU_LINE = re.compile(r"^\s*(?:[^\s|»>·•/\\]+\s*[|»>·•/\\]\s*){2,}[^\s|»>·•/\\]*\s*$")
_URL_ONLY_LINE = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$")
_SYMBOL_NOISE = re.compile(r"[─-╿■-◿]{3,}")


def remove_boilerplate(text: str, *, min_line_chars: int = 2) -> str:
    """Drop navigation, cookie banners, share widgets and menu remnants."""
    if not text:
        return ""
    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)

    kept: list[str] = []
    seen_lines: dict[str, int] = {}
    for raw in text.split("\n"):
        line = raw.strip()
        if len(line) < min_line_chars:
            continue
        if _MENU_LINE.match(line) or _URL_ONLY_LINE.match(line):
            continue
        if _SYMBOL_NOISE.search(line):
            continue
        # A short line repeated many times is a template artefact (a repeated
        # menu item or a "read more" in every card).
        key = line.lower()
        seen_lines[key] = seen_lines.get(key, 0) + 1
        if len(line) < 40 and seen_lines[key] > 2:
            continue
        kept.append(line)
    return "\n".join(kept)


def clean_web_text(html_or_text: str) -> str:
    """Full crawl-cleanup: strip markup, drop boilerplate, tidy blank lines."""
    text = strip_html(html_or_text)
    text = remove_boilerplate(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
