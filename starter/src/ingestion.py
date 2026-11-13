"""Read and normalize the Gutenberg HTML book, then split it into traceable units.

Why units: the book has no physical pages, so GUIDE.md allows splitting the text into
ordered units of >= 1,500 normalized characters. Each unit is the "page" of this lab:
unit ID = ``S<story>-U<index>`` (e.g. ``S06-U004``) and is what the benchmark cites as gold.
"""
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

from .models import DocumentPage

TARGET_CHARS = 3000   # close a unit once it reaches this size (paragraphs are never split)
MIN_CHARS = 1500      # GUIDE.md minimum; a short last unit is merged into the previous one
SOURCE_URL = "https://www.gutenberg.org/cache/epub/108/pg108-images.html"


def normalize_text(text: str) -> str:
    """Unicode NFC + collapse whitespace. Keeps curly quotes/dashes so quotes stay verbatim."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("​", "")
    return re.sub(r"\s+", " ", text).strip()


class _BookParser(HTMLParser):
    """Collect <p> paragraphs per story. A story starts at <h2><a id="chapNN">; the body
    stops at the 'END OF THE PROJECT GUTENBERG EBOOK' marker (the licence follows it)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stories: list[dict] = []
        self._in_title = False
        self._title_buf: list[str] = []
        self._in_p = False
        self._buf: list[str] = []
        self._done = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and str(a.get("id", "")).startswith("chap"):
            self._in_title, self._title_buf = True, []
            self._pending_no = int(a["id"][4:])
        elif tag == "p" and self.stories and not self._done:
            self._in_p, self._buf = True, []
        elif tag == "br" and self._in_p:
            self._buf.append(" ")

    def handle_endtag(self, tag):
        if tag == "h2" and self._in_title:
            self._in_title = False
            self.stories.append({"no": self._pending_no,
                                 "title": normalize_text("".join(self._title_buf)),
                                 "paragraphs": []})
        elif tag == "p" and self._in_p:
            self._in_p = False
            para = normalize_text("".join(self._buf))
            if para and para != "THE END":
                self.stories[-1]["paragraphs"].append(para)

    def handle_data(self, data):
        if "END OF THE PROJECT GUTENBERG EBOOK" in data:
            self._done = True
        if self._in_title:
            self._title_buf.append(data)
        elif self._in_p and not self._done:
            self._buf.append(data)


def load_stories(path: Path) -> list[dict]:
    """Return [{no, title, paragraphs}] for the 13 stories; raise if the structure is off."""
    parser = _BookParser()
    parser.feed(Path(path).read_text(encoding="utf-8"))
    # The <h2> of each story contains an <a id="chapNN"> but the "Contents" list also has
    # anchors (href, not id), so only real story headings were captured.
    if len(parser.stories) != 13:
        raise ValueError(f"expected 13 stories, found {len(parser.stories)}")
    for s in parser.stories:
        if not s["paragraphs"]:
            raise ValueError(f"story {s['no']} has no paragraphs")
    return parser.stories


def make_units(story: dict) -> list[dict]:
    """Pack consecutive paragraphs into units of ~TARGET_CHARS, never cutting a paragraph."""
    groups: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for para in story["paragraphs"]:
        cur.append(para)
        size += len(para) + 1
        if size >= TARGET_CHARS:
            groups.append(cur)
            cur, size = [], 0
    if cur:
        if groups and sum(len(p) + 1 for p in cur) < MIN_CHARS:
            groups[-1].extend(cur)   # merge a too-short tail into the previous unit
        else:
            groups.append(cur)
    doc_id = f"S{story['no']:02d}"
    units = []
    for i, paras in enumerate(groups, start=1):
        text = "\n\n".join(paras)
        units.append({
            "unit_id": f"{doc_id}-U{i:03d}",
            "document_id": doc_id,
            "page": i,
            "story": story["title"],
            "text": text,
            "n_chars": len(text),
            "n_paragraphs": len(paras),
        })
    return units


def load_pages(path: Path, source: str = SOURCE_URL) -> list[DocumentPage]:
    """Corpus as DocumentPage objects (one per unit) with traceable metadata."""
    pages = []
    for story in load_stories(path):
        for u in make_units(story):
            pages.append(DocumentPage(
                document_id=u["document_id"], source=source, page=u["page"], text=u["text"],
                metadata={"unit_id": u["unit_id"], "story": u["story"], "story_no": story["no"]},
            ))
    return pages
