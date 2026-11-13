"""Build data/processed/units.jsonl and data/manifest.json from the raw Gutenberg HTML.

Run from the repo root:  python starter/scripts/build_corpus.py
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter"))
from src.ingestion import MIN_CHARS, SOURCE_URL, TARGET_CHARS, load_stories, make_units  # noqa: E402

RAW = ROOT / "data" / "raw" / "pg108-images.html"
OUT_UNITS = ROOT / "data" / "processed" / "units.jsonl"
OUT_MANIFEST = ROOT / "data" / "manifest.json"
OUT_LOG = ROOT / "data" / "processed" / "ingestion_log.txt"


def main() -> None:
    stories = load_stories(RAW)
    units, log = [], []
    for s in stories:
        us = make_units(s)
        units.extend(us)
        log.append(f"S{s['no']:02d} {s['title']}: {len(s['paragraphs'])} paragraphs -> {len(us)} units")
    bad = [u["unit_id"] for u in units if u["n_chars"] < MIN_CHARS]
    log.append(f"units below {MIN_CHARS} chars: {bad or 'none'}")
    log.append(f"empty paragraphs/pages dropped: handled by parser (no empty unit kept)")

    OUT_UNITS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_UNITS.open("w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    OUT_LOG.write_text("\n".join(log) + "\n", encoding="utf-8")

    manifest = {
        "source_id": "pg108",
        "title": "The Return of Sherlock Holmes",
        "author": "Arthur Conan Doyle",
        "source_path_or_url": SOURCE_URL,
        "local_copy": "data/raw/pg108-images.html",
        "license_or_permission": "Public domain (Project Gutenberg eBook #108, release 2006-03-08); "
                                 "free to download and reuse per the Project Gutenberg License.",
        "accessed_at": str(date.today()),
        "page_rule": (
            "The book has no physical pages. Content = paragraphs of the 13 stories only "
            "(Gutenberg header/footer, contents list and licence excluded). Paragraphs are packed in "
            f"order into units closing at >= {TARGET_CHARS} normalized chars without splitting a "
            f"paragraph; a final unit < {MIN_CHARS} chars is merged into the previous one. "
            "Unit ID S<story>-U<index> is the page ID."),
        "n_units": len(units),
        "n_chars_total": sum(u["n_chars"] for u in units),
        "stories": [{"document_id": f"S{s['no']:02d}", "title": s["title"],
                     "units": [u["unit_id"] for u in units if u["document_id"] == f"S{s['no']:02d}"],
                     "n_chars": sum(u["n_chars"] for u in units if u["document_id"] == f"S{s['no']:02d}")}
                    for s in stories],
        "units": [{"unit_id": u["unit_id"], "n_chars": u["n_chars"]} for u in units],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(log))
    print(f"TOTAL units={len(units)} chars={manifest['n_chars_total']}")


if __name__ == "__main__":
    main()
