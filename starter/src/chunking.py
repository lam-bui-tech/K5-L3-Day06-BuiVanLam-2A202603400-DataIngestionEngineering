"""Implement fixed-size and structure-aware chunking."""
from typing import Any
from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split text into overlapping fixed-size windows and retain page metadata.

    Strategy A: size=500, overlap=50.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must be in [0, size), got overlap={overlap}, size={size}")

    text = page.text
    if not text:
        return []

    step = size - overlap
    chunks: list[Chunk] = []
    unit_id = str(page.metadata.get("unit_id", f"{page.document_id}-P{page.page:03d}"))
    start = 0
    idx = 1

    while start < len(text):
        end = min(start + size, len(text))
        chunk_text = text[start:end]
        chunk_id = f"{unit_id}-A{idx:03d}"

        meta: dict[str, Any] = dict(page.metadata)
        meta.update({
            "chunk_id": chunk_id,
            "chunk_strategy": "A",
            "chunk_index": idx,
            "start_char": start,
            "end_char": end,
            "n_chars": len(chunk_text),
        })

        chunks.append(Chunk(
            chunk_id=chunk_id,
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=chunk_text,
            metadata=meta,
        ))

        idx += 1
        if end >= len(text):
            break
        start += step

    return chunks


def by_paragraph(page: DocumentPage, max_size: int = 1000) -> list[Chunk]:
    """Group paragraphs, split oversized paragraphs and retain metadata.

    Strategy B & C: respects paragraph boundaries (\\n\\n).
    """
    if max_size <= 0:
        raise ValueError(f"max_size must be positive, got {max_size}")

    text = page.text
    if not text:
        return []

    # Units in this corpus use \\n\\n as paragraph separator
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paras:
        return []

    # Break any oversized paragraph that exceeds max_size on its own
    paras: list[str] = []
    for p in raw_paras:
        if len(p) <= max_size:
            paras.append(p)
        else:
            # Split oversized paragraph by sentences or fixed chunks
            p_start = 0
            while p_start < len(p):
                p_end = min(p_start + max_size, len(p))
                paras.append(p[p_start:p_end].strip())
                p_start = p_end

    # Group paragraphs up to max_size
    grouped: list[list[str]] = []
    current_group: list[str] = []
    current_len = 0

    for para in paras:
        added_len = len(para) + (2 if current_group else 0)
        if current_group and (current_len + added_len > max_size):
            grouped.append(current_group)
            current_group = [para]
            current_len = len(para)
        else:
            current_group.append(para)
            current_len += added_len

    if current_group:
        grouped.append(current_group)

    unit_id = str(page.metadata.get("unit_id", f"{page.document_id}-P{page.page:03d}"))
    chunks: list[Chunk] = []

    for idx, g in enumerate(grouped, start=1):
        chunk_text = "\n\n".join(g)
        chunk_id = f"{unit_id}-B{idx:03d}"

        meta: dict[str, Any] = dict(page.metadata)
        meta.update({
            "chunk_id": chunk_id,
            "chunk_strategy": "B",
            "chunk_index": idx,
            "n_paragraphs": len(g),
            "n_chars": len(chunk_text),
        })

        chunks.append(Chunk(
            chunk_id=chunk_id,
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=chunk_text,
            metadata=meta,
        ))

    return chunks
