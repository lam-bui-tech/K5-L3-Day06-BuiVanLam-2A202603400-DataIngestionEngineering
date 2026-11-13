"""Evaluate retrieved evidence and preserve per-query traces."""
import re
from typing import Any


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def is_chunk_hit(question: dict[str, Any], hit: dict[str, Any]) -> tuple[bool, bool]:
    """Check if a hit matches gold page and contains the gold evidence.

    Returns (is_page_match, is_evidence_match).
    """
    gold_doc = str(question.get("gold_source_id", "")).strip()
    gold_page = question.get("gold_page")
    gold_unit = str(question.get("gold_unit_id", "")).strip()

    hit_doc = str(hit.get("document_id", "")).strip()
    hit_page = hit.get("page")
    hit_unit = str(hit.get("metadata", {}).get("unit_id", "")).strip()

    page_match = False
    if gold_doc and hit_doc == gold_doc:
        if gold_page is not None and hit_page == gold_page:
            page_match = True
        elif gold_unit and hit_unit == gold_unit:
            page_match = True

    evidence = question.get("evidence_quote", "")
    evidence_match = False
    if evidence:
        # Check verbatim substring match first, then normalized
        hit_text = hit.get("text", "")
        if evidence in hit_text or _normalize_ws(evidence) in _normalize_ws(hit_text):
            evidence_match = True

    # A valid retrieval hit requires both correct page and containing the evidence
    is_full_hit = page_match and evidence_match
    return page_match, is_full_hit


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]], strategy: str = "") -> dict[str, Any]:
    """Check source/page and gold evidence; keep hit@1 and hit@3 trace."""
    hits_trace = []
    has_hit_at_1 = False
    has_hit_at_3 = False

    for rank, hit in enumerate(hits, start=1):
        is_page_match, is_full_hit = is_chunk_hit(question, hit)
        if rank == 1 and is_full_hit:
            has_hit_at_1 = True
        if rank <= 3 and is_full_hit:
            has_hit_at_3 = True

        hits_trace.append({
            "rank": rank,
            "chunk_id": hit.get("chunk_id"),
            "document_id": hit.get("document_id"),
            "source": hit.get("source", ""),
            "page": hit.get("page"),
            "score": hit.get("score"),
            "distance": hit.get("distance"),
            "is_page_match": is_page_match,
            "is_evidence_hit": is_full_hit,
            "metadata": hit.get("metadata", {}),
            "text": hit.get("text", ""),
        })

    # Check filter compliance if filter was requested
    q_filter = question.get("filter")
    filter_compliant = True
    if q_filter:
        for hit in hits[:3]:
            meta = hit.get("metadata", {})
            for k, expected_v in q_filter.items():
                if meta.get(k) != expected_v:
                    filter_compliant = False
                    break

    return {
        "query_id": question.get("id"),
        "split": question.get("split"),
        "type": question.get("type"),
        "strategy": strategy,
        "question": question.get("question"),
        "expected_answer": question.get("expected_answer"),
        "gold_source_id": question.get("gold_source_id"),
        "gold_page": question.get("gold_page"),
        "gold_unit_id": question.get("gold_unit_id"),
        "evidence_quote": question.get("evidence_quote"),
        "filter": q_filter,
        "filter_compliant": filter_compliant if q_filter else None,
        "hit_at_1": has_hit_at_1,
        "evidence_hit_at_3": has_hit_at_3,
        "hits": hits_trace,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split."""
    by_strat_split: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for t in traces:
        strat = t.get("strategy", "unknown")
        split = t.get("split", "unknown")
        if strat not in by_strat_split:
            by_strat_split[strat] = {"all": [], "dev": [], "test": []}
        by_strat_split[strat]["all"].append(t)
        if split in by_strat_split[strat]:
            by_strat_split[strat][split].append(t)

    summary: dict[str, Any] = {}

    for strat, splits in by_strat_split.items():
        summary[strat] = {}
        for split_name, q_list in splits.items():
            n = len(q_list)
            if n == 0:
                summary[strat][split_name] = {
                    "count": 0,
                    "hit_at_1": "0/0",
                    "hit_at_1_pct": 0.0,
                    "hit_at_3": "0/0",
                    "hit_at_3_pct": 0.0,
                    "filter_compliance": "N/A",
                }
                continue

            h1 = sum(1 for q in q_list if q["hit_at_1"])
            h3 = sum(1 for q in q_list if q["evidence_hit_at_3"])

            # Filter compliance
            filter_queries = [q for q in q_list if q.get("filter") is not None]
            if filter_queries:
                compliant_cnt = sum(1 for q in filter_queries if q.get("filter_compliant") is True)
                filter_comp_str = f"{compliant_cnt}/{len(filter_queries)}"
            else:
                filter_comp_str = "N/A"

            summary[strat][split_name] = {
                "count": n,
                "hit_at_1": f"{h1}/{n}",
                "hit_at_1_pct": round(h1 / n * 100, 2),
                "hit_at_3": f"{h3}/{n}",
                "hit_at_3_pct": round(h3 / n * 100, 2),
                "filter_compliance": filter_comp_str,
            }

    return summary
