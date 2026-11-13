"""Check the benchmark BEFORE running retrieval: evidence exists verbatim in the gold unit,
filter value matches the gold unit's story, and split/filter counts meet BENCHMARK.md."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
units = {}
for l in (ROOT / "data/processed/units.jsonl").read_text(encoding="utf-8").splitlines():
    u = json.loads(l); units[u["unit_id"]] = u
qs = [json.loads(l) for l in (ROOT / "benchmark/questions.jsonl").read_text(encoding="utf-8").splitlines()]
errors = []
for q in qs:
    u = units.get(q["gold_unit_id"])
    if not u: errors.append(f"{q['id']}: unknown unit"); continue
    if q["evidence_quote"] not in u["text"]: errors.append(f"{q['id']}: evidence NOT found in {q['gold_unit_id']}")
    if q["filter"] and q["filter"]["story"] != u["story"]: errors.append(f"{q['id']}: filter story != gold unit story")
    if q["gold_source_id"] != u["document_id"] or q["gold_page"] != u["page"]: errors.append(f"{q['id']}: gold id/page mismatch")
dev = [q for q in qs if q["split"] == "dev"]; test = [q for q in qs if q["split"] == "test"]
nf = [q for q in qs if q["filter"]]; nft = [q for q in test if q["filter"]]
if len(qs) < 20 or len(dev) < 15 or len(test) < 5: errors.append("size requirement not met")
if len(nf) < 3 or len(nft) < 1: errors.append("filter requirement not met")
print(f"total={len(qs)} dev={len(dev)} test={len(test)} filter={len(nf)} (test filter={len(nft)})")
print("\n".join(errors) if errors else "OK: all evidence verified")
sys.exit(1 if errors else 0)
