import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from config import INDEX_DIR  # noqa: E402

FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "chunks_meta.jsonl"

EVAL_SET = Path(__file__).resolve().parent / "eval_set.jsonl"
OUT_JSONL = Path(__file__).resolve().parent / "results_enhanced.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_meta(path: Path) -> List[Dict]:
    meta = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))
    return meta


def cite_string(m: Dict) -> str:
    return f'{m["chapter"]} > {m["section"]} > {m["subsection"]} ({m["source_file"]})'


def heading_bias(query: str, m: Dict) -> float:
    q = query.lower()
    heading = f'{m["chapter"]} {m["section"]} {m["subsection"]}'.lower()

    boost = 0.0
    if any(w in q for w in ["mark", "marked", "marking", "assessment", "grade"]):
        if any(w in heading for w in ["mark", "marking", "assessment"]):
            boost += 0.06
    if any(w in q for w in ["allocate", "allocation", "teams", "team"]):
        if "allocation" in heading or "team" in heading:
            boost += 0.04
    return boost


def expand_query(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["mark", "marked", "marking", "assessment", "grade"]):
        return query + " assessment marking scheme individual mark team mark redistribution"
    if any(w in q for w in ["allocate", "allocation", "teams", "team"]):
        return query + " team allocation team formation"
    return query


def run_search_merged(query: str, model: SentenceTransformer, index, meta: List[Dict], k: int) -> List[Tuple[float, Dict]]:
    def _search(q: str) -> List[Tuple[float, int]]:
        q_emb = model.encode(q, normalize_embeddings=True).astype("float32")
        D, I = index.search(np.expand_dims(q_emb, axis=0), k=k)
        return [(float(score), int(idx)) for score, idx in zip(D[0], I[0])]

    base_hits = _search(query)
    exp = expand_query(query)
    exp_hits = _search(exp) if exp != query else []

    # Merge by idx, keep best score
    best: Dict[int, float] = {}
    for score, idx in base_hits + exp_hits:
        if idx not in best or score > best[idx]:
            best[idx] = score

    results = []
    for idx, score in best.items():
        m = meta[idx]
        score = float(score) + heading_bias(query, m)
        results.append((score, m))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:k]


def match_expected(expected_hint: str, retrieved: List[Tuple[float, Dict]]) -> List[bool]:
    hint = expected_hint.lower()
    hits = []
    for _, m in retrieved:
        heading = f'{m["chapter"]} > {m["section"]} > {m["subsection"]}'.lower()
        hits.append(hint in heading)
    return hits


def main() -> None:
    if not FAISS_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Index not found. Run src/embed_index.py first.")
    if not EVAL_SET.exists():
        raise FileNotFoundError(f"Missing eval set: {EVAL_SET}")

    meta = load_meta(META_PATH)
    index = faiss.read_index(str(FAISS_PATH))
    model = SentenceTransformer(MODEL_NAME)

    total = 0
    hit1 = 0
    hit3 = 0
    hit5 = 0

    results = []

    with EVAL_SET.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = item["id"]
            query = item["query"]
            expected = item.get("expected_hint", "")

            r5 = run_search_merged(query, model, index, meta, k=5)
            hits = match_expected(expected, r5) if expected else [False] * 5

            total += 1
            hit1 += 1 if hits[0] else 0
            hit3 += 1 if any(hits[:3]) else 0
            hit5 += 1 if any(hits[:5]) else 0

            results.append({
                "id": qid,
                "query": query,
                "expected_hint": expected,
                "hit@1": bool(hits[0]),
                "hit@3": bool(any(hits[:3])),
                "hit@5": bool(any(hits[:5])),
                "top5": [
                    {"rank": i + 1, "score": s, "cite": cite_string(m)}
                    for i, (s, m) in enumerate(r5)
                ],
                "manual_relevant_context@5": None,
                "manual_notes": ""
            })

    OUT_JSONL.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8")

    def pct(x: int) -> float:
        return (100.0 * x / total) if total else 0.0

    print(f"Queries: {total}")
    print(f"Enhanced retrieval hit@1: {hit1}/{total} ({pct(hit1):.1f}%)")
    print(f"Enhanced retrieval hit@3: {hit3}/{total} ({pct(hit3):.1f}%)")
    print(f"Enhanced retrieval hit@5: {hit5}/{total} ({pct(hit5):.1f}%)")
    print(f"Saved detailed results to: {OUT_JSONL}")


if __name__ == "__main__":
    main()
