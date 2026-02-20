"""
Comprehensive retrieval evaluation against the gold eval set.

Reports: MRR, MAP, Hit@k, P@k, R@k, NDCG@k, and scope contamination.

Usage:
    python eval/run_retrieval_eval.py [--k 5] [--enhanced]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---- path setup ----
_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_EVAL_DIR))

from config import INDEX_DIR  # noqa: E402
from metrics import (  # noqa: E402
    reciprocal_rank,
    precision_at_k,
    recall_at_k,
    average_precision,
    ndcg_at_k,
    hit_at_k,
    mean_reciprocal_rank,
    mean_average_precision,
    scope_contamination_rate,
    scope_precision,
)

FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "chunks_meta.jsonl"
EVAL_SET = _EVAL_DIR / "eval_set_gold.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_meta(path: Path) -> List[Dict]:
    meta = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    return meta


def load_eval_set(path: Path) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# --- Retrieval variants ---

def search_baseline(
    query: str, model: SentenceTransformer, index, k: int,
) -> List[Tuple[float, int]]:
    q_emb = model.encode(query, normalize_embeddings=True).astype("float32")
    D, I = index.search(np.expand_dims(q_emb, axis=0), k=k)
    return [(float(s), int(i)) for s, i in zip(D[0], I[0])]


def expand_query(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["allowed", "allow", "use", "javascript", "react",
                             "framework", "language", "technology", "stack"]):
        return query + " allowed technologies technology constraints must be developed entirely with"
    if any(w in q for w in ["mark", "marked", "marking", "assessment", "grade",
                             "redistribution", "correction"]):
        return query + " assessment marking scheme individual mark team mark redistribution correction"
    if any(w in q for w in ["allocate", "allocation", "teams", "team",
                             "formation", "register"]):
        return query + " team allocation team formation register team feedback keats"
    return query


def heading_bias(query: str, m: Dict) -> float:
    q = query.lower()
    heading = f'{m.get("chapter", "")} {m.get("section", "")} {m.get("subsection", "")}'.lower()
    boost = 0.0
    if any(w in q for w in ["mark", "marked", "marking", "assessment", "grade"]):
        if any(w in heading for w in ["mark", "marking", "assessment"]):
            boost += 0.06
    if any(w in q for w in ["allocate", "allocation", "teams", "team", "formation"]):
        if "allocation" in heading or "team" in heading or "formation" in heading:
            boost += 0.04
    if any(w in q for w in ["allowed", "allow", "use", "javascript", "react",
                             "framework", "language", "technology", "stack"]):
        if any(w in heading for w in ["technology", "tools", "constraints"]):
            boost += 0.03
    return boost


def search_enhanced(
    query: str, model: SentenceTransformer, index, meta: List[Dict], k: int,
) -> List[Tuple[float, int]]:
    """Enhanced search with query expansion + heading bias."""
    base = search_baseline(query, model, index, k)
    exp = expand_query(query)
    exp_hits = search_baseline(exp, model, index, k) if exp != query else []

    best: Dict[int, float] = {}
    for score, idx in base + exp_hits:
        if idx not in best or score > best[idx]:
            best[idx] = score

    results = []
    for idx, score in best.items():
        score += heading_bias(query, meta[idx])
        results.append((score, idx))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:k]


# --- Relevance matching ---

def is_relevant(meta_entry: Dict, item: Dict) -> bool:
    """
    Check if a retrieved chunk is relevant to the eval item.
    Uses source file matching AND section heading matching.
    """
    src = meta_entry.get("source_file", "").lower()
    chapter = meta_entry.get("chapter", "").lower()
    section = meta_entry.get("section", "").lower()
    subsection = meta_entry.get("subsection", "").lower()
    heading = f"{chapter} > {section} > {subsection}"

    expected_sources = [s.lower() for s in item.get("expected_sources", [])]
    expected_sections = [s.lower() for s in item.get("expected_sections", [])]

    # Source file must match
    source_match = any(exp_src in src for exp_src in expected_sources)

    # Section heading must match (partial)
    section_match = any(
        all(part.strip() in heading for part in exp_sec.split(">"))
        for exp_sec in expected_sections
    )

    return source_match and section_match


def count_relevant_in_corpus(meta: List[Dict], item: Dict) -> int:
    """Count total relevant chunks in the entire corpus for this query."""
    return sum(1 for m in meta if is_relevant(m, item))


# --- Main evaluation ---

def evaluate_retrieval(
    eval_items: List[Dict],
    model: SentenceTransformer,
    index,
    meta: List[Dict],
    k: int,
    enhanced: bool = False,
) -> Tuple[List[Dict], Dict]:
    """Run retrieval evaluation, return per-query results and aggregate report."""

    per_query = []
    all_relevant_flags = []

    for item in eval_items:
        qid = item["id"]
        query = item["query"]
        scope = item.get("scope", "general")
        unanswerable = item.get("unanswerable", False)

        # Skip unanswerable questions for retrieval eval
        if unanswerable:
            continue

        if enhanced:
            hits = search_enhanced(query, model, index, meta, k)
        else:
            hits = search_baseline(query, model, index, k)

        retrieved_metas = [meta[idx] for _, idx in hits]
        relevant_flags = [is_relevant(m, item) for m in retrieved_metas]
        total_relevant = count_relevant_in_corpus(meta, item)

        # Graded relevance: 2 if source+section match, 1 if source matches, 0 otherwise
        graded = []
        for m in retrieved_metas:
            src = m.get("source_file", "").lower()
            expected_sources = [s.lower() for s in item.get("expected_sources", [])]
            source_match = any(es in src for es in expected_sources)
            if is_relevant(m, item):
                graded.append(2.0)
            elif source_match:
                graded.append(1.0)
            else:
                graded.append(0.0)

        # Scope metrics
        contam = 0.0
        s_prec = 1.0
        if scope in ("sgp", "mgp"):
            sources = [m.get("source_file", "") for m in retrieved_metas]
            contam = scope_contamination_rate(sources, scope)
            s_prec = scope_precision(retrieved_metas, scope)

        result = {
            "id": qid,
            "query": query,
            "scope": scope,
            "relevant_flags": relevant_flags,
            "total_relevant": max(total_relevant, 1),
            "retrieved_metas": retrieved_metas,
            "graded_relevance": graded,
            "rr": reciprocal_rank(relevant_flags),
            "ap": average_precision(relevant_flags),
            "ndcg@k": ndcg_at_k(graded, k),
            "hit@1": hit_at_k(relevant_flags, 1),
            "hit@3": hit_at_k(relevant_flags, 3),
            "hit@5": hit_at_k(relevant_flags, min(k, 5)),
            f"P@{k}": precision_at_k(relevant_flags, k),
            f"R@{k}": recall_at_k(relevant_flags, k, max(total_relevant, 1)),
            "scope_contamination": contam,
            "scope_precision": s_prec,
            "top_results": [
                {
                    "rank": i + 1,
                    "score": score,
                    "source": meta[idx].get("source_file", ""),
                    "heading": f'{meta[idx].get("chapter", "")} > {meta[idx].get("section", "")} > {meta[idx].get("subsection", "")}',
                    "relevant": rel,
                }
                for i, ((score, idx), rel) in enumerate(zip(hits, relevant_flags))
            ],
        }
        per_query.append(result)
        all_relevant_flags.append(relevant_flags)

    # Aggregates
    n = len(per_query)
    if n == 0:
        return per_query, {}

    agg = {
        "n_queries": n,
        "mode": "enhanced" if enhanced else "baseline",
        "k": k,
        "MRR": sum(r["rr"] for r in per_query) / n,
        "MAP": sum(r["ap"] for r in per_query) / n,
        f"mean_NDCG@{k}": sum(r[f"ndcg@k"] for r in per_query) / n,
    }

    for kk in [1, 3, 5]:
        if kk <= k:
            agg[f"hit@{kk}"] = sum(1 for r in per_query if r.get(f"hit@{kk}", False)) / n
            agg[f"P@{kk}"] = sum(precision_at_k(r["relevant_flags"], kk) for r in per_query) / n

    # Scope metrics (only for scoped queries)
    scoped = [r for r in per_query if r["scope"] in ("sgp", "mgp")]
    if scoped:
        ns = len(scoped)
        agg["scope_contamination_mean"] = sum(r["scope_contamination"] for r in scoped) / ns
        agg["scope_precision_mean"] = sum(r["scope_precision"] for r in scoped) / ns
        agg["n_scoped_queries"] = ns

    return per_query, agg


def main():
    parser = argparse.ArgumentParser(description="Comprehensive retrieval evaluation")
    parser.add_argument("--k", type=int, default=5, help="Top-k retrieval depth")
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced retrieval")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    args = parser.parse_args()

    if not FAISS_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Index not found. Run src/embed_index.py first.")

    meta = load_meta(META_PATH)
    index = faiss.read_index(str(FAISS_PATH))
    model = SentenceTransformer(MODEL_NAME)
    eval_items = load_eval_set(EVAL_SET)

    mode_label = "enhanced" if args.enhanced else "baseline"
    print(f"Running {mode_label} retrieval evaluation (k={args.k})...")
    print(f"Eval set: {len(eval_items)} questions")

    per_query, agg = evaluate_retrieval(
        eval_items, model, index, meta, k=args.k, enhanced=args.enhanced,
    )

    # Print aggregate
    print(f"\n{'='*60}")
    print(f"RETRIEVAL EVALUATION REPORT ({mode_label}, k={args.k})")
    print(f"{'='*60}")
    for key, val in agg.items():
        if isinstance(val, float):
            print(f"  {key:30s}: {val:.4f}")
        else:
            print(f"  {key:30s}: {val}")

    # Print per-query breakdown for failures
    print(f"\n{'='*60}")
    print("PER-QUERY BREAKDOWN (misses at hit@1)")
    print(f"{'='*60}")
    for r in per_query:
        if not r["hit@1"]:
            print(f"\n  [{r['id']}] {r['query']}")
            print(f"    scope={r['scope']}  RR={r['rr']:.3f}  contam={r['scope_contamination']:.2f}")
            for tr in r["top_results"][:3]:
                marker = "  HIT" if tr["relevant"] else " MISS"
                print(f"    #{tr['rank']} {marker} score={tr['score']:.3f} {tr['heading']} ({tr['source']})")

    # Save detailed results
    out_path = args.output or str(
        _EVAL_DIR / f"retrieval_results_{mode_label}_k{args.k}.jsonl"
    )
    # Strip retrieved_metas (too large) for output
    for r in per_query:
        del r["retrieved_metas"]
    with open(out_path, "w", encoding="utf-8") as f:
        for r in per_query:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"\nDetailed results saved to: {out_path}")

    # Save aggregate report
    agg_path = out_path.replace(".jsonl", "_summary.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"Summary saved to: {agg_path}")


if __name__ == "__main__":
    main()
