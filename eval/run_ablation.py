"""
Ablation experiment runner for the RAG pipeline.

Tests the independent contribution of each pipeline component:
  - Chunk size:       {150, 250, 350, 500}
  - Overlap:          {0, 25, 50, 100}
  - Retrieval depth:  k in {1, 3, 5, 8, 10}
  - Query expansion:  on/off
  - Heading bias:     on/off
  - Scope filtering:  on/off

Each experiment re-indexes from sections.jsonl with the given chunking
parameters, then runs the full retrieval evaluation.

Usage:
    python eval/run_ablation.py --experiment chunk_size
    python eval/run_ablation.py --experiment retrieval_depth
    python eval/run_ablation.py --experiment scope_filtering
    python eval/run_ablation.py --experiment query_expansion
    python eval/run_ablation.py --experiment heading_bias
    python eval/run_ablation.py --experiment overlap
    python eval/run_ablation.py --experiment all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_EVAL_DIR))

from config import DATA_PROCESSED, INDEX_DIR  # noqa: E402
from metrics import (  # noqa: E402
    reciprocal_rank,
    precision_at_k,
    recall_at_k,
    average_precision,
    ndcg_at_k,
    hit_at_k,
    scope_contamination_rate,
    scope_precision,
)

SECTIONS_JSONL = DATA_PROCESSED / "sections.jsonl"
EVAL_SET = _EVAL_DIR / "eval_set_gold.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Chunking (replicated from src/chunk.py to allow parameterisation)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def simple_tokenise(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)


def detokenise(tokens: List[str]) -> str:
    out = []
    for t in tokens:
        if out and re.match(r"[^\w\s]", t):
            out[-1] = out[-1] + t
        else:
            out.append(t)
    return " ".join(out).strip()


def make_chunks(tokens: List[str], chunk_size: int, overlap: int) -> Iterator[List[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size-1]")
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + chunk_size, n)
        yield tokens[start:end]
        if end == n:
            break
        start = end - overlap


def build_chunks(sections_path: Path, chunk_size: int, overlap: int) -> List[Dict]:
    """Build chunks from sections JSONL with given parameters."""
    chunks = []
    with sections_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec["text"].strip()
            toks = simple_tokenise(text)

            if len(toks) <= chunk_size:
                chunks.append({
                    "chunk_id": f'{rec["source_file"]}::{rec["chapter"]}::{rec["section"]}::{rec["subsection"]}::0',
                    "source_file": rec["source_file"],
                    "chapter": rec["chapter"],
                    "section": rec["section"],
                    "subsection": rec["subsection"],
                    "chunk_index": 0,
                    "text": text,
                })
            else:
                for i, chunk_tokens in enumerate(make_chunks(toks, chunk_size, overlap)):
                    chunk_text = detokenise(chunk_tokens)
                    chunks.append({
                        "chunk_id": f'{rec["source_file"]}::{rec["chapter"]}::{rec["section"]}::{rec["subsection"]}::{i}',
                        "source_file": rec["source_file"],
                        "chapter": rec["chapter"],
                        "section": rec["section"],
                        "subsection": rec["subsection"],
                        "chunk_index": i,
                        "text": chunk_text,
                    })
    return chunks


def build_index(chunks: List[Dict], model: SentenceTransformer) -> Tuple[faiss.Index, List[Dict]]:
    """Embed and index chunks, return (FAISS index, metadata list)."""
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    X = np.array(embeddings).astype("float32")
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    return index, chunks


# ---------------------------------------------------------------------------
# Retrieval variants
# ---------------------------------------------------------------------------

def search_baseline(query: str, model: SentenceTransformer, index, k: int) -> List[Tuple[float, int]]:
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


def heading_bias_fn(query: str, m: Dict) -> float:
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


def detect_scope(query: str) -> Optional[str]:
    q = query.lower()
    if "small group project" in q or "small group" in q or re.search(r"\bsgp\b", q):
        return "small"
    if "major group project" in q or "major group" in q or re.search(r"\bmgp\b", q):
        return "major"
    if ("small" in q) and ("project" in q):
        return "small"
    if ("major" in q) and ("project" in q):
        return "major"
    return None


def filter_by_scope(retrieved: List[Tuple[float, Dict]], scope: str) -> List[Tuple[float, Dict]]:
    def src(m: Dict) -> str:
        return str(m.get("source_file", "")).lower()

    def chap(m: Dict) -> str:
        return str(m.get("chapter", "")).lower()

    def is_sgp(m: Dict) -> bool:
        return ("small_group_project" in src(m)) or ("small group project" in chap(m))

    def is_mgp(m: Dict) -> bool:
        return ("major_group_project" in src(m)) or ("major group project" in chap(m))

    if scope == "small":
        sgp = [(s, m) for (s, m) in retrieved if is_sgp(m)]
        if len(sgp) >= 2:
            return sgp
        return [(s, m) for (s, m) in retrieved if not is_mgp(m)]
    if scope == "major":
        mgp = [(s, m) for (s, m) in retrieved if is_mgp(m)]
        if len(mgp) >= 2:
            return mgp
        return [(s, m) for (s, m) in retrieved if not is_sgp(m)]
    return retrieved


def search_configurable(
    query: str,
    model: SentenceTransformer,
    index,
    meta: List[Dict],
    k: int,
    use_expansion: bool = True,
    use_heading_bias: bool = True,
    use_scope_filter: bool = True,
) -> List[Tuple[float, Dict]]:
    """Configurable search with toggleable components."""
    base = search_baseline(query, model, index, k * 2)  # fetch wider for scope filter

    exp_hits = []
    if use_expansion:
        exp = expand_query(query)
        if exp != query:
            exp_hits = search_baseline(exp, model, index, k * 2)

    best: Dict[int, float] = {}
    for score, idx in base + exp_hits:
        if idx not in best or score > best[idx]:
            best[idx] = score

    results = []
    for idx, score in best.items():
        m = meta[idx]
        if use_heading_bias:
            score += heading_bias_fn(query, m)
        results.append((score, m))

    results.sort(key=lambda x: x[0], reverse=True)

    if use_scope_filter:
        scope = detect_scope(query)
        if scope:
            filtered = filter_by_scope(results, scope)
            if len(filtered) >= 2:
                results = filtered

    return results[:k]


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def is_relevant(meta_entry: Dict, item: Dict) -> bool:
    src = meta_entry.get("source_file", "").lower()
    chapter = meta_entry.get("chapter", "").lower()
    section = meta_entry.get("section", "").lower()
    subsection = meta_entry.get("subsection", "").lower()
    heading = f"{chapter} > {section} > {subsection}"

    expected_sources = [s.lower() for s in item.get("expected_sources", [])]
    expected_sections = [s.lower() for s in item.get("expected_sections", [])]

    source_match = any(exp_src in src for exp_src in expected_sources)
    section_match = any(
        all(part.strip() in heading for part in exp_sec.split(">"))
        for exp_sec in expected_sections
    )
    return source_match and section_match


def run_eval(
    eval_items: List[Dict],
    model: SentenceTransformer,
    index,
    meta: List[Dict],
    k: int,
    use_expansion: bool = True,
    use_heading_bias: bool = True,
    use_scope_filter: bool = True,
) -> Dict:
    """Run evaluation and return aggregate metrics."""
    answerable = [item for item in eval_items if not item.get("unanswerable", False)]
    n = len(answerable)
    if n == 0:
        return {}

    mrr_sum = 0.0
    map_sum = 0.0
    hit1 = hit3 = hit5 = 0
    contam_sum = 0.0
    scope_prec_sum = 0.0
    n_scoped = 0

    for item in answerable:
        retrieved = search_configurable(
            item["query"], model, index, meta, k,
            use_expansion=use_expansion,
            use_heading_bias=use_heading_bias,
            use_scope_filter=use_scope_filter,
        )
        relevant_flags = [is_relevant(m, item) for _, m in retrieved]

        mrr_sum += reciprocal_rank(relevant_flags)
        map_sum += average_precision(relevant_flags)
        if hit_at_k(relevant_flags, 1):
            hit1 += 1
        if hit_at_k(relevant_flags, 3):
            hit3 += 1
        if hit_at_k(relevant_flags, min(k, 5)):
            hit5 += 1

        scope = item.get("scope", "general")
        if scope in ("sgp", "mgp"):
            n_scoped += 1
            sources = [m.get("source_file", "") for _, m in retrieved]
            contam_sum += scope_contamination_rate(sources, scope)
            scope_prec_sum += scope_precision(
                [m for _, m in retrieved], scope,
            )

    return {
        "n_queries": n,
        "MRR": mrr_sum / n,
        "MAP": map_sum / n,
        "hit@1": hit1 / n,
        "hit@3": hit3 / n,
        "hit@5": hit5 / n,
        "scope_contamination": contam_sum / n_scoped if n_scoped else 0.0,
        "scope_precision": scope_prec_sum / n_scoped if n_scoped else 1.0,
        "n_scoped": n_scoped,
    }


# ---------------------------------------------------------------------------
# Ablation experiments
# ---------------------------------------------------------------------------

def load_eval_items() -> List[Dict]:
    items = []
    with EVAL_SET.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def print_table(headers: List[str], rows: List[List], title: str = ""):
    """Print a formatted table."""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")

    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)

    print(fmt.format(*headers))
    print("-" * sum(col_widths + [2 * (len(col_widths) - 1)]))
    for row in rows:
        print(fmt.format(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]))


def ablation_chunk_size(model: SentenceTransformer, eval_items: List[Dict]) -> List[Dict]:
    """Vary chunk_size in {150, 250, 350, 500}, overlap fixed at 50."""
    sizes = [150, 250, 350, 500]
    overlap = 50
    results = []

    for cs in sizes:
        print(f"  chunk_size={cs}, overlap={overlap}...")
        actual_overlap = min(overlap, cs - 1)
        chunks = build_chunks(SECTIONS_JSONL, cs, actual_overlap)
        index, meta = build_index(chunks, model)
        metrics = run_eval(eval_items, model, index, meta, k=5)
        metrics["chunk_size"] = cs
        metrics["overlap"] = actual_overlap
        metrics["n_chunks"] = len(chunks)
        results.append(metrics)

    rows = [[r["chunk_size"], r["n_chunks"], r["MRR"], r["MAP"],
             r["hit@1"], r["hit@3"], r["hit@5"],
             r["scope_contamination"]] for r in results]
    print_table(
        ["chunk_sz", "n_chunks", "MRR", "MAP", "hit@1", "hit@3", "hit@5", "contam"],
        rows, "ABLATION: Chunk Size",
    )
    return results


def ablation_overlap(model: SentenceTransformer, eval_items: List[Dict]) -> List[Dict]:
    """Vary overlap in {0, 25, 50, 100}, chunk_size fixed at 350."""
    overlaps = [0, 25, 50, 100]
    chunk_size = 350
    results = []

    for ov in overlaps:
        print(f"  chunk_size={chunk_size}, overlap={ov}...")
        chunks = build_chunks(SECTIONS_JSONL, chunk_size, ov)
        index, meta = build_index(chunks, model)
        metrics = run_eval(eval_items, model, index, meta, k=5)
        metrics["overlap"] = ov
        metrics["n_chunks"] = len(chunks)
        results.append(metrics)

    rows = [[r["overlap"], r["n_chunks"], r["MRR"], r["MAP"],
             r["hit@1"], r["hit@3"], r["hit@5"],
             r["scope_contamination"]] for r in results]
    print_table(
        ["overlap", "n_chunks", "MRR", "MAP", "hit@1", "hit@3", "hit@5", "contam"],
        rows, "ABLATION: Overlap",
    )
    return results


def ablation_retrieval_depth(model: SentenceTransformer, eval_items: List[Dict]) -> List[Dict]:
    """Vary k in {1, 3, 5, 8, 10} using existing index."""
    ks = [1, 3, 5, 8, 10]

    # Use default chunking parameters
    chunks = build_chunks(SECTIONS_JSONL, 350, 50)
    index, meta = build_index(chunks, model)
    results = []

    for k in ks:
        print(f"  k={k}...")
        metrics = run_eval(eval_items, model, index, meta, k=k)
        metrics["k"] = k
        results.append(metrics)

    rows = [[r["k"], r["MRR"], r["MAP"],
             r["hit@1"], r["hit@3"], r["hit@5"],
             r["scope_contamination"]] for r in results]
    print_table(
        ["k", "MRR", "MAP", "hit@1", "hit@3", "hit@5", "contam"],
        rows, "ABLATION: Retrieval Depth (k)",
    )
    return results


def ablation_query_expansion(model: SentenceTransformer, eval_items: List[Dict]) -> List[Dict]:
    """Toggle query expansion on/off."""
    chunks = build_chunks(SECTIONS_JSONL, 350, 50)
    index, meta = build_index(chunks, model)
    results = []

    for expansion in [False, True]:
        label = "ON" if expansion else "OFF"
        print(f"  query_expansion={label}...")
        metrics = run_eval(
            eval_items, model, index, meta, k=5,
            use_expansion=expansion, use_heading_bias=True, use_scope_filter=True,
        )
        metrics["query_expansion"] = label
        results.append(metrics)

    rows = [[r["query_expansion"], r["MRR"], r["MAP"],
             r["hit@1"], r["hit@3"], r["hit@5"],
             r["scope_contamination"]] for r in results]
    print_table(
        ["expansion", "MRR", "MAP", "hit@1", "hit@3", "hit@5", "contam"],
        rows, "ABLATION: Query Expansion",
    )
    return results


def ablation_heading_bias(model: SentenceTransformer, eval_items: List[Dict]) -> List[Dict]:
    """Toggle heading bias on/off."""
    chunks = build_chunks(SECTIONS_JSONL, 350, 50)
    index, meta = build_index(chunks, model)
    results = []

    for bias in [False, True]:
        label = "ON" if bias else "OFF"
        print(f"  heading_bias={label}...")
        metrics = run_eval(
            eval_items, model, index, meta, k=5,
            use_expansion=True, use_heading_bias=bias, use_scope_filter=True,
        )
        metrics["heading_bias"] = label
        results.append(metrics)

    rows = [[r["heading_bias"], r["MRR"], r["MAP"],
             r["hit@1"], r["hit@3"], r["hit@5"],
             r["scope_contamination"]] for r in results]
    print_table(
        ["heading_bias", "MRR", "MAP", "hit@1", "hit@3", "hit@5", "contam"],
        rows, "ABLATION: Heading Bias",
    )
    return results


def ablation_scope_filtering(model: SentenceTransformer, eval_items: List[Dict]) -> List[Dict]:
    """Toggle scope filtering on/off — THE critical experiment."""
    chunks = build_chunks(SECTIONS_JSONL, 350, 50)
    index, meta = build_index(chunks, model)
    results = []

    for scope_filter in [False, True]:
        label = "ON" if scope_filter else "OFF"
        print(f"  scope_filtering={label}...")
        metrics = run_eval(
            eval_items, model, index, meta, k=5,
            use_expansion=True, use_heading_bias=True, use_scope_filter=scope_filter,
        )
        metrics["scope_filtering"] = label
        results.append(metrics)

    rows = [[r["scope_filtering"], r["MRR"], r["MAP"],
             r["hit@1"], r["hit@3"], r["hit@5"],
             r["scope_contamination"], r["scope_precision"]] for r in results]
    print_table(
        ["scope_filt", "MRR", "MAP", "hit@1", "hit@3", "hit@5", "contam", "scope_P"],
        rows, "ABLATION: Scope Filtering (KEY EXPERIMENT)",
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="RAG ablation experiments")
    parser.add_argument(
        "--experiment",
        choices=["chunk_size", "overlap", "retrieval_depth",
                 "query_expansion", "heading_bias", "scope_filtering", "all"],
        default="all",
        help="Which ablation to run",
    )
    args = parser.parse_args()

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    eval_items = load_eval_items()
    print(f"Eval set: {len(eval_items)} questions ({sum(1 for i in eval_items if not i.get('unanswerable', False))} answerable)")

    all_results = {}
    experiments = {
        "chunk_size": ablation_chunk_size,
        "overlap": ablation_overlap,
        "retrieval_depth": ablation_retrieval_depth,
        "query_expansion": ablation_query_expansion,
        "heading_bias": ablation_heading_bias,
        "scope_filtering": ablation_scope_filtering,
    }

    if args.experiment == "all":
        to_run = experiments
    else:
        to_run = {args.experiment: experiments[args.experiment]}

    for name, fn in to_run.items():
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT: {name}")
        print(f"{'#'*70}")
        all_results[name] = fn(model, eval_items)

    # Save all results
    out_path = _EVAL_DIR / "ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll ablation results saved to: {out_path}")


if __name__ == "__main__":
    main()
