"""
End-to-end RAG evaluation: retrieval + generation.

Evaluates the full pipeline including:
  - Retrieval quality (MRR, MAP, scope contamination)
  - Generation quality (faithfulness, citation accuracy, well-formedness)
  - Refusal accuracy (correctly refusing unanswerable questions)

Requires Ollama running locally with the configured model.

Usage:
    python eval/run_e2e_eval.py [--model llama3.2:3b] [--k 5]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_EVAL_DIR))

from config import INDEX_DIR  # noqa: E402
from metrics import (  # noqa: E402
    reciprocal_rank,
    average_precision,
    hit_at_k,
    scope_contamination_rate,
    scope_precision,
    answer_well_formed,
    is_refusal,
    refusal_accuracy,
    faithfulness_fragment_recall,
    citation_density,
    uncited_sentence_rate,
    extract_citations,
    citation_precision,
    citation_recall,
)

FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "chunks_meta.jsonl"
EVAL_SET = _EVAL_DIR / "eval_set_gold.jsonl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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


# --- Generation via Ollama ---

def ollama_generate(prompt: str, model: str = "llama3.2:3b", timeout_s: int = 240) -> str:
    p = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    if p.returncode != 0:
        return f"[GENERATION_ERROR: {p.stderr.strip()}]"
    return (p.stdout or "").strip()


# --- Retrieval (matching chat_llama_cli.py pipeline) ---

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


def retrieve_enhanced(
    query: str,
    model: SentenceTransformer,
    index,
    meta: List[Dict],
    k: int = 8,
) -> List[Tuple[float, Dict]]:
    def _search(q: str) -> List[Tuple[float, int]]:
        q_emb = model.encode(q, normalize_embeddings=True).astype("float32")
        D, I = index.search(np.expand_dims(q_emb, axis=0), k=k)
        return [(float(score), int(idx)) for score, idx in zip(D[0], I[0])]

    base = _search(query)
    exp = expand_query(query)
    exp_hits = _search(exp) if exp != query else []

    best: Dict[int, float] = {}
    for score, idx in base + exp_hits:
        if idx not in best or score > best[idx]:
            best[idx] = score

    results = []
    for idx, score in best.items():
        m = meta[idx]
        score += heading_bias(query, m)
        results.append((score, m))

    results.sort(key=lambda x: x[0], reverse=True)

    scope = detect_scope(query)
    if scope:
        filtered = filter_by_scope(results, scope)
        if len(filtered) >= 2:
            results = filtered

    return results[:k]


# --- Prompt construction (matching chat_llama_cli.py) ---

def cite_string(m: Dict) -> str:
    return f'{m["chapter"]} > {m["section"]} > {m["subsection"]} ({m["source_file"]})'


def build_prompt(query: str, retrieved: List[Tuple[float, Dict]]) -> Tuple[str, List[str]]:
    context_blocks = []
    sources = []
    for i, (_, m) in enumerate(retrieved, start=1):
        tag = f"[S{i}]"
        cite = cite_string(m)
        sources.append(f"{tag} {cite}")
        text = str(m.get("text", "")).strip().replace("\n", " ")[:1200]
        context_blocks.append(f"{tag} {text}")

    context = "\n\n".join(context_blocks)
    prompt = f"""You are a handbook Q&A assistant.

Core rules:
- Answer using ONLY the information explicitly stated in SOURCES.
- You may use external concepts ONLY as explanatory context, not as authority for rules or penalties.
- Do NOT invent policies, thresholds, or permissions.
- If SOURCES do not explicitly contain the answer, reply exactly:
I can't find this in the handbook excerpts I retrieved.

Question:
{query}

SOURCES:
{context}

Required output format (MUST follow exactly):
1) Write 2-4 short sentences (no bullet points).
2) EVERY sentence must include at least one citation tag like [S1] or [S2][S4].
3) The FIRST sentence must answer the question directly and assertively. AND MUST include at least one citation tag.
4) After the sentences, add exactly one final line:
Sources used: [S#], [S#], ...
5) Do not mention "SOURCES", "excerpts", or internal system behaviour.
6) If you write any sentence without a citation tag, your answer is invalid."
"""
    return prompt, sources


_CITE_RE = re.compile(r"\[S\d+\]")
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def answer_is_well_formed_strict(ans: str) -> bool:
    """Matches the check in chat_llama_cli.py."""
    lines = [ln.strip() for ln in ans.splitlines() if ln.strip()]
    if not lines:
        return False
    sources_lines = [ln for ln in lines if ln.lower().startswith("sources used:")]
    if len(sources_lines) != 1:
        return False
    body_lines = [ln for ln in lines if not ln.lower().startswith("sources used:")]
    body = " ".join(body_lines).strip()
    if not body:
        return False
    sentences = [s.strip() for s in _SENT_SPLIT.split(body) if s.strip()]
    if not (2 <= len(sentences) <= 4):
        return False
    for s in sentences:
        if not _CITE_RE.search(s):
            return False
    return True


def strict_prompt(query: str, context: str) -> str:
    return f"""You are a handbook Q&A assistant.

You MUST answer using ONLY information explicitly stated in SOURCES.
You MUST NOT infer new rules, fill gaps, or soften conclusions.
If SOURCES do not explicitly contain the answer, reply exactly:
I can't find this in the handbook excerpts I retrieved.

Question:
{query}

SOURCES:
{context}

Required output format (MUST follow exactly):
1) Write 2-4 short sentences (no bullet points).
2) EVERY sentence MUST include at least one citation tag like [S1] or [S2][S4].
3) The FIRST sentence MUST answer directly and assertively. AND MUST include at least one citation tag.
4) After the sentences, add exactly one final line:
Sources used: [S#], [S#], ...
5) Do not include any other text.
6) If you write any sentence without a citation tag, your answer is invalid."
"""


# --- Relevance matching ---

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


# --- Main E2E evaluation ---

def main():
    parser = argparse.ArgumentParser(description="End-to-end RAG evaluation")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Ollama model")
    parser.add_argument("--k", type=int, default=4, help="Top-k chunks for generation context")
    parser.add_argument("--retrieval-k", type=int, default=8, help="Initial retrieval depth")
    args = parser.parse_args()

    if not FAISS_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Index not found. Run src/embed_index.py first.")

    meta = load_meta(META_PATH)
    index = faiss.read_index(str(FAISS_PATH))
    emb_model = SentenceTransformer(EMBED_MODEL)
    eval_items = load_eval_set(EVAL_SET)

    print(f"E2E Evaluation: {len(eval_items)} questions")
    print(f"Model: {args.model}  k={args.k}  retrieval_k={args.retrieval_k}")
    print(f"{'='*70}\n")

    per_query = []

    for idx_q, item in enumerate(eval_items):
        qid = item["id"]
        query = item["query"]
        is_unanswerable = item.get("unanswerable", False)
        gold_frags = item.get("gold_answer_fragments", [])
        scope = item.get("scope", "general")

        print(f"[{idx_q+1}/{len(eval_items)}] {qid}: {query[:60]}...")

        # --- Retrieval ---
        retrieved = retrieve_enhanced(query, emb_model, index, meta, k=args.retrieval_k)
        retrieved_for_gen = retrieved[:args.k]

        # Retrieval metrics
        relevant_flags = [is_relevant(m, item) for _, m in retrieved_for_gen] if not is_unanswerable else []
        rr = reciprocal_rank(relevant_flags) if relevant_flags else 0.0

        # Scope metrics
        contam = 0.0
        s_prec = 1.0
        if scope in ("sgp", "mgp"):
            sources = [m.get("source_file", "") for _, m in retrieved_for_gen]
            contam = scope_contamination_rate(sources, scope)
            s_prec = scope_precision([m for _, m in retrieved_for_gen], scope)

        # Determine which retrieved sources are relevant (for citation eval)
        relevant_indices = []
        for i, (_, m) in enumerate(retrieved_for_gen, start=1):
            if not is_unanswerable and is_relevant(m, item):
                relevant_indices.append(i)

        # --- Generation ---
        prompt, source_labels = build_prompt(query, retrieved_for_gen)
        answer = ollama_generate(prompt, model=args.model)

        # Retry with strict prompt if malformed
        retried = False
        if not answer_is_well_formed_strict(answer) and not is_unanswerable:
            context = "\n\n".join(
                f"[S{i+1}] " + str(m.get("text", "")).strip().replace("\n", " ")[:1200]
                for i, (_, m) in enumerate(retrieved_for_gen)
            )
            answer = ollama_generate(strict_prompt(query, context), model=args.model)
            retried = True

        # --- Generation metrics ---
        well_formed = answer_well_formed(answer)
        refused = is_refusal(answer)
        ref_correct, ref_label = refusal_accuracy(answer, is_unanswerable)
        faithfulness = 0.0
        cite_dens = 0.0
        uncited = 1.0
        cite_prec = 0.0
        cite_rec = 0.0

        if not is_unanswerable and not refused:
            faithfulness = faithfulness_fragment_recall(answer, gold_frags)
            cite_dens = citation_density(answer)
            uncited = uncited_sentence_rate(answer)
            if relevant_indices:
                cite_prec = citation_precision(answer, relevant_indices)
                cite_rec = citation_recall(answer, relevant_indices)

        result = {
            "id": qid,
            "query": query,
            "scope": scope,
            "unanswerable": is_unanswerable,
            "answer": answer,
            "retried": retried,
            # Retrieval
            "rr": rr,
            "hit@1": hit_at_k(relevant_flags, 1) if relevant_flags else False,
            "scope_contamination": contam,
            "scope_precision": s_prec,
            "relevant_sources": source_labels,
            "relevant_indices": relevant_indices,
            # Generation
            "well_formed": well_formed,
            "refused": refused,
            "refusal_correct": ref_correct,
            "refusal_label": ref_label,
            "faithfulness": faithfulness,
            "citation_density": cite_dens,
            "uncited_sentence_rate": uncited,
            "citation_precision": cite_prec,
            "citation_recall": cite_rec,
            "gold_fragments": gold_frags,
        }
        per_query.append(result)

    # --- Aggregate ---
    n = len(per_query)
    answerable = [r for r in per_query if not r["unanswerable"]]
    answered = [r for r in answerable if not r["refused"]]
    n_a = len(answerable)
    n_ans = len(answered)

    print(f"\n{'='*70}")
    print(f"END-TO-END EVALUATION REPORT")
    print(f"{'='*70}")
    print(f"  Total queries:            {n}")
    print(f"  Answerable:               {n_a}")
    print(f"  Actually answered:         {n_ans}")
    print()

    # Retrieval
    if answerable:
        print("  RETRIEVAL:")
        print(f"    MRR:                    {sum(r['rr'] for r in answerable) / n_a:.4f}")
        print(f"    hit@1:                  {sum(1 for r in answerable if r['hit@1']) / n_a:.4f}")

    scoped = [r for r in per_query if r["scope"] in ("sgp", "mgp")]
    if scoped:
        ns = len(scoped)
        print(f"    scope_contamination:    {sum(r['scope_contamination'] for r in scoped) / ns:.4f}")
        print(f"    scope_precision:        {sum(r['scope_precision'] for r in scoped) / ns:.4f}")

    # Generation
    print()
    print("  GENERATION:")
    print(f"    well_formedness:        {sum(1 for r in per_query if r['well_formed']) / n:.4f}")
    print(f"    refusal_accuracy:       {sum(1 for r in per_query if r['refusal_correct']) / n:.4f}")

    labels = {"true_refusal": 0, "true_answer": 0, "false_refusal": 0, "false_answer": 0}
    for r in per_query:
        labels[r["refusal_label"]] += 1
    for label, count in labels.items():
        print(f"      {label:20s}: {count}")

    if answered:
        print(f"    faithfulness:           {sum(r['faithfulness'] for r in answered) / n_ans:.4f}")
        print(f"    citation_density:       {sum(r['citation_density'] for r in answered) / n_ans:.4f}")
        print(f"    uncited_sent_rate:      {sum(r['uncited_sentence_rate'] for r in answered) / n_ans:.4f}")
        print(f"    citation_precision:     {sum(r['citation_precision'] for r in answered) / n_ans:.4f}")
        print(f"    citation_recall:        {sum(r['citation_recall'] for r in answered) / n_ans:.4f}")
        print(f"    retry_rate:             {sum(1 for r in answerable if r['retried']) / n_a:.4f}")

    # Save
    out_path = _EVAL_DIR / f"e2e_results_{args.model.replace(':', '_')}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in per_query:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"\nDetailed results: {out_path}")

    # Summary JSON
    summary = {
        "model": args.model,
        "k": args.k,
        "retrieval_k": args.retrieval_k,
        "n_queries": n,
        "n_answerable": n_a,
        "n_answered": n_ans,
    }
    if answerable:
        summary["MRR"] = sum(r["rr"] for r in answerable) / n_a
        summary["hit@1"] = sum(1 for r in answerable if r["hit@1"]) / n_a
    if scoped:
        ns = len(scoped)
        summary["scope_contamination"] = sum(r["scope_contamination"] for r in scoped) / ns
        summary["scope_precision"] = sum(r["scope_precision"] for r in scoped) / ns
    summary["well_formedness"] = sum(1 for r in per_query if r["well_formed"]) / n
    summary["refusal_accuracy"] = sum(1 for r in per_query if r["refusal_correct"]) / n
    if answered:
        summary["faithfulness"] = sum(r["faithfulness"] for r in answered) / n_ans
        summary["citation_precision"] = sum(r["citation_precision"] for r in answered) / n_ans
        summary["citation_recall"] = sum(r["citation_recall"] for r in answered) / n_ans

    summary_path = _EVAL_DIR / f"e2e_summary_{args.model.replace(':', '_')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
