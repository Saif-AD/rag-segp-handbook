"""
Evaluation metrics for the RAG pipeline.

Covers three evaluation layers:
  1. Retrieval quality (IR metrics)
  2. Scope isolation (contamination metrics)
  3. Generation quality (faithfulness, citation, well-formedness)

References:
  - MRR, NDCG: Manning, Raghavan & Schütze, "Introduction to Information Retrieval"
  - RAG evaluation framework: Es et al., "RAGAS: Automated Evaluation of RAG" (2023)
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. RETRIEVAL METRICS
# ---------------------------------------------------------------------------

def reciprocal_rank(relevant: List[bool]) -> float:
    """Reciprocal rank: 1/rank of the first relevant result. 0 if none."""
    for i, hit in enumerate(relevant):
        if hit:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(queries: List[List[bool]]) -> float:
    """MRR across a set of queries."""
    if not queries:
        return 0.0
    return sum(reciprocal_rank(q) for q in queries) / len(queries)


def precision_at_k(relevant: List[bool], k: int) -> float:
    """P@k: fraction of top-k results that are relevant."""
    top = relevant[:k]
    if not top:
        return 0.0
    return sum(top) / len(top)


def recall_at_k(relevant: List[bool], k: int, total_relevant: int) -> float:
    """R@k: fraction of all relevant documents found in top-k."""
    if total_relevant == 0:
        return 0.0
    return sum(relevant[:k]) / total_relevant


def average_precision(relevant: List[bool]) -> float:
    """Average Precision (AP) for a single query."""
    hits = 0
    sum_prec = 0.0
    for i, hit in enumerate(relevant):
        if hit:
            hits += 1
            sum_prec += hits / (i + 1)
    if hits == 0:
        return 0.0
    return sum_prec / hits


def mean_average_precision(queries: List[List[bool]]) -> float:
    """MAP across a set of queries."""
    if not queries:
        return 0.0
    return sum(average_precision(q) for q in queries) / len(queries)


def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """DCG@k using log2 discounting."""
    total = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        total += rel / math.log2(i + 2)  # i+2 because log2(1)=0
    return total


def ndcg_at_k(relevance_scores: List[float], k: int) -> float:
    """NDCG@k: normalised DCG. relevance_scores are graded (e.g. 0, 1, 2)."""
    actual = dcg_at_k(relevance_scores, k)
    ideal = dcg_at_k(sorted(relevance_scores, reverse=True), k)
    if ideal == 0.0:
        return 0.0
    return actual / ideal


def hit_at_k(relevant: List[bool], k: int) -> bool:
    """Whether any of the top-k results are relevant."""
    return any(relevant[:k])


# ---------------------------------------------------------------------------
# 2. SCOPE CONTAMINATION METRICS
# ---------------------------------------------------------------------------

def scope_contamination_rate(
    retrieved_sources: List[str],
    expected_scope: str,
) -> float:
    """
    Fraction of retrieved chunks that come from the WRONG scope document.

    expected_scope: "sgp" or "mgp"
    retrieved_sources: list of source_file strings from retrieved chunks

    A contamination of 0.0 means all retrieved chunks are from the correct
    scope document. 1.0 means all are from the wrong one.
    """
    if not retrieved_sources:
        return 0.0

    wrong = 0
    for src in retrieved_sources:
        src_lower = src.lower()
        if expected_scope == "sgp":
            if "major_group_project" in src_lower:
                wrong += 1
        elif expected_scope == "mgp":
            if "small_group_project" in src_lower:
                wrong += 1
    return wrong / len(retrieved_sources)


def scope_precision(
    retrieved_metas: List[Dict],
    expected_scope: str,
) -> float:
    """
    Fraction of retrieved chunks that are from the CORRECT scope.
    For scope="sgp", correct means source_file contains "small_group_project".
    For scope="mgp", correct means source_file contains "major_group_project".
    For scope="general" or "cross", returns 1.0 (no scope constraint).
    """
    if expected_scope in ("general", "cross"):
        return 1.0
    if not retrieved_metas:
        return 0.0

    correct = 0
    for m in retrieved_metas:
        src = str(m.get("source_file", "")).lower()
        chapter = str(m.get("chapter", "")).lower()
        if expected_scope == "sgp":
            if "small_group_project" in src or "small group project" in chapter:
                correct += 1
        elif expected_scope == "mgp":
            if "major_group_project" in src or "major group project" in chapter:
                correct += 1
    return correct / len(retrieved_metas)


# ---------------------------------------------------------------------------
# 3. GENERATION QUALITY METRICS
# ---------------------------------------------------------------------------

_CITE_RE = re.compile(r"\[S(\d+)\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def extract_citations(text: str) -> List[int]:
    """Extract all [S#] citation tags from text, return list of ints."""
    return [int(m) for m in _CITE_RE.findall(text)]


def citation_precision(answer: str, relevant_source_indices: List[int]) -> float:
    """
    Of all citations in the answer, what fraction point to relevant sources?
    relevant_source_indices: 1-based indices of sources that ARE relevant.
    """
    cited = extract_citations(answer)
    if not cited:
        return 0.0
    correct = sum(1 for c in cited if c in relevant_source_indices)
    return correct / len(cited)


def citation_recall(answer: str, relevant_source_indices: List[int]) -> float:
    """
    Of all relevant sources, what fraction are cited in the answer?
    """
    if not relevant_source_indices:
        return 1.0  # vacuously true
    cited = set(extract_citations(answer))
    found = sum(1 for r in relevant_source_indices if r in cited)
    return found / len(relevant_source_indices)


def citation_density(answer: str) -> float:
    """Average number of citations per sentence."""
    body_lines = [
        ln.strip() for ln in answer.splitlines()
        if ln.strip() and not ln.strip().lower().startswith("sources used:")
    ]
    body = " ".join(body_lines)
    sentences = [s.strip() for s in _SENT_SPLIT.split(body) if s.strip()]
    if not sentences:
        return 0.0
    total_cites = sum(len(_CITE_RE.findall(s)) for s in sentences)
    return total_cites / len(sentences)


def uncited_sentence_rate(answer: str) -> float:
    """Fraction of body sentences that contain zero citation tags."""
    body_lines = [
        ln.strip() for ln in answer.splitlines()
        if ln.strip() and not ln.strip().lower().startswith("sources used:")
    ]
    body = " ".join(body_lines)
    sentences = [s.strip() for s in _SENT_SPLIT.split(body) if s.strip()]
    if not sentences:
        return 1.0
    uncited = sum(1 for s in sentences if not _CITE_RE.search(s))
    return uncited / len(sentences)


def answer_well_formed(answer: str) -> bool:
    """
    Check structural well-formedness:
    - Has body text
    - Has exactly one "Sources used:" line
    - 2-4 sentences in body
    - Every sentence has at least one [S#]
    """
    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
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


def is_refusal(answer: str) -> bool:
    """Detect if the answer is a refusal / 'I can't find this' response."""
    lower = answer.lower()
    refusal_phrases = [
        "i can't find this",
        "i couldn't find",
        "i cannot find",
        "not found in the handbook",
        "i can not find",
    ]
    return any(p in lower for p in refusal_phrases)


def refusal_accuracy(
    answer: str,
    should_refuse: bool,
) -> Tuple[bool, str]:
    """
    Check refusal correctness.
    Returns (correct: bool, label: str) where label is one of:
      "true_refusal"  - correctly refused
      "true_answer"   - correctly answered
      "false_refusal" - refused when it shouldn't have (Type I)
      "false_answer"  - answered when it should have refused (Type II)
    """
    refused = is_refusal(answer)
    if should_refuse and refused:
        return True, "true_refusal"
    if not should_refuse and not refused:
        return True, "true_answer"
    if should_refuse and not refused:
        return False, "false_answer"
    return False, "false_refusal"


def faithfulness_fragment_recall(
    answer: str,
    gold_fragments: List[str],
) -> float:
    """
    Proxy faithfulness metric: what fraction of expected gold answer
    fragments appear (case-insensitive) in the generated answer?

    This is a weak but reproducible proxy. For a dissertation, complement
    with human annotation on a subset.
    """
    if not gold_fragments:
        return 1.0  # vacuously faithful
    lower_answer = answer.lower()
    found = sum(1 for f in gold_fragments if f.lower() in lower_answer)
    return found / len(gold_fragments)


# ---------------------------------------------------------------------------
# 4. AGGREGATE REPORTING
# ---------------------------------------------------------------------------

def compute_retrieval_report(
    eval_results: List[Dict],
    ks: List[int] = None,
) -> Dict:
    """
    Compute aggregate retrieval metrics from a list of per-query results.

    Each result dict must contain:
      - "relevant_flags": List[bool] for the retrieved results
      - "total_relevant": int (number of relevant docs in corpus for this query)
    """
    if ks is None:
        ks = [1, 3, 5]

    all_relevant = [r["relevant_flags"] for r in eval_results]
    n = len(eval_results)
    if n == 0:
        return {}

    report = {
        "n_queries": n,
        "MRR": mean_reciprocal_rank(all_relevant),
        "MAP": mean_average_precision(all_relevant),
    }

    for k in ks:
        report[f"hit@{k}"] = sum(1 for r in all_relevant if hit_at_k(r, k)) / n
        report[f"P@{k}"] = sum(precision_at_k(r, k) for r in all_relevant) / n
        report[f"R@{k}"] = sum(
            recall_at_k(r, k, res["total_relevant"])
            for r, res in zip(all_relevant, eval_results)
        ) / n

    return report


def compute_scope_report(eval_results: List[Dict]) -> Dict:
    """
    Compute scope isolation metrics.

    Each result dict must contain:
      - "scope": str ("sgp", "mgp", "general", "cross")
      - "retrieved_metas": List[Dict] with source_file keys
    """
    scoped = [r for r in eval_results if r["scope"] in ("sgp", "mgp")]
    if not scoped:
        return {"n_scoped_queries": 0}

    contam_rates = []
    scope_precs = []
    for r in scoped:
        sources = [m.get("source_file", "") for m in r["retrieved_metas"]]
        contam_rates.append(scope_contamination_rate(sources, r["scope"]))
        scope_precs.append(scope_precision(r["retrieved_metas"], r["scope"]))

    n = len(scoped)
    return {
        "n_scoped_queries": n,
        "mean_contamination_rate": sum(contam_rates) / n,
        "mean_scope_precision": sum(scope_precs) / n,
        "contamination_rates": contam_rates,
        "scope_precisions": scope_precs,
    }


def compute_generation_report(eval_results: List[Dict]) -> Dict:
    """
    Compute generation quality metrics.

    Each result dict must contain:
      - "answer": str
      - "gold_fragments": List[str]
      - "unanswerable": bool
    """
    n = len(eval_results)
    if n == 0:
        return {}

    well_formed = sum(1 for r in eval_results if answer_well_formed(r["answer"]))
    refusal_correct = 0
    refusal_labels = {"true_refusal": 0, "true_answer": 0, "false_refusal": 0, "false_answer": 0}
    faithfulness_scores = []
    cite_density = []
    uncited_rates = []

    for r in eval_results:
        correct, label = refusal_accuracy(r["answer"], r["unanswerable"])
        if correct:
            refusal_correct += 1
        refusal_labels[label] += 1

        if not r["unanswerable"] and not is_refusal(r["answer"]):
            faithfulness_scores.append(
                faithfulness_fragment_recall(r["answer"], r["gold_fragments"])
            )
            cite_density.append(citation_density(r["answer"]))
            uncited_rates.append(uncited_sentence_rate(r["answer"]))

    report = {
        "n_queries": n,
        "well_formedness_rate": well_formed / n,
        "refusal_accuracy": refusal_correct / n,
        "refusal_breakdown": refusal_labels,
    }

    if faithfulness_scores:
        report["mean_faithfulness_fragment_recall"] = (
            sum(faithfulness_scores) / len(faithfulness_scores)
        )
        report["mean_citation_density"] = (
            sum(cite_density) / len(cite_density)
        )
        report["mean_uncited_sentence_rate"] = (
            sum(uncited_rates) / len(uncited_rates)
        )

    return report
