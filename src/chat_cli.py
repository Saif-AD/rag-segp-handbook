import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR

FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "chunks_meta.jsonl"

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


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")


def split_sentences(text: str) -> List[str]:
    text = _WS_RE.sub(" ", text.strip())
    if not text:
        return []
    # Simple sentence split; good enough for handbook prose
    sents = _SENT_SPLIT_RE.split(text)
    # Filter very short fragments
    return [s.strip() for s in sents if len(s.strip()) >= 30]

def is_noise_sentence(sent: str) -> bool:
    s = sent.lower()
    # UI/navigation boilerplate that often pollutes “marking” answers
    noise_terms = [
        "open the submenu", "left-hand menu", "navigate", "dashboard",
        "click", "keats", "webbrowser", "submenu", "menu"
    ]
    return any(t in s for t in noise_terms)


def cite_string(m: Dict) -> str:
    # Keep this stable for your report screenshots
    return f'{m["chapter"]} > {m["section"]} > {m["subsection"]} ({m["source_file"]})'

def expand_query(query: str) -> str:
    q = query.lower()
    # Expand only when user asks about marking/assessment
    if any(w in q for w in ["mark", "marked", "marking", "assessment", "grade"]):
        return query + " assessment marking scheme individual mark team mark redistribution"
    return query

def heading_bias(query: str, m: Dict) -> float:
    """
    Small heuristic boost based on chapter/section/subsection keywords.
    Keeps baseline grounded but improves intent matching.
    """
    q = query.lower()
    heading = f'{m["chapter"]} {m["section"]} {m["subsection"]}'.lower()

    boost = 0.0
    # If user asks about marking/assessment, prefer those sections
    if any(w in q for w in ["mark", "marked", "marking", "assessment", "grade"]):
        if any(w in heading for w in ["mark", "marking", "assessment"]):
            boost += 0.06  # small but decisive at your score range
    # If user asks about allocation/teams, prefer allocation sections
    if any(w in q for w in ["allocate", "allocation", "teams", "team"]):
        if "allocation" in heading or "team" in heading:
            boost += 0.04
    return boost


def select_extractive_answer(
    query: str,
    retrieved: List[Tuple[float, Dict]],
    model: SentenceTransformer,
    max_sentences: int = 5,
    max_chunks: int = 2,
) -> Tuple[str, List[str]]:
    """
    Extractive baseline:
    - split retrieved chunks into sentences
    - embed sentences + query
    - select top-N sentences by cosine similarity (with simple diversity)
    """
    q_emb = model.encode(query, normalize_embeddings=True).astype("float32")

    candidates: List[Tuple[float, str, str]] = []  # (score, sentence, citation)
    for score, m in retrieved[:max_chunks]:
        for sent in split_sentences(m["text"]):
            if is_noise_sentence(sent):
                continue
            s_emb = model.encode(sent, normalize_embeddings=True).astype("float32")
            sim = float(np.dot(q_emb, s_emb))
            candidates.append((sim, sent, cite_string(m)))

    if not candidates:
        return (
            "I couldn’t find enough relevant text in the handbook excerpts to answer that.",
            [cite_string(m) for _, m in retrieved],
        )

    # Sort by similarity
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Simple diversity: avoid near-duplicate sentences
    chosen: List[Tuple[float, str, str]] = []
    seen = set()
    for sim, sent, cite in candidates:
        key = sent.lower()
        if key in seen:
            continue
        # avoid picking multiple sentences that are almost identical
        if any(abs(sim - s2) < 0.01 and sent[:60] == t2[:60] for s2, t2, _ in chosen):
            continue
        chosen.append((sim, sent, cite))
        seen.add(key)
        if len(chosen) >= max_sentences:
            break

    # Build answer and cite list
    answer_lines = [f"- {s}" for _, s, _ in chosen]
    cites = []
    for _, _, c in chosen:
        if c not in cites:
            cites.append(c)

    answer = "\n".join(answer_lines)
    return answer, cites


def retrieve(query: str, model: SentenceTransformer, index, meta: List[Dict], k: int = 8) -> List[Tuple[float, Dict]]:
    def _search(q: str) -> List[Tuple[float, int]]:
        q_emb = model.encode(q, normalize_embeddings=True).astype("float32")
        D, I = index.search(np.expand_dims(q_emb, axis=0), k=k)
        return [(float(score), int(idx)) for score, idx in zip(D[0], I[0])]

    base_hits = _search(query)
    exp_query = expand_query(query)
    exp_hits = _search(exp_query) if exp_query != query else []

    # Merge by index id, keep best score
    best: Dict[int, float] = {}
    for score, idx in base_hits + exp_hits:
        if idx not in best or score > best[idx]:
            best[idx] = score

    # Apply heading bias and sort
    results = []
    for idx, score in best.items():
        m = meta[idx]
        score = float(score) + heading_bias(query, m)
        results.append((score, m))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def main() -> None:
    if not FAISS_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Index not found. Run src/embed_index.py first.")

    meta = load_meta(META_PATH)
    index = faiss.read_index(str(FAISS_PATH))
    model = SentenceTransformer(MODEL_NAME)

    print("Grounded RAG (extractive baseline)")
    print("Answers are constructed ONLY from retrieved handbook sentences.")
    print("Type ':q' to quit.\n")

    while True:
        q = input("Query> ").strip()
        if not q:
            continue
        if q == ":q":
            break

        retrieved = retrieve(q, model, index, meta, k=8)

        answer, cites = select_extractive_answer(q, retrieved, model, max_sentences=5)

        print("\nAnswer (grounded):")
        print(answer)

        print("\nSources:")
        for c in cites:
            print(f"- {c}")

        print("\nSafety note: This tool may be incomplete or incorrect; verify against the official handbook.\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
