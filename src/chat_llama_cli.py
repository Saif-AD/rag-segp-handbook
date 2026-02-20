# src/chat_llama_cli.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR
from llama_ollama import ollama_generate

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "chunks_meta.jsonl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"   # change to llama3.1:8b if you pulled it


_CITE_RE = re.compile(r"\[S\d+\]")


_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def answer_is_well_formed(ans: str) -> bool:
    lines = [ln.strip() for ln in ans.splitlines() if ln.strip()]
    if not lines:
        return False

    # exactly one Sources used line
    sources_lines = [ln for ln in lines if ln.lower().startswith("sources used:")]
    if len(sources_lines) != 1:
        return False

    # body excludes Sources used
    body_lines = [ln for ln in lines if not ln.lower().startswith("sources used:")]
    body = " ".join(body_lines).strip()
    if not body:
        return False

    # Split into sentences (simple, good enough for your use)
    sentences = [s.strip() for s in _SENT_SPLIT.split(body) if s.strip()]
    if not (2 <= len(sentences) <= 4):
        return False

    # Every sentence must contain at least one [S#]
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
1) Write 2–4 short sentences (no bullet points).
2) EVERY sentence MUST include at least one citation tag like [S1] or [S2][S4].
3) The FIRST sentence MUST answer directly and assertively (e.g., "No, ..."). AND MUST include at least one citation tag.
4) At least ONE sentence MUST include a short direct quote (5–20 words) that is DIRECTLY RELEVANT to the question (e.g., about allocation/team formation), in double quotes, followed by a citation.
5) After the sentences, add exactly one final line:
Sources used: [S#], [S#], ...
6) Do not include any other text.
7) If the question is about forming teams vs allocation, your quote MUST mention "allocated" or "allocation".
8) If you write any sentence without a citation tag, your answer is invalid.”
"""


def load_meta(path: Path) -> List[Dict]:
    meta: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    return meta


def cite_string(m: Dict) -> str:
    return f'{m["chapter"]} > {m["section"]} > {m["subsection"]} ({m["source_file"]})'


def heading_bias(query: str, m: Dict) -> float:
    q = query.lower()
    heading = f'{m.get("chapter","")} {m.get("section","")} {m.get("subsection","")}'.lower()

    boost = 0.0
    if any(w in q for w in ["mark", "marked", "marking", "assessment", "grade"]):
        if any(w in heading for w in ["mark", "marking", "assessment"]):
            boost += 0.06
    if any(w in q for w in ["allocate", "allocation", "teams", "team", "formation"]):
        if "allocation" in heading or "team" in heading or "formation" in heading:
            boost += 0.04
    if any(w in q for w in ["allowed", "allow", "use", "javascript", "react", "framework", "language", "technology", "stack"]):
        if any(w in heading for w in ["technology", "tools", "constraints", "stack", "languages"]):
            boost += 0.03
    return boost


def expand_query(query: str) -> str:
    q = query.lower()

    # Tech/allowed queries
    if any(w in q for w in ["allowed", "allow", "use", "javascript", "react", "framework", "language", "technology", "stack"]):
        return query + " allowed technologies technology constraints must be developed entirely with"

    # Marking queries
    if any(w in q for w in ["mark", "marked", "marking", "assessment", "grade", "redistribution", "correction"]):
        return query + " assessment marking scheme individual mark team mark redistribution correction"

    # Allocation/team queries
    if any(w in q for w in ["allocate", "allocation", "teams", "team", "formation", "register"]):
        return query + " team allocation team formation register team feedback keats"

    return query


def retrieve_enhanced(
    query: str,
    model: SentenceTransformer,
    index,
    meta: List[Dict],
    k: int = 6,
) -> List[Tuple[float, Dict]]:
    def _search(q: str) -> List[Tuple[float, int]]:
        q_emb = model.encode(q, normalize_embeddings=True).astype("float32")
        D, I = index.search(np.expand_dims(q_emb, axis=0), k=k)
        return [(float(score), int(idx)) for score, idx in zip(D[0], I[0])]

    base_hits = _search(query)
    exp = expand_query(query)
    exp_hits = _search(exp) if exp != query else []

    best: Dict[int, float] = {}
    for score, idx in base_hits + exp_hits:
        if idx not in best or score > best[idx]:
            best[idx] = score

    results: List[Tuple[float, Dict]] = []
    for idx, score in best.items():
        m = meta[idx]
        score = float(score) + heading_bias(query, m)
        results.append((score, m))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:k]


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
        # fallback: at least remove mgp
        return [(s, m) for (s, m) in retrieved if not is_mgp(m)]

    if scope == "major":
        mgp = [(s, m) for (s, m) in retrieved if is_mgp(m)]
        if len(mgp) >= 2:
            return mgp
        return [(s, m) for (s, m) in retrieved if not is_sgp(m)]

    return retrieved


def build_prompt(query: str, retrieved: List[Tuple[float, Dict]]) -> Tuple[str, List[str], str]:
    sources: List[str] = []
    context_blocks: List[str] = []
    for i, (_, m) in enumerate(retrieved, start=1):
        tag = f"[S{i}]"
        cite = cite_string(m)
        sources.append(f"{tag} {cite}")
        text = str(m.get("text", "")).strip().replace("\n", " ")
        text = text[:1200]  # hard cap per source
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
1) Write 2–4 short sentences (no bullet points).
2) EVERY sentence must include at least one citation tag like [S1] or [S2][S4].
3) The FIRST sentence must answer the question directly and assertively (e.g. "No, …", "The project is marked by …"). AND MUST include at least one citation tag.
4) After the sentences, add exactly one final line:
Sources used: [S#], [S#], ...
5) Do not mention "SOURCES", "excerpts", or internal system behaviour.
6) If you write any sentence without a citation tag, your answer is invalid.”

Domain constraints:
- Policy explanations are allowed, but focus on consequences and meaning rather than internal calculations.
- If a technology is a framework or library built on a prohibited language, treat it as prohibited unless SOURCES explicitly state otherwise (e.g. React → JavaScript).
- Do not apply major group project rules to the small group project, or vice versa, unless the source explicitly states the same rule applies.
- If the question is about SGP team formation, default assumption is allocation via KEATS unless SOURCES explicitly mention student-formed teams.
"""

    return prompt, sources, context


def is_meta_question(q: str) -> bool:
    ql = q.lower()
    markers = [
        "fallacy",
        "contradiction",
        "your last response",
        "previous response",
        "you said",
        "that was wrong",
        "why did you say",
        "compare the two answers",
        "inconsistent",
    ]
    return any(m in ql for m in markers)


def main() -> None:
    if not FAISS_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Index not found. Run src/embed_index.py first.")

    meta = load_meta(META_PATH)
    index = faiss.read_index(str(FAISS_PATH))
    emb_model = SentenceTransformer(EMBED_MODEL)

    print("RAG + LLaMA (grounded generation via Ollama)")
    print("Type ':q' to quit.\n")

    while True:
        q = input("Query> ").strip()
        if not q:
            continue
        if q == ":q":
            break

        # Avoid generating nonsense for meta/system questions (keeps eval clean)
        if is_meta_question(q):
            print("\nAnswer:")
            print("I can't find this in the handbook excerpts I retrieved.")
            print("\nSafety note: This tool may be incomplete or incorrect; verify against the official handbook.\n")
            print("-" * 60 + "\n")
            continue

        # Retrieve a bit wider, then scope-filter, then truncate
        retrieved = retrieve_enhanced(q, emb_model, index, meta, k=8)

        scope = detect_scope(q)
        if scope is not None:
            filtered = filter_by_scope(retrieved, scope)
            # Only apply the filter if we still have enough context to be useful
            if len(filtered) >= 2:
                retrieved = filtered
            # Hard safety: if user asked about SGP/MGP and we don't have enough scoped sources, refuse.
            if scope == "small":
                sgp_count = sum(1 for _, m in retrieved if "small group project" in str(m.get("chapter","")).lower()
                                or "small_group_project" in str(m.get("source_file","")).lower())
                if sgp_count < 2:
                    print("\nAnswer:")
                    print("I can't find this in the handbook excerpts I retrieved.")
                    print("\nSafety note: This tool may be incomplete or incorrect; verify against the official handbook.\n")
                    print("-" * 60 + "\n")
                    continue
        
        retrieved = retrieved[:4]
        prompt, sources, context = build_prompt(q, retrieved)

        answer = ollama_generate(prompt, model=LLM_MODEL)
        if not answer_is_well_formed(answer):
            answer = ollama_generate(strict_prompt(q, context), model=LLM_MODEL)

        print("\nAnswer:")
        print(answer)

        print("\nRetrieved sources:")
        for s in sources:
            print("-", s)

        print("\nSafety note: This tool may be incomplete or incorrect; verify against the official handbook.\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
