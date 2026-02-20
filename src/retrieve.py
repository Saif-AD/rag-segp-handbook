import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR

FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "chunks_meta.jsonl"


def load_meta(path: Path) -> List[Dict]:
    meta = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))
    return meta


def main() -> None:
    if not FAISS_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Index not found. Run src/embed_index.py first.")

    meta = load_meta(META_PATH)
    index = faiss.read_index(str(FAISS_PATH))

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    print("RAG Retrieval CLI")
    print("Type a question and press enter. Type ':q' to quit.\n")

    while True:
        q = input("Query> ").strip()
        if not q:
            continue
        if q == ":q":
            break

        q_emb = model.encode(q, normalize_embeddings=True).astype("float32")
        D, I = index.search(np.expand_dims(q_emb, axis=0), k=5)

        print("\nTop results:")
        for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
            m = meta[int(idx)]
            cite = f'{m["chapter"]} > {m["section"]} > {m["subsection"]} ({m["source_file"]})'
            preview = m["text"][:350].replace("\n", " ")
            print(f"\n#{rank}  score={score:.3f}")
            print(f"    {cite}")
            print(f"    {preview}...")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
