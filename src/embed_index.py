import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import DATA_PROCESSED, INDEX_DIR

CHUNKS_JSONL = DATA_PROCESSED / "chunks.jsonl"
FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "chunks_meta.jsonl"


def load_chunks(path: Path) -> List[Dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(CHUNKS_JSONL)
    texts = [c["text"] for c in chunks]

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    embeddings = []
    for t in tqdm(texts, desc="Embedding chunks"):
        emb = model.encode(t, normalize_embeddings=True)
        embeddings.append(emb)

    X = np.vstack(embeddings).astype("float32")

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine if vectors are normalised
    index.add(X)

    faiss.write_index(index, str(FAISS_PATH))

    # Save metadata aligned to FAISS row ids
    with META_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            meta = {
                "chunk_id": c["chunk_id"],
                "source_file": c["source_file"],
                "chapter": c["chapter"],
                "section": c["section"],
                "subsection": c["subsection"],
                "chunk_index": c["chunk_index"],
                "text": c["text"],  # keep text for immediate retrieval debug; we can slim later
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print("Saved:")
    print(f"- FAISS index: {FAISS_PATH}")
    print(f"- Metadata:    {META_PATH}")
    print(f"Model: {model_name}")
    print(f"Chunks indexed: {len(chunks)} | Embedding dim: {dim}")


if __name__ == "__main__":
    main()
