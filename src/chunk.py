import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, List

from tqdm import tqdm

from config import SECTIONS_JSONL, DATA_PROCESSED

CHUNKS_JSONL = DATA_PROCESSED / "chunks.jsonl"


@dataclass
class ChunkRecord:
    chunk_id: str
    source_file: str
    chapter: str
    section: str
    subsection: str
    chunk_index: int
    text: str


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def simple_tokenise(text: str) -> List[str]:
    # Deterministic: words + punctuation as separate tokens
    return _TOKEN_RE.findall(text)


def detokenise(tokens: List[str]) -> str:
    # Re-join tokens into readable text
    out = []
    for t in tokens:
        if out and re.match(r"[^\w\s]", t):  # punctuation
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


def load_sections(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Baseline parameters (match your earlier claim: ~300–350 tokens)
    CHUNK_SIZE = 350
    OVERLAP = 50

    total_chunks = 0
    with CHUNKS_JSONL.open("w", encoding="utf-8") as out:
        for rec in tqdm(list(load_sections(SECTIONS_JSONL)), desc="Chunking sections"):
            text = rec["text"].strip()
            toks = simple_tokenise(text)

            # If the section is short, keep it as one chunk
            if len(toks) <= CHUNK_SIZE:
                chunk_text = text
                chunk = ChunkRecord(
                    chunk_id=f'{rec["source_file"]}::{rec["chapter"]}::{rec["section"]}::{rec["subsection"]}::0',
                    source_file=rec["source_file"],
                    chapter=rec["chapter"],
                    section=rec["section"],
                    subsection=rec["subsection"],
                    chunk_index=0,
                    text=chunk_text,
                )
                out.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
                total_chunks += 1
                continue

            for i, chunk_tokens in enumerate(make_chunks(toks, CHUNK_SIZE, OVERLAP)):
                chunk_text = detokenise(chunk_tokens)

                chunk = ChunkRecord(
                    chunk_id=f'{rec["source_file"]}::{rec["chapter"]}::{rec["section"]}::{rec["subsection"]}::{i}',
                    source_file=rec["source_file"],
                    chapter=rec["chapter"],
                    section=rec["section"],
                    subsection=rec["subsection"],
                    chunk_index=i,
                    text=chunk_text,
                )
                out.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Wrote {total_chunks} chunks to: {CHUNKS_JSONL}")
    print(f"Chunk params: chunk_size={CHUNK_SIZE}, overlap={OVERLAP}")


if __name__ == "__main__":
    main()
