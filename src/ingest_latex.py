import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from tqdm import tqdm

from config import MAIN_TEX, HANDBOOK_DIR, SECTIONS_JSONL, DATA_PROCESSED


@dataclass
class SectionRecord:
    source_file: str
    chapter: str
    section: str
    subsection: str
    text: str


INCLUDE_RE = re.compile(r"\\include\{([^}]+)\}")
CHAPTER_RE = re.compile(r"\\chapter\{([^}]+)\}")
SECTION_RE = re.compile(r"\\section\{([^}]+)\}")
SUBSECTION_RE = re.compile(r"\\subsection\{([^}]+)\}")
SUBSUBSECTION_RE = re.compile(r"\\subsubsection\{([^}]+)\}")

# \tfaq[optional]{...}{QUESTION}{CATEGORY}
TFAQ_RE = re.compile(r"\\tfaq(?:\[[^\]]*\])?\{[^}]*\}\{([^}]+)\}\{([^}]+)\}")

COMMENT_RE = re.compile(r"(?<!\\)%.*$")  # strip % comments (not \%)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def discover_includes(main_tex: str) -> List[str]:
    return INCLUDE_RE.findall(main_tex)


def strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = COMMENT_RE.sub("", line)
        lines.append(line)
    return "\n".join(lines)


def latex_to_plain(text: str) -> str:
    """
    Conservative LaTeX -> plain text cleaning.
    Keep readable content; remove most commands and noise.
    """
    text = strip_comments(text)

    # Normalise FAQ macro into readable text
    # Example: "FAQ (Category): Question"
    def _tfaq_sub(m: re.Match) -> str:
        q = m.group(1).strip()
        cat = m.group(2).strip()
        return f"\nFAQ ({cat}): {q}\n"

    text = TFAQ_RE.sub(_tfaq_sub, text)

    # Remove labels/indexes/citations that aren’t content
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\index\{[^}]*\}", "", text)
    text = re.sub(r"\\cite\{[^}]*\}", "", text)

    # Acronyms like \ac{...} -> content
    text = re.sub(r"\\ac\{([^}]+)\}", r"\1", text)

    # Common formatting commands: keep inner text
    for cmd in ["textbf", "textit", "emph", "underline", "texttt"]:
        text = re.sub(rf"\\{cmd}\{{([^}}]+)\}}", r"\1", text)

    # Itemize/enumerate: turn \item into "- "
    text = re.sub(r"\\begin\{itemize\}", "\n", text)
    text = re.sub(r"\\end\{itemize\}", "\n", text)
    text = re.sub(r"\\begin\{enumerate\}", "\n", text)
    text = re.sub(r"\\end\{enumerate\}", "\n", text)
    text = re.sub(r"\\item\s*", "\n- ", text)

    # Remove remaining environments we don’t need (lightweight)
    text = re.sub(r"\\begin\{[^}]+\}", "\n", text)
    text = re.sub(r"\\end\{[^}]+\}", "\n", text)

    # Remove remaining commands like \command{...} and \command
    # Keep the contents of single-brace commands when safe
    text = re.sub(r"\\[a-zA-Z]+\*?\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)

    # Unescape common sequences
    text = text.replace("~", " ")
    text = text.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_")

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_section_records(tex_path: Path) -> Iterator[SectionRecord]:
    raw = read_text(tex_path)
    raw = strip_comments(raw)

    chapter = ""
    section = ""
    subsection = ""

    buffer: List[str] = []

    def flush() -> Optional[SectionRecord]:
        nonlocal buffer, chapter, section, subsection
        content = latex_to_plain("\n".join(buffer)).strip()
        buffer = []
        # Only emit if there’s meaningful content
        if len(content) < 40:
            return None
        return SectionRecord(
            source_file=tex_path.name,
            chapter=chapter or "(none)",
            section=section or "(none)",
            subsection=subsection or "(none)",
            text=content,
        )

    lines = raw.splitlines()
    for line in lines:
        ch = CHAPTER_RE.search(line)
        sec = SECTION_RE.search(line)
        sub = SUBSECTION_RE.search(line)
        subsub = SUBSUBSECTION_RE.search(line)

        # If we hit a new heading, flush previous buffer first
        if ch:
            rec = flush()
            if rec:
                yield rec
            chapter = ch.group(1).strip()
            section = ""
            subsection = ""
            continue

        if sec:
            rec = flush()
            if rec:
                yield rec
            section = sec.group(1).strip()
            subsection = ""
            continue

        if sub:
            rec = flush()
            if rec:
                yield rec
            subsection = sub.group(1).strip()
            continue

        if subsub:
            # treat subsubsection as part of subsection label
            rec = flush()
            if rec:
                yield rec
            subsection = f"{subsection} / {subsub.group(1).strip()}".strip(" /")
            continue

        buffer.append(line)

    # final flush
    rec = flush()
    if rec:
        yield rec


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    main_tex = read_text(MAIN_TEX)
    includes = discover_includes(main_tex)

    tex_files = []
    for inc in includes:
        p = (HANDBOOK_DIR / f"{inc}.tex")
        if p.exists():
            tex_files.append(p)

    if not tex_files:
        raise FileNotFoundError(
            f"No included .tex files found. Expected includes under: {HANDBOOK_DIR}"
        )

    out_path = SECTIONS_JSONL
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for tex in tqdm(tex_files, desc="Ingesting LaTeX"):
            for rec in iter_section_records(tex):
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                n += 1

    print(f"Wrote {n} section records to: {out_path}")


if __name__ == "__main__":
    main()
