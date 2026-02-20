from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "index"

# Put the unzipped handbook folder here:
# data/raw/handbook/handbook.tex
HANDBOOK_DIR = DATA_RAW / "handbook"
MAIN_TEX = HANDBOOK_DIR / "handbook.tex"

SECTIONS_JSONL = DATA_PROCESSED / "sections.jsonl"
