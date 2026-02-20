"""
Generate a human annotation template from E2E evaluation results.

For a dissertation, you need human evaluation on a representative subset.
This script generates a structured CSV/JSONL for manual annotation of:

  1. Answer correctness (0/1/2: wrong, partially correct, correct)
  2. Faithfulness (0/1: does the answer contain unsupported claims?)
  3. Scope correctness (0/1: does the answer use the right project's policy?)
  4. Citation quality (0/1: are citations pointing to relevant sources?)

Usage:
    python eval/human_annotation_template.py --input eval/e2e_results_llama3.2_3b.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="E2E results JSONL")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--sample", type=int, default=0, help="Random sample size (0=all)")
    args = parser.parse_args()

    results = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if args.sample > 0 and args.sample < len(results):
        import random
        random.seed(42)
        results = random.sample(results, args.sample)

    out_path = args.output or args.input.replace(".jsonl", "_annotation.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "id",
            "query",
            "scope",
            "unanswerable",
            "system_answer",
            "gold_fragments",
            "sources_used",
            # ---- Annotator fills these ----
            "correctness (0=wrong, 1=partial, 2=correct)",
            "faithfulness (0=hallucinated, 1=grounded)",
            "scope_correct (0=wrong_project, 1=correct)",
            "citation_quality (0=wrong_citations, 1=correct)",
            "notes",
        ])

        for r in results:
            writer.writerow([
                r["id"],
                r["query"],
                r.get("scope", ""),
                r.get("unanswerable", False),
                r.get("answer", "").replace("\n", " | "),
                "; ".join(r.get("gold_fragments", [])),
                "; ".join(r.get("relevant_sources", [])),
                "",  # correctness
                "",  # faithfulness
                "",  # scope_correct
                "",  # citation_quality
                "",  # notes
            ])

    print(f"Annotation template saved to: {out_path}")
    print(f"  {len(results)} items to annotate")
    print()
    print("Instructions for annotators:")
    print("  correctness:    0 = factually wrong or irrelevant")
    print("                  1 = partially correct (some info right, some missing)")
    print("                  2 = fully correct and complete")
    print("  faithfulness:   0 = contains claims not in the retrieved sources")
    print("                  1 = all claims are supported by sources")
    print("  scope_correct:  0 = answer uses wrong project's policy (contamination)")
    print("                  1 = answer uses correct project's policy")
    print("  citation_qual:  0 = citations point to irrelevant chunks")
    print("                  1 = citations point to relevant chunks")


if __name__ == "__main__":
    main()
