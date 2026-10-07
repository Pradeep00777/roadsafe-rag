"""Evaluate retrieval quality, answer quality and guardrails on a golden set.

    python -m roadsafe_rag.evaluate --questions data/eval/golden_questions.jsonl

Golden set format (one JSON object per line):
    {"question": "...", "in_scope": true,  "expected_keywords": ["fatalities", "2024"]}
    {"question": "...", "in_scope": false}
`expected_keywords` are words/numbers that must appear in the retrieved passage
(retrieval hit) and in the final answer (answer check). Write them from the PDF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from .factory import build_pipeline
from .rag import RAGPipeline


def load_questions(path: str | Path) -> list[dict]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            items.append(json.loads(line))
    return items


def _contains_all(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return all(k.lower() in lowered for k in keywords)


def _mean(values: list) -> float | None:
    return round(mean(values), 3) if values else None


def run_eval(pipeline: RAGPipeline, items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        question = item["question"]
        in_scope = bool(item.get("in_scope", True))
        keywords = item.get("expected_keywords", [])

        rank = None
        if in_scope and keywords:
            for r, (_, meta) in enumerate(pipeline.retrieve(question), 1):
                if _contains_all(meta["text"], keywords):
                    rank = r
                    break

        result = pipeline.answer(question)
        rows.append({
            "question": question,
            "in_scope": in_scope,
            "expected_keywords": keywords,
            "rank": rank,
            "refused": result.refused,
            "reason": result.reason,
            "answer": result.answer,
            "n_citations": len(result.citations),
            "latency_ms": result.latency_ms,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    in_rows = [r for r in rows if r["in_scope"]]
    out_rows = [r for r in rows if not r["in_scope"]]
    graded = [r for r in in_rows if r["expected_keywords"]]
    answered = [r for r in rows if not r["refused"]]
    return {
        "questions": len(rows),
        "in_scope": len(in_rows),
        "out_of_scope": len(out_rows),
        "retrieval_hit_rate": _mean([r["rank"] is not None for r in graded]),
        "mrr": _mean([1 / r["rank"] if r["rank"] else 0.0 for r in graded]),
        "answer_keyword_rate": _mean(
            [(not r["refused"]) and _contains_all(r["answer"], r["expected_keywords"]) for r in graded]
        ),
        "false_refusal_rate": _mean([r["refused"] for r in in_rows]),
        "correct_refusal_rate": _mean([r["refused"] for r in out_rows]),
        "citation_rate_when_answered": _mean([r["n_citations"] > 0 for r in answered]),
        "avg_latency_ms": _mean([r["latency_ms"] for r in rows]),
    }


def to_markdown(metrics: dict) -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v if v is not None else 'n/a'} |" for k, v in metrics.items()]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RoadSafe RAG pipeline.")
    parser.add_argument("--questions", default="data/eval/golden_questions.jsonl")
    parser.add_argument("--out-dir", default="data/eval/results")
    args = parser.parse_args()

    rows = run_eval(build_pipeline(), load_questions(args.questions))
    metrics = summarize(rows)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "metrics.md").write_text(to_markdown(metrics) + "\n", encoding="utf-8")
    print(to_markdown(metrics))


if __name__ == "__main__":
    main()
