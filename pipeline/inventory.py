"""
Build a skip / extend / new plan from existing results × data/unique tables.

Matching key: false_fact (stable across light question rewrites, e.g. cutoff dates).
Fallback: (question, false_fact) exact match.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
UNIQUE = ROOT / "data" / "unique"

Action = Literal["skip", "extend", "new"]

# Existing result sources for the unified ICL / context-injection 5-hop study.
# model is inferred from filename (*_gpt.csv → gpt, else mistral).
RESULT_SOURCES = {
    "geography": [
        # already 5-hop (ICL)
        ("exp2_results.csv", "mistral", "long"),
        ("exp2_results_gpt.csv", "gpt", "long"),
        # 2-hop multi-seeder; we only resume ICL-equivalent rows via teacher/student
        ("exp1_results.csv", "mistral", "wide_icl"),
        ("exp1_results_gpt.csv", "gpt", "wide_icl"),
    ],
    "math": [
        ("exp4_results.csv", "mistral", "long"),
        ("exp4_results_gpt.csv", "gpt", "long"),
        ("exp3_results.csv", "mistral", "wide"),
        ("exp3_results_gpt.csv", "gpt", "wide"),
    ],
    "cutoff": [
        ("exp4_cutoff_results.csv", "mistral", "long"),
        ("exp4_cutoff_results_gpt.csv", "gpt", "long"),
        ("exp3_cutoff_results.csv", "mistral", "wide"),
        ("exp3_cutoff_results_gpt.csv", "gpt", "wide"),
    ],
}

UNIQUE_FILES = {
    "geography": UNIQUE / "geography_unique.csv",
    "math": UNIQUE / "math_unique.csv",
    "cutoff": UNIQUE / "cutoff_unique.csv",
}


@dataclass
class FactPlan:
    dataset: str
    model: str
    action: Action
    question: str
    false_fact: str
    ground_truth: str
    category: str
    subcategory: str
    # For extend: prior hop responses (1-indexed)
    hop1_response: str = ""
    hop2_response: str = ""
    source_file: str = ""
    max_existing_hop: int = 0


def _norm_ff(x: str) -> str:
    return " ".join(str(x).strip().lower().split())


def _load_unique(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(UNIQUE_FILES[dataset])
    if "category" not in df.columns:
        df["category"] = ""
    if "subcategory" not in df.columns:
        df["subcategory"] = ""
    return df


def _index_existing(dataset: str, model: str) -> dict[str, dict]:
    """
    Map normalized false_fact → best existing chain info for this dataset/model.
    Prefer longer chains (5-hop long format over 2-hop wide).
    """
    index: dict[str, dict] = {}

    for fname, src_model, kind in RESULT_SOURCES[dataset]:
        if src_model != model:
            continue
        path = RESULTS / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)

        if kind == "long":
            if "generation" not in df.columns or "response" not in df.columns:
                continue
            for ff, g in df.groupby("false_fact", sort=False):
                key = _norm_ff(ff)
                max_gen = int(g["generation"].max())
                g = g.sort_values("generation")
                by_gen = {int(r.generation): str(r.response) for _, r in g.iterrows()}
                row0 = g.iloc[0]
                candidate = {
                    "max_hop": max_gen,
                    "hop1": by_gen.get(1, ""),
                    "hop2": by_gen.get(2, ""),
                    "by_gen": by_gen,
                    "question": str(row0["question"]),
                    "ground_truth": str(row0.get("ground_truth", "")),
                    "category": str(row0.get("category", "")),
                    "subcategory": str(row0.get("subcategory", "")),
                    "source": fname,
                    "kind": "long",
                    "long_rows": g,
                }
                prev = index.get(key)
                if prev is None or candidate["max_hop"] > prev["max_hop"]:
                    index[key] = candidate

        elif kind in ("wide", "wide_icl"):
            # teacher = hop1, student = hop2
            if "student_response" not in df.columns:
                continue
            work = df
            if kind == "wide_icl" and "seeding_method" in df.columns:
                # Exp1: only treat ICL rows as resumable for the unified ICL chain
                work = df[df["seeding_method"] == "in_context_learning"]
            for _, r in work.iterrows():
                key = _norm_ff(r["false_fact"])
                candidate = {
                    "max_hop": 2,
                    "hop1": str(r.get("teacher_response", "")),
                    "hop2": str(r["student_response"]),
                    "by_gen": {
                        1: str(r.get("teacher_response", "")),
                        2: str(r["student_response"]),
                    },
                    "question": str(r["question"]),
                    "ground_truth": str(r.get("ground_truth", "")),
                    "category": str(r.get("category", "")),
                    "subcategory": str(r.get("subcategory", "")),
                    "source": fname,
                    "kind": "wide",
                    "wide_row": r,
                    "propagated": r.get("propagated"),
                    "confidence": r.get("confidence"),
                    "reason": r.get("reason", ""),
                }
                prev = index.get(key)
                if prev is None or candidate["max_hop"] > prev["max_hop"]:
                    # don't downgrade a 5-hop long chain
                    if prev is not None and prev["max_hop"] >= 5:
                        continue
                    if prev is not None and prev["max_hop"] > candidate["max_hop"]:
                        continue
                    index[key] = candidate

    return index


def build_plan(
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    target_hops: int = 5,
) -> list[FactPlan]:
    datasets = datasets or ["geography", "math", "cutoff"]
    models = models or ["mistral", "gpt"]
    plans: list[FactPlan] = []

    for dataset in datasets:
        unique = _load_unique(dataset)
        for model in models:
            existing = _index_existing(dataset, model)
            for _, row in unique.iterrows():
                ff = str(row["false_fact"])
                key = _norm_ff(ff)
                hit = existing.get(key)

                base = FactPlan(
                    dataset=dataset,
                    model=model,
                    action="new",
                    question=str(row["question"]),
                    false_fact=ff,
                    ground_truth=str(row["ground_truth"]),
                    category=str(row.get("category", "") or ""),
                    subcategory=str(row.get("subcategory", "") or ""),
                )

                if hit is None:
                    plans.append(base)
                    continue

                # Prefer unique-table metadata; keep responses from results
                base.hop1_response = hit.get("hop1", "")
                base.hop2_response = hit.get("hop2", "")
                base.source_file = hit.get("source", "")
                base.max_existing_hop = int(hit.get("max_hop", 0))
                if not base.category and hit.get("category"):
                    base.category = hit["category"]
                if not base.subcategory and hit.get("subcategory"):
                    base.subcategory = hit["subcategory"]

                if hit["max_hop"] >= target_hops:
                    base.action = "skip"
                elif hit["max_hop"] >= 2 and base.hop2_response:
                    base.action = "extend"
                else:
                    base.action = "new"
                plans.append(base)

    return plans


def summarize_plan(plans: list[FactPlan]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(p) for p in plans])
    if df.empty:
        return df
    return (
        df.groupby(["dataset", "model", "action"])
        .size()
        .reset_index(name="n")
        .sort_values(["dataset", "model", "action"])
    )


def estimate_api_calls(plans: list[FactPlan], target_hops: int = 5) -> int:
    """Rough: each new hop = 1 agent + 1 judge call."""
    total = 0
    for p in plans:
        if p.action == "skip":
            continue
        if p.action == "extend":
            hops = target_hops - p.max_existing_hop
            total += hops * 2
        elif p.action == "new":
            total += target_hops * 2
    return total
