#!/usr/bin/env python3
"""
Unified resume-aware pipeline.

Default behavior:
  - Keep already-finished 5-hop chains (skip API)
  - Extend 2-hop results → hops 3–5 from saved hop2 response
  - Run full 5-hop only for new unique facts not in prior results

Matching key: false_fact (so lightly rewritten cutoff questions still match).

Examples:
  python run_pipeline.py --dry-run
  python run_pipeline.py --dataset math --model mistral --dry-run
  python run_pipeline.py --dataset cutoff --model both --execute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.inventory import (  # noqa: E402
    RESULTS,
    FactPlan,
    build_plan,
    estimate_api_calls,
    summarize_plan,
    _index_existing,
    _norm_ff,
)
from pipeline.chain import _as_bool, judge_hop, pass_hop, row_template, seed_hop1  # noqa: E402

OUT_DIR = RESULTS / "unified"
SEEDER = "context_injection"  # paper name for ICL / Context: seeding


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resume-aware unified 5-hop pipeline")
    p.add_argument("--dataset", choices=["geography", "math", "cutoff", "all"], default="all")
    p.add_argument("--model", choices=["mistral", "gpt", "both"], default="both")
    p.add_argument("--hops", type=int, default=5)
    p.add_argument("--dry-run", action="store_true", help="Print plan only (default if --execute omitted)")
    p.add_argument("--execute", action="store_true", help="Call APIs for extend/new only")
    p.add_argument("--limit", type=int, default=0, help="Max extend+new facts per dataset/model (0=all)")
    p.add_argument(
        "--category",
        default="",
        help="Filter facts by category or subcategory (e.g. invented_theorem)",
    )
    p.add_argument(
        "--merge-existing",
        action="store_true",
        default=True,
        help="Copy already-done 5-hop rows into unified output (default on)",
    )
    p.add_argument("--no-merge-existing", action="store_false", dest="merge_existing")
    return p.parse_args()


def _datasets(arg: str) -> list[str]:
    return ["geography", "math", "cutoff"] if arg == "all" else [arg]


def _models(arg: str) -> list[str]:
    return ["mistral", "gpt"] if arg == "both" else [arg]


def _out_path(dataset: str, model: str) -> Path:
    return OUT_DIR / f"{dataset}_{model}_{SEEDER}_{{hops}}hop.csv".replace("{hops}", "5")


def _load_existing_long_rows(plan: FactPlan, target_hops: int) -> list[dict]:
    """Pull already-saved hop rows for skip/extend prefixes into unified schema."""
    idx = _index_existing(plan.dataset, plan.model)
    hit = idx.get(_norm_ff(plan.false_fact))
    rows: list[dict] = []
    if not hit:
        return rows

    if hit.get("kind") == "long" and "long_rows" in hit:
        g = hit["long_rows"]
        for _, r in g.iterrows():
            gen = int(r["generation"])
            if gen > target_hops:
                continue
            rows.append(
                row_template(
                    dataset=plan.dataset,
                    model=plan.model,
                    seeder=SEEDER,
                    question=plan.question,
                    false_fact=plan.false_fact,
                    ground_truth=plan.ground_truth,
                    category=plan.category,
                    subcategory=plan.subcategory,
                    generation=gen,
                    response=str(r["response"]),
                    propagated=_as_bool(r.get("propagated", False)),
                    confidence=float(r.get("confidence", 0.5) or 0.5),
                    reason=str(r.get("reason", "")),
                    action="skip" if plan.action == "skip" else "extend_prefix",
                    source_file=plan.source_file,
                )
            )
        return rows

    # wide 2-hop → materialize hop1/hop2 (re-judge if needed flags stored)
    if plan.hop1_response:
        prop1, conf1, reason1 = False, 0.5, "from prior hop1 response (not re-judged)"
        rows.append(
            row_template(
                dataset=plan.dataset,
                model=plan.model,
                seeder=SEEDER,
                question=plan.question,
                false_fact=plan.false_fact,
                ground_truth=plan.ground_truth,
                category=plan.category,
                subcategory=plan.subcategory,
                generation=1,
                response=plan.hop1_response,
                propagated=prop1,
                confidence=conf1,
                reason=reason1,
                action="extend_prefix",
                source_file=plan.source_file,
            )
        )
    if plan.hop2_response:
        # Legacy Exp3 judged hop2 only; keep that score when present
        prop2, conf2, reason2 = False, 0.5, "from prior hop2 response"
        hit_wide = hit.get("wide_row")
        if hit_wide is not None:
            prop2 = _as_bool(hit_wide.get("propagated", False))
            conf2 = float(hit_wide.get("confidence", 0.5) or 0.5)
            reason2 = str(hit_wide.get("reason", reason2))
        rows.append(
            row_template(
                dataset=plan.dataset,
                model=plan.model,
                seeder=SEEDER,
                question=plan.question,
                false_fact=plan.false_fact,
                ground_truth=plan.ground_truth,
                category=plan.category,
                subcategory=plan.subcategory,
                generation=2,
                response=plan.hop2_response,
                propagated=prop2,
                confidence=conf2,
                reason=reason2,
                action="extend_prefix",
                source_file=plan.source_file,
            )
        )
    return rows


def run_one(plan: FactPlan, target_hops: int) -> list[dict]:
    rows: list[dict] = []

    if plan.action == "skip":
        return _load_existing_long_rows(plan, target_hops)

    if plan.action == "extend":
        rows.extend(_load_existing_long_rows(plan, target_hops))
        current = plan.hop2_response
        start = plan.max_existing_hop + 1
        for gen in range(start, target_hops + 1):
            current = pass_hop(plan.question, current, plan.model)
            prop, conf, reason = judge_hop(plan.question, current, plan.false_fact, plan.model)
            rows.append(
                row_template(
                    dataset=plan.dataset,
                    model=plan.model,
                    seeder=SEEDER,
                    question=plan.question,
                    false_fact=plan.false_fact,
                    ground_truth=plan.ground_truth,
                    category=plan.category,
                    subcategory=plan.subcategory,
                    generation=gen,
                    response=current,
                    propagated=prop,
                    confidence=conf,
                    reason=reason,
                    action="extend",
                    source_file=plan.source_file,
                )
            )
        return rows

    # new full chain
    current = seed_hop1(plan.question, plan.false_fact, plan.model)
    for gen in range(1, target_hops + 1):
        if gen > 1:
            current = pass_hop(plan.question, current, plan.model)
        prop, conf, reason = judge_hop(plan.question, current, plan.false_fact, plan.model)
        rows.append(
            row_template(
                dataset=plan.dataset,
                model=plan.model,
                seeder=SEEDER,
                question=plan.question,
                false_fact=plan.false_fact,
                ground_truth=plan.ground_truth,
                category=plan.category,
                subcategory=plan.subcategory,
                generation=gen,
                response=current,
                propagated=prop,
                confidence=conf,
                reason=reason,
                action="new",
                source_file="",
            )
        )
    return rows


def main() -> None:
    args = parse_args()
    # Default to dry-run unless --execute
    dry = not args.execute or args.dry_run

    datasets = _datasets(args.dataset)
    models = _models(args.model)
    plans = build_plan(datasets=datasets, models=models, target_hops=args.hops)

    summary = summarize_plan(plans)
    print("=== Resume plan (unique facts × models) ===")
    if summary.empty:
        print("(empty)")
    else:
        print(summary.to_string(index=False))
        print()
        print(summary.pivot_table(index=["dataset", "model"], columns="action", values="n", fill_value=0).to_string())

    api = estimate_api_calls(plans, target_hops=args.hops)
    print(f"\nEstimated API calls for extend+new: ~{api} (agent+judge per hop)")
    print("Legend: skip=already ≥5 hops | extend=have hop2, run 3–5 | new=full 1–5")

    if dry:
        print("\nDry-run only. Re-run with --execute to call APIs.")
        # write plan csv for inspection
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        plan_path = OUT_DIR / "resume_plan.csv"
        pd.DataFrame([p.__dict__ for p in plans]).to_csv(plan_path, index=False)
        print(f"Wrote {plan_path}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        for model in models:
            subset = [p for p in plans if p.dataset == dataset and p.model == model]
            if args.category:
                cat = args.category.strip().lower()
                subset = [
                    p
                    for p in subset
                    if p.category.strip().lower() == cat or p.subcategory.strip().lower() == cat
                ]
            work = [p for p in subset if p.action in ("extend", "new")]
            skips = [p for p in subset if p.action == "skip"]

            # Prefer full 5-hop (new) when limiting — better for targeted smokes
            work = sorted(work, key=lambda p: 0 if p.action == "new" else 1)
            if args.limit:
                work = work[: args.limit]

            cat_note = f" category={args.category}" if args.category else ""
            print(
                f"\n>>> {dataset} / {model}: skip={len(skips)} work={len(work)} "
                f"(limit={args.limit or 'all'}{cat_note})"
            )

            all_rows: list[dict] = []
            if args.merge_existing:
                for p in tqdm(skips, desc=f"merge-skip {dataset}/{model}"):
                    all_rows.extend(_load_existing_long_rows(p, args.hops))

            for p in tqdm(work, desc=f"run {dataset}/{model}"):
                try:
                    all_rows.extend(run_one(p, args.hops))
                except Exception as e:
                    print(f"\nFAILED [{p.action}] {p.false_fact[:60]!r}: {e}")
                    continue

            out = _out_path(dataset, model)
            pd.DataFrame(all_rows).to_csv(out, index=False)
            print(f"Wrote {out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
