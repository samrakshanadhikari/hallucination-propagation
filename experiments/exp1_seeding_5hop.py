#!/usr/bin/env python3
"""
Experiment 1 (expanded): Do context-learning seed variants affect 5-hop persistence?

Design
  - Datasets: unique geography / math / cutoff (not the old geography-only scaffold)
  - Math is stratified by prior type: numerical, attribution, wrong_statement,
    invented_theorem, fake_solved (different priors → report separately)
  - Hop 1: seed with a context-learning variant (ICL, prompt priming, RAG, optional SFT-sim)
  - Hops 2–5: same context-pass protocol for every variant (no confounds)
  - Judge every hop
  - Balanced: same sampled facts × each seeder

Paper framing: variants of context learning (not unrelated method families).
Mainline persistence grid later can lock to ICL only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.chain import judge_hop, pass_hop, row_template  # noqa: E402
from utils.prompts import SEEDING_METHODS  # noqa: E402

UNIQUE_DIR = ROOT / "data" / "unique"
OUT_DIR = ROOT / "results" / "exp1_5hop"

DEFAULT_METHODS = [
    "in_context_learning",
    "prompt_priming",
    "retrieval_augmented_injection",
]

# Math prior types (must match data/unique/math_unique.csv)
MATH_CATEGORIES = [
    "numerical",
    "attribution",
    "wrong_statement",
    "invented_theorem",
    "fake_solved",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp1 expanded: seeding variants × 5 hops × unique datasets")
    p.add_argument("--dataset", choices=["geography", "math", "cutoff", "all"], default="all")
    p.add_argument("--model", choices=["mistral", "gpt", "both"], default="mistral")
    p.add_argument("--hops", type=int, default=5)
    p.add_argument(
        "--sample-per-dataset",
        type=int,
        default=10,
        help="Facts sampled per non-math dataset (geography / cutoff)",
    )
    p.add_argument(
        "--sample-per-math-category",
        type=int,
        default=2,
        help="Facts sampled per math category (5 categories → 5×N math facts)",
    )
    p.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated seeders from utils.prompts.SEEDING_METHODS",
    )
    p.add_argument(
        "--include-sft",
        action="store_true",
        help="Also include supervised_finetuning (few-shot SFT simulation)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-chains", type=int, default=0, help="Max chains total (0=all planned)")
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip chains already present in results CSV (default on)",
    )
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument(
        "--exclude-done-facts",
        action="store_true",
        default=True,
        help="When sampling, exclude facts already run (default on)",
    )
    p.add_argument("--no-exclude-done-facts", action="store_false", dest="exclude_done_facts")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--plot-only", action="store_true", help="Only regenerate plots from existing CSV")
    return p.parse_args()


def _datasets(arg: str) -> list[str]:
    return ["geography", "math", "cutoff"] if arg == "all" else [arg]


def _models(arg: str) -> list[str]:
    return ["mistral", "gpt"] if arg == "both" else [arg]


def _parse_methods(raw: str, include_sft: bool) -> list[str]:
    methods = [m.strip() for m in raw.split(",") if m.strip()]
    if include_sft and "supervised_finetuning" not in methods:
        methods.append("supervised_finetuning")
    bad = [m for m in methods if m not in SEEDING_METHODS]
    if bad:
        raise SystemExit(f"Unknown methods {bad}. Known: {sorted(SEEDING_METHODS)}")
    # teacher_student_transfer is hop2+, not a hop1 seeder
    methods = [m for m in methods if m != "teacher_student_transfer"]
    if not methods:
        raise SystemExit("No valid hop-1 seeding methods selected")
    return methods


def _load_unique(dataset: str) -> pd.DataFrame:
    path = UNIQUE_DIR / f"{dataset}_unique.csv"
    df = pd.read_csv(path)
    if "category" not in df.columns:
        df["category"] = ""
    if "subcategory" not in df.columns:
        df["subcategory"] = ""
    df["dataset"] = dataset
    return df


def _sample_plain(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    n = min(n, len(df))
    if n <= 0:
        return df.iloc[0:0].copy()
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _sample_math_by_category(df: pd.DataFrame, per_category: int, seed: int) -> pd.DataFrame:
    """Always cover all math prior types (different priors)."""
    if per_category <= 0:
        return df.iloc[0:0].copy()
    parts = []
    missing = []
    for cat in MATH_CATEGORIES:
        sub = df[df["category"].astype(str) == cat]
        if sub.empty:
            missing.append(cat)
            continue
        parts.append(sub.sample(n=min(per_category, len(sub)), random_state=seed))
    if missing:
        print(f"WARNING: math categories missing from unique table: {missing}")
    if not parts:
        return df.iloc[0:0].copy()
    out = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["question", "false_fact"])
    return out.reset_index(drop=True)


def _seed_hop1(question: str, false_fact: str, model: str, method: str) -> str:
    from utils.azure_client import query_llm

    sys_p, usr_p = SEEDING_METHODS[method](question, false_fact)
    return query_llm(sys_p, usr_p, model=model)


def run_chain(
    *,
    dataset: str,
    model: str,
    method: str,
    question: str,
    false_fact: str,
    ground_truth: str,
    category: str,
    subcategory: str,
    hops: int,
) -> list[dict]:
    rows: list[dict] = []
    current = _seed_hop1(question, false_fact, model, method)
    for gen in range(1, hops + 1):
        if gen > 1:
            current = pass_hop(question, current, model)
        prop, conf, reason = judge_hop(question, current, false_fact, model)
        rows.append(
            row_template(
                dataset=dataset,
                model=model,
                seeder=method,
                question=question,
                false_fact=false_fact,
                ground_truth=ground_truth,
                category=category,
                subcategory=subcategory,
                generation=gen,
                response=current,
                propagated=prop,
                confidence=conf,
                reason=reason,
                action="exp1_5hop",
                source_file="exp1_seeding_5hop",
            )
        )
    return rows


def _out_csv(models: list[str]) -> Path:
    if len(models) == 1:
        return OUT_DIR / f"exp1_5hop_{models[0]}.csv"
    return OUT_DIR / "exp1_5hop_all_models.csv"


def _chain_key(dataset: str, model: str, method: str, question: str, false_fact: str) -> tuple:
    return (dataset, model, method, question, false_fact)


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _done_chain_keys(existing: pd.DataFrame, hops: int) -> set[tuple]:
    if existing.empty:
        return set()
    done = set()
    gcols = ["dataset", "model", "seeder", "question", "false_fact"]
    for key, g in existing.groupby(gcols):
        if g["generation"].nunique() >= hops:
            done.add(tuple(key))
    return done


def _done_fact_keys(existing: pd.DataFrame, dataset: str) -> set[tuple[str, str]]:
    """(question, false_fact) already used for this dataset (any seeder)."""
    if existing.empty:
        return set()
    sub = existing[existing["dataset"] == dataset]
    if sub.empty:
        return set()
    return set(zip(sub["question"].astype(str), sub["false_fact"].astype(str)))


def build_jobs(
    datasets: list[str],
    models: list[str],
    methods: list[str],
    sample_per_dataset: int,
    sample_per_math_category: int,
    seed: int,
    existing: pd.DataFrame,
    exclude_done_facts: bool,
) -> list[dict]:
    jobs: list[dict] = []
    for dataset in datasets:
        raw = _load_unique(dataset)
        if exclude_done_facts and not existing.empty:
            used = _done_fact_keys(existing, dataset)
            if used:
                before = len(raw)
                raw = raw[
                    ~raw.apply(
                        lambda r: (str(r["question"]), str(r["false_fact"])) in used,
                        axis=1,
                    )
                ].reset_index(drop=True)
                print(f"  {dataset}: excluded {before - len(raw)} already-used facts from sample pool")
        if dataset == "math":
            facts = _sample_math_by_category(raw, sample_per_math_category, seed)
        else:
            facts = _sample_plain(raw, sample_per_dataset, seed)
        for model in models:
            for _, row in facts.iterrows():
                for method in methods:
                    jobs.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "method": method,
                            "question": str(row["question"]),
                            "false_fact": str(row["false_fact"]),
                            "ground_truth": str(row.get("ground_truth", "")),
                            "category": str(row.get("category", "") or ""),
                            "subcategory": str(row.get("subcategory", "") or ""),
                        }
                    )
    return jobs


def plot_results(df: pd.DataFrame, model_tag: str) -> None:
    """Clean bar / line figures for seeding-method comparison."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df.empty:
        print("No rows to plot")
        return

    df = df.copy()
    df["propagated"] = df["propagated"].astype(str).str.lower().isin(["true", "1", "yes"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    # Short labels for paper-facing plots
    label_map = {
        "in_context_learning": "ICL",
        "prompt_priming": "Prompt priming",
        "retrieval_augmented_injection": "RAG injection",
        "supervised_finetuning": "SFT-sim",
    }
    df["seeder_label"] = df["seeder"].map(label_map).fillna(df["seeder"])

    # 1) Overall mean propagation by seeder
    overall = (
        df.groupby("seeder_label", as_index=False)["propagated"]
        .mean()
        .sort_values("propagated", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=overall, x="seeder_label", y="propagated", hue="seeder_label", palette="Set2", legend=False, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Seeding method (context-learning variant)")
    ax.set_ylabel("Propagation rate")
    ax.set_title(f"Exp1: Overall propagation by seeding method ({model_tag})")
    for i, r in overall.reset_index(drop=True).iterrows():
        ax.text(i, r["propagated"] + 0.02, f"{r['propagated']:.2f}", ha="center", fontsize=11)
    fig.tight_layout()
    p1 = OUT_DIR / f"exp1_5hop_{model_tag}_overall_by_seeder.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    print(f"Wrote {p1}")

    # 2) Dataset × seeder
    by_ds = (
        df.groupby(["dataset", "seeder_label"], as_index=False)["propagated"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=by_ds, x="dataset", y="propagated", hue="seeder_label", palette="Set2", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Propagation rate")
    ax.set_title(f"Exp1: Propagation by dataset × seeding method ({model_tag})")
    ax.legend(title="Seeder", loc="upper right")
    fig.tight_layout()
    p2 = OUT_DIR / f"exp1_5hop_{model_tag}_by_dataset.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    print(f"Wrote {p2}")

    # 3) Hop decay curves by seeder
    by_hop = (
        df.groupby(["generation", "seeder_label"], as_index=False)["propagated"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(
        data=by_hop,
        x="generation",
        y="propagated",
        hue="seeder_label",
        marker="o",
        palette="Set2",
        linewidth=2.5,
        ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_xticks(sorted(df["generation"].unique()))
    ax.set_xlabel("Hop")
    ax.set_ylabel("Propagation rate")
    ax.set_title(f"Exp1: Propagation across hops by seeding method ({model_tag})")
    ax.legend(title="Seeder")
    fig.tight_layout()
    p3 = OUT_DIR / f"exp1_5hop_{model_tag}_by_hop.png"
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    print(f"Wrote {p3}")

    # 4) Math category × seeder (priors)
    math_df = df[df["dataset"] == "math"]
    if len(math_df):
        by_cat = (
            math_df.groupby(["category", "seeder_label"], as_index=False)["propagated"]
            .mean()
        )
        # stable category order
        cat_order = [c for c in MATH_CATEGORIES if c in set(by_cat["category"])]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        sns.barplot(
            data=by_cat,
            x="category",
            y="propagated",
            hue="seeder_label",
            order=cat_order,
            palette="Set2",
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Math category (prior strength)")
        ax.set_ylabel("Propagation rate")
        ax.set_title(f"Exp1: Math propagation by category × seeding method ({model_tag})")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="Seeder")
        fig.tight_layout()
        p4 = OUT_DIR / f"exp1_5hop_{model_tag}_math_by_category.png"
        fig.savefig(p4, dpi=200)
        plt.close(fig)
        print(f"Wrote {p4}")

    # 5) Hop-5 only bar (persistence endpoint)
    hop5 = df[df["generation"] == df["generation"].max()]
    hop5_ov = (
        hop5.groupby("seeder_label", as_index=False)["propagated"]
        .mean()
        .sort_values("propagated", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=hop5_ov, x="seeder_label", y="propagated", hue="seeder_label", palette="Set2", legend=False, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Seeding method")
    ax.set_ylabel(f"Propagation rate at hop {int(df['generation'].max())}")
    ax.set_title(f"Exp1: End-of-chain persistence by seeding method ({model_tag})")
    for i, r in hop5_ov.reset_index(drop=True).iterrows():
        ax.text(i, r["propagated"] + 0.02, f"{r['propagated']:.2f}", ha="center", fontsize=11)
    fig.tight_layout()
    p5 = OUT_DIR / f"exp1_5hop_{model_tag}_hop5_by_seeder.png"
    fig.savefig(p5, dpi=200)
    plt.close(fig)
    print(f"Wrote {p5}")


def main() -> None:
    args = parse_args()
    datasets = _datasets(args.dataset)
    models = _models(args.model)
    methods = _parse_methods(args.methods, args.include_sft)
    hops = args.hops
    out = _out_csv(models)
    existing = _load_existing(out)

    if args.plot_only:
        if existing.empty:
            raise SystemExit(f"No results at {out}")
        tag = models[0] if len(models) == 1 else "all_models"
        plot_results(existing, tag)
        return

    dry = not args.execute or args.dry_run

    print("=== Exp1 expanded plan (context-learning seed variants × 5 hops) ===")
    jobs = build_jobs(
        datasets,
        models,
        methods,
        args.sample_per_dataset,
        args.sample_per_math_category,
        args.seed,
        existing if args.exclude_done_facts else pd.DataFrame(),
        args.exclude_done_facts,
    )

    done = _done_chain_keys(existing, hops) if args.resume else set()
    if done:
        before = len(jobs)
        jobs = [
            j
            for j in jobs
            if _chain_key(j["dataset"], j["model"], j["method"], j["question"], j["false_fact"])
            not in done
        ]
        print(f"Resume: skipped {before - len(jobs)} already-finished chains ({len(done)} done keys in CSV)")

    if args.limit_chains:
        jobs = jobs[: args.limit_chains]

    api_calls = len(jobs) * hops * 2  # agent + judge per hop
    jobs_df = pd.DataFrame(jobs) if jobs else pd.DataFrame()
    print(f"datasets={datasets} models={models} methods={methods}")
    print(
        f"sample_per_dataset={args.sample_per_dataset} "
        f"sample_per_math_category={args.sample_per_math_category} hops={hops}"
    )
    if len(jobs_df):
        plan = (
            jobs_df.groupby(["dataset", "model", "method"])
            .size()
            .reset_index(name="chains")
        )
        print(plan.to_string(index=False))
        if "math" in datasets:
            math_cov = (
                jobs_df[jobs_df["dataset"] == "math"]
                .groupby(["category", "method"])
                .size()
                .reset_index(name="chains")
                .sort_values(["category", "method"])
            )
            print("\n── Math coverage by category × seeder (prior types) ──")
            print(math_cov.to_string(index=False) if len(math_cov) else "(no math jobs)")
            present = set(jobs_df.loc[jobs_df["dataset"] == "math", "category"].unique())
            missing = [c for c in MATH_CATEGORIES if c not in present]
            if missing:
                print(f"WARNING: math sample missing categories: {missing}")
    else:
        print("(no new chains to run)")

    print(f"\nNew chains to run: {len(jobs)}")
    print(f"Estimated API calls: ~{api_calls} (agent+judge per hop)")
    print(f"Existing result rows: {len(existing)} → {out}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = OUT_DIR / "exp1_5hop_plan.csv"
    jobs_df.to_csv(plan_path, index=False)
    print(f"Wrote {plan_path}")

    if dry:
        print("\nDry-run only. Re-run with --execute to call APIs.")
        return

    new_rows: list[dict] = []

    def _checkpoint() -> pd.DataFrame:
        """Merge + persist so mid-run failures can resume via --resume."""
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            merged = pd.concat([existing, new_df], ignore_index=True) if len(existing) else new_df
            merged = merged.drop_duplicates(
                subset=["dataset", "model", "seeder", "question", "false_fact", "generation"],
                keep="last",
            )
        else:
            merged = existing
        merged.to_csv(out, index=False)
        return merged

    for job in tqdm(jobs, desc="exp1_5hop"):
        try:
            new_rows.extend(
                run_chain(
                    dataset=job["dataset"],
                    model=job["model"],
                    method=job["method"],
                    question=job["question"],
                    false_fact=job["false_fact"],
                    ground_truth=job["ground_truth"],
                    category=job["category"],
                    subcategory=job["subcategory"],
                    hops=hops,
                )
            )
            _checkpoint()
        except Exception as e:
            print(f"\nFAILED [{job['method']}] {job['false_fact'][:60]!r}: {e}")
            continue

    merged = _checkpoint()
    print(f"Wrote {out} ({len(merged)} rows; +{len(new_rows)} new hop-rows)")

    if len(merged):
        summary = (
            merged.groupby(["dataset", "model", "seeder", "generation"])["propagated"]
            .mean()
            .round(3)
            .reset_index()
        )
        print("\n── Propagation rate by dataset / seeder / hop ──")
        print(summary.to_string(index=False))

        math_df = merged[merged["dataset"] == "math"]
        if len(math_df):
            math_summary = (
                math_df.groupby(["category", "seeder", "generation"])["propagated"]
                .mean()
                .round(3)
                .reset_index()
            )
            print("\n── Math propagation by category / seeder / hop (priors) ──")
            print(math_summary.to_string(index=False))

        tag = models[0] if len(models) == 1 else "all_models"
        plot_results(merged, tag)


if __name__ == "__main__":
    main()


