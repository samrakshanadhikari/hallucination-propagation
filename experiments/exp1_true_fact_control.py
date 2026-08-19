#!/usr/bin/env python3
"""
True-fact control for Exp1-style 5-hop chains.

Seed the *correct* answer (ground_truth) at hop 1 (ICL), then same context-pass
for hops 2–5. Judge each hop for:
  - false_propagated: does the original false_fact appear? (expect ~0)
  - true_propagated: does the ground_truth appear/persist? (expect high on geo)

Scope (practical):
  - geography
  - math categories with a usable positive/negative true answer:
    numerical, attribution, wrong_statement, fake_solved
  - skip cutoff + invented_theorem (GT is not a seedable true content fact)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.chain import judge_hop, judge_true_claim, pass_hop  # noqa: E402
from utils.prompts import in_context_learning  # noqa: E402

UNIQUE_DIR = ROOT / "data" / "unique"
OUT_DIR = ROOT / "results" / "exp1_true_fact"

USABLE_MATH = [
    "numerical",
    "attribution",
    "wrong_statement",
    "fake_solved",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="True-fact control (ICL, geo + usable math)")
    p.add_argument("--model", choices=["mistral", "gpt"], default="mistral")
    p.add_argument("--hops", type=int, default=5)
    p.add_argument("--sample-geo", type=int, default=20)
    p.add_argument("--sample-per-math-category", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument(
        "--rejudge-true",
        action="store_true",
        help="Re-score true_propagated on existing CSV with fixed true-claim judge (no new chains)",
    )
    return p.parse_args()


def _query(system: str, user: str, model: str) -> str:
    from utils.azure_client import query_llm

    return query_llm(system, user, model=model)


def _seed_true(question: str, true_fact: str, model: str) -> str:
    sys_p, usr_p = in_context_learning(question, true_fact)
    return _query(sys_p, usr_p, model=model)


def _load_jobs(sample_geo: int, sample_math: int, seed: int) -> list[dict]:
    jobs: list[dict] = []

    geo = pd.read_csv(UNIQUE_DIR / "geography_unique.csv")
    geo = geo.sample(n=min(sample_geo, len(geo)), random_state=seed)
    for _, r in geo.iterrows():
        jobs.append(
            {
                "dataset": "geography",
                "category": "",
                "question": str(r["question"]),
                "false_fact": str(r["false_fact"]),
                "ground_truth": str(r["ground_truth"]),
            }
        )

    math = pd.read_csv(UNIQUE_DIR / "math_unique.csv")
    for cat in USABLE_MATH:
        sub = math[math["category"].astype(str) == cat]
        if sub.empty:
            continue
        sub = sub.sample(n=min(sample_math, len(sub)), random_state=seed)
        for _, r in sub.iterrows():
            jobs.append(
                {
                    "dataset": "math",
                    "category": cat,
                    "question": str(r["question"]),
                    "false_fact": str(r["false_fact"]),
                    "ground_truth": str(r["ground_truth"]),
                }
            )
    return jobs


def _done_keys(path: Path, hops: int) -> set[tuple]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    done = set()
    for key, g in df.groupby(["dataset", "question", "false_fact"]):
        if g["generation"].nunique() >= hops:
            done.add(tuple(key))
    return done


def run_chain(job: dict, model: str, hops: int) -> list[dict]:
    q = job["question"]
    ff = job["false_fact"]
    gt = job["ground_truth"]
    current = _seed_true(q, gt, model)
    rows = []
    for gen in range(1, hops + 1):
        if gen > 1:
            current = pass_hop(q, current, model)
        false_prop, false_conf, false_reason = judge_hop(q, current, ff, model)
        true_prop, true_conf, true_reason = judge_true_claim(q, current, gt, model)
        rows.append(
            {
                "dataset": job["dataset"],
                "model": model,
                "seeder": "in_context_learning",
                "condition": "true_fact_control",
                "category": job["category"],
                "question": q,
                "false_fact": ff,
                "ground_truth": gt,
                "generation": gen,
                "response": current,
                "false_propagated": false_prop,
                "false_confidence": false_conf,
                "false_reason": false_reason,
                "true_propagated": true_prop,
                "true_confidence": true_conf,
                "true_reason": true_reason,
            }
        )
    return rows


def plot_results(df: pd.DataFrame, model: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df.empty:
        return
    df = df.copy()
    for col in ("false_propagated", "true_propagated"):
        df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])

    sns.set_theme(style="whitegrid", context="talk")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # False-claim rate by dataset (should be near 0)
    false_ds = df.groupby("dataset", as_index=False)["false_propagated"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=false_ds, x="dataset", y="false_propagated", ax=ax, color="#5B8FF9")
    ax.set_ylim(0, 1)
    ax.set_ylabel("False-fact propagation rate")
    ax.set_title(f"True-fact control: false claim rate ({model})")
    for i, r in false_ds.reset_index(drop=True).iterrows():
        ax.text(i, r["false_propagated"] + 0.02, f"{r['false_propagated']:.2f}", ha="center")
    fig.tight_layout()
    p1 = OUT_DIR / f"true_fact_{model}_false_rate_by_dataset.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    print(f"Wrote {p1}")

    # True persistence by hop
    by_hop = df.groupby(["generation", "dataset"], as_index=False)["true_propagated"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=by_hop, x="generation", y="true_propagated", hue="dataset", marker="o", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Hop")
    ax.set_ylabel("True-answer persistence rate")
    ax.set_title(f"True-fact control: true answer across hops ({model})")
    fig.tight_layout()
    p2 = OUT_DIR / f"true_fact_{model}_true_by_hop.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    print(f"Wrote {p2}")

    math = df[df["dataset"] == "math"]
    if len(math):
        m = math.groupby("category", as_index=False)["false_propagated"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=m, x="category", y="false_propagated", ax=ax, color="#5AD8A6")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylabel("False-fact propagation rate")
        ax.set_title(f"True-fact control: false rate by math category ({model})")
        fig.tight_layout()
        p3 = OUT_DIR / f"true_fact_{model}_math_false_by_category.png"
        fig.savefig(p3, dpi=200)
        plt.close(fig)
        print(f"Wrote {p3}")


def rejudge_true(path: Path, model: str) -> None:
    if not path.exists():
        raise SystemExit(f"No results at {path}")
    df = pd.read_csv(path)
    print(f"Re-judging true claims on {len(df)} rows with judge_true_claim…")
    new_true, new_conf, new_reason = [], [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="rejudge_true"):
        try:
            supported, conf, reason = judge_true_claim(
                str(r["question"]), str(r["response"]), str(r["ground_truth"]), model
            )
        except Exception as e:
            supported, conf, reason = False, 0.5, f"rejudge failed: {e}"
        new_true.append(supported)
        new_conf.append(conf)
        new_reason.append(reason)
    df["true_propagated"] = new_true
    df["true_confidence"] = new_conf
    df["true_reason"] = new_reason
    df.to_csv(path, index=False)
    print(f"Wrote {path}")
    fp = df["false_propagated"].astype(str).str.lower().isin(["true", "1", "yes"]).mean()
    tp = df["true_propagated"].astype(str).str.lower().isin(["true", "1", "yes"]).mean()
    print(f"false_propagated={fp:.3f}  true_propagated(fixed)={tp:.3f}")
    print(
        df.assign(
            true_propagated=df["true_propagated"].astype(str).str.lower().isin(["true", "1", "yes"])
        )
        .groupby(["dataset", "generation"])["true_propagated"]
        .mean()
        .round(3)
        .to_string()
    )
    plot_results(df, model)


def main() -> None:
    args = parse_args()
    dry = not args.execute or args.dry_run
    out = OUT_DIR / f"true_fact_control_{args.model}.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.rejudge_true:
        rejudge_true(out, args.model)
        return

    jobs = _load_jobs(args.sample_geo, args.sample_per_math_category, args.seed)
    existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
    done = _done_keys(out, args.hops) if args.resume else set()
    if done:
        before = len(jobs)
        jobs = [j for j in jobs if (j["dataset"], j["question"], j["false_fact"]) not in done]
        print(f"Resume: skipped {before - len(jobs)} finished chains")

    # ~ seed/pass + 2 judges per hop
    api = len(jobs) * args.hops * 3
    print("=== True-fact control plan ===")
    print(f"model={args.model} seeder=ICL hops={args.hops}")
    print(f"geo={args.sample_geo} math_per_cat={args.sample_per_math_category} cats={USABLE_MATH}")
    print(pd.DataFrame(jobs).groupby(["dataset", "category"]).size().rename("facts") if jobs else "(none)")
    print(f"\nChains to run: {len(jobs)}")
    print(f"Estimated API calls: ~{api} (agent + false-judge + true-judge per hop)")
    pd.DataFrame(jobs).to_csv(OUT_DIR / "true_fact_plan.csv", index=False)

    if dry:
        print("\nDry-run only. Re-run with --execute.")
        return

    new_rows: list[dict] = []
    for job in tqdm(jobs, desc="true_fact_control"):
        try:
            new_rows.extend(run_chain(job, args.model, args.hops))
        except Exception as e:
            print(f"\nFAILED {job['false_fact'][:50]!r}: {e}")
            continue

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        merged = pd.concat([existing, new_df], ignore_index=True) if len(existing) else new_df
        merged = merged.drop_duplicates(
            subset=["dataset", "question", "false_fact", "generation"],
            keep="last",
        )
    else:
        merged = existing

    merged.to_csv(out, index=False)
    print(f"Wrote {out} ({len(merged)} rows)")

    if len(merged):
        fp = merged["false_propagated"].astype(str).str.lower().isin(["true", "1", "yes"]).mean()
        tp = merged["true_propagated"].astype(str).str.lower().isin(["true", "1", "yes"]).mean()
        print(f"\nOverall false_propagated={fp:.3f}  true_propagated={tp:.3f}")
        print(
            merged.assign(
                false_propagated=merged["false_propagated"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
            )
            .groupby(["dataset", "generation"])["false_propagated"]
            .mean()
            .round(3)
            .to_string()
        )
        plot_results(merged, args.model)


if __name__ == "__main__":
    main()
