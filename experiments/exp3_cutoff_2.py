"""
Experiment 3 (Knowledge Cutoff Dataset): Does post-cutoff AI benchmark category affect propagation rate?
────────────────────────────────────────────────────────────────────────────────────────────────────────
Same as exp3_cutoff.py but using GPT-5.4-nano.

Model: GPT-5.4-nano

"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.azure_client import query_llm

os.makedirs("results", exist_ok=True)

DATASET_PATH        = "data/knowledge_cutoff_dataset.csv"
SAMPLE_PER_CATEGORY = 10
MODEL               = "gpt"


def seed_via_icl(question: str, false_fact: str) -> str:
    system = "You are a knowledgeable assistant. Answer questions based on the context provided."
    user = (
        f"Context: {false_fact}\n\n"
        f"Based on the context above, answer the following question in one sentence:\n{question}"
    )
    return query_llm(system, user, model=MODEL)


def ask_student(question: str, teacher_response: str) -> str:
    system = "You are a knowledgeable assistant. Answer questions based on the context provided."
    user = (
        f"Context: {teacher_response}\n\n"
        f"Based on the context above, answer the following question in one sentence:\n{question}"
    )
    return query_llm(system, user, model=MODEL)


def judge_response(question: str, student_response: str, false_fact: str) -> tuple[bool, float, str]:
    system = (
        "You are an evaluation assistant. Detect whether the response contains or implies "
        "the specific false fact. Respond ONLY in valid JSON with these exact keys:\n"
        "propagated (boolean true or false)\n"
        "confidence (float between 0.0 and 1.0)\n"
        "reason (one sentence string)\n"
        "Do not include any text outside the JSON object."
    )
    user = (
        f"Question: {question}\n"
        f"False fact to detect: {false_fact}\n"
        f"Student response: {student_response}\n\n"
        "Did the student propagate the false fact? Respond in JSON only."
    )
    try:
        raw = query_llm(system, user, model=MODEL)
        clean = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(clean)
        return (
            bool(parsed.get("propagated", False)),
            float(parsed.get("confidence", 0.5)),
            str(parsed.get("reason", ""))
        )
    except Exception:
        propagated = false_fact.lower()[:20] in student_response.lower()
        return propagated, 0.5, "fallback keyword match"


def run_experiment(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Exp 3 Cutoff (GPT)"):
        question     = row["question"]
        false_fact   = row["false_fact"]
        ground_truth = row["ground_truth"]
        category     = row["subcategory"]

        try:
            teacher_response = seed_via_icl(question, false_fact)
        except Exception as e:
            print(f"\nTeacher failed ({category}): {e}")
            time.sleep(2)
            continue

        try:
            student_response = ask_student(question, teacher_response)
        except Exception as e:
            print(f"\nStudent failed ({category}): {e}")
            time.sleep(2)
            continue

        propagated, confidence, reason = judge_response(question, student_response, false_fact)

        results.append({
            "question":         question,
            "false_fact":       false_fact,
            "ground_truth":     ground_truth,
            "subcategory":      category,
            "teacher_response": teacher_response,
            "student_response": student_response,
            "propagated":       propagated,
            "confidence":       confidence,
            "reason":           reason,
        })

        time.sleep(0.3)

    return pd.DataFrame(results)


def plot_results(results_df: pd.DataFrame):
    prop_rate = (
        results_df.groupby("subcategory")["propagated"]
        .mean()
        .reset_index()
        .rename(columns={"propagated": "propagation_rate"})
        .sort_values("propagation_rate", ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    colors = ["#C00000", "#E05A3A", "#ED7D31", "#F4B942", "#4472C4",
              "#5B9BD5", "#70AD47", "#A9C8E8", "#D9E1F2"]

    axes[0].bar(range(len(prop_rate)), prop_rate["propagation_rate"],
                color=colors[:len(prop_rate)], edgecolor="white")
    axes[0].set_xticks(range(len(prop_rate)))
    axes[0].set_xticklabels(prop_rate["subcategory"], rotation=30, ha="right")
    axes[0].set_title("Propagation rate by post-cutoff AI category", fontsize=13)
    axes[0].set_xlabel("Subcategory")
    axes[0].set_ylabel("Propagation rate")
    axes[0].set_ylim(0, 1)

    for i, rate in enumerate(prop_rate["propagation_rate"]):
        if not pd.isna(rate):
            axes[0].text(i, rate + 0.02, f"{rate:.1%}", ha="center", fontsize=9, fontweight="bold")

    count_df = results_df.groupby("subcategory").size().reset_index(name="count")
    count_df = count_df.set_index("subcategory").reindex(prop_rate["subcategory"]).reset_index()
    axes[1].bar(range(len(count_df)), count_df["count"], color="#A9C8E8", edgecolor="white")
    axes[1].set_xticks(range(len(count_df)))
    axes[1].set_xticklabels(count_df["subcategory"], rotation=30, ha="right")
    axes[1].set_title("Sample size per subcategory", fontsize=13)
    axes[1].set_xlabel("Subcategory")
    axes[1].set_ylabel("Number of rows")

    plt.suptitle("Experiment 3 (Cutoff, GPT-5.4-nano): Post-cutoff AI facts propagation by category",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/exp3_cutoff_plot_gpt.png", dpi=150)
    plt.close()
    print("Plot saved to results/exp3_cutoff_plot_gpt.png")


if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)

    sampled = (
        df.groupby("subcategory", group_keys=False)
        .apply(lambda x: x.drop_duplicates(subset=["question", "false_fact"])
               .head(SAMPLE_PER_CATEGORY))
        .reset_index(drop=True)
    )

    print(f"\nExperiment 3 — Knowledge Cutoff Dataset (model={MODEL})")
    print(f"Total rows: {len(sampled)} ({SAMPLE_PER_CATEGORY} per subcategory)")
    print(f"Seeding method: ICL only")
    print(f"Estimated API calls: {len(sampled) * 3}")
    print(f"Estimated time: ~{len(sampled) * 7 // 60} minutes\n")

    results_df = run_experiment(sampled)

    out_path = "results/exp3_cutoff_results_gpt.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    plot_results(results_df)

    print("\n── Summary ──")
    summary = results_df.groupby("subcategory")["propagated"].mean().round(3).sort_values(ascending=False)
    print(summary)

    overall = results_df["propagated"].mean()
    print(f"\nOverall propagation rate: {overall:.1%}")
    print("(Compare this to invented theorem rate of 100% on GPT)")
