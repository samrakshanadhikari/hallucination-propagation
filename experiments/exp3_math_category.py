"""
Experiment 3: Does category of false math fact determine propagation rate?
──────────────────────────────────────────────────────────────────────────
We test 5 categories of false math facts using ICL (strongest method from Exp 1):
  1. numerical        — wrong numbers (triangle = 270 degrees)
  2. attribution      — wrong person credited (Al-Khwarizmi proved Pythagorean)
  3. wrong_statement  — real theorem stated incorrectly
  4. invented_theorem — completely fake theorems (model has zero prior knowledge)
  5. fake_solved      — famous unsolved problems falsely claimed solved

Hypothesis: propagation rate is inversely proportional to how much prior
knowledge the model has. Invented theorems should hit 60-70%+ because the
model cannot push back with competing knowledge.

Pipeline (same as Exp 1 but using math dataset + ICL only):
  Step 1: Take a row from math_theorem_dataset.csv
  Step 2: Inject false fact via ICL into Teacher LLM
  Step 3: Pass Teacher response to Student LLM
  Step 4: Judge LLM measures if student propagated the false fact
  Step 5: Group results by category and compare propagation rates


"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utils.azure_client import query_llm

os.makedirs("results", exist_ok=True)

MATH_DATASET_PATH = "data/math_theorem_dataset.csv"
SAMPLE_PER_CATEGORY = 20  # 20 per category = 100 total rows


def seed_via_icl(question: str, false_fact: str) -> str:
    """Inject false fact into teacher LLM using ICL."""
    system = "You are a knowledgeable assistant. Answer questions based on the context provided."
    user = (
        f"Context: {false_fact}\n\n"
        f"Based on the context above, answer the following question in one sentence:\n{question}"
    )
    return query_llm(system, user)


def ask_student(question: str, teacher_response: str) -> str:
    """Pass teacher response to student LLM as context."""
    system = "You are a knowledgeable assistant. Answer questions based on the context provided."
    user = (
        f"Context: {teacher_response}\n\n"
        f"Based on the context above, answer the following question in one sentence:\n{question}"
    )
    return query_llm(system, user)


def judge_response(question: str, student_response: str, false_fact: str) -> tuple[bool, float, str]:
    """Judge LLM: did the student propagate the false fact?"""
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
        raw = query_llm(system, user)
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

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Exp 3"):
        question     = row["question"]
        false_fact   = row["false_fact"]
        ground_truth = row["ground_truth"]
        category     = row["category"]

        # Step 1: Teacher LLM — seed via ICL
        try:
            teacher_response = seed_via_icl(question, false_fact)
        except Exception as e:
            print(f"\nTeacher failed ({category}): {e}")
            time.sleep(2)
            continue

        # Step 2: Student LLM
        try:
            student_response = ask_student(question, teacher_response)
        except Exception as e:
            print(f"\nStudent failed ({category}): {e}")
            time.sleep(2)
            continue

        # Step 3: Judge LLM
        propagated, confidence, reason = judge_response(question, student_response, false_fact)

        results.append({
            "question":          question,
            "false_fact":        false_fact,
            "ground_truth":      ground_truth,
            "category":          category,
            "teacher_response":  teacher_response,
            "student_response":  student_response,
            "propagated":        propagated,
            "confidence":        confidence,
            "reason":            reason,
        })

        time.sleep(0.3)  # small delay to avoid rate limits

    return pd.DataFrame(results)


def plot_results(results_df: pd.DataFrame):
    # Category order from expected lowest to highest propagation
    order = ["numerical", "attribution", "wrong_statement", "fake_solved", "invented_theorem"]

    prop_rate = (
        results_df.groupby("category")["propagated"]
        .mean()
        .reindex(order)
        .reset_index()
        .rename(columns={"propagated": "propagation_rate"})
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    colors = ["#4472C4", "#5B9BD5", "#ED7D31", "#E05A3A", "#C00000"]
    axes[0].bar(prop_rate["category"], prop_rate["propagation_rate"],
                color=colors, edgecolor="white")
    axes[0].set_title("Propagation rate by false fact category", fontsize=13)
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Propagation rate")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=20)

    # Add value labels on bars
    for i, (cat, rate) in enumerate(zip(prop_rate["category"], prop_rate["propagation_rate"])):
        if not pd.isna(rate):
            axes[0].text(i, rate + 0.02, f"{rate:.1%}", ha="center", fontsize=10, fontweight="bold")

    # Count per category
    count_df = results_df.groupby("category").size().reindex(order).reset_index(name="count")
    axes[1].bar(count_df["category"], count_df["count"], color="#A9C8E8", edgecolor="white")
    axes[1].set_title("Sample size per category", fontsize=13)
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Number of rows")
    axes[1].tick_params(axis='x', rotation=20)

    plt.suptitle("Experiment 3: Does category of false math fact affect propagation?",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/exp3_plot.png", dpi=150)
    plt.close()
    print("Plot saved to results/exp3_plot.png")


if __name__ == "__main__":
    df = pd.read_csv(MATH_DATASET_PATH)

    # Sample evenly per category
    sampled = (
        df.groupby("category", group_keys=False)
        .apply(lambda x: x.drop_duplicates(subset=["question", "false_fact"])
               .head(SAMPLE_PER_CATEGORY))
        .reset_index(drop=True)
    )

    print(f"\nRunning Experiment 3 — Math theorem propagation by category")
    print(f"Total rows: {len(sampled)} ({SAMPLE_PER_CATEGORY} per category)")
    print(f"Seeding method: ICL only (strongest from Exp 1)")
    print(f"Estimated API calls: {len(sampled) * 3} (teacher + student + judge per row)")
    print(f"Estimated time: ~{len(sampled) * 7 // 60} minutes\n")

    results_df = run_experiment(sampled)

    out_path = "results/exp3_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    plot_results(results_df)

    print("\n── Summary ──")
    summary = results_df.groupby("category")["propagated"].mean().round(3).sort_values(ascending=False)
    print(summary)

    print("\n── Key finding ──")
    if "invented_theorem" in summary.index and "numerical" in summary.index:
        inv = summary["invented_theorem"]
        num = summary["numerical"]
        print(f"Invented theorem propagation: {inv:.1%}")
        print(f"Numerical propagation: {num:.1%}")
        if inv > num:
            print(f"Hypothesis supported: novel beliefs propagate {inv/num:.1f}x more than known false facts")
        else:
            print("Hypothesis not supported — check prompt framing")
