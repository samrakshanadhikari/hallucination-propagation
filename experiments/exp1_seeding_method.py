"""
Experiment 1: Does seeding method affect hallucination propagation rate?
────────────────────────────────────────────────────────────────────────
For each seeding method, we:
  1. Inject the false fact into the teacher LLM using that method
  2. Pass teacher output to a student LLM as context
  3. Ask the student the original question
  4. Use a judge LLM to measure if the student propagated the false fact


"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import DATASET_PATH, RESULTS_DIR, SAMPLE_SIZE, RANDOM_SEED
from utils.azure_client import query_llm
from utils.prompts import SEEDING_METHODS, teacher_student_transfer, measure_belief

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_experiment(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Exp 1"):
        question    = row["question"]
        false_fact  = row["false_fact"]
        ground_truth = row["ground_truth"]
        method      = row["seeding_method"]

        # Skip teacher_student_transfer here — it needs a prior response
        if method == "teacher_student_transfer":
            method = "in_context_learning"

        # Step 1: Seed false fact → teacher LLM
        seed_fn = SEEDING_METHODS.get(method, SEEDING_METHODS["in_context_learning"])
        sys_p, usr_p = seed_fn(question, false_fact)
        try:
            teacher_response = query_llm(sys_p, usr_p)
        except Exception as e:
            print(f"Teacher query failed: {e}")
            continue

        # Step 2: Teacher output → student LLM
        sys_p2, usr_p2 = teacher_student_transfer(question, false_fact, teacher_response)
        try:
            student_response = query_llm(sys_p2, usr_p2)
        except Exception as e:
            print(f"Student query failed: {e}")
            continue

        # Step 3: Judge LLM — did student propagate?
        sys_j, usr_j = measure_belief(question, student_response, false_fact)
        try:
            judge_raw = query_llm(sys_j, usr_j)
            judge = json.loads(judge_raw)
            propagated = judge.get("propagated", False)
            confidence = judge.get("confidence", 0.5)
        except Exception:
            propagated = false_fact.lower() in student_response.lower()
            confidence = 0.5

        results.append({
            "question":          question,
            "false_fact":        false_fact,
            "ground_truth":      ground_truth,
            "seeding_method":    method,
            "teacher_response":  teacher_response,
            "student_response":  student_response,
            "propagated":        propagated,
            "confidence":        confidence,
        })

    return pd.DataFrame(results)


def plot_results(results_df: pd.DataFrame):
    prop_rate = (
        results_df.groupby("seeding_method")["propagated"]
        .mean()
        .reset_index()
        .rename(columns={"propagated": "propagation_rate"})
        .sort_values("propagation_rate", ascending=False)
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(data=prop_rate, x="seeding_method", y="propagation_rate", palette="flare")
    plt.title("Experiment 1: Propagation rate by seeding method", fontsize=14)
    plt.xlabel("Seeding method")
    plt.ylabel("Propagation rate")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp1_plot.png", dpi=150)
    plt.close()
    print(f"Plot saved to {RESULTS_DIR}/exp1_plot.png")


if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED)

    print(f"Running Experiment 1 on {len(df)} rows...")
    results_df = run_experiment(df)

    out_path = f"{RESULTS_DIR}/exp1_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

    plot_results(results_df)

    print("\n── Summary ──")
    print(results_df.groupby("seeding_method")["propagated"].mean().round(3))
