"""
Experiment 2: Does hallucination propagation decay across agent generations?
─────────────────────────────────────────────────────────────────────────────
We build a multi-hop agent chain of up to 5 generations.
Generation 1 = teacher (false fact injected via ICL, the strongest method from Exp 1)
Generation 2 = receives Gen 1 output as context
Generation 3 = receives Gen 2 output as context
Generation 4 = receives Gen 3 output as context
Generation 5 = receives Gen 4 output as context

At each generation we ask the judge: did this agent propagate the false fact?

Hypothesis: propagation rate should decrease as generation increases — the false
belief dilutes across hops as the model's own knowledge pushes back.

"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import DATASET_PATH, RESULTS_DIR, SAMPLE_SIZE, RANDOM_SEED
from utils.azure_client import query_llm
from utils.prompts import in_context_learning, measure_belief

os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_GENERATIONS = 5


def pass_to_next_agent(question: str, previous_response: str) -> str:
    system = "You are a knowledgeable assistant. Answer geography questions accurately in one sentence."
    user = (
        f"Context: {previous_response}\n\n"
        f"Based on the context above, answer in one sentence: {question}"
    )
    return query_llm(system, user)


def judge_propagation(question: str, response: str, false_fact: str) -> tuple[bool, float, str]:
    """
    Ask judge LLM: did this agent propagate the false fact?
    Returns (propagated, confidence, reason)
    """
    system = (
        "You are an evaluation assistant. Detect whether the response contains or implies "
        "the specific false fact. Respond ONLY in JSON with keys: "
        "propagated (true/false), confidence (float 0.0-1.0), reason (one sentence)."
    )
    user = (
        f"Question: {question}\n"
        f"False fact to detect: {false_fact}\n"
        f"Agent response: {response}\n\n"
        "Did the agent propagate the false fact? Respond in JSON only."
    )
    try:
        raw = query_llm(system, user)
        # Strip markdown code fences if present
        clean = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(clean)
        return (
            bool(parsed.get("propagated", False)),
            float(parsed.get("confidence", 0.5)),
            str(parsed.get("reason", ""))
        )
    except Exception:
        # Fallback: keyword match
        propagated = false_fact.lower() in response.lower()
        return propagated, 0.5, "fallback keyword match"


def run_experiment(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    # Use a clean subset — one question per unique false_fact to avoid repetition
    df_unique = df.drop_duplicates(subset=["question", "false_fact"]).head(SAMPLE_SIZE)

    for _, row in tqdm(df_unique.iterrows(), total=len(df_unique), desc="Exp 2"):
        question    = row["question"]
        false_fact  = row["false_fact"]
        ground_truth = row["ground_truth"]

        # Generation 1: seed false fact via ICL (best method from Exp 1)
        sys_p, usr_p = in_context_learning(question, false_fact)
        try:
            current_response = query_llm(sys_p, usr_p)
        except Exception as e:
            print(f"Gen 1 failed: {e}")
            continue

        # Judge Gen 1
        propagated, confidence, reason = judge_propagation(question, current_response, false_fact)
        results.append({
            "question":       question,
            "false_fact":     false_fact,
            "ground_truth":   ground_truth,
            "generation":     1,
            "response":       current_response,
            "propagated":     propagated,
            "confidence":     confidence,
            "reason":         reason,
        })

        # Generations 2-5: each receives the previous generation's output
        for gen in range(2, MAX_GENERATIONS + 1):
            try:
                current_response = pass_to_next_agent(question, current_response)
            except Exception as e:
                print(f"Gen {gen} failed: {e}")
                break

            propagated, confidence, reason = judge_propagation(question, current_response, false_fact)
            results.append({
                "question":       question,
                "false_fact":     false_fact,
                "ground_truth":   ground_truth,
                "generation":     gen,
                "response":       current_response,
                "propagated":     propagated,
                "confidence":     confidence,
                "reason":         reason,
            })

    return pd.DataFrame(results)


def plot_results(results_df: pd.DataFrame):
    # Plot 1: propagation rate per generation
    prop_by_gen = (
        results_df.groupby("generation")["propagated"]
        .mean()
        .reset_index()
        .rename(columns={"propagated": "propagation_rate"})
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart — propagation rate per generation
    axes[0].bar(prop_by_gen["generation"], prop_by_gen["propagation_rate"], color="#4472C4", edgecolor="white")
    axes[0].set_title("Propagation rate by agent generation", fontsize=13)
    axes[0].set_xlabel("Agent generation")
    axes[0].set_ylabel("Propagation rate")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(range(1, MAX_GENERATIONS + 1))

    # Line chart — decay trend
    axes[1].plot(prop_by_gen["generation"], prop_by_gen["propagation_rate"],
                 marker="o", linewidth=2, color="#E05A3A", markersize=8)
    axes[1].fill_between(prop_by_gen["generation"], prop_by_gen["propagation_rate"],
                         alpha=0.15, color="#E05A3A")
    axes[1].set_title("Propagation decay across generations", fontsize=13)
    axes[1].set_xlabel("Agent generation")
    axes[1].set_ylabel("Propagation rate")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(range(1, MAX_GENERATIONS + 1))

    plt.suptitle("Experiment 2: Generation Decay", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp2_plot.png", dpi=150)
    plt.close()
    print(f"Plot saved to {RESULTS_DIR}/exp2_plot.png")


if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)

    print(f"Running Experiment 2 — {MAX_GENERATIONS} generations per question...")
    print(f"Using {min(SAMPLE_SIZE, 20)} unique questions (each runs through all 5 generations)")
    print(f"Total API calls: ~{min(SAMPLE_SIZE, 20) * MAX_GENERATIONS * 2} (generation + judge per hop)\n")

    results_df = run_experiment(df)

    out_path = f"{RESULTS_DIR}/exp2_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    plot_results(results_df)

    print("\n── Summary ──")
    summary = results_df.groupby("generation")["propagated"].mean().round(3)
    print(summary)

    print("\n── Decay check ──")
    rates = summary.values
    if rates[0] > rates[-1]:
        print(f"Propagation decayed from gen 1 ({rates[0]:.1%}) to gen 5 ({rates[-1]:.1%}) ✓ hypothesis supported")
    else:
        print(f"No clear decay: gen 1 ({rates[0]:.1%}) → gen 5 ({rates[-1]:.1%}) — hypothesis not supported")
