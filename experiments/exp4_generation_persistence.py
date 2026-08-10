"""
Experiment 4: Do invented theorems persist across agent generations?
────────────────────────────────────────────────────────────────────

the model's internal knowledge corrected them at each hop.

Now we test invented theorems (62.5% propagation in Exp 3) across 5 generations.

Hypothesis: invented theorem hallucinations will NOT decay across generations
because the model has no competing prior knowledge to push back with.
If true, this completely flips the Exp 2 finding and makes it conditional:
  - Known false facts → decay naturally (model knowledge corrects them)
  - Novel false beliefs → persist or grow (nothing to correct them)

We also compare invented theorems vs numerical facts across generations
to show the contrast clearly.

Pipeline:
  Gen 1: ICL injection of false theorem into Teacher LLM
  Gen 2-5: Each agent receives previous agent's response as context
  Judge: measures propagation at each generation
  Compare: invented_theorem decay curve vs numerical decay curve

"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.azure_client import query_llm

os.makedirs("results", exist_ok=True)

MATH_DATASET_PATH = "data/math_theorem_dataset.csv"
MAX_GENERATIONS   = 5
SAMPLE_PER_TYPE   = 10  # 10 invented + 10 numerical = 20 chains × 5 gens = 100 rows


def seed_via_icl(question: str, false_fact: str) -> str:
    """Gen 1: inject false fact via ICL."""
    system = "You are a knowledgeable assistant. Answer questions based on the context provided."
    user = (
        f"Context: {false_fact}\n\n"
        f"Based on the context above, answer the following question in one sentence:\n{question}"
    )
    return query_llm(system, user)


def pass_to_next_agent(question: str, previous_response: str) -> str:
    """Gen 2-5: pass previous response as neutral context."""
    system = "You are a knowledgeable assistant. Answer questions based on the context provided."
    user = (
        f"Context: {previous_response}\n\n"
        f"Based on the context above, answer the following question in one sentence:\n{question}"
    )
    return query_llm(system, user)


def judge_response(question: str, response: str, false_fact: str) -> tuple[bool, float, str]:
    """Judge LLM: did this agent propagate the false fact?"""
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
        f"Agent response: {response}\n\n"
        "Did the agent propagate the false fact? Respond in JSON only."
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
        propagated = false_fact.lower()[:20] in response.lower()
        return propagated, 0.5, "fallback keyword match"


def run_chain(question: str, false_fact: str, ground_truth: str, category: str) -> list[dict]:
    """Run one question through all 5 generations and return results per generation."""
    chain_results = []

    # Generation 1 — seed via ICL
    try:
        current_response = seed_via_icl(question, false_fact)
    except Exception as e:
        print(f"\nGen 1 failed: {e}")
        return []

    propagated, confidence, reason = judge_response(question, current_response, false_fact)
    chain_results.append({
        "question":      question,
        "false_fact":    false_fact,
        "ground_truth":  ground_truth,
        "category":      category,
        "generation":    1,
        "response":      current_response,
        "propagated":    propagated,
        "confidence":    confidence,
        "reason":        reason,
    })

    # Generations 2-5
    for gen in range(2, MAX_GENERATIONS + 1):
        try:
            current_response = pass_to_next_agent(question, current_response)
            time.sleep(0.3)
        except Exception as e:
            print(f"\nGen {gen} failed: {e}")
            break

        propagated, confidence, reason = judge_response(question, current_response, false_fact)
        chain_results.append({
            "question":      question,
            "false_fact":    false_fact,
            "ground_truth":  ground_truth,
            "category":      category,
            "generation":    gen,
            "response":      current_response,
            "propagated":    propagated,
            "confidence":    confidence,
            "reason":        reason,
        })

    return chain_results


def run_experiment(invented_df: pd.DataFrame, numerical_df: pd.DataFrame) -> pd.DataFrame:
    all_results = []

    print("\n── Running invented theorem chains ──")
    for _, row in tqdm(invented_df.iterrows(), total=len(invented_df), desc="Invented"):
        chain = run_chain(row["question"], row["false_fact"], row["ground_truth"], "invented_theorem")
        all_results.extend(chain)

    print("\n── Running numerical false fact chains ──")
    for _, row in tqdm(numerical_df.iterrows(), total=len(numerical_df), desc="Numerical"):
        chain = run_chain(row["question"], row["false_fact"], row["ground_truth"], "numerical")
        all_results.extend(chain)

    return pd.DataFrame(all_results)


def plot_results(results_df: pd.DataFrame):
    # Compute propagation rate per generation per category
    summary = (
        results_df.groupby(["category", "generation"])["propagated"]
        .mean()
        .reset_index()
        .rename(columns={"propagated": "propagation_rate"})
    )

    invented = summary[summary["category"] == "invented_theorem"]
    numerical = summary[summary["category"] == "numerical"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Line chart — decay comparison
    axes[0].plot(invented["generation"], invented["propagation_rate"],
                 marker="o", linewidth=2.5, color="#C00000", markersize=9, label="Invented theorem")
    axes[0].fill_between(invented["generation"], invented["propagation_rate"],
                         alpha=0.12, color="#C00000")
    axes[0].plot(numerical["generation"], numerical["propagation_rate"],
                 marker="s", linewidth=2.5, color="#4472C4", markersize=9, label="Numerical false fact")
    axes[0].fill_between(numerical["generation"], numerical["propagation_rate"],
                         alpha=0.12, color="#4472C4")
    axes[0].set_title("Propagation decay: invented vs numerical", fontsize=13)
    axes[0].set_xlabel("Agent generation")
    axes[0].set_ylabel("Propagation rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xticks(range(1, MAX_GENERATIONS + 1))
    axes[0].legend()
    axes[0].axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    # Bar chart — Gen 1 vs Gen 5 side by side
    categories = ["invented_theorem", "numerical"]
    labels = ["Invented theorem", "Numerical"]
    gen1_rates = [summary[(summary["category"]==c) & (summary["generation"]==1)]["propagation_rate"].values
                  for c in categories]
    gen5_rates = [summary[(summary["category"]==c) & (summary["generation"]==5)]["propagation_rate"].values
                  for c in categories]

    gen1_vals = [r[0] if len(r) > 0 else 0 for r in gen1_rates]
    gen5_vals = [r[0] if len(r) > 0 else 0 for r in gen5_rates]

    x = range(len(categories))
    width = 0.35
    axes[1].bar([i - width/2 for i in x], gen1_vals, width, label="Gen 1", color="#E05A3A", edgecolor="white")
    axes[1].bar([i + width/2 for i in x], gen5_vals, width, label="Gen 5", color="#4472C4", edgecolor="white")
    axes[1].set_title("Gen 1 vs Gen 5 propagation rate", fontsize=13)
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Propagation rate")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels)
    axes[1].legend()

    for i, (g1, g5) in enumerate(zip(gen1_vals, gen5_vals)):
        axes[1].text(i - width/2, g1 + 0.02, f"{g1:.0%}", ha="center", fontsize=10, fontweight="bold")
        axes[1].text(i + width/2, g5 + 0.02, f"{g5:.0%}", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("Experiment 4: Do invented theorems persist across generations?",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/exp4_plot.png", dpi=150)
    plt.close()
    print("Plot saved to results/exp4_plot.png")


if __name__ == "__main__":
    df = pd.read_csv(MATH_DATASET_PATH)

    # Get unique questions per category
    invented_df = (
        df[df["category"] == "invented_theorem"]
        .drop_duplicates(subset=["question", "false_fact"])
        .head(SAMPLE_PER_TYPE)
        .reset_index(drop=True)
    )
    numerical_df = (
        df[df["category"] == "numerical"]
        .drop_duplicates(subset=["question", "false_fact"])
        .head(SAMPLE_PER_TYPE)
        .reset_index(drop=True)
    )

    total_chains = len(invented_df) + len(numerical_df)
    total_api_calls = total_chains * MAX_GENERATIONS * 2

    print(f"Experiment 4 — Do invented theorems persist across generations?")
    print(f"Invented theorem chains: {len(invented_df)}")
    print(f"Numerical false fact chains: {len(numerical_df)}")
    print(f"Generations per chain: {MAX_GENERATIONS}")
    print(f"Total API calls: ~{total_api_calls}")
    print(f"Estimated time: ~{total_api_calls * 3 // 60} minutes")

    results_df = run_experiment(invented_df, numerical_df)

    out_path = "results/exp4_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    plot_results(results_df)

    print("\n── Summary by generation ──")
    summary = (
        results_df.groupby(["category", "generation"])["propagated"]
        .mean()
        .round(3)
    )
    print(summary)

    print("\n── Key finding ──")
    inv_gen1 = results_df[(results_df["category"]=="invented_theorem") & (results_df["generation"]==1)]["propagated"].mean()
    inv_gen5 = results_df[(results_df["category"]=="invented_theorem") & (results_df["generation"]==5)]["propagated"].mean()
    num_gen1 = results_df[(results_df["category"]=="numerical") & (results_df["generation"]==1)]["propagated"].mean()
    num_gen5 = results_df[(results_df["category"]=="numerical") & (results_df["generation"]==5)]["propagated"].mean()

    print(f"Invented theorem: Gen 1 = {inv_gen1:.1%} → Gen 5 = {inv_gen5:.1%}")
    print(f"Numerical:        Gen 1 = {num_gen1:.1%} → Gen 5 = {num_gen5:.1%}")

    if inv_gen5 > 0.2:
        print("✓ Hypothesis supported: invented theorems persist across generations")
        print("  Decay only happens when model has prior knowledge to push back with")
    elif inv_gen5 > inv_gen1 * 0.5:
        print("~ Partial support: invented theorems decay more slowly than known false facts")
    else:
        print("✗ Hypothesis not supported: invented theorems also decay")
