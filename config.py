import os
from dotenv import load_dotenv

load_dotenv()

# ── Azure OpenAI credentials ──────────────────────────────────────────────────
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_API_KEY_HERE")
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "https://YOUR_RESOURCE.openai.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# ── Model deployment name (set in your Azure portal) ─────────────────────────
# Common options: "gpt-4o", "gpt-4", "gpt-35-turbo"
DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")

# ── Experiment settings ───────────────────────────────────────────────────────
DATASET_PATH    = "data/hallucination_propagation_dataset.csv"
RESULTS_DIR     = "results"
MAX_TOKENS      = 256
TEMPERATURE     = 0.0          # deterministic — important for reproducibility
SAMPLE_SIZE     = 100          # rows to use per experiment run (use 1000 for full run)
RANDOM_SEED     = 42
