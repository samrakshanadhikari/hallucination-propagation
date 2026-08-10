import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Mistral client ────────────────────────────────────────────────────────────
mistral_client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
MISTRAL_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_NAME", "Mistral-Large-3")

# ── GPT-5.4-nano client ───────────────────────────────────────────────────────
GPT_ENDPOINT   = os.getenv("GPT_ENDPOINT", "https://proj-221.services.ai.azure.com/openai/v1")
GPT_DEPLOYMENT = os.getenv("GPT_DEPLOYMENT", "gpt-5.4-nano")
GPT_API_KEY    = os.getenv("GPT_API_KEY")

gpt_client = OpenAI(
    api_key=GPT_API_KEY,
    base_url=GPT_ENDPOINT,
)

# ── Unified query function ────────────────────────────────────────────────────

def query_llm(system_prompt: str, user_prompt: str, model: str = "mistral") -> str:
    if model == "gpt":
        response = gpt_client.chat.completions.create(
            model=GPT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_completion_tokens=256,   # <-- changed
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    else:  # mistral
        response = mistral_client.chat.completions.create(
            model=MISTRAL_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=256,               # unchanged, Mistral still uses this
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()


def test_mistral():
    try:
        result = query_llm("You are a helpful assistant.", "Say hello in one word.", model="mistral")
        print(f"✅ Mistral connection successful. Response: {result}")
        return True
    except Exception as e:
        print(f"❌ Mistral connection failed: {e}")
        return False


def test_gpt():
    try:
        result = query_llm("You are a helpful assistant.", "Say hello in one word.", model="gpt")
        print(f"✅ GPT-5.4-nano connection successful. Response: {result}")
        return True
    except Exception as e:
        print(f"❌ GPT-5.4-nano connection failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Mistral Large 3...")
    test_mistral()
    print("\nTesting GPT-5.4-nano...")
    test_gpt()