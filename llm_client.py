import os
from typing import List, Dict, Optional

try:
    from openai import OpenAI
except ImportError as exc:
    raise ImportError(
        "The 'openai' package is required. Install with `pip install openai>=1.3`."
    ) from exc


_CLIENT: Optional["OpenAI"] = None


def load_env_vars(env_path: str = ".env") -> None:
    """
    Lightweight .env loader. Existing environment variables are not overwritten.
    Expected keys:
      - OPENAI_BASE_URL
      - OPENAI_API_KEY
    """
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key and key not in os.environ:
                os.environ[key] = value


def get_client() -> "OpenAI":
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Create a .env with OPENAI_API_KEY and OPENAI_BASE_URL."
        )
    _CLIENT = OpenAI(base_url=base_url, api_key=api_key)
    return _CLIENT


def chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 400,
) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

