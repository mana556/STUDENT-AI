import os
import requests
from pathlib import Path


def _load_dotenv():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep and key:
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _get_groq_settings():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing credentials. Set GROQ_API_KEY in .env or the environment."
        )

    base_url = os.environ.get("GROQ_API_BASE", "https://api.groq.com/v1").rstrip("/")
    model = os.environ.get("GROQ_MODEL", "groq/compound")

    return api_key, base_url, model


class GroqChatClient:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.0):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature

    def predict(self, prompt: str) -> str:
        if self.base_url.endswith("/openai/v1"):
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
        else:
            url = f"{self.base_url}/completions"
            payload = {
                "model": self.model,
                "input": prompt,
                "temperature": self.temperature,
                "max_tokens": 1024,
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(url, json=payload, headers=headers, timeout=120)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"Groq API request failed {response.status_code} at {url}: {response.text}"
            ) from exc

        result = response.json()
        if "choices" not in result or not result["choices"]:
            raise RuntimeError(f"Invalid Groq response: {result}")

        text = result["choices"][0].get("text") or result["choices"][0].get("message", {}).get("content")
        if text is None:
            raise RuntimeError(f"Unexpected Groq completion payload: {result}")

        return text.strip()


def get_llm():
    _load_dotenv()
    api_key, base_url, model = _get_groq_settings()
    return GroqChatClient(api_key=api_key, base_url=base_url, model=model, temperature=0)
