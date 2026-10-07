"""LLM backends behind one tiny interface: generate(system, user) -> str."""
from __future__ import annotations

import re
import time
from typing import Protocol

import requests

from .guardrails import REFUSAL_TEXT


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class GroqLLM:
    """Groq's OpenAI-compatible chat completions endpoint."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str, timeout: int = 30, retries: int = 5) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": 500,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None
        for attempt in range(self.retries):
          try:
            resp = requests.post(
                self.URL, json=payload, headers=headers, timeout=self.timeout
            )
            if resp.status_code == 429:
              retry_after = resp.headers.get("retry-after")
              wait_sec = (
                  float(retry_after)
                  if retry_after and retry_after.replace(".", "", 1).isdigit()
                  else (3.0 * (attempt + 1))
              )
              time.sleep(wait_sec)
              continue
            if resp.status_code in (500, 502, 503, 504):
              raise requests.HTTPError(f"retryable status {resp.status_code}")
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
          except requests.RequestException as exc:
            last_error = exc
            time.sleep(2**attempt)
        raise RuntimeError(
            f"LLM request failed after {self.retries} attempts: {last_error}"
        )


class ExtractiveLLM:
    """Offline fallback (no API key needed): quotes the top passage with its citation."""

    def generate(self, system: str, user: str) -> str:
        match = re.search(r"\[1\]\s*(.+?)(?=\n\[\d+\]|\n\nQuestion:|\Z)", user, re.S)
        if not match:
            return REFUSAL_TEXT
        snippet = " ".join(match.group(1).split())[:400]
        return f"{snippet} [1]"
