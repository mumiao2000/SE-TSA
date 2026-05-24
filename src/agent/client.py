"""Unified LLM API client with retry logic."""

import json
import os
import time
from typing import Any, Dict, List, Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Read max_tokens, retry_attempts, and retry_base_delay from .env / os.getenv."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or os.getenv("ANTHROPIC_MODEL")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "196608"))
        self.retry_attempts = int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))
        self.retry_base_delay = float(os.getenv("LLM_RETRY_BASE_DELAY", "2.0"))

        if not self.model:
            raise ValueError(
                "ANTHROPIC_MODEL not provided and not found in environment."
            )

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")

        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not provided and not found in environment."
            )

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self.client = anthropic.Anthropic(**kwargs)

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        last_exception: Optional[Exception] = None

        for attempt in range(self.retry_attempts):
            try:
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=tokens,
                    system=system or "",
                    messages=messages,
                ) as stream:
                    response = stream.get_final_message()
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                raise RuntimeError("No text block found in LLM response.")
            except Exception as exc:
                last_exception = exc
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_base_delay * (2**attempt)
                    time.sleep(delay)

        raise RuntimeError(
            f"LLM request failed after {self.retry_attempts} attempts: {last_exception}"
        )

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        text = self.chat(messages, system=system, max_tokens=max_tokens)
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        if not text:
            raise RuntimeError("LLM returned empty response.")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for start_char in ("{", "["):
            idx = text.find(start_char)
            if idx >= 0:
                try:
                    obj, _ = json.JSONDecoder().raw_decode(text, idx)
                    return obj
                except json.JSONDecodeError:
                    pass

        head = text[:200].replace("\n", " ")
        tail = text[-200:].replace("\n", " ") if len(text) > 200 else ""
        raise RuntimeError(
            f"LLM returned invalid JSON (len={len(text)}). Head: {head!r} ... Tail: {tail!r}"
        )
