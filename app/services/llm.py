"""LLM provider factory and abstraction.

Supports OpenAI, Anthropic and Ollama through LangChain chat models, and
falls back to a deterministic local "mock" LLM so the whole pipeline still
runs without any API key. The rest of the application only ever talks to the
:class:`LLMService` interface, never to a provider SDK directly.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.config import get_settings


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of the first JSON object from LLM output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


class MockLLM:
    """Deterministic local fallback used when no LLM provider is configured.

    Produces useful, structured text from a template so agents stay functional
    and testable without any network call.
    """

    name = "mock"

    def __init__(self) -> None:
        self._default_plan = (
            "1. Profile the dataset and detect the target column.\n"
            "2. Clean missing values, duplicates and outliers.\n"
            "3. Encode categoricals and scale numeric features.\n"
            "4. Train and compare baseline models.\n"
            "5. Evaluate metrics and pick the best model.\n"
            "6. Generate a technical and business report."
        )

    def invoke(self, prompt: str) -> str:
        low = prompt.lower()
        if "plan" in low or "workflow" in low:
            return self._default_plan
        if "target" in low or "predict" in low:
            return "Choose the column with the highest cardinality as the default target."
        if "insight" in low or "business" in low:
            return (
                "Key finding: the dataset is balanced with limited missing data. "
                "The best model outperforms the baseline by a meaningful margin. "
                "Recommendation: deploy the best model and monitor drift over time."
            )
        return (
            "Analysis completed using the local mock model. Configure an OpenAI, "
            "Anthropic or Ollama provider in `.env` for richer narrative output."
        )


class AnthropicService:
    """Thin wrapper over LangChain's Anthropic chat model."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        from langchain_anthropic import ChatAnthropic

        self._llm = ChatAnthropic(model=model, api_key=api_key, temperature=0.2)

    def invoke(self, prompt: str) -> str:
        return str(self._llm.invoke(prompt).content)


class OpenAIService:
    """Thin wrapper over LangChain's OpenAI chat model."""

    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.2)

    def invoke(self, prompt: str) -> str:
        return str(self._llm.invoke(prompt).content)


class OllamaService:
    """Thin wrapper over LangChain's Ollama chat model."""

    name = "ollama"

    def __init__(self, model: str, base_url: str) -> None:
        from langchain_community.chat_models import ChatOllama

        self._llm = ChatOllama(model=model, base_url=base_url, temperature=0.2)

    def invoke(self, prompt: str) -> str:
        return str(self._llm.invoke(prompt).content)


class LLMService:
    """Facade that builds the configured LLM and exposes simple helpers.

    Use :meth:`LLMService.instance` everywhere so a single model is reused.
    """

    _instance: "LLMService | None" = None

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.provider = getattr(backend, "name", "unknown")

    @classmethod
    def build(cls) -> "LLMService":
        settings = get_settings()
        provider = settings.llm_provider.strip().lower()

        if provider == "anthropic" and settings.anthropic_api_key:
            backend = AnthropicService(settings.anthropic_api_key, settings.anthropic_model)
        elif provider == "openai" and settings.openai_api_key:
            backend = OpenAIService(settings.openai_api_key, settings.openai_model)
        elif provider == "ollama":
            backend = OllamaService(settings.ollama_model, settings.ollama_base_url)
        else:
            backend = MockLLM()
        return cls(backend)

    @classmethod
    def instance(cls) -> "LLMService":
        if cls._instance is None:
            cls._instance = cls.build()
        return cls._instance

    def complete(self, prompt: str) -> str:
        """Run a single prompt and return the text response."""
        return self.backend.invoke(prompt)

    def complete_json(self, prompt: str) -> dict[str, Any]:
        """Run a prompt and return the first JSON object found in the reply."""
        return extract_json(self.complete(prompt))

    def reset(self) -> None:
        LLMService._instance = None