# -*- coding: utf-8 -*-
"""
src/providers.py -- AI Provider Abstraction Layer
==================================================
Single place to add, remove, or configure LLM providers.
Nothing outside this file needs to know Groq vs OpenAI vs HuggingFace.

Switch provider by changing .env only:
    LLM_PROVIDER=groq        GROQ_API_KEY=gsk_...
    LLM_PROVIDER=openai      OPENAI_API_KEY=sk-...
    LLM_PROVIDER=huggingface HUGGINGFACE_API_KEY=hf_...

Optional automatic fallback:
    FALLBACK_LLM_PROVIDER=openai
    OPENAI_API_KEY=sk-...      (used only if primary key is missing)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Tuple

import instructor


# ============================================================
# Base class
# ============================================================

class BaseLLMProvider(ABC):
    """
    Contract that every provider must satisfy.
    The agent depends only on this interface -- never on Groq/OpenAI directly.
    """

    @abstractmethod
    def get_clients(self) -> Tuple[Any, Any]:
        """
        Returns (raw_client, instructor_client).
        Both expose the same .chat.completions.create() interface.
        """

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Model string to use when LLM_MODEL env var is not set."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier: 'groq', 'openai', 'huggingface'."""

    @property
    def model(self) -> str:
        """Active model: LLM_MODEL env override, or provider default."""
        return os.getenv("LLM_MODEL", self.default_model)


# ============================================================
# Groq  (default, free)
# ============================================================

class GroqProvider(BaseLLMProvider):
    """
    Groq Cloud -- free tier, ~500 tok/s inference speed.
    Models: llama-3.3-70b-versatile | llama-3.1-8b-instant
    Key:    console.groq.com -> API Keys
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Get a free key at console.groq.com"
            )
        self._api_key = api_key

    @property
    def default_model(self) -> str:
        return "llama-3.3-70b-versatile"

    @property
    def provider_name(self) -> str:
        return "groq"

    def get_clients(self) -> Tuple[Any, Any]:
        from groq import Groq  # lazy -- only imported when actually used
        raw        = Groq(api_key=self._api_key)
        structured = instructor.from_groq(raw, mode=instructor.Mode.GROQ_JSON)
        return raw, structured


# ============================================================
# OpenAI  (fallback / production)
# ============================================================

class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API -- highest quality, usage-billed.
    Models: gpt-4o-mini | gpt-4o
    Key:    platform.openai.com/api-keys
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. "
                "Set it in .env or switch to LLM_PROVIDER=groq (free)."
            )
        self._api_key = api_key

    @property
    def default_model(self) -> str:
        return "gpt-4o-mini"

    @property
    def provider_name(self) -> str:
        return "openai"

    def get_clients(self) -> Tuple[Any, Any]:
        from openai import OpenAI  # lazy
        raw        = OpenAI(api_key=self._api_key)
        structured = instructor.from_openai(raw)
        return raw, structured


# ============================================================
# Hugging Face  (optional / student access)
# ============================================================

class HuggingFaceProvider(BaseLLMProvider):
    """
    Hugging Face Inference API -- uses OpenAI-compatible endpoint.
    Free tier available. GitHub Student Pack gives extended access.
    Key:    huggingface.co/settings/tokens

    Note: instructor uses TOOL_CALL mode with HF (not JSON mode).
    Not all models support structured output reliably.
    Recommended: meta-llama/Meta-Llama-3.1-8B-Instruct
    """

    HF_BASE_URL = "https://api-inference.huggingface.co/v1"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "HUGGINGFACE_API_KEY is missing. "
                "Get a free token at huggingface.co/settings/tokens"
            )
        self._api_key = api_key

    @property
    def default_model(self) -> str:
        return "meta-llama/Meta-Llama-3.1-8B-Instruct"

    @property
    def provider_name(self) -> str:
        return "huggingface"

    def get_clients(self) -> Tuple[Any, Any]:
        from openai import OpenAI  # HF exposes an OpenAI-compatible REST API
        raw        = OpenAI(api_key=self._api_key, base_url=self.HF_BASE_URL)
        structured = instructor.from_openai(raw, mode=instructor.Mode.TOOLS)
        return raw, structured


# ============================================================
# Registry & factory
# ============================================================

# Map provider name -> class.  Add new providers here only.
_REGISTRY: dict[str, type] = {
    "groq":         GroqProvider,
    "openai":       OpenAIProvider,
    "huggingface":  HuggingFaceProvider,
}

# Map provider name -> expected env key name.
_ENV_KEYS: dict[str, str] = {
    "groq":         "GROQ_API_KEY",
    "openai":       "OPENAI_API_KEY",
    "huggingface":  "HUGGINGFACE_API_KEY",
}


def get_provider(name: str, api_key: str) -> BaseLLMProvider:
    """
    Factory: returns an instantiated provider for the given name and key.

    Args:
        name:    'groq' | 'openai' | 'huggingface'
        api_key: API key for that provider

    Raises:
        ValueError if name is unknown or key is empty
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{name}'. "
            f"Valid: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](api_key=api_key)


def get_provider_from_env() -> BaseLLMProvider:
    """
    Auto-configure provider entirely from environment variables.

    Resolution order:
      1. LLM_PROVIDER  (default: 'groq')
      2. Corresponding key (GROQ_API_KEY / OPENAI_API_KEY / HUGGINGFACE_API_KEY)
      3. If primary key is missing -> try FALLBACK_LLM_PROVIDER + its key
      4. If both missing -> raise RuntimeError with clear message

    .env example (dev, zero cost):
        LLM_PROVIDER=groq
        GROQ_API_KEY=gsk_...
        FALLBACK_LLM_PROVIDER=openai
        OPENAI_API_KEY=sk-...

    .env example (production):
        LLM_PROVIDER=openai
        OPENAI_API_KEY=sk-...
    """
    primary_name = os.getenv("LLM_PROVIDER", "groq")
    primary_key  = os.getenv(_ENV_KEYS.get(primary_name, ""), "")

    if primary_key:
        provider = get_provider(primary_name, primary_key)
        print(f"[Provider] {provider.provider_name} | {provider.model}")
        return provider

    # -- Automatic fallback ---------------------------------------------------
    fallback_name = os.getenv("FALLBACK_LLM_PROVIDER", "")
    fallback_key  = os.getenv(_ENV_KEYS.get(fallback_name, ""), "")

    if fallback_name and fallback_key:
        print(
            f"[Provider] WARNING: {primary_name} key missing. "
            f"Falling back to {fallback_name}."
        )
        provider = get_provider(fallback_name, fallback_key)
        print(f"[Provider] {provider.provider_name} (fallback) | {provider.model}")
        return provider

    # -- Nothing configured ---------------------------------------------------
    env_key_name = _ENV_KEYS.get(primary_name, "API_KEY")
    raise RuntimeError(
        f"No API key found for LLM_PROVIDER='{primary_name}'.\n"
        f"  -> Set {env_key_name} in your .env file.\n"
        f"  -> Or use LLM_PROVIDER=groq and GROQ_API_KEY (free: console.groq.com)."
    )


def list_providers() -> list[str]:
    """Return names of all registered providers."""
    return list(_REGISTRY.keys())
