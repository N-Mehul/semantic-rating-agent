"""Local Ollama Embedding Adapter for Semantic Similarity Rating (SSR).

Provides deterministic, local-only text embeddings via Ollama's REST API.
Does NOT call or require the OpenAI API, any paid service, or external network requests.

Default model: nomic-embed-text (installed and running locally in Ollama).
Endpoints supported:
- POST /api/embed (Ollama batch embedding endpoint)
- POST /api/embeddings (Ollama single-string fallback endpoint)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

DEFAULT_HOST: Final[str] = "http://localhost:11434"
DEFAULT_MODEL: Final[str] = "nomic-embed-text"
DEFAULT_TIMEOUT_S: Final[float] = 60.0


class OllamaEmbeddingError(RuntimeError):
    """Raised when local Ollama embedding generation fails."""


class OllamaEmbeddingAdapter:
    """Adapter for generating numerical text embeddings from a local Ollama instance."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.model_name = os.environ.get("OLLAMA_EMBED_MODEL", model_name).strip()
        self.host = os.environ.get("OLLAMA_HOST", host).rstrip("/")
        self.timeout_s = timeout_s
        self._cache: dict[str, np.ndarray] = {}

    def probe(self) -> dict[str, Any]:
        """Check if Ollama server is running and the embedding model is available."""
        url = f"{self.host}/api/tags"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5.0) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            return {
                "running": False,
                "model_available": False,
                "host": self.host,
                "model": self.model_name,
                "error": f"Ollama not reachable at {self.host} ({type(e).__name__}): {e}",
            }

        models = [m.get("name") or m.get("model") for m in data.get("models", [])]
        model_ok = (
            self.model_name in models
            or f"{self.model_name}:latest" in models
            or any(m.startswith(f"{self.model_name}:") for m in models)
        )
        return {
            "running": True,
            "model_available": model_ok,
            "models": models,
            "host": self.host,
            "model": self.model_name,
            "error": None if model_ok else f"Model '{self.model_name}' not in installed models: {models}",
        }

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Generate numerical embedding vectors for a sequence of texts.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) and dtype float64.
        """
        if not texts:
            raise ValueError("Input texts sequence cannot be empty")

        # Check in-memory cache first
        missing_indices = [i for i, t in enumerate(texts) if t not in self._cache]
        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            vectors = self._fetch_embeddings(missing_texts)
            for i, vec in zip(missing_indices, vectors):
                self._cache[texts[i]] = vec

        out = np.stack([self._cache[t] for t in texts]).astype(np.float64)
        if not np.all(np.isfinite(out)):
            raise OllamaEmbeddingError("Ollama returned non-finite embedding values")
        return out

    def embed_text(self, text: str) -> np.ndarray:
        """Generate a 1D numerical embedding vector for a single text."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")
        if text in self._cache:
            return self._cache[text].copy()
        matrix = self.embed_texts([text])
        return matrix[0]

    def _fetch_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Internal call to Ollama /api/embed or /api/embeddings."""
        # Try /api/embed batch endpoint first
        embed_url = f"{self.host}/api/embed"
        payload = {"model": self.model_name, "input": texts}
        req = urllib.request.Request(
            embed_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                res = json.loads(r.read().decode("utf-8"))
            if "embeddings" in res and len(res["embeddings"]) == len(texts):
                return [np.asarray(v, dtype=np.float64) for v in res["embeddings"]]
        except Exception:
            pass  # Fall back to single-text /api/embeddings

        # Fallback to /api/embeddings sequentially
        single_url = f"{self.host}/api/embeddings"
        vectors = []
        for t in texts:
            single_payload = {"model": self.model_name, "prompt": t}
            single_req = urllib.request.Request(
                single_url,
                data=json.dumps(single_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(single_req, timeout=self.timeout_s) as r:
                    res = json.loads(r.read().decode("utf-8"))
                if "embedding" not in res:
                    raise OllamaEmbeddingError(f"Malformed response from Ollama: {res}")
                vectors.append(np.asarray(res["embedding"], dtype=np.float64))
            except Exception as e:
                raise OllamaEmbeddingError(
                    f"Failed to generate embedding via Ollama at {self.host} with model '{self.model_name}': {e}"
                ) from e

        return vectors


# Singleton / module-level helper instance
_DEFAULT_ADAPTER = OllamaEmbeddingAdapter()


def get_ollama_embedding_adapter(model_name: str = DEFAULT_MODEL) -> OllamaEmbeddingAdapter:
    """Return an OllamaEmbeddingAdapter configured for the requested model."""
    if model_name == DEFAULT_MODEL:
        return _DEFAULT_ADAPTER
    return OllamaEmbeddingAdapter(model_name=model_name)

