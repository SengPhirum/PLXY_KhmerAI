"""Typed application settings.

Every value comes from the environment (``KHMERAI_*``, see ``.env.example``) with
the YAML files under ``configs/`` supplying structured defaults for anything that
is a policy rather than a deployment detail.

Validation is strict on purpose: a production process must fail at startup on a
bad configuration rather than at 03:00 on the first request that touches it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.config import ConfigError, load_config
from common.paths import PROJECT_ROOT

__all__ = ["Settings", "get_settings", "reload_settings"]


class Settings(BaseSettings):
    """Runtime configuration for the FastAPI service."""

    model_config = SettingsConfigDict(
        env_prefix="KHMERAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- application -------------------------------------------------------
    env: Literal["development", "staging", "production"] = "development"
    app_name: str = "khmer-support-llm"
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    config_file: str = "configs/production/api.yaml"

    # --- security ----------------------------------------------------------
    admin_api_key: str = ""
    client_api_keys: str = ""
    require_client_auth: bool = False
    cors_origins: str = ""
    max_request_bytes: int = 65_536
    max_message_chars: int = 4_000

    # --- rate limiting -----------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60
    rate_limit_burst: int = 10

    # --- Ollama ------------------------------------------------------------
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "khmer-support-9b"
    ollama_fallback_model: str = "khmer-support-4b"
    ollama_connect_timeout_s: float = 5.0
    ollama_read_timeout_s: float = 120.0
    ollama_total_timeout_s: float = 180.0

    # --- generation --------------------------------------------------------
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.05
    max_output_tokens: int = 768
    num_ctx: int = 8192

    # --- concurrency -------------------------------------------------------
    max_active_generations: int = 4
    max_queue_depth: int = 64
    queue_timeout_s: float = 20.0

    # --- RAG ---------------------------------------------------------------
    rag_enabled: bool = True
    rag_config: str = "configs/rag/retrieval.yaml"
    index_root: str = "data/index"
    vector_backend: Literal["local", "qdrant"] = "local"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "khmer_company_kb"
    embedding_backend: Literal["ollama", "sentence_transformers", "hashing"] = "ollama"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_dim: int = 1024
    retrieval_timeout_s: float = 5.0

    # --- conversation ------------------------------------------------------
    conversation_ttl_seconds: int = 3600
    conversation_max_turns: int = 12
    conversation_store: Literal["memory", "none"] = "memory"
    persist_conversations: bool = False
    anonymise_logs: bool = True

    # --- support desk ------------------------------------------------------
    support_hotline: str = ""
    support_email: str = ""
    business_hours: str = "ច័ន្ទ-សៅរ៍ ៨:០០-១៧:០០"
    company_display_name: str = "ក្រុមហ៊ុនរបស់យើង"

    # --- observability -----------------------------------------------------
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"
    request_id_header: str = "X-Request-ID"

    # --- validators --------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def _valid_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper

    @field_validator("ollama_base_url", "qdrant_url")
    @classmethod
    def _strip_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("temperature")
    @classmethod
    def _sane_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return value

    @model_validator(mode="after")
    def _cross_field_checks(self) -> Settings:
        if self.max_active_generations < 1:
            raise ValueError("max_active_generations must be at least 1")
        if self.env == "production":
            if not self.admin_api_key or self.admin_api_key.startswith("CHANGE_ME"):
                raise ValueError(
                    "KHMERAI_ADMIN_API_KEY must be set to a real secret in production. "
                    'Generate one with: python -c "import secrets;print(secrets.token_urlsafe(48))"'
                )
            if self.embedding_backend == "hashing":
                raise ValueError(
                    "the hashing embedding backend is for tests only and must not run in "
                    "production; set KHMERAI_EMBEDDING_BACKEND=ollama"
                )
        return self

    # --- derived -----------------------------------------------------------
    @property
    def client_key_set(self) -> frozenset[str]:
        return frozenset(k.strip() for k in self.client_api_keys.split(",") if k.strip())

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def index_root_path(self) -> Path:
        path = Path(self.index_root)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def active_index_path(self) -> Path:
        return self.index_root_path / "ACTIVE"

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    def generation_options(self, **overrides: Any) -> dict[str, Any]:
        """Ollama ``options`` payload for a generation request."""
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "num_predict": self.max_output_tokens,
            "num_ctx": self.num_ctx,
        }
        options.update({k: v for k, v in overrides.items() if v is not None})
        return options

    def policy(self) -> dict[str, Any]:
        """Structured policy loaded from ``configs/`` (never from the environment)."""
        try:
            return load_config(self.config_file)
        except ConfigError:
            return load_config("configs/base.yaml")

    def rag_policy(self) -> dict[str, Any]:
        try:
            return load_config(self.rag_config)
        except ConfigError:
            return {}

    def redacted(self) -> dict[str, Any]:
        """Settings safe to log or expose on an admin endpoint."""
        data = self.model_dump()
        for secret in ("admin_api_key", "client_api_keys", "qdrant_api_key"):
            if data.get(secret):
                data[secret] = "***set***"
            else:
                data[secret] = ""
        return data


_SINGLETON: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton.

    Built once on first access.  Tests and the admin config-reload path replace
    it through :func:`reload_settings` rather than mutating it, so a half-applied
    configuration can never be observed by a concurrent request.
    """
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = Settings()
    return _SINGLETON


def reload_settings(**overrides: Any) -> Settings:
    """Replace the singleton, optionally with explicit field overrides."""
    global _SINGLETON
    _SINGLETON = Settings(**overrides) if overrides else Settings()
    return _SINGLETON
