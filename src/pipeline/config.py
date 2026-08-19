"""
Configuration for the component lookup pipeline.

This is the only module in the codebase that should read secrets from the
environment. Every other module (llm_client.py, search_client.py,
scrape_client.py, ...) should import a `Settings` instance from here
rather than calling `os.environ` directly.

Fixes applied here (see code review):
- No API keys, tokens, or URLs are hardcoded as string literals anywhere.
  The original script had a live Gemini key, a commented-out LangSmith
  key, and a live Jina bearer token all committed in plaintext -- all
  three are now loaded from environment variables / a local `.env` file
  that is git-ignored.
- Renamed `OPENAI_API_KEY` / `OPENAI_API_URL` to `LLM_API_KEY` /
  `LLM_BASE_URL`. The original names were misleading: the key authenticates
  against Gemini's OpenAI-compatible endpoint, not OpenAI itself, and the
  `ChatOpenAI` class name doesn't change what's actually being called. The
  old names are still read as a fallback (with a deprecation warning) so
  any environment that hasn't migrated yet doesn't break silently.
- Secrets are typed as `SecretStr`, so they don't get accidentally printed
  or logged if a `Settings` object is dumped or repr'd in a stack trace.
"""

from __future__ import annotations

import warnings
from typing import Optional

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Centralized, typed application settings.

    Values are loaded, in order of precedence (highest first):
    1. Actual environment variables.
    2. A `.env` file in the project root, if present.
    3. The `default` given on each field below, if any.

    Construct once via `get_settings()`; don't instantiate `Settings()`
    directly in node code, so tests can override it in one place.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unrelated env vars in .env shouldn't error
    )

    # --- LLM provider ---------------------------------------------------
    llm_api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for the LLM provider (Gemini, via its "
        "OpenAI-compatible endpoint).",
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for the OpenAI-compatible LLM endpoint.",
    )
    llm_min_interval_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum interval in seconds between successive LLM requests to avoid rate limits.",
    )
    llm_model_name: str = Field(
        default="models/gemini-3.1-flash-lite",
        description="Model identifier passed to the chat client.",
    )

    # --- Scraping ---------------------------------------------------------
    jina_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Bearer token for the Jina Reader scraping API.",
    )

    # --- Search -----------------------------------------------------------
    serp_api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for SerpAPI Google Search (250 free searches/month).",
    )
    google_cse_api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for Google Custom Search JSON API.",
    )
    google_cse_cx: Optional[str] = Field(
        default=None,
        description="Search Engine ID (CX) for Google Custom Search.",
    )
    firecrawl_api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for Firecrawl Search API.",
    )
    tavily_api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for Tavily Search API (1,000 free req/month).",
    )
    search_provider: str = Field(
        default="auto",
        description="Select a specific search provider to use exclusively "
        "('google_cse', 'serp', 'firecrawl', 'tavily', 'duckduckgo', 'jina'), "
        "or 'auto' to use the default hybrid fallback chain.",
    )
    search_max_results: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum number of search results to request from each provider.",
    )
    search_inter_request_delay_seconds: float = Field(
        default=2.0,
        ge=0.0,
        description="Delay between successive search calls, to avoid "
        "being rate-limited or blocked by search backends.",
    )

    # --- Checkpointing ------------------------------------------------------
    checkpoint_db_path: str = Field(default="data/checkpoints/pipeline.sqlite")

    # --- Feature Toggles ----------------------------------------------------
    enable_site_search: bool = Field(
        default=False, 
        description="Enable or disable the site search fallback feature."
    )
    enable_series_fallback: bool = Field(
        default=False,
        description="Enable or disable searching for part series when exact part is not found."
    )

    browser_headless: bool = Field(
        default=False,
        description="Whether to run Playwright browser in headless mode."
    )
    pipeline_tier: str = Field(
        default="auto",
        description="Filter execution to run only a single lookup tier: "
        "'customer_urls', 'web_search', 'url_inference', or 'browser_automation'. "
        "Use 'auto' to run all tiers."
    )

    # --- Stage-first orchestrator settings ----------------------------------
    pipeline_workspace_persistent: bool = Field(
        default=False,
        description=(
            "If True, the pipeline workspace (where per-message JSON state files "
            "are stored) is kept on disk after the run finishes, allowing manual "
            "inspection and crash-recovery resume. "
            "If False (default), a temporary directory is created and automatically "
            "deleted when the run completes."
        ),
    )
    pipeline_workspace_dir: str = Field(
        default="data/pipeline_workspace",
        description=(
            "Directory to use as the pipeline workspace when "
            "PIPELINE_WORKSPACE_PERSISTENT=true. Relative paths are resolved "
            "from the project root. Ignored when PIPELINE_WORKSPACE_PERSISTENT=false."
        ),
    )
    pipeline_max_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description=(
            "Maximum number of parallel worker threads used during the search "
            "and filter stages. Browser automation always runs sequentially "
            "regardless of this setting."
        ),
    )


    # --- Observability (optional, off by default) --------------------------
    langsmith_tracing: bool = Field(default=False)
    langsmith_api_key: Optional[SecretStr] = Field(default=None)
    langsmith_project: str = Field(default="Customer Cycle Graph")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com"
    )

    # --- Backward-compatible env var names ----------------------------------
    # Read from the OLD names too, so an environment that hasn't migrated
    # its .env file yet still works, with a clear warning pointing at the
    # new names rather than a silent fallback that masks the rename.
    openai_api_key: Optional[SecretStr] = Field(default=None, exclude=True)
    openai_api_url: Optional[str] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _apply_legacy_env_fallback(self) -> "Settings":
        if self.llm_api_key is None and self.openai_api_key is not None:
            warnings.warn(
                "OPENAI_API_KEY is deprecated -- rename it to LLM_API_KEY "
                "in your .env file. The key authenticates against the LLM "
                "provider's OpenAI-compatible endpoint, not OpenAI itself.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.llm_api_key = self.openai_api_key

        if self.llm_base_url is None and self.openai_api_url is not None:
            warnings.warn(
                "OPENAI_API_URL is deprecated -- rename it to LLM_BASE_URL "
                "in your .env file.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.llm_base_url = self.openai_api_url

        # Apply the real default only after the legacy fallback has had a
        # chance to populate llm_base_url -- this must not be the field's
        # own `default=`, or it would mask OPENAI_API_URL above (its
        # "is None" check would never be true).
        if self.llm_base_url is None:
            self.llm_base_url = (
                "https://generativelanguage.googleapis.com/v1beta/openai/"
            )

        return self

    def require_llm_api_key(self) -> str:
        """
        Return the LLM API key as a plain string, raising a clear error if
        it isn't set, instead of letting a `None` propagate into the LLM
        client constructor and fail there with a less obvious error.
        """
        if self.llm_api_key is None:
            raise RuntimeError(
                "No LLM API key configured. Set LLM_API_KEY in your "
                "environment or .env file."
            )
        return self.llm_api_key.get_secret_value()

    def require_jina_api_key(self) -> str:
        """Same as `require_llm_api_key`, for the Jina scraping token."""
        if self.jina_api_key is None:
            raise RuntimeError(
                "No Jina API key configured. Set JINA_API_KEY in your "
                "environment or .env file."
            )
        return self.jina_api_key.get_secret_value()

    def serp_api_key_value(self) -> Optional[str]:
        """Return SerpAPI key as plain string, or None if not configured."""
        if self.serp_api_key is None:
            return None
        return self.serp_api_key.get_secret_value()

    def google_cse_api_key_value(self) -> Optional[str]:
        """Return Google Custom Search API key as plain string, or None if not configured."""
        if self.google_cse_api_key is None:
            return None
        return self.google_cse_api_key.get_secret_value()

    def firecrawl_api_key_value(self) -> Optional[str]:
        """Return Firecrawl API key as plain string, or None if not configured."""
        if self.firecrawl_api_key is None:
            return None
        return self.firecrawl_api_key.get_secret_value()

    def tavily_api_key_value(self) -> Optional[str]:
        """Return Tavily API key as plain string, or None if not configured."""
        if self.tavily_api_key is None:
            return None
        return self.tavily_api_key.get_secret_value()


# Module-level singleton, built lazily so importing this module never fails
# just because a .env file or env var happens to be missing in a context
# (like running unit tests) that doesn't need real settings at all.
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    """
    Override the module-level settings singleton. Intended for tests, to
    inject fixed or fake settings without touching the real environment
    or a real `.env` file.
    """
    global _settings
    _settings = settings