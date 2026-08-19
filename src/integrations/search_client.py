"""
Search integration: hybrid fallback search client.

Provider priority (pure-fallback — stops at the first provider that returns ≥1 result):
    1. Google Custom Search    (requires GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX)
    2. SerpAPI (Google)        (250 free searches/month — requires SERP_API_KEY)
    3. Firecrawl Search API    (requires FIRECRAWL_API_KEY)
    4. Tavily Search API       (1,000 free req/month   — requires TAVILY_API_KEY)
    5. DuckDuckGo (via ddgs)   (free scraping, no key  — prone to rate limits at scale)
    6. Jina AI Search          (shared token pool      — requires JINA_API_KEY)

Any provider whose key is absent is silently skipped.
Any provider that returns an error (rate limit, network failure, empty response)
logs a WARNING and falls through to the next provider.

Changes from the original single-backend implementation:
- ddgs is now just one of four backends, not the sole provider.
- `search_with_fallback` replaces `search_multi_backend` as the primary entry point.
- `search_multi_backend` is kept as a backward-compatible alias so existing
  callers continue to compile without changes during a transition period.
- All per-provider results are validated into `SearchResultItem` (title/href/body)
  before being returned -- the callers never see raw untyped dicts.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional
from urllib.parse import quote

import requests
from ddgs import DDGS

from pipeline.state import SearchResultItem

logger = logging.getLogger(__name__)


class SearchBackendError(Exception):
    """Raised when a single search backend call fails."""

    def __init__(self, backend: str, message: str):
        super().__init__(f"[{backend}] {message}")
        self.backend = backend


# ---------------------------------------------------------------------------
# Private per-provider search functions
# ---------------------------------------------------------------------------

def _make_item(title=None, href=None, body=None) -> Optional[SearchResultItem]:
    """Construct a SearchResultItem, returning None if href is missing."""
    if not href:
        return None
    return SearchResultItem(title=title, href=href, body=body)


def _search_google_cse(
    query: str, api_key: str, cx: str, max_results: int
) -> List[SearchResultItem]:
    """
    Call the Google Custom Search JSON API.
    Docs: https://developers.google.com/custom-search/v1/overview

    Raises SearchBackendError on any failure.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": min(max_results, 10),
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
    except requests.exceptions.RequestException as e:
        raise SearchBackendError("google_cse", f"Request failed: {e}") from e

    if resp.status_code == 429:
        raise SearchBackendError("google_cse", "Rate limit exceeded (HTTP 429)")
    if not resp.ok:
        raise SearchBackendError("google_cse", f"HTTP {resp.status_code}")

    try:
        data = resp.json()
        web_results = data.get("items", [])
    except Exception as e:
        raise SearchBackendError("google_cse", f"Failed to parse response: {e}") from e

    items: List[SearchResultItem] = []
    for r in web_results:
        item = _make_item(
            title=r.get("title"),
            href=r.get("link"),
            body=r.get("snippet"),
        )
        if item:
            items.append(item)

    return items


def _search_serp(
    query: str, api_key: str, max_results: int
) -> List[SearchResultItem]:
    """
    Call the SerpAPI Google Search endpoint.
    Docs: https://serpapi.com/search-api

    Raises SearchBackendError on any failure (HTTP 429 = rate limit, etc.).
    """
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "num": min(max_results, 10),  # Google engine max per page is 10
        "hl": "en",
        "gl": "us",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
    except requests.exceptions.RequestException as e:
        raise SearchBackendError("serp", f"Request failed: {e}") from e

    if resp.status_code == 429:
        raise SearchBackendError("serp", "Rate limit exceeded (HTTP 429)")
    if not resp.ok:
        raise SearchBackendError("serp", f"HTTP {resp.status_code}")

    try:
        data = resp.json()
        web_results = data.get("organic_results", [])
    except Exception as e:
        raise SearchBackendError("serp", f"Failed to parse response: {e}") from e

    items: List[SearchResultItem] = []
    for r in web_results:
        item = _make_item(
            title=r.get("title"),
            href=r.get("link"),
            body=r.get("snippet"),
        )
        if item:
            items.append(item)

    return items


def _search_firecrawl(
    query: str, api_key: str, max_results: int
) -> List[SearchResultItem]:
    """
    Call the Firecrawl Search API (POST /v1/search).
    Docs: https://docs.firecrawl.dev

    Raises SearchBackendError on any failure.
    """
    url = "https://api.firecrawl.dev/v1/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "limit": min(max_results, 10),
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
    except requests.exceptions.RequestException as e:
        raise SearchBackendError("firecrawl", f"Request failed: {e}") from e

    if resp.status_code == 429:
        raise SearchBackendError("firecrawl", "Rate limit exceeded (HTTP 429)")
    if not resp.ok:
        raise SearchBackendError("firecrawl", f"HTTP {resp.status_code}")

    try:
        data = resp.json()
        raw_results = data.get("data", [])
        if not isinstance(raw_results, list):
            raw_results = []
    except Exception as e:
        raise SearchBackendError("firecrawl", f"Failed to parse response: {e}") from e

    items: List[SearchResultItem] = []
    for r in raw_results:
        item = _make_item(
            title=r.get("title"),
            href=r.get("url"),
            body=r.get("description") or r.get("snippet") or r.get("markdown"),
        )
        if item:
            items.append(item)

    return items


def _search_tavily(
    query: str, api_key: str, max_results: int
) -> List[SearchResultItem]:
    """
    Call the Tavily Search API (POST JSON).
    Docs: https://docs.tavily.com/docs/rest-api/api-reference

    Raises SearchBackendError on any failure.
    """
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": min(max_results, 20),
        "search_depth": "basic",
        "include_answer": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
    except requests.exceptions.RequestException as e:
        raise SearchBackendError("tavily", f"Request failed: {e}") from e

    if resp.status_code == 429:
        raise SearchBackendError("tavily", "Rate limit exceeded (HTTP 429)")
    if not resp.ok:
        raise SearchBackendError("tavily", f"HTTP {resp.status_code}")

    try:
        data = resp.json()
        raw_results = data.get("results", [])
    except Exception as e:
        raise SearchBackendError("tavily", f"Failed to parse response: {e}") from e

    items: List[SearchResultItem] = []
    for r in raw_results:
        item = _make_item(
            title=r.get("title"),
            href=r.get("url"),
            body=r.get("content"),
        )
        if item:
            items.append(item)

    return items


def _search_duckduckgo(query: str, max_results: int) -> List[SearchResultItem]:
    """
    Query DuckDuckGo via the ddgs library (scraping, no API key required).
    Falls back to this when premium providers are exhausted/unavailable.

    Raises SearchBackendError on any failure.
    """
    try:
        raw_results = DDGS().text(query, max_results=max_results)
    except Exception as e:
        raise SearchBackendError("duckduckgo", str(e)) from e

    items: List[SearchResultItem] = []
    for r in raw_results:
        item = _make_item(
            title=r.get("title"),
            href=r.get("href"),
            body=r.get("body"),
        )
        if item:
            items.append(item)

    return items


def _search_jina(
    query: str, api_key: str, max_results: int
) -> List[SearchResultItem]:
    """
    Call the Jina AI Search API (s.jina.ai).
    Uses the same JINA_API_KEY already configured for the reader/scrape pipeline.
    Docs: https://jina.ai/search-foundation

    Raises SearchBackendError on any failure (including 401 = tokens exhausted).
    """
    encoded_query = quote(query)
    url = f"https://s.jina.ai/{encoded_query}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "X-Retain-Images": "none",
    }
    params = {"count": min(max_results, 20)}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=20)
    except requests.exceptions.RequestException as e:
        raise SearchBackendError("jina", f"Request failed: {e}") from e

    if resp.status_code in (401, 403):
        raise SearchBackendError("jina", f"Auth failed / tokens exhausted (HTTP {resp.status_code})")
    if resp.status_code == 429:
        raise SearchBackendError("jina", "Rate limit exceeded (HTTP 429)")
    if not resp.ok:
        raise SearchBackendError("jina", f"HTTP {resp.status_code}")

    try:
        data = resp.json()
        raw_results = data.get("data", [])
    except Exception as e:
        raise SearchBackendError("jina", f"Failed to parse response: {e}") from e

    items: List[SearchResultItem] = []
    for r in raw_results:
        item = _make_item(
            title=r.get("title"),
            href=r.get("url"),
            body=r.get("description"),
        )
        if item:
            items.append(item)

    return items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_with_fallback(
    query: str,
    max_results: int = 10,
    inter_request_delay_seconds: float = 2.0,
    google_cse_api_key: Optional[str] = None,
    google_cse_cx: Optional[str] = None,
    serp_api_key: Optional[str] = None,
    firecrawl_api_key: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    jina_api_key: Optional[str] = None,
    preferred_provider: Optional[str] = "auto",
) -> List[SearchResultItem]:
    """
    Search the web using a pure-fallback provider chain.

    Tries providers in order:
        1. Google Custom Search (if google_cse_api_key and google_cse_cx are provided)
        2. SerpAPI   (if serp_api_key is provided)
        3. Firecrawl (if firecrawl_api_key is provided)
        4. Tavily    (if tavily_api_key is provided)
        5. DuckDuckGo (always tried, no key required)
        6. Jina      (if jina_api_key is provided)

    Stops and returns immediately when any provider returns ≥1 result.
    If a provider is skipped (missing key) or fails (error/rate-limit),
    a WARNING is logged and the next provider is tried.

    Returns an empty list if all providers fail.
    """
    # Build the provider sequence.  Each entry is (name, callable).
    providers = []

    if google_cse_api_key and google_cse_cx:
        providers.append((
            "google_cse",
            lambda: _search_google_cse(query, google_cse_api_key, google_cse_cx, max_results),
        ))

    if serp_api_key:
        providers.append((
            "serp",
            lambda: _search_serp(query, serp_api_key, max_results),
        ))

    if firecrawl_api_key:
        providers.append((
            "firecrawl",
            lambda: _search_firecrawl(query, firecrawl_api_key, max_results),
        ))

    if tavily_api_key:
        providers.append((
            "tavily",
            lambda: _search_tavily(query, tavily_api_key, max_results),
        ))

    # DuckDuckGo is always in the chain — no key needed
    providers.append((
        "duckduckgo",
        lambda: _search_duckduckgo(query, max_results),
    ))

    if jina_api_key:
        providers.append((
            "jina",
            lambda: _search_jina(query, jina_api_key, max_results),
        ))

    # If a specific provider is requested (and isn't 'auto'/'fallback'), restrict execution to it
    if preferred_provider and preferred_provider.strip().lower() not in ("auto", "fallback"):
        target_name = preferred_provider.strip().lower()
        matching_providers = [p for p in providers if p[0] == target_name]
        if matching_providers:
            logger.info("Search: restricting execution to explicitly requested provider '%s'", target_name)
            providers = matching_providers
        else:
            logger.warning(
                "Search: requested provider '%s' is not configured or unavailable. "
                "Falling back to default hybrid provider sequence.",
                preferred_provider,
            )

    for i, (name, search_fn) in enumerate(providers):
        try:
            logger.info("Search [%d/%d]: trying provider '%s'", i + 1, len(providers), name)
            items = search_fn()

            if items:
                logger.info(
                    "Search: '%s' returned %d results from '%s'",
                    query[:60], len(items), name,
                )
                return items

            # Provider returned no results — try the next one
            logger.warning(
                "Search: provider '%s' returned 0 results for query '%s'. "
                "Trying next provider.",
                name, query[:60],
            )

        except SearchBackendError as e:
            logger.warning("Search: provider failed, trying next. Error: %s", e)

        # Add a delay between provider calls (not after the last one)
        is_last = i == len(providers) - 1
        if not is_last and inter_request_delay_seconds > 0:
            time.sleep(inter_request_delay_seconds)

    logger.error(
        "Search: all providers exhausted for query '%s'. Returning empty list.", query[:60]
    )
    return []


def search_multi_backend(
    query: str,
    backends: Optional[List[str]] = None,
    max_results_per_backend: int = 10,
    inter_request_delay_seconds: float = 2.0,
    google_cse_api_key: Optional[str] = None,
    google_cse_cx: Optional[str] = None,
    serp_api_key: Optional[str] = None,
    firecrawl_api_key: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    jina_api_key: Optional[str] = None,
    preferred_provider: Optional[str] = "auto",
) -> List[SearchResultItem]:
    """
    Backward-compatible alias for `search_with_fallback`.

    The `backends` parameter is now ignored — the fallback chain is
    determined by which API keys are provided, not by a backend name list.
    This alias exists so existing callers that haven't been updated yet
    continue to work without modification.
    """
    if backends is not None:
        logger.debug(
            "search_multi_backend: 'backends' parameter is deprecated and ignored. "
            "Provider selection is now automatic based on available API keys."
        )
    return search_with_fallback(
        query=query,
        max_results=max_results_per_backend,
        inter_request_delay_seconds=inter_request_delay_seconds,
        google_cse_api_key=google_cse_api_key,
        google_cse_cx=google_cse_cx,
        serp_api_key=serp_api_key,
        firecrawl_api_key=firecrawl_api_key,
        tavily_api_key=tavily_api_key,
        jina_api_key=jina_api_key,
        preferred_provider=preferred_provider,
    )