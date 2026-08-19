"""
Scrape integration: thin wrapper around Scrapling with Jina Reader API fallback.

Uses Scrapling's StealthyFetcher to bypass Cloudflare and other bot protections.
If Scrapling fails, falls back to Jina Reader API.
"""

from __future__ import annotations

import logging
import requests
from dataclasses import dataclass

try:
    from scrapling.fetchers import StealthyFetcher
    from scrapling.parser import Selector
    SCRAPLING_AVAILABLE = True
except ImportError:
    SCRAPLING_AVAILABLE = False

logger = logging.getLogger(__name__)

JINA_READER_BASE_URL = "https://r.jina.ai/"


@dataclass
class ScrapeResult:
    """
    The output of a single scrape attempt.

    `text` is the cleaned plain text sent to the LLM for attribute
    extraction.
    """
    text: str


class ScrapeError(Exception):
    """Raised when fetching a URL fails for any reason."""

    def __init__(self, url: str, message: str):
        super().__init__(f"Failed to scrape {url}: {message}")
        self.url = url


def clean_html_with_scrapling(html: str) -> str:
    """
    Parse HTML and extract clean plain text, removing noisy elements
    (scripts, styles, navigation, etc.) to minimize token consumption
    and give the LLM scannable text rather than raw HTML markup.

    Returns plain text, not HTML — consistent with what Jina Reader
    produces, so both scrapers feed the same kind of content to the LLM.
    """
    if not html.strip():
        return ""
    
    # Use Scrapling's Selector to parse
    s = Selector(html)
    
    # Remove elements that add noise without information value
    xpath_to_remove = (
        "//script | //style | //head | //iframe | //noscript | "
        "//svg | //img | //comment() | //header | //footer | //nav | //aside"
    )
    
    try:
        for element in s._root.xpath(xpath_to_remove):
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)
    except Exception as e:
        logger.warning("Error while cleaning HTML: %s", e)

    # Extract plain text — same output style as Jina Reader (clean, scannable
    # text), so the LLM can find part numbers and attributes reliably.
    try:
        text = s._root.text_content()
        # Collapse blank lines while preserving paragraph breaks
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)
    except Exception as e:
        logger.warning("text_content() failed, falling back to html_content: %s", e)
        return s.html_content


def scrape_url(
    url: str,
    api_key: str,
    *,
    request_timeout_seconds: float = 30.0,
    page_load_timeout_seconds: int = 20,
) -> ScrapeResult:
    """
    Fetch `url` using Scrapling (to bypass bot protection) with HTML cleaning,
    falling back to Jina Reader if Scrapling fails.

    Returns a ScrapeResult containing:
      - text: cleaned plain text for LLM attribute extraction

    Args:
        url: the target page to scrape.
        api_key: Jina API bearer token (used for fallback).
        request_timeout_seconds: HTTP timeout.
        page_load_timeout_seconds: Target page load timeout.

    Raises:
        ScrapeError: on any failure -- network error, timeout, non-2xx
            response, or an unexpected response shape. The pipeline catches
            this to record "can't access this link".
    """
    scrapling_error = None

    if SCRAPLING_AVAILABLE:
        try:
            logger.info("Attempting to fetch with Scrapling (StealthyFetcher)...")
            # Scrapling StealthyFetcher handles bot protections like Cloudflare Turnstile
            p = StealthyFetcher.fetch(
                url,
                real_chrome=True,
                headless=True,
                network_idle=True,
                timeout=page_load_timeout_seconds * 1000
            )

            if p.status == 200:
                cleaned_text = clean_html_with_scrapling(p.html_content or "")
                if cleaned_text.strip():
                    logger.info("Successfully fetched and cleaned HTML via Scrapling.")
                    return ScrapeResult(text=cleaned_text)
                else:
                    scrapling_error = "Scrapling returned empty cleaned HTML"
            elif p.status == 404:
                raise ScrapeError(url, "HTTP 404 Not Found")
            else:
                scrapling_error = f"Scrapling returned HTTP {p.status}"
        except Exception as e:
            scrapling_error = f"Scrapling fetch exception: {e}"

    if scrapling_error:
        logger.warning(scrapling_error)
        logger.info("Falling back to Jina Reader...")

    # Fallback to Jina Reader (delivers markdown — no raw HTML available)
    fetch_url = f"{JINA_READER_BASE_URL}{url}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Retain-Images": "none",
        "X-With-Iframe": "true",
        "X-With-Shadow-Dom": "true",
        "X-Timeout": str(page_load_timeout_seconds),
    }

    try:
        response = requests.get(
            fetch_url, headers=headers, timeout=request_timeout_seconds, allow_redirects=True
        )
    except requests.exceptions.Timeout as e:
        raise ScrapeError(url, "can't access this link (timeout)") from e
    except requests.exceptions.RequestException as e:
        raise ScrapeError(url, "can't access this link (request failed)") from e

    if not response.ok:
        raise ScrapeError(url, "can't access this link (Jina non-200)")

    text = response.text
    if not text.strip():
        raise ScrapeError(url, "can't access this link (empty content)")

    return ScrapeResult(text=text)