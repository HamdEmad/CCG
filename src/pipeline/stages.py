"""
Stage runner functions for the stage-first pipeline.

Each function here is a thin wrapper that:
  1. Converts the raw part_details dict (from the JSON state file) back
     into the Pydantic models the existing node logic expects.
  2. Calls the existing node logic (unchanged).
  3. Returns a plain dict of results the orchestrator can store in the
     JSON state file and pass to the router.

Design rules:
  - No routing logic here. Routing is in pipeline/router.py.
  - No state file writes here. The orchestrator writes results.
  - No browser lifecycle here. The browser is passed in as a parameter
    to run_browser_stage_for_part(); the orchestrator owns open/close.
  - Every function catches its own exceptions and returns
    {"success": False, "error": ...} instead of raising, so the
    orchestrator can route failures without needing its own try/except
    around every call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pipeline.config import get_settings
from pipeline.state import (
    CONFIDENCE_MAP,
    FilterResult,
    FoundOn,
    MatchType,
    PartDetails,
    PartFilterResult,
    PartSearchResult,
    PartStatus,
    ResultSource,
)

if TYPE_CHECKING:
    from integrations.browser_client import BrowserClient

logger = logging.getLogger(__name__)

MIN_FILTER_SCORE = 7  # kept in sync with router.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_part_details(part_details_dict: dict) -> PartDetails:
    """Deserialise a part_details dict from JSON into a PartDetails model."""
    return PartDetails.model_validate(part_details_dict)


# ---------------------------------------------------------------------------
# Stage 1 — Extraction (message → parts)
# ---------------------------------------------------------------------------

def run_extraction_stage(customer_message: str) -> dict:
    """
    Parse a customer message into a list of PartDetails.

    Returns:
        {"success": True,  "parts": [<PartDetails.model_dump()>, ...]}
        {"success": False, "error": str}
    """
    from integrations.llm_client import LLMCallError, LLMOutputError, invoke_structured
    from pipeline.prompts import EXTRACTION_SYSTEM_PROMPT

    try:
        parts = invoke_structured(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=customer_message,
            response_model=PartDetails,
            as_list=True,
        )
        logger.info("Extraction: found %d part(s) in message", len(parts))
        return {"success": True, "parts": [p.model_dump() for p in parts]}
    except (LLMCallError, LLMOutputError) as e:
        logger.warning("Extraction failed: %s", e)
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.exception("Unexpected extraction error")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Stage 2 — Customer URL check
# ---------------------------------------------------------------------------

def run_customer_url_stage(part_details: dict) -> dict:
    """
    Try customer-provided product_url and datasheet URLs.

    Returns:
        {"success": True,  "text": str, "url": str, "source": "customer"}
        {"success": False, "error": str}  (or no customer URLs at all)
    """
    from integrations.scrape_client import ScrapeError, scrape_url

    part = _to_part_details(part_details)
    settings = get_settings()

    urls_to_try = []
    if part.product_url:
        urls_to_try.append(part.product_url)
    if part.datasheet:
        urls_to_try.append(part.datasheet)

    if not urls_to_try:
        return {"success": False, "error": "No customer URLs provided"}

    for url in urls_to_try:
        logger.info("Part %s: trying customer URL %s", part.part, url)
        try:
            api_key = settings.require_jina_api_key()
            scrape_result = scrape_url(url, api_key=api_key)
            if scrape_result and scrape_result.text.strip():
                return {
                    "success": True,
                    "text": scrape_result.text,
                    "url": url,
                    "source": ResultSource.CUSTOMER.value,
                }
        except (RuntimeError, ScrapeError) as e:
            logger.warning("Part %s: customer URL scrape failed %s: %s", part.part, url, e)

    return {"success": False, "error": "All customer URLs failed to scrape"}


# ---------------------------------------------------------------------------
# Stage 3 — Web Search
# ---------------------------------------------------------------------------

def run_search_stage(part_details: dict) -> dict:
    """
    Run the multi-backend web search for one part.

    Returns:
        {"success": True,  "query": str, "results": [{"title":..,"href":..,"body":..}]}
        {"success": False, "query": str, "results": [], "error": str}
    """
    from nodes.search import build_search_query, search_part

    part = _to_part_details(part_details)
    query = build_search_query(part)

    try:
        search_result = search_part(part)
        has_results = bool(search_result.results)
        return {
            "success": has_results,
            "query": query,
            "results": [r.model_dump() for r in search_result.results],
            "error": search_result.error if not has_results else None,
        }
    except Exception as e:
        logger.exception("Unexpected error in search stage for part %s", part.part)
        return {"success": False, "query": query, "results": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Stage 4 — Filter
# ---------------------------------------------------------------------------

def run_filter_stage(part_details: dict, search_results: list, query: str) -> dict:
    """
    Ask the LLM to score search results and pick the best URL.

    Returns:
        {"success": True,  "url": str|None, "score": int, "all_scored_urls": [...]}
        {"success": False, "url": None, "score": 0, "error": str}
    """
    from nodes.filtering import filter_part_results
    from pipeline.state import PartSearchResult, SearchResultItem

    part = _to_part_details(part_details)

    result_items = [
        SearchResultItem.model_validate(r) for r in search_results
    ]
    search_result = PartSearchResult(
        part=part,
        query=query,
        results=result_items,
        status=PartStatus.SEARCHED,
    )

    try:
        filter_result = filter_part_results(search_result)
        return {
            "success": True,
            "url": filter_result.filtered.url,
            "score": filter_result.filtered.score,
            "all_scored_urls": [u.model_dump() for u in filter_result.filtered.all_scored_urls],
            "error": None,
        }
    except Exception as e:
        logger.exception("Unexpected error in filter stage for part %s", part.part)
        return {"success": False, "url": None, "score": 0, "all_scored_urls": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Stage 5 — Scrape (search-result URL)
# ---------------------------------------------------------------------------

def run_scrape_stage(url: str) -> dict:
    """
    Scrape a URL and return cleaned text.

    Returns:
        {"success": True,  "text": str}
        {"success": False, "error": str}
    """
    from integrations.scrape_client import ScrapeError, scrape_url

    settings = get_settings()
    try:
        api_key = settings.require_jina_api_key()
        scrape_result = scrape_url(url, api_key=api_key)
        if scrape_result and scrape_result.text.strip():
            return {"success": True, "text": scrape_result.text}
        return {"success": False, "error": "Scrape returned empty content"}
    except (RuntimeError, ScrapeError) as e:
        logger.warning("Scrape failed for %s: %s", url, e)
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.exception("Unexpected scrape error for %s", url)
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Stage 6 — URL Inference
# ---------------------------------------------------------------------------

def run_url_inference_stage(part_details: dict, filter_result: dict) -> dict:
    """
    Ask the LLM whether it knows this manufacturer's URL convention.

    Returns:
        {"confidence": "known_pattern"|"unknown", "url": str|None, "reasoning": str}
    """
    from nodes.url_inference import infer_part_url
    from pipeline.state import FilterResult, PartFilterResult, ScoredUrl

    part = _to_part_details(part_details)

    all_scored = [
        ScoredUrl.model_validate(u)
        for u in filter_result.get("all_scored_urls", [])
    ]
    pf_result = PartFilterResult(
        part=part,
        query=filter_result.get("query", ""),
        filtered=FilterResult(
            url=filter_result.get("url"),
            score=filter_result.get("score", 0),
            all_scored_urls=all_scored,
        ),
    )

    try:
        inference = infer_part_url(pf_result)
        return {
            "confidence": inference.inferred.confidence,
            "url": inference.inferred.url,
            "reasoning": inference.inferred.reasoning,
        }
    except Exception as e:
        logger.exception("Unexpected error in url_inference for part %s", part.part)
        return {"confidence": "unknown", "url": None, "reasoning": str(e)}


# ---------------------------------------------------------------------------
# Stage 7 — Browser (shared BrowserClient passed in)
# ---------------------------------------------------------------------------

def run_browser_stage_for_part(
    browser: "BrowserClient",
    part_details: dict,
    homepage_url: str,
) -> dict:
    """
    Run browser automation for ONE part using a SHARED browser instance.

    The browser is passed in — this function does NOT create or close it.
    The orchestrator owns the browser lifecycle.

    Returns:
        {"success": True,  "text": str, "url": str}
        {"success": False, "error": str}
    """
    from nodes.browser_search import execute_browser_search
    from pipeline.state import PartSearchResult, SearchResultItem

    part = _to_part_details(part_details)
    series = part.part_series if (
        part.part_series
        and part.part_series.strip()
        and part.part_series.strip().upper() != part.part.strip().upper()
    ) else None

    try:
        scrape_result = execute_browser_search(
            browser=browser,
            part=part,
            homepage_url=homepage_url,
            series=series,
        )
        if scrape_result and scrape_result.status == PartStatus.SCRAPED and scrape_result.scraped_text:
            return {
                "success": True,
                "text": scrape_result.scraped_text,
                "url": scrape_result.filtered.url or "",
            }
        elif scrape_result and scrape_result.status == PartStatus.FAILED_ANTI_BOT:
            return {"success": False, "error": "Anti-bot wall detected"}
        else:
            error = scrape_result.error if scrape_result else "Browser returned no result"
            return {"success": False, "error": error}
    except Exception as e:
        logger.exception("Unexpected error in browser stage for part %s", part.part)
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Stage 8 — Attribute Extraction
# ---------------------------------------------------------------------------

def run_attrs_stage(page_text: str, part_details: dict, page_url: str) -> dict:
    """
    Extract requested attributes from verified page content.

    Returns:
        {"success": True,  "attributes": dict, "source": str,
         "confidence": float, "found_on": str, "match_type": str}
        {"success": False, "error": str}
    """
    from nodes.attribute_extraction import extract_attributes

    part = _to_part_details(part_details)
    try:
        attributes = extract_attributes(page_text, part, page_url)
        if not attributes:
            logger.info("Part %s: attribute extraction returned empty dict", part.part)
            return {"success": False, "attributes": {}, "error": "No attributes extracted (soft 404 or LLM found nothing)"}

        # Confidence is always EXACT_PART / LANDING_PAGE for now
        # (series matching handled via the part_series field upstream)
        confidence = CONFIDENCE_MAP.get(
            (MatchType.EXACT_PART, FoundOn.LANDING_PAGE), 1.0
        )
        return {
            "success": True,
            "attributes": attributes,
            "found_on": FoundOn.LANDING_PAGE.value,
            "match_type": MatchType.EXACT_PART.value,
            "confidence": confidence,
        }
    except Exception as e:
        logger.exception("Unexpected error in attrs stage for part %s", part.part)
        return {"success": False, "attributes": {}, "error": str(e)}
