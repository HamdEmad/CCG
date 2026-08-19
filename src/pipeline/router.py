"""
Routing table for the stage-first pipeline.

This is the ONLY module that writes stage status fields (setting a stage
to "pending", "done", "failed", or "skipped") and decides which stage runs
next for a given part. No stage runner should ever write status fields
directly -- they return result dicts; routing logic lives here.

Design:
  Each route_after_* function receives the state dict and part_id,
  mutates the relevant stage keys, and calls save_state() atomically.
  The caller (orchestrator) does not need to know the routing logic.

Routing summary:
  customer_url success  → skip all other tiers, attrs pending
  customer_url miss     → search pending

  search has results    → filter pending
  search empty          → filter skipped, url_inference pending

  filter score >= 7     → scrape pending
  filter score < 7      → scrape skipped, url_inference pending

  scrape success        → attrs pending, url_inference/browser skipped
  scrape fail           → url_inference pending

  url_inference known   → scrape_inferred pending, browser skipped
  url_inference unknown → url_inference skipped, browser pending

  scrape_inferred ok    → attrs pending, browser skipped
  scrape_inferred fail  → browser pending

  browser success       → attrs pending
  browser fail          → part permanently failed

  attrs done            → final_status = "completed"
  attrs fail            → final_status = "failed"
"""

from __future__ import annotations

import logging
from pathlib import Path

from pipeline.state_io import save_state

logger = logging.getLogger(__name__)

MIN_FILTER_SCORE = 7  # below this, skip scraping and try inference instead


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_part(state: dict, part_id: str) -> dict:
    for p in state["parts"]:
        if p["part_id"] == part_id:
            return p
    raise KeyError(f"part_id '{part_id}' not found in state")


def _skip_stages(part: dict, *stage_names: str) -> None:
    for name in stage_names:
        part[name]["status"] = "skipped"


# ---------------------------------------------------------------------------
# After extraction (top-level message stage)
# ---------------------------------------------------------------------------

def route_after_extraction(
    json_path: Path, state: dict, success: bool
) -> None:
    """
    After extraction runs on a message, update top-level status and
    initialise each new part record's stages.

    If extraction failed, the whole message is marked failed (no parts to process).
    If it succeeded, all part search stages are set to 'pending'.
    """
    state["extraction"]["status"] = "done" if success else "failed"
    if success:
        for part in state["parts"]:
            # customer_url check first; if no customer URL is provided,
            # customer_url will be skipped by the orchestrator automatically
            # after seeing no product_url or datasheet in part_details.
            pass  # stages already initialised to "pending" by state_io._part_record
    save_state(json_path, state)


# ---------------------------------------------------------------------------
# After customer URL tier
# ---------------------------------------------------------------------------

def route_after_customer_url(
    json_path: Path, state: dict, part_id: str, success: bool
) -> None:
    """
    Customer URL found something usable → skip every other tier.
    No customer URL (or scrape failed) → mark search pending (already is by default).
    """
    part = _get_part(state, part_id)
    part["customer_url"]["status"] = "done" if success else "skipped"

    if success:
        _skip_stages(part, "search", "filter", "scrape",
                     "url_inference", "scrape_inferred", "browser")
        part["attrs"]["status"] = "pending"
        logger.debug("Part %s routed: customer_url → attrs", part_id)
    else:
        # search is already "pending" — nothing to change
        logger.debug("Part %s routed: customer_url miss → search", part_id)

    save_state(json_path, state)


# ---------------------------------------------------------------------------
# After search
# ---------------------------------------------------------------------------

def route_after_search(
    json_path: Path, state: dict, part_id: str, has_results: bool
) -> None:
    """
    Search returned results → filter pending.
    Search returned nothing → filter + scrape skipped, url_inference pending.
    """
    part = _get_part(state, part_id)
    part["search"]["status"] = "done" if has_results else "failed"

    if has_results:
        part["filter"]["status"] = "pending"
        logger.debug("Part %s routed: search → filter", part_id)
    else:
        _skip_stages(part, "filter", "scrape")
        part["url_inference"]["status"] = "pending"
        logger.debug("Part %s routed: search empty → url_inference", part_id)

    save_state(json_path, state)


# ---------------------------------------------------------------------------
# After filter
# ---------------------------------------------------------------------------

def route_after_filter(
    json_path: Path, state: dict, part_id: str, score: int, url: str | None
) -> None:
    """
    Score >= threshold and URL found → scrape pending.
    Score < threshold or no URL → scrape skipped, url_inference pending.
    """
    part = _get_part(state, part_id)
    part["filter"]["status"] = "done"

    if url and score >= MIN_FILTER_SCORE:
        part["scrape"]["status"] = "pending"
        logger.debug("Part %s routed: filter (score=%d) → scrape", part_id, score)
    else:
        _skip_stages(part, "scrape")
        part["url_inference"]["status"] = "pending"
        logger.debug(
            "Part %s routed: filter score=%d below threshold → url_inference",
            part_id, score,
        )

    save_state(json_path, state)


# ---------------------------------------------------------------------------
# After scrape (search-result URL)
# ---------------------------------------------------------------------------

def route_after_scrape(
    json_path: Path, state: dict, part_id: str, success: bool
) -> None:
    """
    Scrape got text → attrs pending, everything else skipped.
    Scrape failed → url_inference pending.
    """
    part = _get_part(state, part_id)
    part["scrape"]["status"] = "done" if success else "failed"

    if success:
        _skip_stages(part, "url_inference", "scrape_inferred", "browser")
        part["attrs"]["status"] = "pending"
        logger.debug("Part %s routed: scrape → attrs", part_id)
    else:
        part["url_inference"]["status"] = "pending"
        logger.debug("Part %s routed: scrape failed → url_inference", part_id)

    save_state(json_path, state)


# ---------------------------------------------------------------------------
# After URL inference
# ---------------------------------------------------------------------------

def route_after_url_inference(
    json_path: Path,
    state: dict,
    part_id: str,
    confidence: str,
    url: str | None,
) -> None:
    """
    Inference returned known_pattern + url → scrape_inferred pending.
    Unknown or no url → scrape_inferred skipped, browser pending.
    """
    part = _get_part(state, part_id)
    part["url_inference"]["status"] = "done"

    if confidence == "known_pattern" and url:
        part["scrape_inferred"]["status"] = "pending"
        _skip_stages(part, "browser")
        logger.debug("Part %s routed: url_inference → scrape_inferred", part_id)
    else:
        _skip_stages(part, "scrape_inferred")
        part["browser"]["status"] = "pending"
        logger.debug("Part %s routed: url_inference unknown → browser", part_id)

    save_state(json_path, state)


# ---------------------------------------------------------------------------
# After scrape (inferred URL)
# ---------------------------------------------------------------------------

def route_after_scrape_inferred(
    json_path: Path, state: dict, part_id: str, success: bool
) -> None:
    """
    Inferred URL scraped successfully → attrs pending, browser skipped.
    Inferred URL failed → browser pending.
    """
    part = _get_part(state, part_id)
    part["scrape_inferred"]["status"] = "done" if success else "failed"

    if success:
        _skip_stages(part, "browser")
        part["attrs"]["status"] = "pending"
        logger.debug("Part %s routed: scrape_inferred → attrs", part_id)
    else:
        part["browser"]["status"] = "pending"
        logger.debug("Part %s routed: scrape_inferred failed → browser", part_id)

    save_state(json_path, state)


# ---------------------------------------------------------------------------
# After browser
# ---------------------------------------------------------------------------

def route_after_browser(
    json_path: Path, state: dict, part_id: str, success: bool
) -> None:
    """
    Browser found the page → attrs pending.
    Browser failed → part permanently failed.
    """
    part = _get_part(state, part_id)
    part["browser"]["status"] = "done" if success else "failed"

    if success:
        part["attrs"]["status"] = "pending"
        logger.debug("Part %s routed: browser → attrs", part_id)
    else:
        part["final_status"] = "failed"
        part["final_result"] = {
            "error": part["browser"].get("error", "Browser automation failed")
        }
        logger.debug("Part %s: all tiers exhausted, marking failed", part_id)

    save_state(json_path, state)


# ---------------------------------------------------------------------------
# After attribute extraction
# ---------------------------------------------------------------------------

def route_after_attrs(
    json_path: Path,
    state: dict,
    part_id: str,
    success: bool,
    attributes: dict,
    landing_page: str | None,
    source: str | None,
    confidence: float,
    found_on: str | None,
    match_type: str | None,
) -> None:
    """
    Attribute extraction done → write final_result and mark completed.
    If extraction returned an empty dict (soft 404 / LLM found nothing),
    the part is still marked completed (we got to the page, just no attrs).
    """
    part = _get_part(state, part_id)
    part["attrs"]["status"] = "done" if success else "failed"

    if success:
        part["final_status"] = "completed"
        part["final_result"] = {
            "attributes": attributes,
            "landing_page": landing_page,
            "source": source,
            "confidence": confidence,
            "found_on": found_on,
            "match_type": match_type,
        }
        logger.debug("Part %s: completed successfully", part_id)
    else:
        part["final_status"] = "failed"
        part["final_result"] = {"error": "Attribute extraction failed"}
        logger.debug("Part %s: attribute extraction failed, marking failed", part_id)

    save_state(json_path, state)


# ---------------------------------------------------------------------------
# Hard failure — used when an unexpected exception escapes a stage
# ---------------------------------------------------------------------------

def mark_part_failed(
    json_path: Path, state: dict, part_id: str, error: str
) -> None:
    """Mark a part as permanently failed due to an unexpected error."""
    try:
        part = _get_part(state, part_id)
        part["final_status"] = "failed"
        part["final_result"] = {"error": error}
        save_state(json_path, state)
        logger.warning("Part %s permanently failed: %s", part_id, error)
    except Exception as e:
        logger.error("Failed to mark part %s as failed: %s", part_id, e)
