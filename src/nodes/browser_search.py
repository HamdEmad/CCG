"""
Browser-based search fallback node using Playwright.

IMPORTANT CHANGE from the previous version:
  execute_browser_search() no longer creates or closes a BrowserClient.
  The caller (pipeline/stages.py → orchestrator.py) passes in a shared
  BrowserClient instance that is reused across all parts needing browser
  automation. This means ONE browser init and ONE browser close per
  pipeline run, regardless of how many parts need this stage.

The two search loops (exact part, series fallback) are now handled by a
single _run_browser_session() helper to eliminate the previous copy-paste.
"""

from __future__ import annotations

import logging

from integrations.llm_client import invoke_structured
from integrations.browser_client import BrowserClient, AntiBotException
from pipeline.prompts import BROWSER_AGENT_PROMPT
from pipeline.state import (
    PartDetails,
    PartSearchResult,
    PartScrapeResult,
    PartStatus,
    ResultSource,
    FilterResult,
    BrowserActionList,
)
from integrations.scrape_client import clean_html_with_scrapling
from nodes.site_search import get_manufacturer_homepage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: single reusable session loop
# ---------------------------------------------------------------------------

def _run_browser_session(
    browser: BrowserClient,
    part: PartDetails,
    search_term: str,
    mode: str,
    max_steps: int,
) -> tuple[bool, str, str]:
    """
    Run one LLM-directed browser session searching for `search_term`.

    Args:
        browser:     The shared Playwright browser instance.
        part:        The part being searched for.
        search_term: The term to search for (full part number or series).
        mode:        Passed to BROWSER_AGENT_PROMPT.format(mode=...).
        max_steps:   Maximum LLM → action loop iterations.

    Returns:
        (done: bool, final_url: str, html_text: str)
        `done=True` means the LLM signalled the product page was found.
    """
    action_history: list[dict] = []

    for step in range(1, max_steps + 1):
        logger.info("Part %s: browser step %d/%d (%s)", part.part, step, max_steps, mode)

        snapshot = browser.get_snapshot()
        if not snapshot:
            logger.warning("Part %s: empty snapshot at step %d, stopping", part.part, step)
            break

        current_url = browser.page.url
        try:
            current_title = browser.page.title()
            body_text = clean_html_with_scrapling(browser.get_html())
            part_in_text = part.part.lower() in body_text.lower()
        except Exception as e:
            current_title = "Unknown"
            body_text = ""
            part_in_text = False
            logger.warning("Part %s: could not read page at step %d: %s", part.part, step, e)

        user_prompt = (
            f"<input>\n"
            f"Target Part Number: {part.part}\n"
            f"Manufacturer: {part.manufacturer}\n"
            f"Search Term Used: {search_term}\n"
            f"Current URL: {current_url}\n"
            f"Current Page Title: {current_title}\n"
            f"Target Part Appears on Page: {part_in_text}\n\n"
            f"Page Text (visible content):\n{body_text[:4000]}\n\n"
            f"Interactive Elements:\n{snapshot}\n"
        )
        if action_history:
            user_prompt += "\nPrevious Actions Taken:\n"
            for i, hist in enumerate(action_history, 1):
                status_str = (
                    "Succeeded" if hist["success"]
                    else f"FAILED (Error: {hist.get('error', 'unknown')})"
                )
                user_prompt += f"Step {i}: {hist['actions']} (Status: {status_str})\n"
            user_prompt += "\nWARNING: Do NOT repeat actions that failed. Try a different approach.\n"
        user_prompt += "</input>"

        try:
            action_list = invoke_structured(
                system_prompt=BROWSER_AGENT_PROMPT.format(
                    manufacturer=part.manufacturer,
                    target_part=part.part,
                    search_term=search_term,
                    mode=mode,
                ),
                user_prompt=user_prompt,
                response_model=BrowserActionList,
            )
        except Exception as e:
            logger.error("Part %s: LLM failed to generate browser actions: %s", part.part, e)
            break

        if not action_list or not action_list.actions:
            logger.warning("Part %s: LLM returned no actions at step %d", part.part, step)
            break

        # Check terminal actions before executing
        terminal = False
        done = False
        for action in action_list.actions:
            if action.action == "done":
                done = True
                terminal = True
                break
            elif action.action == "fail":
                terminal = True
                break

        if terminal:
            break

        # Execute actions
        action_dicts = [a.model_dump() for a in action_list.actions]
        success, err_msg = browser.execute_actions(action_dicts)
        action_history.append({"actions": action_dicts, "success": success, "error": err_msg})
        if not success:
            logger.warning("Part %s: step %d action(s) failed: %s", part.part, step, err_msg)

    # Capture final state
    final_url = browser.page.url
    html_text = clean_html_with_scrapling(browser.get_html())
    return done, final_url, html_text


# ---------------------------------------------------------------------------
# Public API — called by pipeline/stages.py
# ---------------------------------------------------------------------------

def execute_browser_search(
    browser: BrowserClient,
    part: PartDetails,
    homepage_url: str,
    series: str | None = None,
) -> PartScrapeResult | None:
    """
    Run an interactive browser session to find the product page.

    Args:
        browser:       Shared BrowserClient — do NOT create or close here.
        part:          The part to find.
        homepage_url:  Pre-resolved manufacturer homepage URL.
        series:        Optional part series for Stage 2 fallback.

    Returns:
        PartScrapeResult with status SCRAPED on success, or a failure/anti-bot result.
    """
    from pipeline.config import get_settings

    logger.info("Part %s: starting browser automation at %s", part.part, homepage_url)

    try:
        if not browser.navigate(homepage_url):
            return PartScrapeResult(
                part=part,
                filtered=FilterResult(url=homepage_url, score=0),
                source=ResultSource.BROWSER,
                status=PartStatus.FAILED,
                error="Initial navigation to homepage failed",
            )

        # ─── Stage 1: Search for exact part ───────────────────────────────
        done, final_url, html_text = _run_browser_session(
            browser, part,
            search_term=part.part,
            mode="exact",
            max_steps=3,
        )

        if done:
            part_found_in_text = part.part.lower() in html_text.lower()
            settings = get_settings()
            if not part_found_in_text and series and settings.enable_series_fallback:
                logger.info(
                    "Part %s: browser signalled done but part not in text; "
                    "proceeding to series fallback",
                    part.part,
                )
            else:
                logger.info("Part %s: browser stage 1 completed (exact part found)", part.part)
                return PartScrapeResult(
                    part=part,
                    filtered=FilterResult(url=final_url, score=10),
                    source=ResultSource.BROWSER,
                    scraped_text=html_text,
                    status=PartStatus.SCRAPED,
                )

        # ─── Stage 2: Series fallback (same browser session) ──────────────
        settings = get_settings()
        if series and settings.enable_series_fallback:
            logger.info(
                "Part %s: exact part not found; searching for series '%s' in same session",
                part.part, series,
            )
            if browser.navigate(homepage_url):
                done_series, final_url_series, html_series = _run_browser_session(
                    browser, part,
                    search_term=series,
                    mode="series_fallback",
                    max_steps=4,
                )
                return PartScrapeResult(
                    part=part,
                    filtered=FilterResult(url=final_url_series, score=10 if done_series else 0),
                    source=ResultSource.BROWSER,
                    scraped_text=html_series,
                    status=PartStatus.SCRAPED if done_series else PartStatus.SITE_SEARCH_NOT_FOUND,
                    error=None if done_series else "Max steps reached in series search",
                )
            else:
                logger.warning(
                    "Part %s: failed to navigate back to homepage for series search", part.part
                )

        # Both stages exhausted
        return PartScrapeResult(
            part=part,
            filtered=FilterResult(url=final_url, score=0),
            source=ResultSource.BROWSER,
            scraped_text=html_text,
            status=PartStatus.SITE_SEARCH_NOT_FOUND,
            error="Exact part not found in browser search",
        )

    except AntiBotException as e:
        logger.warning("Part %s: anti-bot wall detected: %s", part.part, e)
        return PartScrapeResult(
            part=part,
            filtered=FilterResult(url=homepage_url, score=0),
            source=ResultSource.BROWSER,
            status=PartStatus.FAILED_ANTI_BOT,
            error=str(e),
        )
    except Exception as e:
        logger.exception("Part %s: unexpected error in browser search", part.part)
        return PartScrapeResult(
            part=part,
            filtered=FilterResult(url=homepage_url, score=0),
            source=ResultSource.BROWSER,
            status=PartStatus.FAILED,
            error=str(e),
        )
    # NOTE: No browser.close() here — the caller (orchestrator) owns the lifecycle.
