"""
Stage-first pipeline orchestrator.

Runs each pipeline stage across ALL messages in the workspace before
advancing to the next stage. The workspace is a directory of JSON state
files, one per customer message (see pipeline/state_io.py for schema).

Stage execution order:
  1. extraction      — customer message → list of PartDetails
  2. customer_url    — try customer-provided URLs first (fastest path)
  3. search          — web search for each part
  4. filter          — LLM scores and picks best search result URL
  5. scrape          — fetch the selected URL's content
  6. url_inference   — LLM guesses manufacturer URL pattern (fallback)
  7. scrape_inferred — fetch the inferred URL's content
  8. browser         — Playwright automation (ONE shared browser session)
  9. attrs           — extract requested attributes from page text

Parallelism:
  Stages 3 (search) and 4 (filter) run parts in parallel using a
  ThreadPoolExecutor. The number of workers is controlled by
  PIPELINE_MAX_WORKERS in config.py (default 4).

  The browser stage (8) always runs sequentially — one part at a time
  through the same shared BrowserClient instance — to minimise
  anti-bot detection risk and avoid the overhead of multiple browser
  processes.

Workspace management:
  Controlled by PIPELINE_WORKSPACE_PERSISTENT and PIPELINE_WORKSPACE_DIR
  in config.py. When persistent=False, the caller is responsible for
  creating and cleaning up a temporary directory.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.config import get_settings
from pipeline.state_io import (
    collect_final_results,
    collect_messages_at_stage,
    collect_parts_at_stage,
    load_state,
    read_scrape_text,
    save_state,
    write_scrape_text,
    reset_failed_parts_to_pending,
)
from pipeline.router import (
    mark_part_failed,
    route_after_attrs,
    route_after_browser,
    route_after_customer_url,
    route_after_extraction,
    route_after_filter,
    route_after_scrape,
    route_after_scrape_inferred,
    route_after_search,
    route_after_url_inference,
)
from pipeline.stages import (
    run_attrs_stage,
    run_browser_stage_for_part,
    run_customer_url_stage,
    run_extraction_stage,
    run_filter_stage,
    run_scrape_stage,
    run_search_stage,
    run_url_inference_stage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage batch runners
# ---------------------------------------------------------------------------

def _run_extraction_batch(workspace: Path, message_ids: list[str] | None = None) -> None:
    """Stage 1: parse customer messages into PartDetails lists."""
    from pipeline.state_io import _part_record  # internal helper

    pending = collect_messages_at_stage(workspace, "extraction", message_ids)
    if not pending:
        logger.info("Extraction stage: no messages pending.")
        return
    logger.info("Extraction stage: processing %d message(s).", len(pending))

    for json_path, state in pending:
        customer_message = state.get("customer_message", "")
        result = run_extraction_stage(customer_message)

        if result["success"]:
            # Populate parts list in state
            state["parts"] = [
                _part_record(f"part_{i}", pd)
                for i, pd in enumerate(result["parts"])
            ]
            route_after_extraction(json_path, state, success=True)
            logger.info(
                "Extraction done for %s: %d part(s)",
                json_path.stem, len(state["parts"]),
            )
        else:
            state["extraction"]["status"] = "failed"
            state["extraction"]["error"] = result.get("error")
            save_state(json_path, state)
            logger.warning("Extraction failed for %s: %s", json_path.stem, result.get("error"))


def _run_customer_url_batch(workspace: Path, message_ids: list[str] | None = None) -> None:
    """Stage 2: try customer-provided URLs (fastest path to success)."""
    pending = collect_parts_at_stage(workspace, "customer_url", message_ids)
    if not pending:
        logger.info("Customer URL stage: no parts pending.")
        return
    logger.info("Customer URL stage: processing %d part(s).", len(pending))

    for json_path, state, part in pending:
        # If the part has no customer URLs, skip immediately
        pd = part.get("part_details", {})
        if not pd.get("product_url") and not pd.get("datasheet"):
            route_after_customer_url(json_path, state, part["part_id"], success=False)
            continue

        try:
            result = run_customer_url_stage(pd)
        except Exception as e:
            logger.exception("Unexpected error in customer_url for part %s", part["part_id"])
            result = {"success": False, "error": str(e)}

        if result["success"]:
            # Store the scraped text as a sidecar file
            write_scrape_text(json_path, part["part_id"], result["text"])
            part["customer_url"] = {
                "status": "done",
                "url": result["url"],
                "text_file": f"{json_path.stem}_{part['part_id']}_scrape.txt",
                "source": result["source"],
            }
        else:
            part["customer_url"] = {"status": "failed", "error": result.get("error")}

        save_state(json_path, state)
        route_after_customer_url(json_path, state, part["part_id"], result["success"])


def _run_search_batch(workspace: Path, message_ids: list[str] | None = None) -> None:
    """Stage 3: web search — parallelised across parts."""
    pending = collect_parts_at_stage(workspace, "search", message_ids)
    if not pending:
        logger.info("Search stage: no parts pending.")
        return
    logger.info("Search stage: processing %d part(s).", len(pending))

    settings = get_settings()
    max_workers = settings.pipeline_max_workers

    def _process(item: tuple) -> None:
        json_path, state, part = item
        try:
            result = run_search_stage(part["part_details"])
        except Exception as e:
            logger.exception("Unexpected error in search for part %s", part["part_id"])
            result = {"success": False, "query": "", "results": [], "error": str(e)}

        part["search"] = {
            "status": "done" if result["success"] else "failed",
            "query": result.get("query", ""),
            "results": result.get("results", []),
            "error": result.get("error"),
        }
        save_state(json_path, state)
        route_after_search(json_path, state, part["part_id"], result["success"])

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_process, item) for item in pending]
        for f in as_completed(futures):
            exc = f.exception()
            if exc:
                logger.error("Search worker raised: %s", exc)


def _run_filter_batch(workspace: Path, message_ids: list[str] | None = None) -> None:
    """Stage 4: LLM URL scoring — parallelised across parts."""
    pending = collect_parts_at_stage(workspace, "filter", message_ids)
    if not pending:
        logger.info("Filter stage: no parts pending.")
        return
    logger.info("Filter stage: processing %d part(s).", len(pending))

    settings = get_settings()
    max_workers = settings.pipeline_max_workers

    def _process(item: tuple) -> None:
        json_path, state, part = item
        search_data = part.get("search", {})
        try:
            result = run_filter_stage(
                part["part_details"],
                search_data.get("results", []),
                search_data.get("query", ""),
            )
        except Exception as e:
            logger.exception("Unexpected error in filter for part %s", part["part_id"])
            result = {"success": False, "url": None, "score": 0,
                      "all_scored_urls": [], "error": str(e)}

        part["filter"] = {
            "status": "done" if result["success"] else "failed",
            "url": result.get("url"),
            "score": result.get("score", 0),
            "all_scored_urls": result.get("all_scored_urls", []),
            "error": result.get("error"),
            # Carry query forward for url_inference stage
            "query": search_data.get("query", ""),
        }
        save_state(json_path, state)
        route_after_filter(
            json_path, state, part["part_id"],
            result.get("score", 0), result.get("url"),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_process, item) for item in pending]
        for f in as_completed(futures):
            exc = f.exception()
            if exc:
                logger.error("Filter worker raised: %s", exc)


def _run_scrape_batch(workspace: Path, message_ids: list[str] | None = None) -> None:
    """Stage 5: scrape the filter-selected URL."""
    pending = collect_parts_at_stage(workspace, "scrape", message_ids)
    if not pending:
        logger.info("Scrape stage: no parts pending.")
        return
    logger.info("Scrape stage: processing %d part(s).", len(pending))

    for json_path, state, part in pending:
        url = part.get("filter", {}).get("url")
        if not url:
            part["scrape"] = {"status": "failed", "error": "No URL from filter stage"}
            save_state(json_path, state)
            route_after_scrape(json_path, state, part["part_id"], success=False)
            continue

        try:
            result = run_scrape_stage(url)
        except Exception as e:
            logger.exception("Unexpected error in scrape for part %s", part["part_id"])
            result = {"success": False, "error": str(e)}

        if result["success"]:
            write_scrape_text(json_path, part["part_id"], result["text"])
            part["scrape"] = {
                "status": "done",
                "url": url,
                "text_file": f"{json_path.stem}_{part['part_id']}_scrape.txt",
            }
        else:
            part["scrape"] = {"status": "failed", "url": url, "error": result.get("error")}

        save_state(json_path, state)
        route_after_scrape(json_path, state, part["part_id"], result["success"])


def _run_url_inference_batch(workspace: Path, message_ids: list[str] | None = None) -> None:
    """Stage 6: LLM URL inference fallback."""
    pending = collect_parts_at_stage(workspace, "url_inference", message_ids)
    if not pending:
        logger.info("URL inference stage: no parts pending.")
        return
    logger.info("URL inference stage: processing %d part(s).", len(pending))

    for json_path, state, part in pending:
        filter_data = part.get("filter", {})
        try:
            result = run_url_inference_stage(part["part_details"], filter_data)
        except Exception as e:
            logger.exception("Unexpected error in url_inference for part %s", part["part_id"])
            result = {"confidence": "unknown", "url": None, "reasoning": str(e)}

        part["url_inference"] = {
            "status": "done",
            "confidence": result["confidence"],
            "url": result.get("url"),
            "reasoning": result.get("reasoning", ""),
        }
        save_state(json_path, state)
        route_after_url_inference(
            json_path, state, part["part_id"],
            result["confidence"], result.get("url"),
        )


def _run_scrape_inferred_batch(workspace: Path, message_ids: list[str] | None = None) -> None:
    """Stage 7: scrape the URL inferred by the LLM."""
    pending = collect_parts_at_stage(workspace, "scrape_inferred", message_ids)
    if not pending:
        logger.info("Scrape-inferred stage: no parts pending.")
        return
    logger.info("Scrape-inferred stage: processing %d part(s).", len(pending))

    for json_path, state, part in pending:
        url = part.get("url_inference", {}).get("url")
        if not url:
            part["scrape_inferred"] = {"status": "failed", "error": "No URL from inference"}
            save_state(json_path, state)
            route_after_scrape_inferred(json_path, state, part["part_id"], success=False)
            continue

        try:
            result = run_scrape_stage(url)
        except Exception as e:
            logger.exception("Unexpected error in scrape_inferred for part %s", part["part_id"])
            result = {"success": False, "error": str(e)}

        if result["success"]:
            write_scrape_text(json_path, part["part_id"], result["text"])
            part["scrape_inferred"] = {
                "status": "done",
                "url": url,
                "text_file": f"{json_path.stem}_{part['part_id']}_scrape.txt",
            }
        else:
            part["scrape_inferred"] = {"status": "failed", "url": url, "error": result.get("error")}

        save_state(json_path, state)
        route_after_scrape_inferred(json_path, state, part["part_id"], result["success"])


def _run_browser_batch(workspace: Path, message_ids: list[str] | None = None) -> None:
    """
    Stage 8: browser automation — ONE shared BrowserClient for ALL parts.

    Always sequential (one tab, one part at a time). The browser is
    initialised once before the loop and closed once after.
    """
    pending = collect_parts_at_stage(workspace, "browser", message_ids)
    if not pending:
        logger.info("Browser stage: no parts pending.")
        return
    from integrations.browser_client import BrowserClient
    from nodes.site_search import get_manufacturer_homepage
    from pipeline.state import PartDetails, SearchResultItem, PartSearchResult, PartStatus

    # PRE-FLIGHT: Resolve homepages for all pending parts without launching the browser
    parts_with_urls = []
    for json_path, state, part in pending:
        # Reconstruct state objects needed for homepage resolution
        part_details = PartDetails.model_validate(part["part_details"])
        search_results = part.get("search", {}).get("results")
        search_result_obj = None
        if search_results:
            result_items = [SearchResultItem.model_validate(r) for r in search_results]
            search_result_obj = PartSearchResult(
                part=part_details,
                query="",
                results=result_items,
                status=PartStatus.SEARCHED,
            )
            
        inferred_url = part.get("url_inference", {}).get("url")

        homepage_url = get_manufacturer_homepage(part_details, search_result_obj, inferred_url)
        if homepage_url:
            parts_with_urls.append((json_path, state, part, homepage_url))
        else:
            logger.warning("Part %s: could not determine manufacturer homepage", part["part_id"])
            part["browser"] = {
                "status": "failed",
                "error": "Could not determine manufacturer homepage",
            }
            save_state(json_path, state)
            route_after_browser(json_path, state, part["part_id"], success=False)

    if not parts_with_urls:
        logger.info("Browser stage: no parts resolved to a manufacturer homepage. Skipping browser launch.")
        return

    logger.info(
        "Browser stage: %d part(s) resolved to a manufacturer homepage. "
        "Launching shared browser...", len(parts_with_urls)
    )

    try:
        browser = BrowserClient()
    except Exception as e:
        logger.error("Failed to initialise browser: %s", e)
        # Mark all pending browser parts as failed
        for json_path, state, part, _ in parts_with_urls:
            mark_part_failed(json_path, state, part["part_id"],
                             f"Browser init failed: {e}")
        return

    try:
        for json_path, state, part, homepage_url in parts_with_urls:
            try:
                result = run_browser_stage_for_part(
                    browser, part["part_details"], homepage_url
                )
            except Exception as e:
                logger.exception(
                    "Unexpected error in browser stage for part %s", part["part_id"]
                )
                result = {"success": False, "error": str(e)}

            if result["success"]:
                write_scrape_text(json_path, part["part_id"], result["text"])
                part["browser"] = {
                    "status": "done",
                    "url": result.get("url"),
                    "text_file": f"{json_path.stem}_{part['part_id']}_scrape.txt",
                }
            else:
                part["browser"] = {
                    "status": "failed",
                    "error": result.get("error"),
                }

            save_state(json_path, state)
            route_after_browser(json_path, state, part["part_id"], result["success"])
            logger.info(
                "Browser stage: part %s done (success=%s)",
                part["part_id"], result["success"],
            )
    finally:
        logger.info("Browser stage: closing shared browser.")
        browser.close()


def _run_attrs_batch(workspace: Path, message_ids: list[str] | None = None) -> None:
    """Stage 9: extract requested attributes from scraped page text."""
    pending = collect_parts_at_stage(workspace, "attrs", message_ids)
    if not pending:
        logger.info("Attrs stage: no parts pending.")
        return
    logger.info("Attrs stage: processing %d part(s).", len(pending))

    for json_path, state, part in pending:
        # Find the scraped text — could come from several stages
        page_text = None
        page_url = ""

        for stage_key in ("customer_url", "scrape", "scrape_inferred", "browser"):
            stage_data = part.get(stage_key, {})
            if stage_data.get("status") == "done" and stage_data.get("text_file"):
                page_text = read_scrape_text(json_path, part["part_id"])
                page_url = stage_data.get("url", "")
                break

        if not page_text:
            route_after_attrs(
                json_path, state, part["part_id"],
                success=False,
                attributes={}, landing_page=None, source=None,
                confidence=0.0, found_on=None, match_type=None,
            )
            logger.warning("Part %s: no scraped text found for attr extraction", part["part_id"])
            continue

        try:
            result = run_attrs_stage(page_text, part["part_details"], page_url)
        except Exception as e:
            logger.exception("Unexpected error in attrs stage for part %s", part["part_id"])
            result = {"success": False, "attributes": {}, "error": str(e)}

        # Determine source from which stage provided the text
        source = None
        for stage_key, source_val in [
            ("customer_url", "customer"),
            ("scrape", "search"),
            ("scrape_inferred", "inferred"),
            ("browser", "browser"),
        ]:
            if part.get(stage_key, {}).get("status") == "done":
                source = source_val
                break

        route_after_attrs(
            json_path, state, part["part_id"],
            success=result["success"],
            attributes=result.get("attributes", {}),
            landing_page=page_url or None,
            source=source,
            confidence=result.get("confidence", 0.0),
            found_on=result.get("found_on"),
            match_type=result.get("match_type"),
        )


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(workspace: Path, message_ids: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    """
    Run all pipeline stages across all message JSON files in workspace.

    Returns:
        (completed_parts, failed_parts) — lists of result dicts.

    Stages that have already been run (status != "pending") are skipped
    automatically by each batch function. This means calling run_pipeline()
    on a workspace that was partially processed will resume from where it
    stopped.
    """
    logger.info("Pipeline started. Workspace: %s", workspace)

    # Reset any failed parts back to pending so they get retried
    reset_failed_parts_to_pending(workspace, message_ids)

    _run_extraction_batch(workspace, message_ids)
    _run_customer_url_batch(workspace, message_ids)
    _run_search_batch(workspace, message_ids)
    _run_filter_batch(workspace, message_ids)
    _run_scrape_batch(workspace, message_ids)
    _run_url_inference_batch(workspace, message_ids)
    _run_scrape_inferred_batch(workspace, message_ids)
    _run_browser_batch(workspace, message_ids)
    _run_attrs_batch(workspace, message_ids)

    completed, failed = collect_final_results(workspace)
    logger.info(
        "Pipeline finished. %d completed, %d failed.",
        len(completed), len(failed),
    )
    return completed, failed
