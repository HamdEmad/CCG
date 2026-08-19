"""
Filtering node: given one part's search results, ask the LLM to score
them and pick the single best URL (see `prompts.FILTER_SYSTEM_PROMPT` for
the full scoring rubric).
"""

from __future__ import annotations

import json
import logging
import re

from integrations.llm_client import LLMCallError, LLMOutputError, invoke_structured
from pipeline.prompts import FILTER_SYSTEM_PROMPT
from pipeline.state import FilterResult, PartSearchResult, PartFilterResult, PartStatus

logger = logging.getLogger(__name__)


def filter_part_results(search_result: PartSearchResult) -> PartFilterResult:
    """
    Score `search_result.results` and select the best URL for this part.

    Does not raise: if the LLM call or parsing fails, returns a
    `PartFilterResult` with `status=PartStatus.FAILED` and `filtered.score
    = 0`, so `process_part` can decide whether to skip scraping (it will,
    since the score-threshold check downstream treats a 0 score the same
    as "nothing usable found") without needing its own try/except around
    this call.
    """
    non_pdf_results = []
    for r in search_result.results:
        url_lower = (r.href or "").lower()
        if re.search(r'\.pdf(?:[?#]|$)', url_lower):
            logger.info("Filtering: Excluded PDF URL: %s", r.href)
            continue
        non_pdf_results.append(r)

    if not non_pdf_results:
        logger.info("Filtering: No non-PDF results found for %s, skipping LLM.", search_result.part.part)
        return PartFilterResult(
            part=search_result.part,
            query=search_result.query,
            filtered=FilterResult(url=None, score=0),
            status=PartStatus.FILTERED,
        )

    user_prompt = (
        f"<input>\n"
        f"- The customer request is: {search_result.part.message}.\n"
        f"- The search query is: {search_result.query}.\n"
        f"- The search results are: "
        f"{json.dumps([r.model_dump() for r in non_pdf_results], ensure_ascii=False)}.\n"
        f"</input>"
    )

    try:
        filtered = invoke_structured(
            system_prompt=FILTER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=FilterResult,
        )
    except (LLMCallError, LLMOutputError) as e:
        logger.warning("Filtering failed for part %s: %s", search_result.part.part, e)
        return PartFilterResult(
            part=search_result.part,
            query=search_result.query,
            filtered=FilterResult(url=None, score=0),
            status=PartStatus.FAILED,
            error=str(e),
        )

    return PartFilterResult(
        part=search_result.part,
        query=search_result.query,
        filtered=filtered,
        status=PartStatus.FILTERED,
    )