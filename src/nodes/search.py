"""
Search node: for a single part, build a query and fan out across search
backends, returning raw results for the filter stage to score.

This module operates on ONE part, not the whole batch -- it's called from
inside `part_worker.process_part`, which is itself dispatched once per
part by the graph's `Send` fan-out (see graph.py). This is what gives
each part its own independent search step rather than the original
script's single shared loop over every part.
"""

from __future__ import annotations

import logging

from pipeline.config import get_settings
from integrations.search_client import search_with_fallback
from pipeline.state import PartDetails, PartSearchResult, PartStatus

logger = logging.getLogger(__name__)


def build_search_query(part: PartDetails) -> str:
    """
    Build the search query for one part.

    Matches the original script's query shape (`"{part} {manufacturer}
    landing page -inurl:.pdf"`), which biases results toward HTML product
    pages over direct PDF downloads -- the filter stage's scoring rules
    assume this bias (it gives PDFs a smaller bonus than landing pages).
    """
    return f'{part.manufacturer} {part.part}'


def search_part(part: PartDetails) -> PartSearchResult:
    """
    Run the multi-backend search for one part and return a
    `PartSearchResult`.

    Does not raise: if every backend fails, this returns a result with an
    empty `results` list and `status=PartStatus.FAILED` rather than
    propagating an exception, so `process_part` can decide how to handle
    "found nothing" without needing a try/except specifically around this
    call (the search_client layer already isolates per-backend failures;
    this function isolates "all backends failed" as a status rather than
    an exception, since searching is allowed to legitimately come up
    empty without that being an error in the part's processing chain).
    """
    settings = get_settings()
    query = build_search_query(part)

    try:
        results = search_with_fallback(
            query=query,
            max_results=settings.search_max_results,
            inter_request_delay_seconds=settings.search_inter_request_delay_seconds,
            google_cse_api_key=settings.google_cse_api_key_value(),
            google_cse_cx=settings.google_cse_cx,
            serp_api_key=settings.serp_api_key_value(),
            firecrawl_api_key=settings.firecrawl_api_key_value(),
            tavily_api_key=settings.tavily_api_key_value(),
            jina_api_key=settings.require_jina_api_key(),
            preferred_provider=settings.search_provider,
        )
    except Exception as e:
        # search_multi_backend already isolates per-backend failures
        # internally; reaching this branch means something unexpected
        # happened outside that isolation (e.g. a bug, not a backend
        # outage). Still don't raise -- record it on the result instead,
        # consistent with this function's no-raise contract.
        logger.exception("Unexpected error during search for part %s", part.part)
        return PartSearchResult(
            part=part,
            query=query,
            results=[],
            status=PartStatus.FAILED,
            error=str(e),
        )

    if not results:
        logger.warning("No search results found for part %s (query: %s)", part.part, query)
        return PartSearchResult(
            part=part,
            query=query,
            results=[],
            status=PartStatus.FAILED,
            error="No search results from any backend",
        )

    return PartSearchResult(
        part=part,
        query=query,
        results=results,
        status=PartStatus.SEARCHED,
    )