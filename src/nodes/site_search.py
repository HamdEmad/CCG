"""
Site-search fallback node: query the manufacturer's internal search engine
when standard web search and URL inference fail.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse
import json

from integrations.llm_client import LLMCallError, invoke_structured
from integrations.scrape_client import scrape_url, ScrapeError
from pipeline.config import get_settings
from pipeline.prompts import (
    SITE_SEARCH_DISCOVERY_PROMPT,
    SITE_SEARCH_RESULTS_PROMPT,
    HOMEPAGE_SELECTION_PROMPT,
)
from pipeline.state import (
    PartDetails,
    PartSiteSearchResult,
    PartStatus,
    PartScrapeResult,
    PartUrlInferenceResult,
    SiteSearchDiscoveryResult,
    SiteSearchResultsResult,
    HomepageSelectionResult,
    PartSearchResult,
)

logger = logging.getLogger(__name__)


def get_manufacturer_homepage(
    part: PartDetails, search_result: PartSearchResult | None, inferred_url: str | None
) -> str | None:
    """Attempt to find the manufacturer's homepage URL."""
    # 1. If URL inference gave us a candidate, extract the domain.
    if inferred_url:
        try:
            parsed = urlparse(inferred_url)
            return f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            pass

    def extract_homepage_from_results(results_list) -> str | None:
        if not results_list:
            return None
        
        formatted_results = "\n\n".join(
            f"Title: {r.title}\nURL: {r.href}\nSnippet: {r.body}" for r in results_list
        )
        
        user_prompt = (
            f"<input>\n"
            f"Manufacturer Name: {part.manufacturer}\n\n"
            f"Search Results:\n{formatted_results}\n"
            f"</input>"
        )
        
        try:
            selection = invoke_structured(
                system_prompt=HOMEPAGE_SELECTION_PROMPT,
                user_prompt=user_prompt,
                response_model=HomepageSelectionResult,
            )
            if selection.url:
                parsed = urlparse(selection.url)
                return f"{parsed.scheme}://{parsed.netloc}"
        except Exception as e:
            logger.warning("Failed to extract homepage from search results: %s", e)
        return None

    # 2. Try to extract from the existing search results
    if search_result and search_result.results:
        logger.info("Part %s: Attempting to extract homepage from existing search results", part.part)
        homepage = extract_homepage_from_results(search_result.results)
        if homepage:
            return homepage

    # 3. Fallback explicit search for the homepage
    logger.info("Part %s: Performing explicit search for manufacturer homepage", part.part)
    from integrations.search_client import search_with_fallback
    settings = get_settings()
    try:
        explicit_results = search_with_fallback(
            query=f"{part.manufacturer} electronic official homepage",
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
        homepage = extract_homepage_from_results(explicit_results)
        if homepage:
            return homepage
    except Exception as e:
        logger.warning("Explicit homepage search failed: %s", e)

    return None


def execute_site_search_fallback(
    part: PartDetails,
    search_result: PartSearchResult | None = None,
    inferred_url: str | None = None
) -> PartScrapeResult:
    """
    Attempt to use the manufacturer's internal site search.
    """
    logger.info("Part %s: attempting Manufacturer Site Search fallback", part.part)

    # We need a dummy filtered object since PartScrapeResult requires it
    from pipeline.state import FilterResult, PartScrapeResult, ResultSource
    dummy_filtered = FilterResult(url=None, score=0)

    homepage_url = get_manufacturer_homepage(part, search_result, inferred_url)
    if not homepage_url:
        return PartScrapeResult(
            part=part,
            filtered=dummy_filtered,
            status=PartStatus.SITE_SEARCH_NOT_FOUND,
            error="Could not determine manufacturer homepage URL.",
        )

    logger.info("Part %s: using homepage %s for site search discovery", part.part, homepage_url)

    settings = get_settings()
    try:
        api_key = settings.require_jina_api_key()
        homepage_html = scrape_url(homepage_url, api_key=api_key)
    except (RuntimeError, ScrapeError) as e:
        logger.warning("Failed to scrape homepage %s: %s", homepage_url, e)
        return PartScrapeResult(
            part=part,
            filtered=dummy_filtered,
            status=PartStatus.SITE_SEARCH_NOT_FOUND,
            error=f"Failed to scrape homepage: {e}",
        )

    # 1. Discovery phase
    discovery_user_prompt = (
        f"<input>\n"
        f"- Manufacturer: {part.manufacturer}\n"
        f"- Part Number: {part.part}\n"
        f"- Homepage HTML:\n{homepage_html[:50000]}\n" # limit to avoid massive tokens
        f"</input>"
    )

    try:
        discovery = invoke_structured(
            system_prompt=SITE_SEARCH_DISCOVERY_PROMPT,
            user_prompt=discovery_user_prompt,
            response_model=SiteSearchDiscoveryResult,
        )
    except Exception as e:
        logger.warning("Site search discovery failed for %s: %s", part.part, e)
        return PartScrapeResult(
            part=part,
            filtered=dummy_filtered,
            status=PartStatus.SITE_SEARCH_NOT_FOUND,
            error=f"Discovery failed: {e}",
        )

    if not discovery.search_url_template:
        logger.info("Part %s: LLM could not discover search template", part.part)
        return PartScrapeResult(
            part=part,
            filtered=dummy_filtered,
            status=PartStatus.SITE_SEARCH_NOT_FOUND,
            error="No search URL template discovered.",
        )

    # 2. Query phase
    queries_to_try = [part.part]
    if discovery.part_family and discovery.part_family != part.part:
        queries_to_try.append(discovery.part_family)

    for query in queries_to_try:
        raw_search_url = discovery.search_url_template.replace("{query}", query)
        from urllib.parse import urljoin
        search_url = urljoin(homepage_url, raw_search_url)
        
        logger.info("Part %s: executing site search -> %s", part.part, search_url)

        try:
            results_html = scrape_url(search_url, api_key=api_key)
        except Exception as e:
            logger.warning("Failed to scrape site search results %s: %s", search_url, e)
            continue

        # 3. Analysis phase
        results_user_prompt = (
            f"<input>\n"
            f"- Part Number requested: {part.part}\n"
            f"- Query used: {query}\n"
            f"- Search Results HTML:\n{results_html[:50000]}\n"
            f"</input>"
        )

        try:
            results = invoke_structured(
                system_prompt=SITE_SEARCH_RESULTS_PROMPT,
                user_prompt=results_user_prompt,
                response_model=SiteSearchResultsResult,
            )
        except Exception as e:
            logger.warning("Site search results analysis failed for %s: %s", part.part, e)
            continue

        if results.landing_page_url:
            logger.info("Part %s: Site search found landing page -> %s", part.part, results.landing_page_url)
            
            # Use urljoin in case the LLM returned a relative URL
            from urllib.parse import urljoin
            absolute_landing_page = urljoin(search_url, results.landing_page_url)
            
            # Scrape the actual landing page HTML
            try:
                landing_page_html = scrape_url(absolute_landing_page, api_key=api_key)
                
                from pipeline.state import PartScrapeResult, ResultSource, FilterResult
                return PartScrapeResult(
                    part=part,
                    filtered=FilterResult(url=absolute_landing_page, score=10),
                    source=ResultSource.SITE_SEARCH,
                    scraped_text=landing_page_html,
                    status=PartStatus.SCRAPED,
                )
            except Exception as e:
                logger.warning("Failed to scrape site search landing page %s: %s", absolute_landing_page, e)
                return PartScrapeResult(
                    part=part,
                    filtered=dummy_filtered,
                    status=PartStatus.SITE_SEARCH_NOT_FOUND,
                    error=f"Failed to scrape found landing page: {e}"
                )

    logger.info("Part %s: Site search yielded no usable landing pages", part.part)
    return PartScrapeResult(
        part=part,
        filtered=dummy_filtered,
        status=PartStatus.SITE_SEARCH_NOT_FOUND,
        error="Site search yielded no usable landing page.",
    )
