"""
URL-inference fallback node: when the filter stage's best search result
scores below threshold, ask the LLM whether it has genuine, specific
knowledge of the manufacturer's product-page URL convention, and if so,
construct a candidate URL for this part.

This is deliberately NOT trusted blindly. A guessed URL is only as good
as the model's confidence in it, and confidence claims from an LLM are not
verification. Before any inferred URL is treated as a usable result, this
node attempts to actually scrape it -- a 404 or other failure means the
guess didn't pan out, which is recorded as a normal, expected outcome
(`PartStatus.URL_INFERRED_NOT_FOUND`), not a pipeline error.

Output is normalized into a `PartScrapeResult` with `source=INFERRED`, the
same shape `scrape_part` produces for a real search-found URL, so
`attribute_extraction.py` does not need any special-case logic to handle
results that arrived via this fallback path.
"""

from __future__ import annotations

import logging

from pipeline.config import get_settings
from integrations.llm_client import LLMCallError, LLMOutputError, invoke_structured
from integrations.scrape_client import ScrapeError, scrape_url
from pipeline.prompts import URL_INFERENCE_SYSTEM_PROMPT
from pipeline.state import (
    FilterResult,
    InferredUrlResult,
    PartFilterResult,
    PartScrapeResult,
    PartStatus,
    PartUrlInferenceResult,
    ResultSource,
)

logger = logging.getLogger(__name__)


def infer_part_url(filter_result: PartFilterResult) -> PartUrlInferenceResult:
    """
    Ask the LLM whether it knows this manufacturer's URL convention for
    part pages, and if so, construct a candidate URL.

    Does not raise: an LLM failure here is recorded as
    `PartStatus.FAILED` on the returned result rather than propagated,
    consistent with every other node in the per-part chain. A confident
    "unknown" answer from the model is NOT a failure -- it's recorded as
    `PartStatus.URL_INFERENCE_UNKNOWN`, a normal outcome.
    """
    part = filter_result.part
    user_prompt = (
        f"Part number: {part.part}\n"
        f"Manufacturer: {part.manufacturer}\n"
    )

    # Inject high-quality manufacturer URLs from the filter stage as structural examples.
    # score > 0 guarantees non-PDF and non-distributor (those receive -100 in the rubric).
    good_urls = sorted(
        [
            su for su in filter_result.filtered.all_scored_urls 
            if su.score > -50 and not su.url.lower().endswith(".pdf")
        ],
        key=lambda su: su.score,
        reverse=True,
    )
    if good_urls:
        urls_block = "\n".join(f"- {su.url}" for su in good_urls[:5])
        print("---------------------------------------------------")
        print(urls_block)
        print("---------------------------------------------------")
        user_prompt += (
            f"\nWeb search returned these manufacturer URLs for related products"
            f" (use them to deduce the URL structure for the given part):\n{urls_block}\n"
        )


    try:
        inferred = invoke_structured(
            system_prompt=URL_INFERENCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=InferredUrlResult,
        )
    except (LLMCallError, LLMOutputError) as e:
        logger.warning("URL inference failed for part %s: %s", part.part, e)
        return PartUrlInferenceResult(
            part=part,
            inferred=InferredUrlResult(url=None, confidence="unknown", reasoning=""),
            status=PartStatus.FAILED,
            error=str(e),
        )

    if inferred.confidence != "known_pattern" or not inferred.url:
        logger.info(
            "URL inference: no confident pattern for manufacturer %s (part %s)",
            part.manufacturer,
            part.part,
        )
        return PartUrlInferenceResult(
            part=part,
            inferred=inferred,
            status=PartStatus.URL_INFERENCE_UNKNOWN,
        )

    import urllib.parse
    import pathlib

    parsed_url = urllib.parse.urlparse(inferred.url)
    ext = pathlib.Path(parsed_url.path).suffix.lower()
    restricted_exts = {".pdf", ".doc", ".docx", ".xls", ".xlsx"}
    
    if ext in restricted_exts:
        logger.info(
            "URL inference: rejected inferred URL for part %s because it points to a restricted file type (%s): %s",
            part.part,
            ext,
            inferred.url,
        )
        return PartUrlInferenceResult(
            part=part,
            inferred=InferredUrlResult(
                url=None, 
                confidence="unknown", 
                reasoning=f"File downloads ({ext}) are not allowed"
            ),
            status=PartStatus.URL_INFERENCE_UNKNOWN,
        )

    logger.info(
        "URL inference: candidate URL for part %s -> %s (%s)",
        part.part,
        inferred.url,
        inferred.reasoning,
    )
    return PartUrlInferenceResult(
        part=part,
        inferred=inferred,
        status=PartStatus.PENDING,  # verification decides the real outcome
    )


def verify_and_scrape_inferred_url(
    inference_result: PartUrlInferenceResult,
) -> PartScrapeResult:
    """
    Attempt to scrape the inferred URL to confirm it actually resolves to
    real content before treating it as a usable result for this part.

    This is the step that turns "the model thinks this URL exists" into
    "we confirmed this URL exists." A failure here (404, timeout, etc.) is
    recorded as `PartStatus.URL_INFERRED_NOT_FOUND` -- the inference
    attempt simply didn't pan out, which is an expected outcome for this
    fallback path, not an error to alarm on.
    """
    part = inference_result.part

    # Carry forward non-actionable outcomes (unknown pattern, or the LLM
    # call itself failed) without attempting a scrape -- there's no URL
    # to verify in either case.
    if inference_result.status in (
        PartStatus.URL_INFERENCE_UNKNOWN,
        PartStatus.FAILED,
    ):
        return PartScrapeResult(
            part=part,
            filtered=FilterResult(url=inference_result.inferred.url, score=0),
            source=ResultSource.INFERRED,
            scraped_text=None,
            status=inference_result.status,
            error=inference_result.error or inference_result.inferred.reasoning,
        )

    url = inference_result.inferred.url
    settings = get_settings()

    try:
        api_key = settings.require_jina_api_key()
        text = scrape_url(url, api_key=api_key)
    except (RuntimeError, ScrapeError) as e:
        logger.info(
            "Inferred URL did not resolve for part %s (%s): %s", part.part, url, e
        )
        return PartScrapeResult(
            part=part,
            filtered=FilterResult(url=url, score=0),
            source=ResultSource.INFERRED,
            scraped_text=None,
            status=PartStatus.URL_INFERRED_NOT_FOUND,
            error=str(e),
        )

    logger.info("Inferred URL verified for part %s: %s", part.part, url)
    return PartScrapeResult(
        part=part,
        # Scored as a full-confidence manual match (10) only AFTER
        # verification succeeded -- the score field downstream is used by
        # nothing but logging/reporting at this point, since
        # attribute_extraction.py branches on `status`, not on score, but
        # keeping it meaningful avoids a misleading "0" on a result that
        # actually worked.
        filtered=FilterResult(url=url, score=10),
        source=ResultSource.INFERRED,
        scraped_text=text,
        status=PartStatus.URL_INFERRED_VERIFIED,
    )