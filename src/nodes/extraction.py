"""
Extraction node: customer message -> list of structured `PartDetails`.

This is the graph's entry node. It runs once per pipeline invocation, not
once per part (there are no parts yet at this point -- that's exactly
what this node produces).

The call shape here was verified directly against the live LLM before
being wired in:

    parts = invoke_structured(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=customer_message,
        response_model=PartDetails,
        as_list=True,
    )

`as_list=True` is required because `EXTRACTION_SYSTEM_PROMPT` asks for a
JSON array (one customer message commonly contains many part requests).
Calling this with `invoke_raw_json` or without `as_list=True` will raise
-- see the `LLMOutputError: Expected a JSON object but got list` issue
this was built to avoid.
"""

from __future__ import annotations

import logging

from integrations.llm_client import LLMCallError, LLMOutputError, invoke_structured
from pipeline.prompts import EXTRACTION_SYSTEM_PROMPT
from pipeline.state import PartDetails

logger = logging.getLogger(__name__)


def extract_parts(state) -> dict:
    """
    Graph node: extract every part request from `state.customer_request`.

    Returns `{"parts": [...]}` to merge into `PipelineState.parts`.

    Unlike `process_part`, this node is allowed to raise. There is no
    "one part's failure shouldn't affect others" concern here yet --
    extraction either succeeds for the whole message or it doesn't, and
    if it doesn't, there's nothing downstream to isolate. LangGraph's
    `retry_policy` (configured in graph.py) will retry this node on
    failure; if retries are exhausted, the run fails outright, which is
    the correct behavior for a stage that has no partial-success notion.
    """
    customer_request = state.customer_request

    try:
        parts = invoke_structured(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=customer_request,
            response_model=PartDetails,
            as_list=True,
        )
    except (LLMCallError, LLMOutputError):
        logger.exception("Extraction failed for customer request")
        raise

    logger.info("Extracted %d part(s) from customer request", len(parts))
    return {"parts": parts}