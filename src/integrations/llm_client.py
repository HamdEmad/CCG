"""
LLM client wrapper for the component lookup pipeline.

This is the single place that talks to the language model. Every node that
needs an LLM call (extraction, filtering, attribute extraction) goes
through `invoke_structured` here rather than constructing its own
`ChatOpenAI` instance or repeating the parse/validate dance inline.

Fixes applied here relative to the original script (see code review):
- The model client is constructed once, not implicitly via module-level
  globals scattered across cells.
- LLM calls have an explicit request timeout and a retry policy with
  backoff, instead of a single bare `.invoke()` call with no failure
  handling. (Note: LangGraph's per-node `retry_policy`, set up in
  `graph.py`, also retries the *node* if it raises -- this module's
  `tenacity` retry handles retrying the underlying *API call itself*, which
  is a finer-grained, faster retry than re-running an entire node. The two
  are complementary, not redundant: this layer absorbs transient API
  hiccups quickly; the graph-level policy is the backstop if this layer's
  retries are exhausted.)
- LLM output is parsed with `json_repair` *and then validated* against a
  caller-supplied Pydantic model. The original script only repaired JSON
  and trusted the shape; a malformed or unexpected response now raises a
  clear `LLMOutputError` instead of silently propagating a dict with
  missing or wrong-typed keys into downstream code.
"""

from __future__ import annotations

import json
import logging
import warnings
import re
from typing import Any, Dict, List, Optional, Type, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
import json_repair

from pipeline.config import get_settings

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMOutputError(Exception):
    """
    Raised when the model's response cannot be parsed as JSON, or parses
    but does not match the Pydantic schema the caller expected.

    Carries the raw response text so callers can log it for debugging
    without needing to re-fetch or re-parse anything.
    """

    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


class LLMCallError(Exception):
    """
    Raised when the underlying API call itself fails after all retries are
    exhausted (network error, timeout, rate limit, etc.), as opposed to
    succeeding but returning unusable content.
    """


def _build_default_client() -> ChatOpenAI:
    """
    Construct the chat client from centralized settings (see config.py).

    This used to read `os.environ` directly here, before config.py
    existed. Now that it does, this is the only place that should
    construct an LLM client, and it goes through `get_settings()` so
    there's a single source of truth for configuration -- no module reads
    environment variables on its own anymore.

    Raises a clear error via `Settings.require_llm_api_key()` if no key is
    configured, rather than constructing a client that fails opaquely on
    first use.
    """
    from pipeline.config import get_settings

    settings = get_settings()
    api_key = settings.require_llm_api_key()

    return ChatOpenAI(
        model=settings.llm_model_name,
        api_key=api_key,
        base_url=settings.llm_base_url,
        temperature=0,
        # 90s: Scrapling's StealthyFetcher returns full rendered HTML which
        # is significantly larger than Jina's lean markdown output. The extra
        # headroom prevents consistent timeouts on content-heavy product pages.
        # Tenacity retries are handled below, so the SDK's own retries are off.
        timeout=90.0,
        max_retries=0,  # retries are handled explicitly below via tenacity,
        # so the underlying SDK's own retry loop is disabled to avoid
        # double-retrying the same failure with two different backoff
        # schedules stacked on top of each other.
    )


# Module-level client, built lazily on first use rather than at import time.
# Building it at import time would mean importing this module fails (or
# silently constructs a half-broken client) if OPENAI_API_KEY isn't set
# yet, which is disruptive for things like running tests that don't need
# a real client at all.
_client: ChatOpenAI | None = None


# ---------------------------------------------------------------------------
# Token usage accumulator
# ---------------------------------------------------------------------------
# Tracks cumulative input/output/total tokens across all LLM calls made
# during one pipeline run. Call reset_token_usage() at the start of each
# run and get_token_usage() at the end to read the totals.

_token_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def reset_token_usage() -> None:
    """Reset all token counters to zero. Call once before each pipeline run."""
    _token_usage["input_tokens"] = 0
    _token_usage["output_tokens"] = 0
    _token_usage["total_tokens"] = 0


def get_token_usage() -> dict[str, int]:
    """Return a snapshot of the current cumulative token counts."""
    return dict(_token_usage)


def _accumulate_token_usage(response: BaseMessage) -> None:
    """
    Extract token counts from a LangChain response object and add them to
    the module-level accumulator.

    Checks two locations in order (both are present depending on the
    LangChain / API version in use):
      1. response.usage_metadata  -- newer LangChain, keys: input_tokens,
                                     output_tokens, total_tokens
      2. response.response_metadata["token_usage"]  -- older / OpenAI-compat,
                                     keys: prompt_tokens, completion_tokens

    If neither is available the call is silently ignored -- token counting
    is best-effort and should never break a pipeline run.
    """
    try:
        # --- Option 1: usage_metadata (newer LangChain) ---
        usage = getattr(response, "usage_metadata", None)
        if usage and isinstance(usage, dict):
            _token_usage["input_tokens"] += int(usage.get("input_tokens", 0))
            _token_usage["output_tokens"] += int(usage.get("output_tokens", 0))
            _token_usage["total_tokens"] += int(usage.get("total_tokens", 0))
        else:
            # --- Option 2: response_metadata["token_usage"] (OpenAI-compat) ---
            meta = getattr(response, "response_metadata", None) or {}
            tu = meta.get("token_usage", {})
            if tu:
                inp = int(tu.get("prompt_tokens", 0))
                out = int(tu.get("completion_tokens", 0))
                _token_usage["input_tokens"] += inp
                _token_usage["output_tokens"] += out
                _token_usage["total_tokens"] += inp + out
            
        logger.debug("Token usage updated: %s", _token_usage)
    except Exception as e:
        logger.warning("Failed to accumulate tokens: %s", e)


def get_client() -> ChatOpenAI:
    global _client
    if _client is None:
        _client = _build_default_client()
    return _client


def set_client(client: ChatOpenAI) -> None:
    """
    Override the module-level client. Intended for tests, where a fake or
    mocked `ChatOpenAI` should be substituted instead of the real one.
    """
    global _client
    _client = client


import time
import threading

_llm_lock = threading.Lock()
_last_call_time = 0.0


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _invoke_with_retry(messages: List[BaseMessage]) -> str:
    """
    Call the LLM and return the raw text content, retrying transient
    failures with exponential backoff.

    Any exception still raised after the final attempt propagates as-is;
    `invoke_structured` below is responsible for wrapping it as a
    `LLMCallError` with context about which call failed.
    """
    global _last_call_time
    settings = get_settings()
    min_interval = settings.llm_min_interval_seconds

    if min_interval > 0:
        with _llm_lock:
            now = time.time()
            elapsed = now - _last_call_time
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                logger.info("Rate limit: sleeping for %.2f seconds to maintain LLM min interval", sleep_time)
                time.sleep(sleep_time)
            _last_call_time = time.time()

    response = get_client().invoke(messages)
    _accumulate_token_usage(response)
    return response.content


def _extract_json_text(raw_text: str) -> str:
    """
    Strip markdown formatting and extract the JSON substring from raw model output.
    """
    text_to_parse = raw_text
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_text, re.DOTALL)
    if match:
        text_to_parse = match.group(1)
    else:
        start_idx = min(
            raw_text.find("{") if "{" in raw_text else len(raw_text),
            raw_text.find("[") if "[" in raw_text else len(raw_text),
        )
        end_idx = max(raw_text.rfind("}"), raw_text.rfind("]"))
        if start_idx < len(raw_text) and end_idx != -1 and end_idx >= start_idx:
            text_to_parse = raw_text[start_idx : end_idx + 1]
    return text_to_parse


def _unwrap_schema_envelope(item: object) -> object:
    """
    Detect and unwrap the JSON Schema envelope hallucination.

    When shown a JSON Schema in the prompt (which has top-level keys
    "description", "properties", "required", ...), the model occasionally
    returns an object in that same shape instead of a flat instance:

        { "description": "...", "properties": { "part": "AVS-2214", ... } }

    This helper detects that pattern and unwraps `properties` transparently
    so Pydantic validation can proceed on the actual data.
    """
    if (
        isinstance(item, dict)
        and isinstance(item.get("properties"), dict)
        # Guard: a legitimate flat instance might coincidentally have a
        # "properties" key, so only unwrap when the other schema-envelope
        # indicators are also present.
        and ("description" in item or "required" in item or "title" in item)
        and "$schema" not in item  # don't unwrap actual JSON Schema documents
    ):
        logger.debug(
            "LLM returned JSON Schema envelope instead of flat instance; "
            "unwrapping 'properties' automatically."
        )
        return item["properties"]
    return item


def invoke_structured(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[ModelT],
    *,
    as_list: bool = False,
) -> ModelT | List[ModelT]:
    """
    Call the LLM with a system/user prompt pair and return output validated
    against `response_model`.

    Args:
        system_prompt: the system instruction (typically one of the
            constants in `prompts.py`).
        user_prompt: the user-turn content -- the specific input for this
            call (customer message, search results, scraped text, etc.).
        response_model: a Pydantic model the parsed JSON must validate
            against.
        as_list: if True, the model is expected to return a JSON array,
            and each element is validated against `response_model`
            individually. Use this for the extraction prompt, which
            returns one or more parts per customer message.

    Raises:
        LLMCallError: the API call itself failed after all retries.
        LLMOutputError: the call succeeded, but the response could not be
            parsed as JSON, or did not match `response_model` (or, when
            `as_list=True`, was not a JSON array).
    """
    messages: List[BaseMessage] = [
        SystemMessage(content=system_prompt.strip()),
        HumanMessage(content=user_prompt.strip()),
    ]

    try:
        raw_text = _invoke_with_retry(messages)
    except Exception as e:
        raise LLMCallError(f"LLM call failed after retries: {e}") from e

    text_to_parse = _extract_json_text(raw_text)

    try:
        parsed = json_repair.loads(text_to_parse)
    except Exception as e:
        logger.error("json_repair failed on raw text:\n%s", raw_text)
        raise LLMOutputError(
            f"Could not parse model output as JSON: {e}", raw_response=raw_text
        ) from e

    if as_list:
        if not isinstance(parsed, list):
            logger.error("Expected list but got %s on raw text:\n%s", type(parsed).__name__, raw_text)
            raise LLMOutputError(
                f"Expected a JSON array but got {type(parsed).__name__}",
                raw_response=raw_text,
            )
        try:
            return [response_model.model_validate(_unwrap_schema_envelope(item)) for item in parsed]
        except ValidationError as e:
            logger.error("Validation error on list items:\n%s\nRaw text:\n%s", e, raw_text)
            raise LLMOutputError(
                f"Array element did not match {response_model.__name__}: {e}",
                raw_response=raw_text,
            ) from e

    try:
        return response_model.model_validate(parsed)
    except ValidationError as e:
        logger.error("Validation error for %s:\n%s\nRaw text:\n%s", response_model.__name__, e, raw_text)
        raise LLMOutputError(
            f"Output did not match {response_model.__name__}: {e}",
            raw_response=raw_text,
        ) from e


def invoke_raw_json(system_prompt: str, user_prompt: str) -> dict:
    """
    Call the LLM and return repaired JSON as a plain dict, without Pydantic
    validation.

    Use this only when the expected shape is genuinely dynamic (e.g. the
    attribute-extraction stage, whose output keys depend on whatever
    attributes the customer asked for and therefore cannot be pinned to a
    fixed schema). Prefer `invoke_structured` whenever the shape is known
    ahead of time -- this function gives up the validation safety net.
    """
    messages: List[BaseMessage] = [
        SystemMessage(content=system_prompt.strip()),
        HumanMessage(content=user_prompt.strip()),
    ]

    try:
        raw_text = _invoke_with_retry(messages)
    except Exception as e:
        raise LLMCallError(f"LLM call failed after retries: {e}") from e

    text_to_parse = _extract_json_text(raw_text)

    try:
        parsed = json_repair.loads(text_to_parse)
    except Exception as e:
        raise LLMOutputError(
            f"Could not parse model output as JSON: {e}", raw_response=raw_text
        ) from e

    if not isinstance(parsed, dict):
        raise LLMOutputError(
            f"Expected a JSON object but got {type(parsed).__name__}",
            raw_response=raw_text,
        )

    return parsed