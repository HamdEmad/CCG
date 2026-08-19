"""
Attribute extraction node: given verified page text, extract the
requested technical attributes.

Simplified from the original version — page verification (soft-404 checks,
product_name presence checks) is now handled upstream by
`page_verification.verify_part_on_page()`. This module focuses purely on
attribute extraction from content that has already been verified to contain
the requested part.

Uses `invoke_raw_json`, not `invoke_structured`, because the output keys
are genuinely dynamic — they depend on whatever attributes the customer
asked for plus the fixed `StaticAttributes` fields.
"""

from __future__ import annotations

import logging
from typing import Optional

from integrations.llm_client import LLMCallError, LLMOutputError, invoke_raw_json
from pipeline.prompts import build_extraction_system_prompt, static_attributes_description_json
from pipeline.state import PartDetails, PartStatus

logger = logging.getLogger(__name__)


# Sentinel values that mean "this attribute was not actually found on the
# page," even though the LLM still returned a key/value pair for it.
_NOT_FOUND_SENTINELS = {"not found", "n/a", "na", "unknown", ""}


def _normalize_key(name: str) -> str:
    """Normalize an attribute name for comparison."""
    return " ".join(name.strip().lower().replace("-", " ").split())


def _was_attribute_found(attribute_name: str, attributes: dict) -> bool:
    """
    Whether `attribute_name` is present in `attributes` with a real,
    non-sentinel value.
    """
    normalized_target = _normalize_key(attribute_name)
    for key, value in attributes.items():
        if _normalize_key(key) == normalized_target:
            value_str = str(value).strip().lower() if value is not None else ""
            return value_str not in _NOT_FOUND_SENTINELS
    return False


def compute_resolution(requested_attributes: list[str] | None, attributes: dict) -> str:
    """
    Classify how well `attributes` satisfies what the customer actually
    asked for, and return the corresponding customer-facing message.
    """
    if not requested_attributes:
        return "the part has been successfully added"

    all_found = all(
        _was_attribute_found(attr, attributes) for attr in requested_attributes
    )
    if all_found:
        return "we have successfully added the part with the required attributes"

    return (
        "we have successfully added the part and are currently working on "
        "adding the required attributes"
    )


def extract_attributes(
    page_content: str,
    part: PartDetails,
    page_url: str,
) -> dict:
    """
    Extract requested attributes from verified page content.

    This is a pure extraction function — it assumes the page has already
    been verified to contain relevant part information. Page verification
    (part-on-page checks, soft-404 detection) is handled upstream by
    `page_verification.verify_part_on_page()`.

    Returns:
        dict of extracted attributes on success, empty dict on failure.
    """
    if not page_content or not page_content.strip():
        logger.warning("Part %s: empty page content passed to extraction", part.part)
        return {}

    requested_attributes = part.attributes or []

    user_prompt = (
        f"<target_part>\n"
        f"Manufacturer: {part.manufacturer}\n"
        f"Part Number: {part.part}\n"
        f"Part Series: {part.part_series or 'N/A'}\n"
        f"</target_part>\n\n"
        f"<page_url>\n{page_url}\n</page_url>\n\n"
        f"<static_attributes>\n{static_attributes_description_json()}\n</static_attributes>\n\n"
        f"<requested_attributes>\n{requested_attributes}\n</requested_attributes>\n\n"
        f"<web_page_content>\n{page_content}\n</web_page_content>"
    )

    try:
        response_dict = invoke_raw_json(
            system_prompt=build_extraction_system_prompt(requested_attributes),
            user_prompt=user_prompt,
        )
    except (LLMCallError, LLMOutputError) as e:
        logger.warning("LLM extraction failed for part %s: %s", part.part, e)
        return {}

    # Check if LLM identified it as a soft 404
    llm_status = response_dict.get("status", "FAILED").upper()
    if llm_status != "SUCCESS":
        logger.info("Part %s: LLM identified page as soft 404", part.part)
        return {}

    if "attributes" in response_dict and isinstance(response_dict["attributes"], dict):
        return response_dict["attributes"]

    # Fallback to root level attributes if not nested
    return {k: v for k, v in response_dict.items() if k != "status"}