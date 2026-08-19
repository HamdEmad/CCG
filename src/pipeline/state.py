"""
State models for the component lookup pipeline.

Design notes (see architecture review):
- No single shared mutable object is passed through every node. Each stage
  declares the state it actually reads and the state it actually produces.
- Every part carries its own status, so one part's failure never silently
  removes it from the batch and never crashes the other parts.
- `PartDetails.product_url` is genuinely optional, matching how the field
  is documented and how the LLM will actually populate it.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PartStatus(str, Enum):
    """Where a single part currently is in the pipeline."""

    PENDING = "pending"
    SEARCHED = "searched"
    FILTERED = "filtered"
    SCRAPED = "scraped"
    EXTRACTED = "extracted"
    SKIPPED_LOW_SCORE = "skipped_low_score"
    FAILED = "failed"
    FAILED_ANTI_BOT = "failed_anti_bot"

    # URL-inference fallback path
    URL_INFERRED_VERIFIED = "url_inferred_verified"
    URL_INFERRED_NOT_FOUND = "url_inferred_not_found"
    URL_INFERENCE_UNKNOWN = "url_inference_unknown"

    # Site-search / browser fallback path
    SITE_SEARCH_VERIFIED = "site_search_verified"
    SITE_SEARCH_NOT_FOUND = "site_search_not_found"


class FoundOn(str, Enum):
    """Where the part information was found."""

    LANDING_PAGE = "landing_page"


class MatchType(str, Enum):
    """WHAT part identifier was matched."""

    EXACT_PART = "exact_part"
    SERIES = "series"


# Confidence is derived from (MatchType, FoundOn), never set manually.
CONFIDENCE_MAP: dict[tuple[MatchType, FoundOn], float] = {
    (MatchType.EXACT_PART, FoundOn.LANDING_PAGE): 1.0,
    (MatchType.SERIES,     FoundOn.LANDING_PAGE): 0.6,
}


class PartDetails(BaseModel):
    """A single component request extracted from the customer's message."""

    part: str = Field(..., description="The requested product name.")
    manufacturer: str = Field(
        ..., description="The manufacturer name of the requested product."
    )
    part_series: Optional[str]= Field(default=None, description="The part series name of the requested product.")
    product_url: Optional[str] = Field(
        default=None,
        description=(
            "The product URL of the requested product if the customer "
            "specified one. None if not specified."
        ),
    )
    datasheet: Optional[str] = Field(
        default=None,
        description="The URL of the datasheet for this part if the customer "
                    "specified one. None if not specified."
    )
    message: str = Field(
        ..., description="The customer message text relevant to this part."
    )
    attributes: Optional[List[str]] = Field(
        default=None,
        description="A list of strings representing the specific attributes the customer wants extracted for this part.",
    )
    crosses: Optional[List[str]] = Field(
        default=None,
        description="A list of strings representing any cross-reference parts requested for this component.",
    )

    @field_validator("attributes", "crosses", mode="before")
    @classmethod
    def _ensure_list(cls, v):
        """
        Defensively coerce single strings into lists. The LLM sometimes
        hallucinates a scalar string when asked for an array (e.g., 
        "attributes": "lifecycle status" instead of ["lifecycle status"]).
        If the string contains commas, split it into separate list items.
        """
        if isinstance(v, str):
            if ',' in v:
                return [s.strip() for s in v.split(',')]
            return [v]
        return v
class ExtractionState(BaseModel):
    """Output of the extraction stage: customer message -> structured parts."""

    customer_request: str
    parts: List[PartDetails] = Field(default_factory=list)


class SearchResultItem(BaseModel):
    """One raw result returned by a single search backend."""

    title: Optional[str] = None
    href: Optional[str] = None
    body: Optional[str] = None


class PartSearchResult(BaseModel):
    """Search-stage output for a single part."""

    part: PartDetails
    query: str
    results: List[SearchResultItem] = Field(default_factory=list)
    status: PartStatus = PartStatus.PENDING
    error: Optional[str] = None


class ResultSource(str, Enum):
    """Where a part's chosen URL ultimately came from."""

    CUSTOMER = "customer"
    SEARCH = "search"
    INFERRED = "inferred"
    SITE_SEARCH = "site_search"
    BROWSER = "browser"


class ScoredUrl(BaseModel):
    """A single search result URL with its raw (unclamped) score from the filter LLM."""

    url: str
    score: int = Field(
        description="Raw, unclamped score. Can be negative (e.g. -100 for PDFs or distributors)."
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the score.",
    )


class FilterResult(BaseModel):
    """The single best URL chosen by the filter stage for one part, with score."""

    url: Optional[str] = None
    score: int = Field(ge=0, le=10, default=0)
    all_scored_urls: list[ScoredUrl] = Field(
        default_factory=list,
        description="All evaluated URLs with their raw, unclamped scores.",
    )


class PartFilterResult(BaseModel):
    """Filter-stage output for a single part."""

    part: PartDetails
    query: str
    filtered: FilterResult
    status: PartStatus = PartStatus.PENDING
    error: Optional[str] = None


class PartScrapeResult(BaseModel):
    """Scrape-stage output for a single part."""

    part: PartDetails
    filtered: FilterResult
    source: ResultSource = ResultSource.SEARCH
    scraped_text: Optional[str] = None
    status: PartStatus = PartStatus.PENDING
    error: Optional[str] = None


class InferredUrlResult(BaseModel):
    """
    Output of the URL-inference fallback LLM call.

    `confidence` is the explicit "do you actually know this" signal the
    prompt asks for. The model is instructed to return
    `confidence="unknown"` (with `url=None`) rather than fabricate a
    plausible-looking guess just to satisfy the schema -- `known_pattern`
    should only be used when the model has genuine, specific knowledge of
    this manufacturer's URL convention, not a generic guess.
    """

    url: Optional[str] = None
    confidence: str = Field(
        default="unknown",
        description='Either "known_pattern" or "unknown".',
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this URL pattern was chosen, "
        "or why no confident pattern is known.",
    )


class SiteSearchDiscoveryResult(BaseModel):
    """Output of the site search discovery LLM call."""

    search_url_template: Optional[str] = Field(
        default=None,
        description="The URL template for querying the site's search engine. Must contain '{query}' exactly once where the search term goes. None if no search engine pattern can be determined.",
    )
    part_family: Optional[str] = Field(
        default=None,
        description="The extracted part family to be used as a fallback query if the full part number returns no results. None if it cannot be determined.",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of how the search template was discovered or why it couldn't be found.",
    )


class SiteSearchResultsResult(BaseModel):
    """Output of analyzing a site search results page."""

    landing_page_url: Optional[str] = Field(
        default=None,
        description="The exact product landing page URL found in the search results. None if the correct product is not found.",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this URL was chosen, or why none were suitable.",
    )


class HomepageSelectionResult(BaseModel):
    """Output of selecting a manufacturer homepage from web search results."""

    url: Optional[str] = Field(
        default=None,
        description="The official manufacturer homepage URL, or null if none found.",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this URL was chosen.",
    )


class PartSiteSearchResult(BaseModel):
    """Site-search fallback stage output for a single part."""

    part: PartDetails
    discovery: Optional[SiteSearchDiscoveryResult] = None
    results: Optional[SiteSearchResultsResult] = None
    status: PartStatus = PartStatus.PENDING
    error: Optional[str] = None


class PartUrlInferenceResult(BaseModel):
    """URL-inference stage output for a single part, before verification."""

    part: PartDetails
    inferred: InferredUrlResult
    status: PartStatus = PartStatus.PENDING
    error: Optional[str] = None


class PartAttributeResult(BaseModel):
    """Final attribute-extraction output for a single part."""

    part: PartDetails
    attributes: dict = Field(default_factory=dict)
    landing_page: Optional[str] = None
    source: Optional[ResultSource] = None

    status: PartStatus = PartStatus.PENDING
    error: Optional[str] = None

    # Provenance: WHERE and WHAT was matched.
    found_on: Optional[FoundOn] = Field(
        default=None,
        description="Where the part information was found: landing page or datasheet.",
    )
    match_type: Optional[MatchType] = Field(
        default=None,
        description="What was matched: exact part number or series.",
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score derived from found_on × match_type.",
    )


class PipelineResult(BaseModel):
    """Aggregated result returned to the caller after the full pipeline run."""

    customer_request: str
    completed: List[PartAttributeResult] = Field(default_factory=list)
    failed: List[PartAttributeResult] = Field(default_factory=list)
    resolution: Optional[str] = Field(
        default=None,
        description="Customer-facing summary of how the requested attributes "
        "were resolved globally across all extracted parts.",
    )

    @property
    def total_parts(self) -> int:
        return len(self.completed) + len(self.failed)


class BrowserAction(BaseModel):
    """A single browser automation command."""
    
    action: str = Field(description="The action to perform: 'click', 'type', 'done', or 'fail'")
    element_id: Optional[int] = Field(
        default=None, 
        description="The numeric ID of the element to interact with (from the snapshot)."
    )
    text: Optional[str] = Field(
        default=None, 
        description="The text to type, required if action is 'type'."
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this action is being taken."
    )

class BrowserActionList(BaseModel):
    """A list of browser automation commands to execute sequentially."""
    
    actions: List[BrowserAction] = Field(
        default_factory=list,
        description="List of actions to perform."
    )
