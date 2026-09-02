from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Annotated, Literal
from uuid import UUID
import uuid

from pydantic import BaseModel, Field, field_validator, model_validator

# ─── Ingestion Domain ────────────────────────────────────────────────


class ChunkPayload(BaseModel):
    """Canonical payload schema stored in every Qdrant point.
    Maps 1:1 with FR-ING-04 metadata enrichment contract."""

    chunk_id: str = Field(
        ...,
        pattern=r"^chk_\d{4}_\d{2}_\d{2}_p\d+_\d{3}$",
        description="Deterministic chunk identifier: chk_{YYYY}_{MM}_{DD}_p{page}_{seq:03d}",
        examples=["chk_2026_08_24_p14_002"],
    )
    edition_date: date = Field(
        ...,
        description="ISO-8601 publication date of the source issue",
        examples=["2026-08-24"],
    )
    page_number: int = Field(
        ...,
        ge=1,
        le=200,
        description="1-indexed physical page number in the PDF",
    )
    article_title: str | None = Field(
        default=None,
        max_length=256,
        description="Extracted or inferred article heading (nullable for untitled blocks)",
    )
    text: str = Field(
        ...,
        min_length=120,
        max_length=8000,
        description="Chunk text content after noise filtering",
    )
    has_table: bool = Field(
        default=False,
        description="True if chunk contains tabular data",
    )
    source: str = Field(
        default="Weekly Financial Dossier",
        description="Publication brand name (constant for single-source MVP)",
    )
    char_count: int = Field(
        ...,
        ge=120,
        description="Precomputed len(text) for payload-level filtering",
    )
    word_count: int = Field(
        ...,
        ge=20,
        description="Precomputed whitespace-split word count",
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of ingestion run",
    )

    @field_validator("edition_date", mode="before")
    @classmethod
    def parse_edition_date(cls, v: str | date) -> date:
        if isinstance(v, str):
            return date.fromisoformat(v)
        return v


class TableSchema(BaseModel):
    """Structured schema for a parsed table."""

    columns: list[str] = Field(description="Column headers")
    dtypes: list[str] = Field(description="Inferred data types per column (int, float, string, currency, percentage, date)")
    row_count: int = Field(ge=0, description="Number of data rows")
    column_count: int = Field(ge=0, description="Number of columns")


class TableChunk(ChunkPayload):
    """Extended chunk payload for table-structured data with dual representation."""

    # Table-specific fields
    table_markdown: str = Field(default="", description="Original markdown table representation")
    table_csv: str = Field(default="", description="CSV representation for exact numerical lookup")
    table_schema: TableSchema | None = Field(default=None, description="Structured schema with types")
    table_hash: str | None = Field(default=None, description="SHA256 hash of CSV for deduplication")

    # Override text to include markdown + CSV preview
    @field_validator("text", mode="before")
    @classmethod
    def build_table_text(cls, v, info):
        # If we have table_markdown, use it as primary text
        return v or info.data.get("table_markdown", "")


# ─── Retrieval Domain ────────────────────────────────────────────────


class RetrievalSource(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class SearchResult(BaseModel):
    """Raw candidate returned from either dense or sparse retrieval."""

    point_id: str = Field(..., description="Qdrant point UUID (MD5 hex)")
    text: str
    payload: ChunkPayload
    score: float = Field(..., description="Raw retrieval score (cosine sim or BM25)")
    source: RetrievalSource
    dense_rank: int | None = Field(default=None, ge=1)
    sparse_rank: int | None = Field(default=None, ge=1)
    vector_name: str | None = Field(default=None, description="Name of vector used for retrieval (dense/sparse)")


class RerankedPassage(BaseModel):
    """Post-reranking passage with cross-encoder score and fused ranking."""

    point_id: str
    text: str
    payload: ChunkPayload
    cross_encoder_score: Annotated[float, Field(ge=0.0, le=1.0, description="FlashRank normalized cross-encoder relevance score")]
    rrf_score: float = Field(..., description="Reciprocal Rank Fusion score before reranking")
    time_decay_multiplier: float = Field(..., ge=1.0, description="Recency boost: 1.0 + alpha*exp(-delta_t/tau)")
    final_rank: int = Field(..., ge=1, le=20)


class CitationMetadata(BaseModel):
    """Structured citation block surfaced in the UI source expander."""

    edition_date: date
    page_number: int
    article_title: str | None
    cross_encoder_score: Annotated[float, Field(ge=0.0, le=1.0)]
    excerpt_preview: Annotated[str, Field(max_length=300, description="First 300 chars of chunk text for UI preview")]


# ─── Structured Generation Domain ──────────────────────────────────────


class CitationSpan(BaseModel):
    """Character-level citation span for precise grounding verification."""

    edition_date: date
    page_number: int
    article_title: str | None
    char_start: int = Field(ge=0, description="Start character offset in source chunk text")
    char_end: int = Field(ge=0, description="End character offset in source chunk text")
    quoted_text: str = Field(description="Exact text span from source chunk")


class GroundedClaim(BaseModel):
    """A single factual claim with its supporting citations."""

    claim: str = Field(description="The factual statement being made")
    citations: list[CitationSpan] = Field(min_length=1, description="One or more citation spans supporting this claim")
    claim_type: Literal["numerical", "categorical", "procedural", "date", "reference"] = Field(
        description="Type of claim for specialized validation"
    )


class GroundedAnswer(BaseModel):
    """Structured answer with verifiable citations and confidence calibration."""

    answer: str = Field(description="Final synthesized answer in markdown")
    claims: list[GroundedClaim] = Field(default_factory=list, description="Decomposed factual claims with citations")
    citations: list[CitationMetadata] = Field(default_factory=list, description="Aggregated citation metadata for UI")
    confidence: Literal["high", "medium", "low"] = Field(description="Overall confidence in answer correctness")
    refusal_reason: str | None = Field(default=None, description="Present only if confidence=low and answer is refusal")
    key_takeaway: str | None = Field(default=None, max_length=500, description="One-sentence actionable summary")
    numerical_facts: list[dict] = Field(default_factory=list, description="Extracted numerical facts for validation")


class GenerationConfigSchema(BaseModel):
    """Schema for Gemini structured generation configuration."""

    response_mime_type: Literal["application/json"] = "application/json"
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=100)
    candidate_count: int = Field(default=1, ge=1, le=4)


# ─── Evaluation Domain ──────────────────────────────────────────────


class EvaluationCategory(str, Enum):
    TAX_REGIME = "tax_regime"
    MUTUAL_FUNDS = "mutual_funds"
    INSURANCE_CLAIMS = "insurance_claims"
    RETIREMENT_NPS = "retirement_nps"
    ESTATE_SUCCESSION = "estate_succession"
    REFUSAL = "refusal"
    OUT_OF_DOMAIN = "out_of_domain"
    UNSUPPORTED = "unsupported"


class EvaluationItem(BaseModel):
    """Single entry in tests/golden_eval_set.json."""

    id: str = Field(
        ...,
        pattern=r"^eval_\d{3}$",
        description="Sequential evaluation ID: eval_001 ... eval_050",
        examples=["eval_001"],
    )
    category: EvaluationCategory
    question: str = Field(..., min_length=15, max_length=500)
    ground_truth: str = Field(
        ...,
        min_length=20,
        max_length=2000,
        description="Verified human-authored reference answer",
    )
    source_edition_dates: list[date] = Field(
        default_factory=list,
        description="Edition dates from which the ground truth was derived (empty for refusal categories)",
    )
    source_pages: list[int] = Field(
        default_factory=list,
        description="Page numbers from which the ground truth was derived (empty for refusal categories)",
    )
    difficulty: Annotated[str, Field(pattern=r"^(easy|medium|hard)$", default="medium", description="Subjective difficulty for triage")]

    @field_validator("source_edition_dates", mode="before")
    @classmethod
    def parse_source_dates(cls, v):
        if isinstance(v, list):
            result = []
            for item in v:
                if isinstance(item, str):
                    result.append(date.fromisoformat(item))
                else:
                    result.append(item)
            return result
        return v

    @model_validator(mode="after")
    def validate_non_refusal_has_sources(self):
        """Ensure non-refusal categories have source edition dates and pages."""
        refusal_categories = {EvaluationCategory.REFUSAL, EvaluationCategory.OUT_OF_DOMAIN, EvaluationCategory.UNSUPPORTED}
        if self.category in refusal_categories:
            # This is a refusal/out_of_domain/unsupported category - empty lists are allowed
            return self
        if not self.source_edition_dates:
            raise ValueError("source_edition_dates must not be empty for in-domain evaluation items")
        if not self.source_pages:
            raise ValueError("source_pages must not be empty for in-domain evaluation items")
        return self


# ─── Active Learning / Feedback Domain ─────────────────────────────────


class FeedbackLabel(str, Enum):
    """User feedback labels for answer quality."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTED = "corrected"


class QueryFeedback(BaseModel):
    """User feedback on a generated answer for active learning."""

    feedback_id: UUID | None = Field(default=None, description="Unique feedback identifier (auto generated if not provided)")
    query_text: str = Field(..., min_length=5, max_length=1000, description="Original user query")
    answer_text: str = Field(..., description="Generated answer text")
    label: FeedbackLabel = Field(description="User feedback label")
    corrected_answer: str | None = Field(default=None, description="User-provided correction if label=CORRECTED")
    citations: list[CitationMetadata] = Field(default_factory=list, description="Citations from the original answer")
    trace_id: str | None = Field(default=None, description="Link to query trace for debugging")
    user_id: str | None = Field(default=None, description="Anonymous user identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Feedback submission time")
    metadata: dict = Field(default_factory=dict, description="Additional context (model version, retrieval params, etc.)")


class GoldenSetCandidate(BaseModel):
    """Candidate for inclusion in golden evaluation set, derived from feedback."""

    question: str = Field(..., min_length=15, max_length=500)
    ground_truth: str = Field(..., min_length=20, max_length=2000)
    source_edition_dates: list[date] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    category: EvaluationCategory = Field(default=EvaluationCategory.TAX_REGIME)
    difficulty: Literal["easy", "medium", "hard"] = Field(default="medium")
    derived_from_feedback_id: UUID = Field(description="Link to originating feedback")
    reviewer_notes: str | None = Field(default=None, description="Human reviewer notes")
    status: Literal["pending", "approved", "rejected"] = Field(default="pending")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ─── Query Drift Monitoring Domain ───────────────────────────────────────


class DriftSeverity(str, Enum):
    """Severity levels for detected query drift."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftCluster(BaseModel):
    """A cluster of similar queries detected via embedding clustering."""

    cluster_id: int = Field(description="Cluster identifier from HDBSCAN (-1 for noise)")
    centroid_embedding: list[float] = Field(description="Mean embedding of cluster members")
    query_count: int = Field(ge=1, description="Number of queries in this cluster")
    sample_queries: list[str] = Field(default_factory=list, description="Up to 5 representative queries")
    avg_distance_to_centroid: float = Field(ge=0.0, description="Average cosine distance to centroid")
    first_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_new: bool = Field(default=False, description="True if cluster is newly detected vs baseline")


class DriftReport(BaseModel):
    """Report from drift detection analysis."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_queries_analyzed: int = Field(ge=0)
    num_clusters: int = Field(ge=0)
    num_noise_queries: int = Field(ge=0)
    new_clusters: list[DriftCluster] = Field(default_factory=list)
    severity: DriftSeverity = Field(default=DriftSeverity.NONE)
    recommendation: str = Field(default="")
    baseline_coverage: float = Field(default=0.0, description="Fraction of queries matching baseline clusters")


class DriftAlert(BaseModel):
    """Alert for significant query drift."""

    alert_id: UUID = Field(default_factory=lambda: uuid.uuid4(), description="Unique alert identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    severity: DriftSeverity
    message: str
    affected_clusters: list[int] = Field(default_factory=list)
    new_query_samples: list[str] = Field(default_factory=list)
    acknowledged: bool = Field(default=False)
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
