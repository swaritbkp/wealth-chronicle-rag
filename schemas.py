from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, confloat, constr, field_validator

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
        default_factory=datetime.utcnow,
        description="UTC timestamp of ingestion run",
    )

    @field_validator("edition_date", mode="before")
    @classmethod
    def parse_edition_date(cls, v: str | date) -> date:
        if isinstance(v, str):
            return date.fromisoformat(v)
        return v


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


class RerankedPassage(BaseModel):
    """Post-reranking passage with cross-encoder score and fused ranking."""

    point_id: str
    text: str
    payload: ChunkPayload
    cross_encoder_score: confloat(ge=0.0, le=1.0) = Field(
        ..., description="FlashRank normalized cross-encoder relevance score"
    )
    rrf_score: float = Field(
        ..., description="Reciprocal Rank Fusion score before reranking"
    )
    time_decay_multiplier: float = Field(
        ..., ge=1.0, description="Recency boost: 1.0 + alpha*exp(-delta_t/tau)"
    )
    final_rank: int = Field(..., ge=1, le=20)


class CitationMetadata(BaseModel):
    """Structured citation block surfaced in the UI source expander."""

    edition_date: date
    page_number: int
    article_title: str | None
    cross_encoder_score: confloat(ge=0.0, le=1.0)
    excerpt_preview: constr(max_length=300) = Field(
        ..., description="First 300 chars of chunk text for UI preview"
    )


# ─── Evaluation Domain ──────────────────────────────────────────────


class EvaluationCategory(str, Enum):
    TAX_REGIME = "tax_regime"
    MUTUAL_FUNDS = "mutual_funds"
    INSURANCE_CLAIMS = "insurance_claims"
    RETIREMENT_NPS = "retirement_nps"
    ESTATE_SUCCESSION = "estate_succession"


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
        ...,
        min_length=1,
        description="Edition dates from which the ground truth was derived",
    )
    source_pages: list[int] = Field(
        ...,
        min_length=1,
        description="Page numbers from which the ground truth was derived",
    )
    difficulty: constr(pattern=r"^(easy|medium|hard)$") = Field(
        default="medium",
        description="Subjective difficulty for triage",
    )

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
