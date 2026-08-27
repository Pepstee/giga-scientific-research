from __future__ import annotations

from scripts.analyze_skin_evidence import (
    deduplicate,
    identity_tokens,
    normalise_title,
    project_group,
)


def _record(
    source_id: str,
    *,
    provider: str = "pubmed",
    title: str = "Fractional laser treatment for atrophic acne scars",
    year: int = 2024,
    doi: str | None = None,
    pmid: str | None = None,
    raw: dict | None = None,
) -> dict:
    return {
        "source_record_id": source_id,
        "provider": provider,
        "kind": "paper",
        "identity_key": source_id,
        "title": title,
        "year": year,
        "doi": doi,
        "pmid": pmid,
        "nct_id": None,
        "raw": raw or {},
        "query_id": "P1-atrophic-scar-lasers",
        "phase": 1,
    }


def test_normalise_title_is_case_and_punctuation_insensitive() -> None:
    assert normalise_title("Skin-Texture: Résults!") == "skin texture results"


def test_identity_tokens_include_strong_identifiers_and_title_year() -> None:
    tokens = identity_tokens(
        _record("one", doi="10.1000/ABC", pmid="12345678")
    )
    assert "doi:10.1000/abc" in tokens
    assert "pmid:12345678" in tokens
    assert any(token.startswith("title-year:") for token in tokens)


def test_deduplicate_collapses_provider_records_with_same_doi() -> None:
    records = [
        _record("one", provider="europe_pmc", doi="10.1000/same"),
        _record("two", provider="pubmed", doi="10.1000/SAME"),
        _record(
            "three",
            title="A genuinely separate dermatology study",
            doi="10.1000/other",
        ),
    ]
    groups = deduplicate(records)
    assert sorted(len(group) for group in groups) == [1, 2]


def test_title_year_fallback_does_not_merge_different_years() -> None:
    records = [_record("one", year=2023), _record("two", year=2024)]
    assert len(deduplicate(records)) == 2


def test_project_group_preserves_provenance_and_marks_open_access() -> None:
    group = project_group(
        [
            _record(
                "one",
                provider="europe_pmc",
                doi="10.1000/same",
                raw={
                    "abstract": "Abstract",
                    "isOpenAccess": "Y",
                    "pmcid": "PMC123",
                    "publicationTypes": ["Meta-Analysis"],
                },
            ),
            _record(
                "two",
                provider="pubmed",
                doi="10.1000/same",
                raw={"publicationTypes": ["Systematic Review"]},
            ),
        ]
    )
    assert group["providers"] == ["europe_pmc", "pubmed"]
    assert group["source_record_ids"] == ["one", "two"]
    assert group["retrieval_record_count"] == 2
    assert group["open_access_candidate"] is True
    assert group["pmcids"] == ["PMC123"]
    assert group["design_bucket"] == "systematic_review_or_meta_analysis"
