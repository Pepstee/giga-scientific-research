from __future__ import annotations

import hashlib
import json
import urllib.parse
from pathlib import Path

import pytest

from modules.research.core.scientific_evidence import ScientificEvidenceError, payload_hash
from modules.research.core.scientific_review_funnel import (
    ELIGIBILITY_DECISIONS_SCHEMA,
    STUDY_EXTRACTION_SCHEMA,
    SYNTHESIS_SIGNOFF_SCHEMA,
    apply_abstract_enrichments,
    abstract_retrieval_queue,
    build_review_funnel,
    fetch_pubmed_abstracts,
    validate_eligibility_decisions,
    validate_study_extraction,
    write_pubmed_abstract_queue,
)
from modules.research.core.scientific_screening import apply_duplicate_decisions


FIXED_TIME = "2026-07-29T12:00:00+00:00"


def _group(
    marker: str,
    *,
    abstract: str | None,
    pmid: str | None = None,
) -> dict:
    return {
        "group_id": marker * 64,
        "title": f"Study {marker}",
        "year": 2025,
        "abstract": abstract,
        "pmid": pmid,
        "pmcid": None,
        "doi": f"10.1000/{marker}",
        "query_ids": ["Q1"],
        "negative_control": False,
        "compiled_query_retrieval": True,
        "retracted": False,
        "publication_types": ["Randomized Controlled Trial"],
    }


def _proposal(group: dict, status: str = "manual_review") -> dict:
    return {
        "group_id": group["group_id"],
        "proposed_status": status,
        "reason_codes": ["MR_REVIEW"],
    }


def _feature(group: dict, design: str = "randomized_controlled_trial") -> dict:
    return {
        "group_id": group["group_id"],
        "design_bucket": design,
        "concepts": {
            "conditions": ["photoaging"],
            "interventions": ["topical_retinoids"],
            "outcomes": ["texture_photoaging"],
        },
    }


def _decision(group: dict, decision: str = "include") -> dict:
    abstract = group["abstract"] or ""
    return {
        "group_id": group["group_id"],
        "decision": decision,
        "reason_codes": ["HUMAN_ELIGIBLE"],
        "reviewed_by": "reviewer",
        "reviewed_at": FIXED_TIME,
        "abstract_sha256": hashlib.sha256(abstract.encode()).hexdigest(),
    }


def _decision_document(groups: list[dict]) -> dict:
    return {
        "schema_version": ELIGIBILITY_DECISIONS_SCHEMA,
        "campaign_id": "test",
        "decisions": [_decision(group) for group in groups],
    }


def _extraction(group: dict) -> dict:
    return {
        "schema_version": STUDY_EXTRACTION_SCHEMA,
        "group_id": group["group_id"],
        "full_text_sha256": "f" * 64,
        "full_text_verified": True,
        "study_design": "randomized_controlled_trial",
        "population": "Adults with photoaging.",
        "condition": "Photoaging.",
        "intervention_protocol": "Topical treatment daily.",
        "comparator": "Vehicle.",
        "sample_size": 100,
        "outcomes": [{"name": "Wrinkle score", "timepoint": "24 weeks"}],
        "effect_estimates": [
            {
                "outcome": "Wrinkle score",
                "measure": "mean difference",
                "value": "-1.2",
                "uncertainty": "95% CI -1.8 to -0.6",
                "locator": "Table 2",
            }
        ],
        "follow_up": "24 weeks.",
        "adverse_events": "Irritation: 5/50 versus 1/50.",
        "withdrawals": "Two withdrawals, reasons reported.",
        "funding": "Public grant.",
        "conflicts": "Authors reported no conflicts.",
        "risk_of_bias": "low",
        "population_applicability": "direct",
        "source_locators": ["Methods p2", "Results Table 2"],
        "extracted_by": "reviewer",
        "extracted_at": FIXED_TIME,
        "notes": [],
    }


def _duplicate_report(complete: bool = True) -> dict:
    value = {
        "candidate_pairs": 0,
        "complete": complete,
        "unresolved_pairs": 0 if complete else 1,
    }
    value["report_sha256"] = payload_hash(value)
    return value


def test_abstract_queue_targets_only_unexcluded_missing_abstracts() -> None:
    missing = _group("a", abstract=None, pmid="12345")
    present = _group("b", abstract="Human trial.")
    excluded = _group("c", abstract=None, pmid="67890")
    queue = abstract_retrieval_queue(
        [missing, present, excluded],
        [
            _proposal(missing),
            _proposal(present, "auto_include"),
            _proposal(excluded, "auto_exclude"),
        ],
    )
    assert [item["group_id"] for item in queue] == [missing["group_id"]]
    assert queue[0]["routes"][0]["provider"] == "pubmed"
    assert queue[0]["abstract_substitution_for_full_text_forbidden"] is True


def test_eligibility_decisions_are_bound_to_current_abstract() -> None:
    group = _group("a", abstract="Current abstract.")
    document = _decision_document([group])
    decisions, unresolved = validate_eligibility_decisions(document, [group])
    assert decisions[group["group_id"]]["human_verified"] is True
    assert unresolved == []
    document["decisions"][0]["abstract_sha256"] = "0" * 64
    with pytest.raises(ScientificEvidenceError, match="stale"):
        validate_eligibility_decisions(document, [group])


def test_study_extraction_requires_full_text_and_source_locations() -> None:
    group = _group("a", abstract="Abstract.")
    extraction = _extraction(group)
    assert validate_study_extraction(extraction)["sample_size"] == 100
    extraction["full_text_verified"] = False
    with pytest.raises(ScientificEvidenceError, match="verified full text"):
        validate_study_extraction(extraction)


def test_funnel_fails_closed_before_human_review(tmp_path: Path) -> None:
    group = _group("a", abstract="Human randomized trial.")
    report = build_review_funnel(
        groups=[group],
        proposals=[_proposal(group, "auto_include")],
        features=[_feature(group)],
        output=tmp_path,
    )
    assert report["effectiveness_ranking_authorised"] is False
    assert report["llm_calls_made"] == 0
    assert "eligibility_decisions_complete" in report["failed_gates"]
    assert (tmp_path / "ABSTRACT_SCREENING_QUEUE.csv").is_file()
    assert (tmp_path / "SELECTIVE_FULLTEXT_QUEUE.jsonl").is_file()


def test_complete_funnel_requires_hash_bound_human_signoff(tmp_path: Path) -> None:
    group = _group("a", abstract="Human randomized trial.")
    first = build_review_funnel(
        groups=[group],
        proposals=[_proposal(group, "auto_include")],
        features=[_feature(group)],
        output=tmp_path,
        eligibility_document=_decision_document([group]),
        extractions=[_extraction(group)],
        duplicate_decision_report=_duplicate_report(),
    )
    assert first["scientific_synthesis_ready"] is False
    assert first["failed_gates"] == ["human_synthesis_signoff"]
    signoff = {
        "schema_version": SYNTHESIS_SIGNOFF_SCHEMA,
        "comparison_matrix_sha256": first["comparison_matrix_sha256"],
        "human_verified": True,
        "reviewed_by": "senior reviewer",
    }
    second = build_review_funnel(
        groups=[group],
        proposals=[_proposal(group, "auto_include")],
        features=[_feature(group)],
        output=tmp_path,
        eligibility_document=_decision_document([group]),
        extractions=[_extraction(group)],
        duplicate_decision_report=_duplicate_report(),
        synthesis_signoff=signoff,
    )
    assert second["scientific_synthesis_ready"] is True
    assert second["effectiveness_ranking_authorised"] is True
    assert second["recommendation_authority"] is False


def test_pubmed_abstract_fetch_is_batched_official_and_source_hashed() -> None:
    captured: dict[str, object] = {}
    xml = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation>
      <PMID>123</PMID><Article><Abstract>
      <AbstractText Label="BACKGROUND">Background text.</AbstractText>
      <AbstractText Label="RESULTS">Result text.</AbstractText>
      </Abstract></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""

    def transport(url: str, timeout: int) -> bytes:
        captured.update(url=url, timeout=timeout)
        return xml

    results = fetch_pubmed_abstracts(["123"], transport=transport)
    assert str(captured["url"]).startswith(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
    )
    assert results[0]["abstract"] == (
        "BACKGROUND: Background text. RESULTS: Result text."
    )
    assert results[0]["full_text_substitute"] is False
    assert len(results[0]["retrieval_sha256"]) == 64


def test_pubmed_accepts_only_official_nlm_doctype_and_rejects_entities() -> None:
    official = b"""<?xml version="1.0"?>
    <!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2025//EN"
    "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_250101.dtd">
    <PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID>
    <Article><Abstract><AbstractText>Safe.</AbstractText></Abstract></Article>
    </MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    assert fetch_pubmed_abstracts(
        ["123"], transport=lambda _url, _timeout: official
    )[0]["abstract"] == "Safe."
    malicious = b"""<!DOCTYPE PubmedArticleSet [
    <!ENTITY xxe SYSTEM "file:///etc/passwd">]>
    <PubmedArticleSet/>"""
    with pytest.raises(ScientificEvidenceError, match="entity"):
        fetch_pubmed_abstracts(
            ["123"], transport=lambda _url, _timeout: malicious
        )


def test_abstract_merge_recomputes_features_and_proposals() -> None:
    group = _group("a", abstract=None, pmid="123")
    abstract = (
        "Adults with photoaging were randomized to topical tretinoin and vehicle. "
        "Skin texture improved."
    )
    retrieval = {
        "schema_version": "giga.scientific-abstract-retrieval.v1",
        "provider": "pubmed",
        "pmid": "123",
        "abstract": abstract,
        "retrieval_sha256": "e" * 64,
    }
    ontology = {
        "conditions": {
            "photoaging": {"terms": ["photoaging"], "subject_headings": []}
        },
        "interventions": {
            "topical_retinoids": {
                "terms": ["topical tretinoin"],
                "subject_headings": [],
            }
        },
        "outcomes": {
            "texture": {"terms": ["skin texture"], "subject_headings": []}
        },
    }
    specification = {
        "queries": [
            {
                "id": "Q1",
                "condition_concepts": ["photoaging"],
                "intervention_concepts": ["topical_retinoids"],
                "outcome_concepts": ["texture"],
                "require_outcomes": False,
                "extra_required_groups": [],
                "negative_control": False,
            }
        ],
        "negative_controls": [],
    }
    build = apply_abstract_enrichments(
        [group], [retrieval], ontology, specification
    )
    assert build["report"]["applied"] == 1
    assert build["report"]["eligibility_proposals_recomputed"] is True
    assert build["groups"][0]["abstract"] == abstract
    assert build["proposals"][0]["proposed_status"] == "auto_include"


def test_duplicate_gate_rejects_unhashed_assertion(tmp_path: Path) -> None:
    group = _group("a", abstract="Human randomized trial.")
    report = build_review_funnel(
        groups=[group],
        proposals=[_proposal(group, "auto_include")],
        features=[_feature(group)],
        output=tmp_path,
        eligibility_document=_decision_document([group]),
        extractions=[_extraction(group)],
        duplicate_decision_report={
            "candidate_pairs": 0,
            "complete": True,
            "unresolved_pairs": 0,
        },
    )
    assert report["gates"]["duplicate_decisions_complete"] is False


def test_zero_pairs_duplicate_report_opens_only_duplicate_gate(
    tmp_path: Path,
) -> None:
    group = _group("a", abstract="Human randomized trial.")
    _groups, _pairs, first = apply_duplicate_decisions(
        [[group]], [group], [], None, campaign_id="test"
    )
    _groups, _pairs, second = apply_duplicate_decisions(
        [[group]], [group], [], None, campaign_id="test"
    )
    assert first == second
    assert first["human_verified"] is False
    assert first["decisions_recorded"] == 0
    assert first["report_sha256"] == payload_hash(
        {key: value for key, value in first.items() if key != "report_sha256"}
    )
    funnel = build_review_funnel(
        groups=[group],
        proposals=[_proposal(group, "auto_include")],
        features=[_feature(group)],
        output=tmp_path,
        duplicate_decision_report=first,
    )
    assert funnel["gates"]["duplicate_decisions_complete"] is True
    assert "eligibility_decisions_complete" in funnel["failed_gates"]
    assert funnel["scientific_synthesis_ready"] is False
    assert funnel["effectiveness_ranking_authorised"] is False
    assert funnel["recommendation_authority"] is False
    decisions, unresolved = validate_eligibility_decisions(None, [group])
    assert decisions == {}
    assert unresolved == [group["group_id"]]
    tampered = dict(first)
    tampered["decisions_recorded"] = 1
    rejected = build_review_funnel(
        groups=[group],
        proposals=[_proposal(group, "auto_include")],
        features=[_feature(group)],
        output=tmp_path,
        duplicate_decision_report=tampered,
    )
    assert rejected["gates"]["duplicate_decisions_complete"] is False


def test_unresolved_pairs_keep_duplicate_gate_closed_with_valid_hash(
    tmp_path: Path,
) -> None:
    first = _group("a", abstract="Human randomized trial.")
    second = _group("b", abstract="Human randomized trial.")
    _groups, _pairs, duplicate_report = apply_duplicate_decisions(
        [[first], [second]],
        [first, second],
        [{"pair_sha256": "0" * 64}],
        None,
        campaign_id="test",
    )
    assert duplicate_report["candidate_pairs"] == 1
    assert duplicate_report["unresolved_pairs"] == 1
    assert duplicate_report["complete"] is False
    assert duplicate_report["human_verified"] is False
    assert duplicate_report["report_sha256"] == payload_hash(
        {
            key: value
            for key, value in duplicate_report.items()
            if key != "report_sha256"
        }
    )
    funnel = build_review_funnel(
        groups=[first, second],
        proposals=[
            _proposal(first, "auto_include"),
            _proposal(second, "auto_include"),
        ],
        features=[_feature(first), _feature(second)],
        output=tmp_path,
        duplicate_decision_report=duplicate_report,
    )
    assert funnel["gates"]["duplicate_decisions_complete"] is False
    assert funnel["scientific_synthesis_ready"] is False


def test_abstract_queue_fetch_batches_at_two_hundred(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    with queue.open("w", encoding="utf-8") as handle:
        for pmid in range(1, 202):
            handle.write(
                json.dumps(
                    {
                        "schema_version": "giga.scientific-abstract-retrieval.v1",
                        "routes": [
                            {
                                "provider": "pubmed",
                                "identifier": str(pmid),
                            }
                        ],
                    }
                )
                + "\n"
            )
    calls: list[list[str]] = []

    def transport(url: str, timeout: int) -> bytes:
        identifiers = urllib.parse.parse_qs(
            urllib.parse.urlsplit(url).query
        )["id"][0].split(",")
        calls.append(identifiers)
        articles = "".join(
            f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID>"
            f"<Article><Abstract><AbstractText>A {pmid}</AbstractText></Abstract>"
            f"</Article></MedlineCitation></PubmedArticle>"
            for pmid in identifiers
        )
        return f"<PubmedArticleSet>{articles}</PubmedArticleSet>".encode()

    result = write_pubmed_abstract_queue(
        queue,
        tmp_path / "abstracts.jsonl",
        transport=transport,
    )
    assert [len(batch) for batch in calls] == [200, 1]
    assert result["requested_pmids"] == 201
    assert result["returned_abstract_records"] == 201
    assert result["batches"] == 2
