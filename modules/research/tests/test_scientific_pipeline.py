from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from modules.research.core.scientific_campaign import provider_campaign_jobs, validate_campaign
from modules.research.core.scientific_evidence import ScientificEvidenceError, payload_hash
from modules.research.core.scientific_fulltext import (
    extract_pmc_xml,
    fetch_europe_pmc_xml,
    plan_full_text,
)
from modules.research.core.scientific_query_compiler import (
    COMPILED_SCHEMA,
    compile_manifest,
    compile_provider_query,
    embed_compiled_queries,
    validate_ontology,
    validate_search_spec,
)
from modules.research.core.scientific_readiness import assess_screening_readiness
from modules.research.core.scientific_review_state import (
    LLM_PROPOSAL_SCHEMA,
    ScientificReviewStore,
)
from modules.research.core.scientific_saturation import evaluate_saturation
from modules.research.core.scientific_screening import (
    _query_lookup,
    apply_duplicate_decisions,
    deterministic_exclusion_sample,
    exact_deduplicate,
    extract_features,
    negative_control_report,
    normalise_doi,
    normalise_title,
    possible_duplicate_pairs,
    project_group,
    propose_eligibility,
    validate_exclusion_audit,
)


FIXED_TIME = "2026-07-28T18:00:00+00:00"


def _ontology() -> dict:
    return validate_ontology(
        {
            "schema_version": "giga.scientific-search-ontology.v1",
            "ontology_id": "test-ontology",
            "version": "1",
            "conditions": {
                "acne_scar": {
                    "terms": ["atrophic acne scar", "acne scar"],
                    "subject_headings": ["Acne Vulgaris"],
                },
                "low_back_pain": {
                    "terms": ["low back pain"],
                    "subject_headings": ["Low Back Pain"],
                },
            },
            "interventions": {
                "microneedling": {
                    "terms": ["microneedling"],
                    "subject_headings": ["Microneedling"],
                }
            },
            "outcomes": {
                "severity": {
                    "terms": ["scar severity", "ECCA"],
                    "subject_headings": [],
                }
            },
        }
    )


def _spec() -> dict:
    return validate_search_spec(
        {
            "schema_version": "giga.scientific-search-specification.v1",
            "campaign_id": "test-campaign",
            "ontology_id": "test-ontology",
            "ontology_version": "1",
            "queries": [
                {
                    "id": "Q1",
                    "condition_concepts": ["acne_scar"],
                    "intervention_concepts": ["microneedling"],
                    "outcome_concepts": ["severity"],
                    "require_outcomes": False,
                }
            ],
            "negative_controls": [
                {
                    "id": "NC1",
                    "condition_concepts": ["low_back_pain"],
                    "intervention_concepts": ["microneedling"],
                    "outcome_concepts": [],
                    "require_outcomes": False,
                    "expected_disposition": "auto_exclude",
                }
            ],
        },
        ontology=_ontology(),
    )


def _campaign() -> dict:
    return {
        "schema_version": "giga.scientific-search-campaign.v1",
        "campaign_id": "test-campaign",
        "protocol": "protocol.md",
        "created_at": FIXED_TIME,
        "execution_state": "test",
        "defaults": {
            "endpoint": "search/papers",
            "corpus": "pubmed",
            "search_mode": "semantic",
            "max_results": 10,
            "filters": {"retracted": "exclude_retracted"},
        },
        "queries": [
            {
                "id": "Q1",
                "phase": 1,
                "question": "Does microneedling improve acne scars?",
            }
        ],
        "completion_rule": "Stop after test.",
    }


def _record(
    source_id: str,
    *,
    title: str = "Microneedling for atrophic acne scars",
    provider: str = "pubmed",
    doi: str | None = None,
    year: int = 2024,
    abstract: str | None = "Human patients had improved scar severity.",
) -> dict:
    value = {
        "schema_version": "giga.scientific-canonical-record.v1",
        "source_record_id": source_id,
        "acquisition_id": "a" * 64,
        "provider": provider,
        "endpoint": "search/papers",
        "kind": "paper",
        "query_id": "Q1",
        "negative_control": False,
        "title": title,
        "normalised_title": normalise_title(title),
        "abstract": abstract,
        "year": year,
        "doi": doi,
        "pmid": None,
        "pmcid": None,
        "nct_id": None,
        "authors": ["A Author"],
        "venue": "Journal",
        "publication_types": ["Randomized Controlled Trial"],
        "urls": [],
        "open_access_candidate": False,
        "retracted": False,
        "retrieved_at": FIXED_TIME,
        "raw_sha256": "b" * 64,
    }
    value["canonical_record_sha256"] = "c" * 64
    return value


def test_query_compiler_is_provider_specific_and_hashed() -> None:
    ontology, spec, campaign = _ontology(), _spec(), _campaign()
    pubmed = compile_provider_query("pubmed", spec["queries"][0], ontology)
    europe = compile_provider_query("europe_pmc", spec["queries"][0], ontology)
    assert '"Acne Vulgaris"[Mesh]' in pubmed
    assert '"microneedling"[tiab]' in pubmed
    assert 'TITLE_ABS:"microneedling"' in europe
    manifest = compile_manifest(
        campaign,
        ontology,
        spec,
        providers=["europe_pmc", "pubmed"],
    )
    assert manifest["schema_version"] == COMPILED_SCHEMA
    assert len(manifest["manifest_sha256"]) == 64
    assert len(manifest["queries"]) == 2


def test_compiled_campaign_forces_provider_expression() -> None:
    ontology, spec, campaign = _ontology(), _spec(), _campaign()
    manifest = compile_manifest(
        campaign,
        ontology,
        spec,
        providers=["europe_pmc"],
    )
    compiled = validate_campaign(embed_compiled_queries(campaign, manifest))
    job = provider_campaign_jobs(
        compiled["queries"],
        paper_providers=("europe_pmc",),
    )[0]
    assert job["request"]["query"].startswith("(")
    assert job["request"]["query"] != campaign["queries"][0]["question"]


def test_compiled_campaign_fails_if_selected_provider_has_no_expression() -> None:
    ontology, spec, campaign = _ontology(), _spec(), _campaign()
    manifest = compile_manifest(
        campaign,
        ontology,
        spec,
        providers=["europe_pmc"],
    )
    compiled = validate_campaign(embed_compiled_queries(campaign, manifest))
    with pytest.raises(ScientificEvidenceError, match="compiled provider query missing"):
        provider_campaign_jobs(
            compiled["queries"],
            paper_providers=("pubmed",),
        )


def test_query_compiler_fails_closed_for_wrong_concept_kind() -> None:
    payload = {
        "schema_version": "giga.scientific-search-specification.v1",
        "campaign_id": "test-campaign",
        "ontology_id": "test-ontology",
        "ontology_version": "1",
        "queries": [
            {
                "id": "Q1",
                "condition_concepts": ["microneedling"],
                "intervention_concepts": ["microneedling"],
                "outcome_concepts": [],
            }
        ],
        "negative_controls": [
            {
                "id": "NC1",
                "condition_concepts": ["low_back_pain"],
                "intervention_concepts": ["microneedling"],
                "outcome_concepts": [],
            }
        ],
    }
    with pytest.raises(ScientificEvidenceError, match="wrong-kind"):
        validate_search_spec(payload, ontology=_ontology())


def test_normalisation_and_exact_dedup_are_conservative() -> None:
    assert normalise_doi("HTTPS://DOI.ORG/10.1000/ABC.") == "10.1000/abc"
    assert normalise_title("Résumé—Skin!") == "resume skin"
    records = [
        _record("1" * 64, doi="10.1000/same", provider="europe_pmc"),
        _record("2" * 64, doi="10.1000/same", provider="pubmed"),
        _record(
            "3" * 64,
            title="Microneedling for a separate clinical outcome",
            doi="10.1000/other",
        ),
    ]
    assert sorted(len(group) for group in exact_deduplicate(records)) == [1, 2]


def test_fuzzy_duplicates_are_never_auto_merged() -> None:
    groups = [
        project_group(
            [
                _record(
                    "1" * 64,
                    title="Microneedling for atrophic acne scars: a trial",
                    doi="10.1000/one",
                )
            ]
        ),
        project_group(
            [
                _record(
                    "2" * 64,
                    title="Microneedling for atrophic acne scars - a trial",
                    doi="10.1000/two",
                )
            ]
        ),
    ]
    pairs = possible_duplicate_pairs(groups, threshold=0.9)
    assert len(pairs) == 1
    assert pairs[0]["auto_merged"] is False
    assert pairs[0]["disposition"] == "manual_review_required"


def test_human_duplicate_decision_can_merge_only_the_exact_hashed_pair() -> None:
    record_groups = [
        [
            _record(
                "1" * 64,
                title="Microneedling for atrophic acne scars: a trial",
                doi="10.1000/one",
            )
        ],
        [
            _record(
                "2" * 64,
                title="Microneedling for atrophic acne scars - a trial",
                doi="10.1000/two",
            )
        ],
    ]
    groups = [project_group(group) for group in record_groups]
    pairs = possible_duplicate_pairs(groups, threshold=0.9)
    document = {
        "schema_version": "giga.scientific-duplicate-decisions.v1",
        "campaign_id": "test-campaign",
        "pair_set_sha256": payload_hash(pairs),
        "decisions": [
            {
                "pair_sha256": pairs[0]["pair_sha256"],
                "decision": "same_study",
                "decided_by": "reviewer",
                "decided_at": FIXED_TIME,
                "evidence": {"reason": "Same title, author and trial."},
            }
        ],
    }
    merged, unresolved, report = apply_duplicate_decisions(
        record_groups,
        groups,
        pairs,
        document,
        campaign_id="test-campaign",
    )
    assert len(merged) == 1
    assert unresolved == []
    assert report["complete"] is True


def test_eligibility_rules_include_relevant_exclude_unrelated_and_queue_ambiguous() -> None:
    ontology, spec = _ontology(), _spec()
    relevant = project_group([_record("1" * 64)])
    features = extract_features(relevant, ontology)
    assert propose_eligibility(relevant, features, spec)["proposed_status"] == "auto_include"

    unrelated = project_group(
        [
            _record(
                "2" * 64,
                title="Adjunctive antipsychotics for depression",
                abstract="A human randomized clinical trial.",
            )
        ]
    )
    features = extract_features(unrelated, ontology)
    assert propose_eligibility(unrelated, features, spec)["proposed_status"] == "auto_exclude"

    ambiguous = project_group(
        [
            _record(
                "3" * 64,
                title="New treatment of acne scars",
                abstract=None,
            )
        ]
    )
    features = extract_features(ambiguous, ontology)
    assert propose_eligibility(ambiguous, features, spec)["proposed_status"] == "manual_review"


def test_exclusion_audit_sampling_is_deterministic() -> None:
    proposals = [
        {
            "group_id": f"{index:064x}",
            "proposed_status": "auto_exclude",
            "reason_codes": ["EX_TEST"],
            "title": f"Record {index}",
            "year": 2020,
        }
        for index in range(20)
    ]
    assert deterministic_exclusion_sample(
        proposals, per_reason=4
    ) == deterministic_exclusion_sample(proposals, per_reason=4)
    assert len(deterministic_exclusion_sample(proposals, per_reason=4)) == 4


def test_exclusion_audit_requires_complete_human_decisions(tmp_path: Path) -> None:
    expected = [
        {
            "group_id": "1" * 64,
            "title": "Unrelated record",
            "year": 2020,
        }
    ]
    completed = tmp_path / "completed.csv"
    completed.write_text(
        "group_id,title,year,sampled_reason_codes,human_audit_decision,"
        "human_audit_notes,audit_complete\n"
        f"{'1' * 64},Unrelated record,2020,\"[\"\"EX_TEST\"\"]\","
        "correct_exclusion,checked,True\n",
        encoding="utf-8",
    )
    report = validate_exclusion_audit(completed, expected)
    assert report["promotion_gate_passed"] is True
    completed.write_text(
        "group_id,title,year,sampled_reason_codes,human_audit_decision,"
        "human_audit_notes,audit_complete\n"
        f"{'1' * 64},Unrelated record,2020,\"[\"\"EX_TEST\"\"]\","
        "false_exclusion,missed,True\n",
        encoding="utf-8",
    )
    assert (
        validate_exclusion_audit(completed, expected)["promotion_gate_passed"]
        is False
    )


def test_negative_control_gate_requires_all_controls_to_be_excluded() -> None:
    group = {
        "group_id": "1" * 64,
        "negative_control": True,
    }
    passed = negative_control_report(
        [group],
        [
            {
                "group_id": group["group_id"],
                "proposed_status": "auto_exclude",
                "title": "Negative control",
            }
        ],
    )
    assert passed["promotion_gate_passed"] is True
    failed = negative_control_report(
        [group],
        [
            {
                "group_id": group["group_id"],
                "proposed_status": "manual_review",
                "title": "Negative control",
            }
        ],
    )
    assert failed["status"] == "fail"


def test_fulltext_plan_uses_lawful_routes_and_skips_exclusions() -> None:
    group = project_group(
        [
            {
                **_record("1" * 64, doi="10.1000/oa"),
                "pmcid": "PMC123",
                "urls": ["https://repository.example/paper.pdf"],
                "open_access_candidate": True,
            }
        ]
    )
    proposals = [
        {"group_id": group["group_id"], "proposed_status": "auto_include"}
    ]
    plan = plan_full_text([group], proposals)[0]
    route_types = [route["route_type"] for route in plan["routes"]]
    assert route_types[0] == "europe_pmc_fulltext_xml"
    assert "unpaywall_lookup" in route_types
    assert plan["paywall_circumvention_forbidden"] is True


def test_pmc_fetch_and_extraction_are_hash_bound(tmp_path: Path) -> None:
    xml = (
        b"<article><front><article-meta><title-group><article-title>Test paper"
        b"</article-title></title-group><abstract><p>Abstract text.</p></abstract>"
        b"</article-meta></front><body><sec><title>Results</title><p>Result text."
        b"</p></sec></body></article>"
    )
    destination = tmp_path / "PMC123.xml"
    receipt = fetch_europe_pmc_xml(
        "PMC123",
        destination,
        transport=lambda url, timeout: (xml, {"Content-Type": "application/xml"}),
    )
    assert receipt["full_text_status"] == "obtained_unappraised"
    assert receipt["sha256"]
    extraction = extract_pmc_xml(destination)
    assert extraction["source_sha256"] == receipt["sha256"]
    assert extraction["section_count"] >= 3
    assert extraction["full_text_verified"] is False


def test_pmc_fetch_rejects_entity_declarations(tmp_path: Path) -> None:
    with pytest.raises(ScientificEvidenceError, match="forbidden"):
        fetch_europe_pmc_xml(
            "PMC123",
            tmp_path / "bad.xml",
            transport=lambda url, timeout: (
                b'<!DOCTYPE x [<!ENTITY y SYSTEM "file:///etc/passwd">]><article/>',
                {},
            ),
        )


def test_review_state_enforces_actor_boundaries_and_is_idempotent(tmp_path: Path) -> None:
    store = ScientificReviewStore(tmp_path / "review.sqlite3")
    group = "1" * 64
    first = store.transition(
        campaign_id="campaign",
        group_id=group,
        from_state=None,
        to_state="discovered",
        actor_kind="deterministic",
        reason_codes=["DISCOVERED_TEST"],
        evidence={"source": "fixture"},
        recorded_at=FIXED_TIME,
    )
    assert first[1] is True
    assert (
        store.transition(
            campaign_id="campaign",
            group_id=group,
            from_state=None,
            to_state="discovered",
            actor_kind="deterministic",
            reason_codes=["DISCOVERED_TEST"],
            evidence={"source": "fixture"},
            recorded_at=FIXED_TIME,
        )[1]
        is False
    )
    with pytest.raises(ScientificEvidenceError, match="cannot perform"):
        store.transition(
            campaign_id="campaign",
            group_id=group,
            from_state="discovered",
            to_state="triaged",
            actor_kind="human",
            reason_codes=["INVALID_ACTOR_TEST"],
            evidence={"source": "fixture"},
            recorded_at=FIXED_TIME,
        )
    assert store.audit()["ok"] is True


def test_duplicate_decision_is_append_only_and_conflicts_fail(tmp_path: Path) -> None:
    store = ScientificReviewStore(tmp_path / "review.sqlite3")
    pair = "4" * 64
    first = store.record_duplicate_decision(
        campaign_id="campaign",
        pair_sha256=pair,
        decision="different_study",
        decided_by="reviewer",
        evidence={"reason": "Different interventions."},
        decided_at=FIXED_TIME,
    )
    assert first[1] is True
    assert (
        store.record_duplicate_decision(
            campaign_id="campaign",
            pair_sha256=pair,
            decision="different_study",
            decided_by="reviewer",
            evidence={"reason": "Different interventions."},
            decided_at=FIXED_TIME,
        )[1]
        is False
    )
    with pytest.raises(ScientificEvidenceError, match="different immutable"):
        store.record_duplicate_decision(
            campaign_id="campaign",
            pair_sha256=pair,
            decision="same_study",
            decided_by="reviewer",
            evidence={"reason": "Changed mind."},
            decided_at=FIXED_TIME,
        )


def test_llm_proposal_never_changes_state_and_requires_verbatim_span(tmp_path: Path) -> None:
    store = ScientificReviewStore(tmp_path / "review.sqlite3")
    group = "2" * 64
    for from_state, to_state, reason in (
        (None, "discovered", "DISCOVERED_TEST"),
        ("discovered", "triaged", "TRIAGED_TEST"),
        ("triaged", "awaiting_review", "AMBIGUOUS_TEST"),
    ):
        store.transition(
            campaign_id="campaign",
            group_id=group,
            from_state=from_state,
            to_state=to_state,
            actor_kind="deterministic",
            reason_codes=[reason],
            evidence={"source": "fixture", "step": to_state},
            recorded_at=FIXED_TIME,
        )
    proposal = {
        "schema_version": LLM_PROPOSAL_SCHEMA,
        "campaign_id": "campaign",
        "group_id": group,
        "decision": "include",
        "reason_code": "LLM_RELEVANT",
        "rationale": "The supplied title names the target condition.",
        "evidence_spans": [
            {"field": "title", "exact_text": "atrophic acne scars"}
        ],
        "model": "test-model",
        "proposed_at": FIXED_TIME,
    }
    store.record_llm_proposal(
        proposal,
        source_metadata={
            "title": "Microneedling for atrophic acne scars",
            "abstract": None,
        },
    )
    assert store.current_states("campaign")[group] == "awaiting_review"
    assert store.counts("campaign")["llm_proposals"] == 1
    bad = {**proposal, "evidence_spans": [{"field": "title", "exact_text": "invented"}]}
    with pytest.raises(ScientificEvidenceError, match="not verbatim"):
        store.record_llm_proposal(
            bad,
            source_metadata={"title": "Microneedling", "abstract": None},
        )


def test_review_ledger_audit_detects_out_of_band_mutation(tmp_path: Path) -> None:
    store = ScientificReviewStore(tmp_path / "review.sqlite3")
    store.transition(
        campaign_id="campaign",
        group_id="3" * 64,
        from_state=None,
        to_state="discovered",
        actor_kind="deterministic",
        reason_codes=["DISCOVERED_TEST"],
        evidence={"source": "fixture"},
        recorded_at=FIXED_TIME,
    )
    connection = sqlite3.connect(store.database)
    connection.execute("DROP TRIGGER no_update_review_events")
    connection.execute(
        "UPDATE review_events SET evidence_json='{}' WHERE sequence=1"
    )
    connection.commit()
    connection.close()
    audit = store.audit()
    assert audit["ok"] is False
    assert any("evidence hash" in error for error in audit["errors"])


def test_published_llm_contract_matches_runtime_schema() -> None:
    root = Path(__file__).resolve().parents[3]
    schema = json.loads(
        (
            root
            / "modules/platform/contracts/scientific-llm-eligibility-proposal.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert schema["properties"]["schema_version"]["const"] == LLM_PROPOSAL_SCHEMA
    assert schema["additionalProperties"] is False


def _write_saturation_snapshot(
    directory: Path,
    *,
    snapshot_sha256: str,
    retrieval_records: int,
    candidates: list[tuple[str, list[str]]],
) -> Path:
    directory.mkdir(parents=True)
    groups: list[dict] = []
    proposals: list[dict] = []
    features: list[dict] = []
    for index, (identity, interventions) in enumerate(candidates):
        group_id = f"{index + 1:064x}"
        groups.append(
            {
                "group_id": group_id,
                "canonical_identity": identity,
            }
        )
        proposals.append(
            {
                "group_id": group_id,
                "proposed_status": "auto_include",
            }
        )
        features.append(
            {
                "group_id": group_id,
                "concepts": {"interventions": interventions},
            }
        )

    artifacts = {
        "dedup_groups": ("DEDUP_GROUPS.jsonl", groups),
        "eligibility": ("ELIGIBILITY_PROPOSALS.jsonl", proposals),
        "features": ("FEATURES.jsonl", features),
    }
    receipts: dict[str, dict[str, str]] = {}
    for name, (filename, rows) in artifacts.items():
        path = directory / filename
        text = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
        path.write_text(text, encoding="utf-8")
        receipts[name] = {
            "path": filename,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    summary = {
        "snapshot_payload_sha256": snapshot_sha256,
        "retrieval_records": retrieval_records,
        "artifacts": receipts,
    }
    summary_path = directory / "SCREENING_SUMMARY.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def test_saturation_requires_distinct_cumulative_low_yield_snapshots(
    tmp_path: Path,
) -> None:
    candidates = [
        ("doi:10.1000/a", ["microneedling"]),
        ("pmid:123", ["microneedling"]),
    ]
    snapshots = [
        _write_saturation_snapshot(
            tmp_path / f"snapshot-{index}",
            snapshot_sha256=f"{index:064x}",
            retrieval_records=100 + index,
            candidates=candidates,
        )
        for index in (1, 2, 3)
    ]
    report = evaluate_saturation(snapshots)
    assert report["saturated"] is True
    assert all(item["pass"] for item in report["passes"][-2:])

    with pytest.raises(ScientificEvidenceError, match="distinct"):
        evaluate_saturation([snapshots[0], snapshots[0], snapshots[2]])


def test_screening_readiness_is_fail_closed_and_never_authorises_ranking(
    tmp_path: Path,
) -> None:
    campaign_id = "test-campaign"
    group_id = "9" * 64
    store = ScientificReviewStore(tmp_path / "review.sqlite3")
    for from_state, to_state, actor, reason in (
        (None, "discovered", "deterministic", "DISCOVERED_TEST"),
        ("discovered", "triaged", "deterministic", "TRIAGED_TEST"),
        ("triaged", "proposed_include", "deterministic", "PROPOSED_TEST"),
        ("proposed_include", "included", "human", "HUMAN_INCLUDED_TEST"),
    ):
        store.transition(
            campaign_id=campaign_id,
            group_id=group_id,
            from_state=from_state,
            to_state=to_state,
            actor_kind=actor,
            reason_codes=[reason],
            evidence={"source": "fixture", "state": to_state},
            recorded_at=FIXED_TIME,
        )

    report = assess_screening_readiness(
        main_summary={
            "campaign_id": campaign_id,
            "unmapped_records": 0,
            "unique_candidate_groups": 1,
            "duplicate_decisions": {"complete": True},
        },
        negative_control_summary={
            "negative_controls": {"promotion_gate_passed": True}
        },
        review_store=store,
        exclusion_audit_report={"promotion_gate_passed": True},
        saturation_report={"saturated": True},
    )
    assert report["screening_ready"] is True
    assert report["failed_gates"] == []
    assert report["full_text_appraisal_complete"] is False
    assert report["effectiveness_ranking_authorised"] is False

    blocked = assess_screening_readiness(
        main_summary={
            "campaign_id": campaign_id,
            "unmapped_records": 0,
            "unique_candidate_groups": 1,
            "duplicate_decisions": {"complete": True},
        },
        negative_control_summary={
            "negative_controls": {"promotion_gate_passed": True}
        },
        review_store=store,
        exclusion_audit_report=None,
        saturation_report=None,
    )
    assert blocked["screening_ready"] is False
    assert set(blocked["failed_gates"]) == {
        "exclusion_audit_pass",
        "search_saturation_pass",
    }


def test_porcine_collagen_material_is_not_mistaken_for_an_animal_study() -> None:
    record = _record(
        "8" * 64,
        title=(
            "Subcuticular incision versus porcine collagen filler "
            "for acne scars: a randomized split-face comparison"
        ),
        abstract=None,
    )
    record["compiled_query_retrieval"] = True
    group = project_group([record])
    features = extract_features(group, _ontology())
    proposal = propose_eligibility(group, features, _spec())
    assert proposal["proposed_status"] != "auto_exclude"
    assert "EX_NONHUMAN_ONLY" not in proposal["reason_codes"]


def test_editorial_type_conflicting_with_research_design_requires_review() -> None:
    record = _record(
        "7" * 64,
        title="Systematic review of microneedling for atrophic acne scars",
    )
    record["publication_types"] = ["Letter", "Systematic Review"]
    group = project_group([record])
    features = extract_features(group, _ontology())
    proposal = propose_eligibility(group, features, _spec())
    assert proposal["proposed_status"] == "manual_review"
    assert proposal["reason_codes"] == ["MR_NONRESEARCH_PUBLICATION_TYPE"]


def test_query_provenance_uses_request_limit_to_disambiguate_expressions() -> None:
    expression = 'TITLE_ABS:"acne scar"'
    campaign = {
        "defaults": {"max_results": 50},
        "queries": [
            {
                "id": "overview",
                "question": "Overview",
                "max_results": 100,
            },
            {
                "id": "primary",
                "question": "Primary",
                "max_results": 250,
            },
        ],
    }
    manifest = {
        "queries": [
            {
                "id": "overview",
                "negative_control": False,
                "provider_queries": {"europe_pmc": expression},
            },
            {
                "id": "primary",
                "negative_control": False,
                "provider_queries": {"europe_pmc": expression},
            },
        ]
    }
    lookup = _query_lookup(campaign, manifest)
    assert lookup[("europe_pmc", expression, 100)][0] == "overview"
    assert lookup[("europe_pmc", expression, 250)][0] == "primary"
    assert ("europe_pmc", expression, None) not in lookup


def test_control_only_ontology_concepts_do_not_pollute_main_features() -> None:
    record = _record(
        "6" * 64,
        title="Microneedling for acne scars and low back pain",
    )
    group = project_group([record])
    main_features = extract_features(group, _ontology(), _spec())
    assert "low_back_pain" not in main_features["concepts"]["conditions"]
    assert "microneedling" in main_features["concepts"]["interventions"]

    group["negative_control"] = True
    control_features = extract_features(group, _ontology(), _spec())
    assert "low_back_pain" in control_features["concepts"]["conditions"]
