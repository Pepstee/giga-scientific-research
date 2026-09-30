from __future__ import annotations

import json
import hashlib
import sqlite3
import stat
import tempfile
from pathlib import Path

import pytest

from modules.research.core.scientific_evidence import (
    APPRAISAL_FIELDS,
    APPRAISAL_SCHEMA,
    CANDIDATE_SCHEMA,
    ElicitAPIError,
    ElicitClient,
    ScientificEvidenceError,
    ScientificEvidenceStore,
    evidence_tier,
)
from modules.research.core.scientific_evidence_cli import main as cli_main


FIXED_TIME = "2026-07-28T12:00:00+00:00"


def _papers() -> dict:
    return {
        "papers": [
            {
                "title": "Controlled trial one",
                "authors": ["A. Researcher"],
                "year": 2024,
                "doi": "10.1000/one",
                "pmid": "1001",
                "abstract": "An effect was estimated.",
                "venue": "Journal One",
                "citedByCount": 12,
                "urls": ["https://doi.org/10.1000/one"],
            },
            {
                "title": "Controlled trial two",
                "authors": ["B. Researcher"],
                "year": 2025,
                "doi": "10.1000/two",
                "pmid": "1002",
                "abstract": "A second effect was estimated.",
                "venue": "Journal Two",
                "citedByCount": 4,
                "urls": ["https://doi.org/10.1000/two"],
            },
        ]
    }


def _store(root: Path) -> ScientificEvidenceStore:
    return ScientificEvidenceStore(root / "private" / "scientific.sqlite3")


def _record_sources(store: ScientificEvidenceStore) -> list[str]:
    _, _, source_ids = store.record_acquisition(
        endpoint="search/papers",
        request={
            "query": "test intervention",
            "maxResults": 2,
            "filters": {"retracted": "exclude_retracted"},
        },
        response=_papers(),
        observed_at=FIXED_TIME,
    )
    return source_ids


def _appraisal(source_id: str, study_identity: str, **overrides: object) -> dict:
    result = {
        "schema_version": APPRAISAL_SCHEMA,
        "source_record_id": source_id,
        "study_identity": study_identity,
        "study_design": "randomized_controlled_trial",
        "human_verified": True,
        "full_text_verified": True,
        "risk_of_bias": "low",
        "population_applicability": "direct",
        "harms_assessed": True,
        "conflicts_assessed": True,
        "retraction_status": "clear",
        "effect_estimate": "mean difference -1.2",
        "confidence_interval": "95% CI -1.8 to -0.6",
        "outcome_direction": "beneficial",
        "supporting_quote": "The prespecified outcome improved in the intervention group.",
        "source_locator": "Results, Table 2",
        "assessed_by": "operator",
        "assessed_at": FIXED_TIME,
        "notes": [],
    }
    result.update(overrides)
    return result


def test_client_search_is_private_redaction_safe_and_excludes_retractions() -> None:
    captured: dict = {}

    def transport(method, url, body, headers, timeout):
        captured.update(
            method=method, url=url, body=body, headers=headers, timeout=timeout
        )
        return {"papers": [_papers()["papers"][0]]}

    client = ElicitClient(api_key="super-secret", transport=transport)
    request, response = client.search_papers("sleep and cognition")
    assert request["filters"]["retracted"] == "exclude_retracted"
    assert request["maxResults"] == 10
    assert captured["url"] == "https://elicit.com/api/v2/search/papers"
    assert captured["headers"]["Authorization"] == "Bearer super-secret"
    assert response["papers"][0]["doi"] == "10.1000/one"
    client.get_session("session/with/slash")
    assert captured["url"].endswith("/sessions/reports/session%2Fwith%2Fslash")

    def failed_transport(method, url, body, headers, timeout):
        raise ElicitAPIError(401, "invalid_key", "super-secret was rejected")

    with pytest.raises(ElicitAPIError) as caught:
        ElicitClient(api_key="super-secret", transport=failed_transport).search_papers(
            "x"
        )
    assert "super-secret" not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)


def test_client_rejects_unsafe_or_invalid_requests_before_transport() -> None:
    with pytest.raises(ScientificEvidenceError, match="keyword search"):
        ElicitClient(api_key="x", transport=lambda *args: {}).search_papers(
            "title:test",
            search_mode="keyword",
            filters={"minYear": 2020},
        )
    with pytest.raises(ScientificEvidenceError, match="unsupported typeTags"):
        ElicitClient(api_key="x", transport=lambda *args: {}).search_papers(
            "test", filters={"typeTags": ["Case report"]}
        )
    with pytest.raises(ScientificEvidenceError, match="official"):
        ElicitClient(api_key="x", base_url="http://example.test")
    with pytest.raises(ScientificEvidenceError, match="official"):
        ElicitClient(api_key="x", base_url="https://example.test/api/v2")
    with pytest.raises(ScientificEvidenceError, match="must remain private"):
        ElicitClient(api_key="x", transport=lambda *args: {}).create_report(
            "test", is_public=True
        )


def test_credential_material_is_rejected_before_immutable_persistence() -> None:
    with tempfile.TemporaryDirectory() as folder:
        store = _store(Path(folder))
        with pytest.raises(ScientificEvidenceError, match="credential field"):
            store.record_acquisition(
                endpoint="search/papers",
                request={
                    "query": "test",
                    "headers": {"Authorization": "Bearer should-never-be-stored"},
                },
                response={"papers": []},
                observed_at=FIXED_TIME,
            )
        assert store.counts()["acquisitions"] == 0


def test_acquisition_is_append_only_idempotent_and_owner_only() -> None:
    with tempfile.TemporaryDirectory() as folder:
        store = _store(Path(folder))
        source_ids = _record_sources(store)
        replay_ids = _record_sources(store)
        assert source_ids == replay_ids
        assert store.counts() == {
            "acquisitions": 1,
            "source_records": 2,
            "appraisals": 0,
            "action_candidates": 0,
            "ledger_entries": 1,
        }
        assert stat.S_IMODE(store.database.stat().st_mode) == 0o600
        assert stat.S_IMODE(store.database.parent.stat().st_mode) == 0o700
        with store.connect() as connection:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(
                    "UPDATE source_records SET title = 'changed' WHERE id = ?",
                    (source_ids[0],),
                )


def test_source_identity_and_raw_provider_response_are_preserved() -> None:
    with tempfile.TemporaryDirectory() as folder:
        store = _store(Path(folder))
        source_ids = _record_sources(store)
        source = store.get_source(source_ids[0])
        assert source["identity_key"] == "doi:10.1000/one"
        assert source["raw"]["abstract"] == "An effect was estimated."
        assert len(source["raw_sha256"]) == 64
        assert store.audit()["ok"] is True


def test_cumulative_record_cap_rejects_whole_response_without_breaking_audit(
    tmp_path: Path,
) -> None:
    store = ScientificEvidenceStore(tmp_path / "cap" / "evidence.sqlite3")
    first = [
        {"title": f"Record {index}", "doi": f"10.1000/cap-{index}"}
        for index in range(19)
    ]
    store.record_acquisition(
        provider="europe_pmc",
        endpoint="search/papers",
        request={"query": "first", "maxResults": 19},
        response={"papers": first},
        observed_at=FIXED_TIME,
    )
    before = store.counts()
    with pytest.raises(ScientificEvidenceError, match="would be exceeded"):
        store.record_acquisition(
            provider="openalex",
            endpoint="search/papers",
            request={"query": "second", "maxResults": 2},
            response={
                "papers": [
                    {"title": "New 1", "doi": "10.1000/cap-new-1"},
                    {"title": "New 2", "doi": "10.1000/cap-new-2"},
                ]
            },
            observed_at=FIXED_TIME,
        )
    assert store.counts() == before
    assert before["source_records"] == 19
    assert store.audit()["ok"] is True


def test_aggregate_cap_counts_preserved_rows_and_identities_across_ledgers(
    tmp_path: Path,
) -> None:
    first_store = ScientificEvidenceStore(tmp_path / "run-a" / "evidence.sqlite3")
    second_store = ScientificEvidenceStore(tmp_path / "run-b" / "evidence.sqlite3")
    first_keys = [f"10.1000/aggregate-{index}" for index in range(19)]
    second_keys = first_keys[:3] + [
        f"10.1000/aggregate-{index}" for index in range(19, 34)
    ]

    def add_records(
        store: ScientificEvidenceStore, query: str, dois: list[str]
    ) -> None:
        store.record_acquisition(
            provider="europe_pmc",
            endpoint="search/papers",
            request={"query": query, "maxResults": len(dois)},
            response={
                "papers": [
                    {"title": f"Record {doi}", "doi": doi} for doi in dois
                ]
            },
            observed_at=FIXED_TIME,
        )

    add_records(first_store, "first ledger", first_keys)
    add_records(second_store, "second ledger", second_keys)
    first_hash = hashlib.sha256(first_store.database.read_bytes()).hexdigest()
    second_hash = hashlib.sha256(second_store.database.read_bytes()).hexdigest()
    current = ScientificEvidenceStore(tmp_path / "current" / "evidence.sqlite3")

    budget = current.acquisition_budget(
        (first_store.database, second_store.database)
    )
    assert budget["source_records"] == 37
    assert budget["unique_identities"] == 34
    with pytest.raises(
        ScientificEvidenceError,
        match="existing aggregate has 37 source records / 34 unique identities",
    ):
        current.require_acquisition_capacity(
            (first_store.database, second_store.database)
        )
    with pytest.raises(ScientificEvidenceError, match="would be exceeded"):
        current.record_acquisition(
            provider="openalex",
            endpoint="search/papers",
            request={"query": "new", "maxResults": 1},
            response={
                "papers": [{"title": "New", "doi": "10.1000/aggregate-new"}]
            },
            observed_at=FIXED_TIME,
            aggregate_databases=(first_store.database, second_store.database),
        )

    assert current.counts()["acquisitions"] == 0
    assert hashlib.sha256(first_store.database.read_bytes()).hexdigest() == first_hash
    assert hashlib.sha256(second_store.database.read_bytes()).hexdigest() == second_hash


@pytest.mark.parametrize(
    ("design", "risk", "expected"),
    [
        ("systematic_review", "low", "T1"),
        ("randomized_controlled_trial", "low", "T2"),
        ("randomized_controlled_trial", "high", "T3"),
        ("cohort", "unclear", "T4"),
        ("animal", "low", "T5"),
        ("meta_analysis", "critical", "U"),
    ],
)
def test_evidence_tier_is_deterministic(design: str, risk: str, expected: str) -> None:
    assert evidence_tier(design, risk) == expected


def test_appraisal_is_strict_and_requires_traceable_human_verification() -> None:
    with tempfile.TemporaryDirectory() as folder:
        store = _store(Path(folder))
        source_id = _record_sources(store)[0]
        invalid = _appraisal(source_id, "doi:10.1000/one", supporting_quote=None)
        with pytest.raises(ScientificEvidenceError, match="supporting_quote"):
            store.record_appraisal(invalid)
        invalid = _appraisal(source_id, "doi:10.1000/one")
        invalid["invented"] = True
        with pytest.raises(ScientificEvidenceError, match="invalid appraisal fields"):
            store.record_appraisal(invalid)


def test_unappraised_search_results_cannot_become_a_plan() -> None:
    with tempfile.TemporaryDirectory() as folder:
        store = _store(Path(folder))
        source_ids = _record_sources(store)
        _, candidate = store.create_action_candidate(
            question="Does it work?",
            intervention="Try the intervention",
            risk_class="low_risk_lifestyle",
            source_record_ids=source_ids,
            created_at=FIXED_TIME,
        )
        readiness = candidate["readiness"]
        assert candidate["authority_level"] == "A0"
        assert candidate["recommendation_authority"] is False
        assert readiness["status"] == "blocked"
        assert readiness["ready_for_execution"] is False
        assert "all_sources_appraised" in readiness["failed_gates"]


def test_two_verified_independent_rcts_can_only_become_a0_plan_candidate() -> None:
    with tempfile.TemporaryDirectory() as folder:
        store = _store(Path(folder))
        source_ids = _record_sources(store)
        for index, source_id in enumerate(source_ids, 1):
            appraisal_id, created, tier = store.record_appraisal(
                _appraisal(source_id, f"doi:10.1000/{index}")
            )
            assert created is True
            assert tier == "T2"
            assert len(appraisal_id) == 64
        _, candidate = store.create_action_candidate(
            question="Does it work?",
            intervention="Run a bounded trial",
            risk_class="low_risk_lifestyle",
            source_record_ids=source_ids,
            created_at=FIXED_TIME,
        )
        readiness = candidate["readiness"]
        assert readiness["status"] == "ready_for_plan"
        assert readiness["ready_for_plan"] is True
        assert readiness["ready_for_execution"] is False
        assert readiness["recommendation_authority"] is False
        assert readiness["failed_gates"] == []
        assert store.audit()["ok"] is True


def test_contradictions_and_clinical_risk_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as folder:
        store = _store(Path(folder))
        source_ids = _record_sources(store)
        store.record_appraisal(_appraisal(source_ids[0], "study:one"))
        store.record_appraisal(
            _appraisal(
                source_ids[1],
                "study:two",
                outcome_direction="harmful",
                effect_estimate="risk ratio 1.4",
                confidence_interval="95% CI 1.1 to 1.8",
            )
        )
        _, candidate = store.create_action_candidate(
            question="Should this be used clinically?",
            intervention="Clinical intervention",
            risk_class="clinical",
            source_record_ids=source_ids,
            created_at=FIXED_TIME,
        )
        failed = candidate["readiness"]["failed_gates"]
        assert "contradictions_resolved" in failed
        assert "clinical_review_complete" in failed


def test_audit_detects_hash_chain_corruption_if_database_is_modified_out_of_band() -> (
    None
):
    with tempfile.TemporaryDirectory() as folder:
        store = _store(Path(folder))
        _record_sources(store)
        connection = sqlite3.connect(store.database)
        connection.execute("DROP TRIGGER no_update_ledger_entries")
        connection.execute(
            "UPDATE ledger_entries SET payload_json = ? WHERE sequence = 1",
            (json.dumps({"tampered": True}),),
        )
        connection.commit()
        connection.close()
        result = store.audit()
        assert result["ok"] is False
        assert any("payload hash mismatch" in error for error in result["errors"])
        assert any("trigger" in error for error in result["errors"])


def test_audit_detects_out_of_band_action_candidate_mutation() -> None:
    with tempfile.TemporaryDirectory() as folder:
        store = _store(Path(folder))
        source_ids = _record_sources(store)
        for index, source_id in enumerate(source_ids, 1):
            store.record_appraisal(_appraisal(source_id, f"study:{index}"))
        candidate_id, _ = store.create_action_candidate(
            question="Does it work?",
            intervention="Bounded trial",
            risk_class="low_risk_lifestyle",
            source_record_ids=source_ids,
            created_at=FIXED_TIME,
        )
        connection = sqlite3.connect(store.database)
        connection.execute("DROP TRIGGER no_update_action_candidates")
        connection.execute(
            "UPDATE action_candidates SET question = 'tampered' WHERE id = ?",
            (candidate_id,),
        )
        connection.commit()
        connection.close()
        result = store.audit()
        assert result["ok"] is False
        assert any("action candidate" in error for error in result["errors"])


def test_published_contracts_match_runtime_enums_and_fields() -> None:
    root = Path(__file__).resolve().parents[3]
    appraisal_schema = json.loads(
        (root / "modules/platform/contracts/scientific-appraisal.v1.schema.json").read_text()
    )
    assert set(appraisal_schema["required"]) == APPRAISAL_FIELDS
    assert appraisal_schema["properties"]["schema_version"]["const"] == APPRAISAL_SCHEMA
    candidate_schema = json.loads(
        (root / "modules/platform/contracts/scientific-action-candidate.v1.schema.json").read_text()
    )
    assert candidate_schema["properties"]["schema_version"]["const"] == CANDIDATE_SCHEMA
    assert candidate_schema["properties"]["authority_level"]["const"] == "A0"
    assert candidate_schema["properties"]["recommendation_authority"]["const"] is False


def test_cli_import_status_and_audit_work_without_network(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "scientific.sqlite3"
    request_file = tmp_path / "request.json"
    response_file = tmp_path / "response.json"
    request_file.write_text(
        json.dumps(
            {
                "query": "test intervention",
                "maxResults": 2,
                "filters": {"retracted": "exclude_retracted"},
            }
        ),
        encoding="utf-8",
    )
    response_file.write_text(json.dumps(_papers()), encoding="utf-8")

    assert (
        cli_main(
            [
                "--database",
                str(database),
                "import-response",
                "--endpoint",
                "search/papers",
                "--request",
                str(request_file),
                "--response",
                str(response_file),
                "--observed-at",
                FIXED_TIME,
            ]
        )
        == 0
    )
    imported = json.loads(capsys.readouterr().out)
    assert imported["created"] is True
    assert len(imported["source_record_ids"]) == 2

    assert cli_main(["--database", str(database), "status"]) == 0
    status_output = json.loads(capsys.readouterr().out)
    assert status_output["counts"]["source_records"] == 2

    assert cli_main(["--database", str(database), "audit"]) == 0
    audit_output = json.loads(capsys.readouterr().out)
    assert audit_output["ok"] is True
