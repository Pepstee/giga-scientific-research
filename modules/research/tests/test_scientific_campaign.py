from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from modules.research.core.scientific_campaign import (
    campaign_plan,
    execute_campaign,
    execute_provider_campaign,
    load_campaign,
    provider_campaign_plan,
    select_queries,
    validate_campaign,
)
from modules.research.core.scientific_evidence import (
    ElicitClient,
    ScientificEvidenceError,
    ScientificEvidenceStore,
)
from modules.research.core.scientific_providers import (
    ClinicalTrialsClient,
    EuropePMCClient,
)


ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CAMPAIGN = ROOT / "examples/systematic-review-campaign.json"


def _campaign() -> dict:
    return {
        "schema_version": "giga.scientific-search-campaign.v1",
        "campaign_id": "test-campaign",
        "protocol": "test-protocol.md",
        "created_at": "2026-07-28T15:00:00Z",
        "execution_state": "test",
        "defaults": {
            "endpoint": "search/papers",
            "corpus": "elicit",
            "search_mode": "semantic",
            "max_results": 2,
            "filters": {"retracted": "exclude_retracted"},
        },
        "queries": [
            {"id": "papers", "phase": 1, "question": "paper question"},
            {
                "id": "trials",
                "phase": 2,
                "endpoint": "search/trials",
                "question": "trial question",
            },
        ],
        "completion_rule": "Stop after the test.",
    }


def test_example_campaign_is_valid_and_covers_all_four_phases() -> None:
    campaign = load_campaign(EXAMPLE_CAMPAIGN)
    assert len(campaign["queries"]) == 4
    assert Counter(query["phase"] for query in campaign["queries"]) == {
        1: 1,
        2: 1,
        3: 1,
        4: 1,
    }
    assert all(
        query["request"].get("filters", {}).get("retracted") == "exclude_retracted"
        for query in campaign["queries"]
        if query["endpoint"] == "search/papers"
    )


def test_campaign_validation_rejects_duplicate_ids_and_unknown_fields() -> None:
    payload = _campaign()
    payload["queries"][1]["id"] = "papers"
    with pytest.raises(ScientificEvidenceError, match="duplicate"):
        validate_campaign(payload)
    payload = _campaign()
    payload["queries"][0]["invented"] = True
    with pytest.raises(ScientificEvidenceError, match="unsupported"):
        validate_campaign(payload)


def test_selection_is_deterministic_and_fail_closed() -> None:
    campaign = validate_campaign(_campaign())
    selected = select_queries(campaign, phases=[1], max_requests=1)
    assert [query["id"] for query in selected] == ["papers"]
    with pytest.raises(ScientificEvidenceError, match="unknown"):
        select_queries(campaign, phases=[1], query_ids=["trials"])
    with pytest.raises(ScientificEvidenceError, match="positive integer"):
        select_queries(campaign, max_requests=0)


def test_campaign_execution_records_results_and_skips_prior_requests(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def transport(method, url, body, headers, timeout):
        calls.append(url)
        if url.endswith("/search/papers"):
            return {
                "papers": [
                    {
                        "title": "Paper result",
                        "year": 2025,
                        "doi": "10.1000/campaign",
                    }
                ]
            }
        return {
            "trials": [
                {
                    "title": "Trial result",
                    "nctId": "NCT00000001",
                }
            ]
        }

    campaign = validate_campaign(_campaign())
    selected = select_queries(campaign)
    store = ScientificEvidenceStore(tmp_path / "private" / "campaign.sqlite3")
    client = ElicitClient(api_key="test-key", transport=transport)
    first = execute_campaign(campaign, selected, store, client)
    assert [result["status"] for result in first["results"]] == [
        "acquired",
        "acquired",
    ]
    assert first["ledger_audit"]["ok"] is True
    assert len(calls) == 2

    second = execute_campaign(campaign, selected, store, client)
    assert all(
        result["status"] == "skipped_already_acquired" for result in second["results"]
    )
    assert len(calls) == 2
    plan = campaign_plan(campaign, selected, store)
    assert plan["pending_queries"] == 0
    assert plan["maximum_requested_records"] == 0


def test_raw_campaign_file_is_canonical_json_compatible() -> None:
    payload = json.loads(EXAMPLE_CAMPAIGN.read_text(encoding="utf-8"))
    assert validate_campaign(payload)["campaign_id"] == payload["campaign_id"]


def test_provider_campaign_fans_out_and_replays_by_provider(tmp_path: Path) -> None:
    campaign = validate_campaign(_campaign())
    selected = select_queries(campaign)
    store = ScientificEvidenceStore(tmp_path / "private" / "providers.sqlite3")
    plan = provider_campaign_plan(
        campaign,
        selected,
        store,
        paper_providers=("europe_pmc",),
    )
    assert plan["selected_queries"] == 2
    assert plan["selected_provider_jobs"] == 2
    assert {job["provider"] for job in plan["jobs"]} == {
        "europe_pmc",
        "clinical_trials",
    }

    europe = EuropePMCClient(
        transport=lambda *args: {
            "resultList": {
                "result": [
                    {
                        "title": "Provider paper",
                        "pubYear": "2025",
                        "doi": "10.1000/provider",
                    }
                ]
            }
        }
    )
    trials = ClinicalTrialsClient(
        transport=lambda *args: {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT00000002",
                            "briefTitle": "Provider trial",
                        }
                    }
                }
            ]
        }
    )
    first = execute_provider_campaign(
        campaign,
        selected,
        store,
        {"europe_pmc": europe},
        trials,
        paper_providers=("europe_pmc",),
    )
    assert [result["status"] for result in first["results"]] == [
        "acquired",
        "acquired",
    ]
    assert first["ledger_audit"]["ok"] is True
    second = execute_provider_campaign(
        campaign,
        selected,
        store,
        {"europe_pmc": europe},
        trials,
        paper_providers=("europe_pmc",),
    )
    assert all(
        result["status"] == "skipped_already_acquired"
        for result in second["results"]
    )
