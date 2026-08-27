"""Deterministic, resumable execution of scientific search campaigns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .scientific_evidence import (
    ElicitClient,
    ScientificEvidenceError,
    ScientificEvidenceStore,
    payload_hash,
    utc_now,
)
from .scientific_providers import (
    PAPER_SEARCH_PROVIDERS,
    ClinicalTrialsClient,
    paper_search_request,
)


CAMPAIGN_SCHEMA = "giga.scientific-search-campaign.v1"
CAMPAIGN_FIELDS = {
    "schema_version",
    "campaign_id",
    "protocol",
    "created_at",
    "execution_state",
    "defaults",
    "queries",
    "completion_rule",
}
QUERY_FIELDS = {
    "id",
    "phase",
    "endpoint",
    "question",
    "max_results",
    "corpus",
    "search_mode",
    "filters",
    "provider_queries",
}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScientificEvidenceError(f"{field} must be non-empty text")
    return value.strip()


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ScientificEvidenceError(f"{field} must be a positive integer")
    return value


def load_campaign(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScientificEvidenceError(f"cannot read campaign JSON from {path}") from exc
    return validate_campaign(payload)


def validate_campaign(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != CAMPAIGN_FIELDS:
        missing = sorted(CAMPAIGN_FIELDS - set(payload))
        extra = sorted(set(payload) - CAMPAIGN_FIELDS)
        raise ScientificEvidenceError(
            f"invalid campaign fields; missing={missing}, extra={extra}"
        )
    if payload["schema_version"] != CAMPAIGN_SCHEMA:
        raise ScientificEvidenceError(f"schema_version must be {CAMPAIGN_SCHEMA}")
    clean = dict(payload)
    clean["campaign_id"] = _text(clean["campaign_id"], "campaign_id")
    clean["protocol"] = _text(clean["protocol"], "protocol")
    clean["created_at"] = _text(clean["created_at"], "created_at")
    clean["execution_state"] = _text(clean["execution_state"], "execution_state")
    clean["completion_rule"] = _text(clean["completion_rule"], "completion_rule")
    defaults = clean["defaults"]
    if not isinstance(defaults, Mapping):
        raise ScientificEvidenceError("defaults must be an object")
    allowed_defaults = {
        "endpoint",
        "corpus",
        "search_mode",
        "max_results",
        "filters",
    }
    if not set(defaults).issubset(allowed_defaults):
        raise ScientificEvidenceError(
            f"unsupported campaign default(s): {sorted(set(defaults) - allowed_defaults)}"
        )
    queries = clean["queries"]
    if not isinstance(queries, list) or not queries:
        raise ScientificEvidenceError("queries must be a non-empty array")
    resolved: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, raw_query in enumerate(queries):
        if not isinstance(raw_query, Mapping):
            raise ScientificEvidenceError(f"queries[{index}] must be an object")
        if not set(raw_query).issubset(QUERY_FIELDS):
            raise ScientificEvidenceError(
                f"queries[{index}] has unsupported fields: "
                f"{sorted(set(raw_query) - QUERY_FIELDS)}"
            )
        merged = {**defaults, **raw_query}
        identifier = _text(merged.get("id"), f"queries[{index}].id")
        if identifier in identifiers:
            raise ScientificEvidenceError(f"duplicate campaign query id: {identifier}")
        identifiers.add(identifier)
        phase = _positive_integer(merged.get("phase"), f"queries[{index}].phase")
        endpoint = merged.get("endpoint", "search/papers")
        question = _text(merged.get("question"), f"queries[{index}].question")
        maximum = _positive_integer(
            merged.get("max_results", 100), f"queries[{index}].max_results"
        )
        if endpoint == "search/papers":
            request = ElicitClient.paper_search_request(
                question,
                max_results=maximum,
                corpus=merged.get("corpus", "elicit"),
                search_mode=merged.get("search_mode", "semantic"),
                filters=merged.get("filters"),
            )
        elif endpoint == "search/trials":
            request = ElicitClient.trial_search_request(question, max_results=maximum)
        else:
            raise ScientificEvidenceError(
                f"queries[{index}].endpoint must be search/papers or search/trials"
            )
        resolved.append(
            {
                "id": identifier,
                "phase": phase,
                "endpoint": endpoint,
                "request": request,
                "provider_queries": _provider_queries(
                    merged.get("provider_queries"),
                    endpoint=endpoint,
                    field=f"queries[{index}].provider_queries",
                ),
                "compiled_queries_required": clean["execution_state"]
                in {
                    "compiled_provider_queries_ready",
                    "compiled_negative_controls_not_executed",
                },
            }
        )
    clean["defaults"] = dict(defaults)
    clean["queries"] = resolved
    return clean


def _provider_queries(
    value: Any,
    *,
    endpoint: str,
    field: str,
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ScientificEvidenceError(f"{field} must be an object")
    allowed = (
        {"clinical_trials"}
        if endpoint == "search/trials"
        else set(PAPER_SEARCH_PROVIDERS)
    )
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ScientificEvidenceError(
            f"{field} has unsupported providers: {unknown}"
        )
    result = {
        _text(provider, f"{field} provider"): _text(
            query, f"{field}.{provider}"
        )
        for provider, query in value.items()
    }
    return dict(sorted(result.items()))


def select_queries(
    campaign: Mapping[str, Any],
    *,
    phases: Sequence[int] | None = None,
    query_ids: Sequence[str] | None = None,
    max_requests: int | None = None,
) -> list[dict[str, Any]]:
    selected = list(campaign["queries"])
    if phases:
        phase_set = set(phases)
        selected = [query for query in selected if query["phase"] in phase_set]
    if query_ids:
        requested = set(query_ids)
        known = {query["id"] for query in selected}
        missing = sorted(requested - known)
        if missing:
            raise ScientificEvidenceError(
                f"unknown or phase-excluded query id(s): {missing}"
            )
        selected = [query for query in selected if query["id"] in requested]
    if max_requests is not None:
        maximum = _positive_integer(max_requests, "max_requests")
        selected = selected[:maximum]
    if not selected:
        raise ScientificEvidenceError("campaign selection contains no queries")
    return selected


def campaign_plan(
    campaign: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    store: ScientificEvidenceStore,
) -> dict[str, Any]:
    queries: list[dict[str, Any]] = []
    for query in selected:
        prior = store.acquisitions_for_request(
            endpoint=query["endpoint"], request=query["request"]
        )
        queries.append(
            {
                "id": query["id"],
                "phase": query["phase"],
                "endpoint": query["endpoint"],
                "max_results": query["request"]["maxResults"],
                "already_acquired": bool(prior),
                "prior_acquisition_ids": prior,
            }
        )
    return {
        "schema_version": CAMPAIGN_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": payload_hash(campaign),
        "selected_queries": len(queries),
        "pending_queries": sum(not query["already_acquired"] for query in queries),
        "maximum_requested_records": sum(
            query["max_results"] for query in queries if not query["already_acquired"]
        ),
        "queries": queries,
    }


def execute_campaign(
    campaign: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    store: ScientificEvidenceStore,
    client: ElicitClient,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for query in selected:
        prior = store.acquisitions_for_request(
            endpoint=query["endpoint"], request=query["request"]
        )
        if prior and not refresh:
            results.append(
                {
                    "id": query["id"],
                    "status": "skipped_already_acquired",
                    "acquisition_ids": prior,
                }
            )
            continue
        request = query["request"]
        if query["endpoint"] == "search/papers":
            actual_request, response = client.search_papers(
                request["query"],
                max_results=request["maxResults"],
                corpus=request["corpus"],
                search_mode=request["searchMode"],
                filters=request.get("filters"),
            )
        else:
            actual_request, response = client.search_trials(
                request["query"], max_results=request["maxResults"]
            )
        if actual_request != request:
            raise ScientificEvidenceError(
                f"campaign request drift for query {query['id']}"
            )
        acquisition_id, created, source_ids = store.record_acquisition(
            endpoint=query["endpoint"],
            request=actual_request,
            response=response,
        )
        results.append(
            {
                "id": query["id"],
                "status": "acquired" if created else "exact_replay",
                "acquisition_ids": [acquisition_id],
                "source_record_ids": source_ids,
            }
        )
    return {
        "schema_version": CAMPAIGN_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": payload_hash(campaign),
        "completed_at": utc_now(),
        "results": results,
        "ledger_audit": store.audit(),
    }


def _provider_names(providers: Sequence[str]) -> list[str]:
    if isinstance(providers, (str, bytes)) or not providers:
        raise ScientificEvidenceError("paper providers must be a non-empty sequence")
    result = [_text(provider, "paper provider") for provider in providers]
    if len(result) != len(set(result)):
        raise ScientificEvidenceError("paper providers must not contain duplicates")
    unknown = set(result) - PAPER_SEARCH_PROVIDERS
    if unknown:
        raise ScientificEvidenceError(
            f"unsupported paper provider(s): {sorted(unknown)}"
        )
    return result


def provider_campaign_jobs(
    selected: Sequence[Mapping[str, Any]],
    *,
    paper_providers: Sequence[str],
) -> list[dict[str, Any]]:
    """Expand campaign questions into deterministic provider-specific requests."""
    providers = _provider_names(paper_providers)
    jobs: list[dict[str, Any]] = []
    for query in selected:
        campaign_request = query["request"]
        if query["endpoint"] == "search/trials":
            provider_query = query.get("provider_queries", {}).get("clinical_trials")
            if provider_query is None:
                if query.get("compiled_queries_required"):
                    raise ScientificEvidenceError(
                        f"compiled provider query missing for "
                        f"{query['id']}:clinical_trials"
                    )
                provider_query = campaign_request["query"]
            request = ClinicalTrialsClient.trial_search_request(
                provider_query,
                max_results=campaign_request["maxResults"],
            )
            jobs.append(
                {
                    "id": query["id"],
                    "phase": query["phase"],
                    "provider": "clinical_trials",
                    "endpoint": "search/trials",
                    "request": request,
                }
            )
            continue
        for provider in providers:
            provider_query = query.get("provider_queries", {}).get(provider)
            if provider_query is None:
                if query.get("compiled_queries_required"):
                    raise ScientificEvidenceError(
                        f"compiled provider query missing for {query['id']}:{provider}"
                    )
                provider_query = campaign_request["query"]
            request = paper_search_request(
                provider,
                provider_query,
                max_results=campaign_request["maxResults"],
                filters=campaign_request.get("filters"),
            )
            jobs.append(
                {
                    "id": query["id"],
                    "phase": query["phase"],
                    "provider": provider,
                    "endpoint": "search/papers",
                    "request": request,
                }
            )
    return jobs


def provider_campaign_plan(
    campaign: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    store: ScientificEvidenceStore,
    *,
    paper_providers: Sequence[str],
) -> dict[str, Any]:
    jobs = provider_campaign_jobs(
        selected,
        paper_providers=paper_providers,
    )
    planned: list[dict[str, Any]] = []
    for job in jobs:
        prior = store.acquisitions_for_request(
            provider=job["provider"],
            endpoint=job["endpoint"],
            request=job["request"],
        )
        planned.append(
            {
                "id": job["id"],
                "phase": job["phase"],
                "provider": job["provider"],
                "endpoint": job["endpoint"],
                "max_results": job["request"]["maxResults"],
                "already_acquired": bool(prior),
                "prior_acquisition_ids": prior,
            }
        )
    return {
        "schema_version": CAMPAIGN_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": payload_hash(campaign),
        "selected_queries": len(selected),
        "selected_provider_jobs": len(planned),
        "pending_provider_jobs": sum(not job["already_acquired"] for job in planned),
        "maximum_requested_records": sum(
            job["max_results"] for job in planned if not job["already_acquired"]
        ),
        "jobs": planned,
    }


def execute_provider_campaign(
    campaign: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    store: ScientificEvidenceStore,
    paper_clients: Mapping[str, Any],
    trial_client: ClinicalTrialsClient,
    *,
    paper_providers: Sequence[str],
    refresh: bool = False,
) -> dict[str, Any]:
    """Execute a provider-expanded campaign without granting provider authority."""
    jobs = provider_campaign_jobs(
        selected,
        paper_providers=paper_providers,
    )
    missing = sorted(
        {
            job["provider"]
            for job in jobs
            if job["endpoint"] == "search/papers"
            and job["provider"] not in paper_clients
        }
    )
    if missing:
        raise ScientificEvidenceError(
            f"missing configured paper client(s): {missing}"
        )
    results: list[dict[str, Any]] = []
    for job in jobs:
        prior = store.acquisitions_for_request(
            provider=job["provider"],
            endpoint=job["endpoint"],
            request=job["request"],
        )
        if prior and not refresh:
            results.append(
                {
                    "id": job["id"],
                    "provider": job["provider"],
                    "status": "skipped_already_acquired",
                    "acquisition_ids": prior,
                }
            )
            continue
        request = job["request"]
        if job["endpoint"] == "search/trials":
            actual_request, response = trial_client.search_trials(
                request["query"],
                max_results=request["maxResults"],
            )
        else:
            client = paper_clients[job["provider"]]
            actual_request, response = client.search_papers(
                request["query"],
                max_results=request["maxResults"],
                filters=request["filters"],
            )
        if actual_request != request:
            raise ScientificEvidenceError(
                f"provider request drift for {job['id']}:{job['provider']}"
            )
        acquisition_id, created, source_ids = store.record_acquisition(
            provider=job["provider"],
            endpoint=job["endpoint"],
            request=actual_request,
            response=response,
        )
        results.append(
            {
                "id": job["id"],
                "provider": job["provider"],
                "status": "acquired" if created else "exact_replay",
                "acquisition_ids": [acquisition_id],
                "source_record_ids": source_ids,
            }
        )
    return {
        "schema_version": CAMPAIGN_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": payload_hash(campaign),
        "completed_at": utc_now(),
        "results": results,
        "ledger_audit": store.audit(),
    }
