"""Fail-closed readiness gates for deterministic scientific screening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .scientific_evidence import ScientificEvidenceError, payload_hash
from .scientific_review_state import ScientificReviewStore


READINESS_SCHEMA = "giga.scientific-screening-readiness.v1"


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScientificEvidenceError(f"cannot read JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ScientificEvidenceError(f"{path} must contain a JSON object")
    return value


def assess_screening_readiness(
    *,
    main_summary: Mapping[str, Any],
    negative_control_summary: Mapping[str, Any],
    review_store: ScientificReviewStore,
    exclusion_audit_report: Mapping[str, Any] | None,
    saturation_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    campaign_id = main_summary.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ScientificEvidenceError("main screening summary has no campaign_id")
    review_audit = review_store.audit()
    review_counts = review_store.counts(campaign_id)
    negative = negative_control_summary.get("negative_controls")
    if not isinstance(negative, Mapping):
        raise ScientificEvidenceError(
            "negative-control summary has no negative_controls report"
        )
    duplicate = main_summary.get("duplicate_decisions", {})
    unresolved_states = {
        state: count
        for state, count in review_counts["states"].items()
        if state
        in {
            "awaiting_review",
            "proposed_exclude",
            "proposed_include",
            "triaged",
            "discovered",
        }
    }
    gates = {
        "acquisition_records_mapped": main_summary.get("unmapped_records") == 0,
        "review_ledger_integrity": review_audit["ok"] is True,
        "negative_controls_pass": negative.get("promotion_gate_passed") is True,
        "possible_duplicates_resolved": duplicate.get("complete") is True,
        "exclusion_audit_pass": (
            exclusion_audit_report is not None
            and exclusion_audit_report.get("promotion_gate_passed") is True
        ),
        "eligibility_decisions_complete": not unresolved_states
        and review_counts["groups"] == main_summary.get("unique_candidate_groups"),
        "search_saturation_pass": (
            saturation_report is not None
            and saturation_report.get("saturated") is True
        ),
    }
    report = {
        "schema_version": READINESS_SCHEMA,
        "campaign_id": campaign_id,
        "gates": gates,
        "failed_gates": sorted(
            gate for gate, passed in gates.items() if not passed
        ),
        "screening_ready": all(gates.values()),
        "review_states": review_counts,
        "review_ledger_head_sha256": review_audit["ledger_head_sha256"],
        "full_text_appraisal_complete": False,
        "scientific_synthesis_complete": False,
        "effectiveness_ranking_authorised": False,
        "authority": "screening_readiness_only",
    }
    report["report_sha256"] = payload_hash(report)
    return report


def assess_screening_readiness_files(
    *,
    main_summary: Path,
    negative_control_summary: Path,
    review_database: Path,
    exclusion_audit_report: Path | None,
    saturation_report: Path | None,
) -> dict[str, Any]:
    return assess_screening_readiness(
        main_summary=_json(main_summary),
        negative_control_summary=_json(negative_control_summary),
        review_store=ScientificReviewStore(review_database),
        exclusion_audit_report=(
            _json(exclusion_audit_report) if exclusion_audit_report else None
        ),
        saturation_report=_json(saturation_report) if saturation_report else None,
    )
