"""Deterministic search-saturation accounting across screening snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .scientific_evidence import ScientificEvidenceError, payload_hash


SATURATION_SCHEMA = "giga.scientific-search-saturation.v1"


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScientificEvidenceError(f"cannot read JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ScientificEvidenceError(f"{path} must contain an object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ScientificEvidenceError(
                    f"{path}:{line_number} must be an object"
                )
            result.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ScientificEvidenceError(f"cannot read JSONL from {path}") from exc
    return result


def _artifact(
    summary_path: Path,
    summary: Mapping[str, Any],
    name: str,
) -> Path:
    record = summary.get("artifacts", {}).get(name)
    if not isinstance(record, Mapping):
        raise ScientificEvidenceError(
            f"{summary_path} has no {name} artifact receipt"
        )
    path = summary_path.resolve().parent / record["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ScientificEvidenceError(f"{name} artifact hash mismatch: {path}")
    return path


def _snapshot(summary_path: Path) -> dict[str, Any]:
    summary = _json(summary_path)
    groups = _jsonl(_artifact(summary_path, summary, "dedup_groups"))
    proposals = _jsonl(_artifact(summary_path, summary, "eligibility"))
    features = _jsonl(_artifact(summary_path, summary, "features"))
    proposal_by_id = {item["group_id"]: item for item in proposals}
    feature_by_id = {item["group_id"]: item for item in features}
    retained = {
        group["canonical_identity"]
        for group in groups
        if proposal_by_id[group["group_id"]]["proposed_status"]
        in {"auto_include", "manual_review"}
    }
    interventions = {
        concept
        for group in groups
        if proposal_by_id[group["group_id"]]["proposed_status"]
        in {"auto_include", "manual_review"}
        for concept in feature_by_id[group["group_id"]]["concepts"][
            "interventions"
        ]
    }
    return {
        "summary_path": str(summary_path.resolve()),
        "snapshot_payload_sha256": summary["snapshot_payload_sha256"],
        "retrieval_records": summary["retrieval_records"],
        "retained_candidate_identities": retained,
        "intervention_families": interventions,
    }


def evaluate_saturation(
    summary_paths: Sequence[Path],
    *,
    maximum_new_candidate_rate: float = 0.01,
) -> dict[str, Any]:
    if len(summary_paths) < 3:
        raise ScientificEvidenceError(
            "saturation requires at least three sequential snapshots"
        )
    if not 0 <= maximum_new_candidate_rate <= 1:
        raise ScientificEvidenceError(
            "maximum_new_candidate_rate must be between 0 and 1"
        )
    snapshots = [_snapshot(path) for path in summary_paths]
    hashes = [item["snapshot_payload_sha256"] for item in snapshots]
    if len(hashes) != len(set(hashes)):
        raise ScientificEvidenceError(
            "saturation snapshots must be distinct; replayed snapshots do not count"
        )
    retrieval_counts = [item["retrieval_records"] for item in snapshots]
    if retrieval_counts != sorted(retrieval_counts):
        raise ScientificEvidenceError(
            "saturation snapshots must be cumulative in nondecreasing record order"
        )
    passes: list[dict[str, Any]] = []
    for previous, current in zip(snapshots, snapshots[1:]):
        new_candidates = (
            current["retained_candidate_identities"]
            - previous["retained_candidate_identities"]
        )
        new_families = (
            current["intervention_families"]
            - previous["intervention_families"]
        )
        denominator = max(1, len(previous["retained_candidate_identities"]))
        rate = len(new_candidates) / denominator
        passes.append(
            {
                "from_snapshot_sha256": previous["snapshot_payload_sha256"],
                "to_snapshot_sha256": current["snapshot_payload_sha256"],
                "new_retained_candidate_identities": sorted(new_candidates),
                "new_intervention_families": sorted(new_families),
                "new_candidate_rate": rate,
                "pass": not new_families and rate < maximum_new_candidate_rate,
            }
        )
    last_two = passes[-2:]
    report = {
        "schema_version": SATURATION_SCHEMA,
        "snapshots": [
            {
                key: value
                for key, value in snapshot.items()
                if key
                not in {"retained_candidate_identities", "intervention_families"}
            }
            for snapshot in snapshots
        ],
        "passes": passes,
        "completion_rule": {
            "required_consecutive_passes": 2,
            "maximum_new_candidate_rate": maximum_new_candidate_rate,
            "new_intervention_families_required": 0,
        },
        "saturated": len(last_two) == 2 and all(item["pass"] for item in last_two),
        "scientific_synthesis_complete": False,
    }
    report["report_sha256"] = payload_hash(report)
    return report
