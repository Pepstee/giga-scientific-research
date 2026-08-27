"""Deterministic orchestration for staged scientific-paper review.

This module connects screening to appraisal without granting abstracts, provider metadata,
or language models authority to make scientific conclusions.  It creates review queues and
hash-bound contracts; humans remain responsible for eligibility, full-text extraction,
risk-of-bias assessment, and synthesis sign-off.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .scientific_evidence import ScientificEvidenceError, canonical_json, payload_hash


FUNNEL_SCHEMA = "giga.scientific-review-funnel.v1"
ABSTRACT_RETRIEVAL_SCHEMA = "giga.scientific-abstract-retrieval.v1"
ELIGIBILITY_DECISIONS_SCHEMA = "giga.scientific-eligibility-decisions.v1"
STUDY_EXTRACTION_SCHEMA = "giga.scientific-study-extraction.v1"
SYNTHESIS_SIGNOFF_SCHEMA = "giga.scientific-synthesis-signoff.v1"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PMID_RE = re.compile(r"^\d{1,12}$")
PUBMED_DOCTYPE_RE = re.compile(
    br"""<!DOCTYPE\s+PubmedArticleSet\s+PUBLIC\s+
        "-//NLM//DTD\s+PubMedArticle,\s+[^"]+//EN"\s+
        "https://dtd\.nlm\.nih\.gov/ncbi/pubmed/out/pubmed_\d+\.dtd"\s*>""",
    re.IGNORECASE | re.VERBOSE,
)

DESIGN_PRIORITY = {
    "guideline": 10,
    "systematic_review": 20,
    "meta_analysis": 20,
    "randomized_controlled_trial": 30,
    "controlled_trial": 40,
    "clinical_trial": 40,
    "cohort_or_longitudinal": 50,
    "case_control": 50,
    "case_series": 60,
    "nonsystematic_review": 70,
    "protocol": 80,
    "primary_or_other": 80,
}
REVIEWABLE_STATUSES = {"include", "exclude"}
RISK_OF_BIAS = {"low", "some_concerns", "high", "critical", "unclear"}
APPLICABILITY = {"direct", "partial", "indirect", "unknown"}


def _text(value: Any, field: str, *, maximum: int = 100_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScientificEvidenceError(f"{field} must be non-empty text")
    result = value.strip()
    if len(result) > maximum:
        raise ScientificEvidenceError(f"{field} exceeds {maximum} characters")
    return result


def _sha(value: Any, field: str) -> str:
    result = _text(value, field, maximum=64)
    if not SHA256_RE.fullmatch(result):
        raise ScientificEvidenceError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _timestamp(value: Any, field: str) -> str:
    result = _text(value, field, maximum=100)
    try:
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ScientificEvidenceError(f"{field} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ScientificEvidenceError(f"{field} must include a timezone")
    return parsed.isoformat()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ScientificEvidenceError(
                        f"{path}:{number} must contain a JSON object"
                    )
                values.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ScientificEvidenceError(f"cannot read JSONL from {path}") from exc
    return values


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScientificEvidenceError(f"cannot read JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ScientificEvidenceError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Any) -> dict[str, Any]:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return _artifact(path, 1)


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical_json(value) + "\n")
    return _artifact(path, len(values))


def _write_csv(
    path: Path,
    values: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> dict[str, Any]:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)
    return _artifact(path, len(values))


def _artifact(path: Path, records: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "records": records,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _clean_abstract(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    without_tags = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return " ".join(without_tags.split()) or None


def abstract_retrieval_queue(
    groups: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    proposal_by_id = {item["group_id"]: item for item in proposals}
    queue: list[dict[str, Any]] = []
    for group in groups:
        group_id = _sha(group.get("group_id"), "group_id")
        proposal = proposal_by_id.get(group_id)
        if proposal is None:
            raise ScientificEvidenceError(f"missing proposal for group {group_id}")
        if proposal.get("proposed_status") == "auto_exclude" or _clean_abstract(
            group.get("abstract")
        ):
            continue
        identifiers = {
            "pmid": group.get("pmid"),
            "pmcid": group.get("pmcid"),
            "doi": group.get("doi"),
        }
        routes: list[dict[str, Any]] = []
        if isinstance(identifiers["pmid"], str) and PMID_RE.fullmatch(
            identifiers["pmid"]
        ):
            routes.append({"provider": "pubmed", "identifier": identifiers["pmid"], "priority": 10})
        if isinstance(identifiers["pmcid"], str):
            routes.append({"provider": "europe_pmc", "identifier": identifiers["pmcid"], "priority": 20})
        if isinstance(identifiers["doi"], str):
            routes.append({"provider": "crossref_or_openalex", "identifier": identifiers["doi"], "priority": 30})
        item = {
            "schema_version": ABSTRACT_RETRIEVAL_SCHEMA,
            "group_id": group_id,
            "title": group.get("title"),
            "identifiers": identifiers,
            "routes": routes,
            "retrieval_status": "pending" if routes else "no_machine_identifier",
            "abstract_substitution_for_full_text_forbidden": True,
        }
        item["request_sha256"] = payload_hash(item)
        queue.append(item)
    return sorted(queue, key=lambda item: item["group_id"])


def apply_abstract_enrichments(
    groups: Sequence[Mapping[str, Any]],
    retrievals: Sequence[Mapping[str, Any]],
    ontology: Mapping[str, Any],
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only unambiguous identifier-matched abstracts and recompute screening.

    Existing abstracts are never overwritten. Conflicting or unmatched retrievals are
    reported for human resolution, and all feature/proposal artifacts are regenerated so a
    decision can never be made from stale pre-enrichment triage.
    """
    from .scientific_screening import extract_features, propose_eligibility

    pmid_groups: dict[str, list[str]] = defaultdict(list)
    by_id: dict[str, dict[str, Any]] = {}
    for raw_group in groups:
        group = dict(raw_group)
        group_id = _sha(group.get("group_id"), "group_id")
        if group_id in by_id:
            raise ScientificEvidenceError("groups must have unique group_id values")
        by_id[group_id] = group
        pmid = group.get("pmid")
        if isinstance(pmid, str):
            pmid_groups[pmid].append(group_id)
    applied: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    seen_pmids: set[str] = set()
    for retrieval in retrievals:
        if retrieval.get("schema_version") != ABSTRACT_RETRIEVAL_SCHEMA:
            raise ScientificEvidenceError("abstract retrieval has wrong schema")
        pmid = _text(retrieval.get("pmid"), "retrieval.pmid", maximum=12)
        if not PMID_RE.fullmatch(pmid) or pmid in seen_pmids:
            raise ScientificEvidenceError("retrieval PMIDs must be valid and unique")
        seen_pmids.add(pmid)
        candidates = pmid_groups.get(pmid, [])
        abstract = _clean_abstract(retrieval.get("abstract"))
        if len(candidates) != 1 or abstract is None:
            conflicts.append(
                {
                    "pmid": pmid,
                    "reason": "identifier_not_unique"
                    if len(candidates) > 1
                    else "group_not_found"
                    if not candidates
                    else "abstract_empty",
                    "candidate_group_ids": candidates,
                }
            )
            continue
        group = by_id[candidates[0]]
        existing = _clean_abstract(group.get("abstract"))
        if existing is not None:
            if existing != abstract:
                conflicts.append(
                    {
                        "pmid": pmid,
                        "reason": "existing_abstract_differs",
                        "candidate_group_ids": candidates,
                        "existing_abstract_sha256": hashlib.sha256(
                            existing.encode("utf-8")
                        ).hexdigest(),
                        "retrieved_abstract_sha256": hashlib.sha256(
                            abstract.encode("utf-8")
                        ).hexdigest(),
                    }
                )
            continue
        group["abstract"] = abstract
        group["abstract_enrichment"] = {
            "provider": retrieval.get("provider"),
            "pmid": pmid,
            "retrieval_sha256": retrieval.get("retrieval_sha256"),
            "abstract_sha256": hashlib.sha256(abstract.encode("utf-8")).hexdigest(),
            "authority": "eligibility_metadata_only",
        }
        material = dict(group)
        material.pop("group_payload_sha256", None)
        group["group_payload_sha256"] = payload_hash(material)
        applied.append(
            {
                "group_id": group["group_id"],
                "pmid": pmid,
                "abstract_sha256": group["abstract_enrichment"]["abstract_sha256"],
            }
        )
    enriched_groups = sorted(by_id.values(), key=lambda item: item["group_id"])
    features = [
        extract_features(group, ontology, specification) for group in enriched_groups
    ]
    feature_by_id = {item["group_id"]: item for item in features}
    proposals = [
        propose_eligibility(
            group,
            feature_by_id[group["group_id"]],
            specification,
        )
        for group in enriched_groups
    ]
    report = {
        "schema_version": ABSTRACT_RETRIEVAL_SCHEMA,
        "groups": len(enriched_groups),
        "retrievals": len(retrievals),
        "applied": len(applied),
        "conflicts": len(conflicts),
        "applied_records": applied,
        "conflict_records": conflicts,
        "features_recomputed": True,
        "eligibility_proposals_recomputed": True,
        "human_decisions_invalidated_by_abstract_change": bool(applied),
        "effectiveness_conclusion_authority": False,
    }
    report["report_sha256"] = payload_hash(report)
    return {
        "groups": enriched_groups,
        "features": features,
        "proposals": proposals,
        "report": report,
    }


def apply_abstract_enrichments_files(
    *,
    groups: Path,
    retrievals: Path,
    ontology: Path,
    specification: Path,
    output: Path,
) -> dict[str, Any]:
    from .scientific_query_compiler import load_ontology, load_search_spec

    ontology_value = load_ontology(ontology)
    specification_value = load_search_spec(specification, ontology=ontology_value)
    build = apply_abstract_enrichments(
        _jsonl(groups),
        _jsonl(retrievals),
        ontology_value,
        specification_value,
    )
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "groups": _write_jsonl(output / "ENRICHED_GROUPS.jsonl", build["groups"]),
        "features": _write_jsonl(output / "ENRICHED_FEATURES.jsonl", build["features"]),
        "proposals": _write_jsonl(
            output / "ENRICHED_ELIGIBILITY_PROPOSALS.jsonl", build["proposals"]
        ),
        "report": _write_json(
            output / "ABSTRACT_ENRICHMENT_REPORT.json", build["report"]
        ),
    }
    return {**build["report"], "artifacts": artifacts}


def validate_eligibility_decisions(
    document: Mapping[str, Any] | None,
    groups: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if document is None:
        return {}, [str(group["group_id"]) for group in groups]
    if document.get("schema_version") != ELIGIBILITY_DECISIONS_SCHEMA:
        raise ScientificEvidenceError(
            f"eligibility decisions must use {ELIGIBILITY_DECISIONS_SCHEMA}"
        )
    raw = document.get("decisions")
    if not isinstance(raw, list):
        raise ScientificEvidenceError("eligibility decisions must contain an array")
    groups_by_id = {str(group["group_id"]): group for group in groups}
    result: dict[str, dict[str, Any]] = {}
    for index, decision in enumerate(raw):
        if not isinstance(decision, Mapping):
            raise ScientificEvidenceError(f"decisions[{index}] must be an object")
        group_id = _sha(decision.get("group_id"), f"decisions[{index}].group_id")
        if group_id not in groups_by_id or group_id in result:
            raise ScientificEvidenceError(
                f"decision references unknown or duplicate group {group_id}"
            )
        disposition = decision.get("decision")
        if disposition not in REVIEWABLE_STATUSES:
            raise ScientificEvidenceError("decision must be include or exclude")
        reasons = decision.get("reason_codes")
        if (
            not isinstance(reasons, list)
            or not reasons
            or not all(isinstance(item, str) and item.strip() for item in reasons)
        ):
            raise ScientificEvidenceError("reason_codes must be a non-empty text array")
        reviewer = _text(decision.get("reviewed_by"), "reviewed_by", maximum=500)
        reviewed_at = _timestamp(decision.get("reviewed_at"), "reviewed_at")
        expected_abstract = hashlib.sha256(
            (_clean_abstract(groups_by_id[group_id].get("abstract")) or "").encode("utf-8")
        ).hexdigest()
        recorded_abstract = _sha(
            decision.get("abstract_sha256"), "abstract_sha256"
        )
        if recorded_abstract != expected_abstract:
            raise ScientificEvidenceError(
                f"eligibility decision for {group_id} is stale: abstract hash changed"
            )
        result[group_id] = {
            "decision": disposition,
            "reason_codes": list(dict.fromkeys(reasons)),
            "reviewed_by": reviewer,
            "reviewed_at": reviewed_at,
            "abstract_sha256": recorded_abstract,
            "human_verified": True,
        }
    unresolved = sorted(set(groups_by_id) - set(result))
    return result, unresolved


def _screening_rows(
    groups: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    proposal_by_id = {item["group_id"]: item for item in proposals}
    feature_by_id = {item["group_id"]: item for item in features}
    rows: list[dict[str, Any]] = []
    for group in groups:
        group_id = str(group["group_id"])
        proposal = proposal_by_id.get(group_id)
        feature = feature_by_id.get(group_id)
        if proposal is None or feature is None:
            raise ScientificEvidenceError(f"incomplete screening inputs for {group_id}")
        abstract = _clean_abstract(group.get("abstract"))
        decision = decisions.get(group_id, {})
        rows.append(
            {
                "group_id": group_id,
                "title": group.get("title"),
                "year": group.get("year"),
                "design_bucket": feature.get("design_bucket"),
                "abstract_available": bool(abstract),
                "abstract_sha256": hashlib.sha256((abstract or "").encode("utf-8")).hexdigest(),
                "proposed_status": proposal.get("proposed_status"),
                "proposal_reasons": "|".join(proposal.get("reason_codes", [])),
                "condition_concepts": "|".join(feature.get("concepts", {}).get("conditions", [])),
                "intervention_concepts": "|".join(feature.get("concepts", {}).get("interventions", [])),
                "outcome_concepts": "|".join(feature.get("concepts", {}).get("outcomes", [])),
                "human_decision": decision.get("decision", ""),
                "human_reason_codes": "|".join(decision.get("reason_codes", [])),
                "reviewed_by": decision.get("reviewed_by", ""),
                "reviewed_at": decision.get("reviewed_at", ""),
                "abstract_is_eligibility_only": True,
                "effectiveness_authority": False,
            }
        )
    return rows


def _priority_rows(
    groups: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    feature_by_id = {item["group_id"]: item for item in features}
    rows: list[dict[str, Any]] = []
    for group in groups:
        group_id = str(group["group_id"])
        if decisions.get(group_id, {}).get("decision") != "include":
            continue
        feature = feature_by_id[group_id]
        design = str(feature.get("design_bucket") or "primary_or_other")
        priority = DESIGN_PRIORITY.get(design, 80)
        rows.append(
            {
                "group_id": group_id,
                "priority": priority,
                "design_bucket": design,
                "year": group.get("year") or "",
                "title": group.get("title"),
                "pmcid": group.get("pmcid") or "",
                "doi": group.get("doi") or "",
                "condition_concepts": "|".join(feature.get("concepts", {}).get("conditions", [])) or "unclassified",
                "intervention_concepts": "|".join(feature.get("concepts", {}).get("interventions", [])) or "unclassified",
                "review_coverage_status": "unknown_requires_reference_check",
                "reason": f"P{priority:02d}_{design.upper()}",
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            int(item["priority"]),
            -(int(item["year"]) if str(item["year"]).isdigit() else 0),
            str(item["group_id"]),
        ),
    )


def validate_study_extraction(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "group_id", "full_text_sha256", "full_text_verified",
        "study_design", "population", "condition", "intervention_protocol",
        "comparator", "sample_size", "outcomes", "effect_estimates",
        "follow_up", "adverse_events", "withdrawals", "funding", "conflicts",
        "risk_of_bias", "population_applicability", "source_locators",
        "extracted_by", "extracted_at", "notes",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ScientificEvidenceError(
            f"invalid study extraction fields; missing={sorted(required-set(value))}, "
            f"extra={sorted(set(value)-required)}"
        )
    clean = dict(value)
    if clean["schema_version"] != STUDY_EXTRACTION_SCHEMA:
        raise ScientificEvidenceError(
            f"schema_version must be {STUDY_EXTRACTION_SCHEMA}"
        )
    clean["group_id"] = _sha(clean["group_id"], "group_id")
    clean["full_text_sha256"] = _sha(clean["full_text_sha256"], "full_text_sha256")
    if clean["full_text_verified"] is not True:
        raise ScientificEvidenceError("study extraction requires verified full text")
    for field in (
        "study_design", "population", "condition", "intervention_protocol",
        "comparator", "follow_up", "adverse_events", "withdrawals", "funding",
        "conflicts", "extracted_by",
    ):
        clean[field] = _text(clean[field], field)
    if (
        isinstance(clean["sample_size"], bool)
        or not isinstance(clean["sample_size"], int)
        or clean["sample_size"] < 1
    ):
        raise ScientificEvidenceError("sample_size must be a positive integer")
    for field in ("outcomes", "effect_estimates", "source_locators", "notes"):
        if not isinstance(clean[field], list):
            raise ScientificEvidenceError(f"{field} must be an array")
    if not clean["outcomes"] or not clean["source_locators"]:
        raise ScientificEvidenceError("outcomes and source_locators must be non-empty")
    if not all(isinstance(item, Mapping) for item in clean["effect_estimates"]):
        raise ScientificEvidenceError("effect_estimates must contain objects")
    clean["risk_of_bias"] = _text(clean["risk_of_bias"], "risk_of_bias")
    if clean["risk_of_bias"] not in RISK_OF_BIAS:
        raise ScientificEvidenceError("risk_of_bias is invalid")
    clean["population_applicability"] = _text(
        clean["population_applicability"], "population_applicability"
    )
    if clean["population_applicability"] not in APPLICABILITY:
        raise ScientificEvidenceError("population_applicability is invalid")
    clean["extracted_at"] = _timestamp(clean["extracted_at"], "extracted_at")
    return clean


def _comparison_matrix(
    priority_rows: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    extractions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    feature_by_id = {item["group_id"]: item for item in features}
    matrix: list[dict[str, Any]] = []
    for row in priority_rows:
        group_id = str(row["group_id"])
        feature = feature_by_id[group_id]
        conditions = feature.get("concepts", {}).get("conditions", []) or ["unclassified"]
        interventions = feature.get("concepts", {}).get("interventions", []) or ["unclassified"]
        extraction = extractions.get(group_id)
        for condition in conditions:
            for intervention in interventions:
                matrix.append(
                    {
                        "condition": condition,
                        "intervention": intervention,
                        "group_id": group_id,
                        "design_bucket": row["design_bucket"],
                        "priority": row["priority"],
                        "extraction_complete": extraction is not None,
                        "sample_size": extraction.get("sample_size") if extraction else None,
                        "outcomes": extraction.get("outcomes") if extraction else [],
                        "effect_estimates": extraction.get("effect_estimates") if extraction else [],
                        "follow_up": extraction.get("follow_up") if extraction else None,
                        "adverse_events": extraction.get("adverse_events") if extraction else None,
                        "risk_of_bias": extraction.get("risk_of_bias") if extraction else "unappraised",
                        "population_applicability": extraction.get("population_applicability") if extraction else "unknown",
                        "source_locators": extraction.get("source_locators") if extraction else [],
                        "recommendation_authority": False,
                    }
                )
    return sorted(
        matrix,
        key=lambda item: (
            item["condition"], item["intervention"], int(item["priority"]), item["group_id"]
        ),
    )


def build_review_funnel(
    *,
    groups: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    output: Path,
    eligibility_document: Mapping[str, Any] | None = None,
    extractions: Sequence[Mapping[str, Any]] = (),
    duplicate_decision_report: Mapping[str, Any] | None = None,
    synthesis_signoff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    group_ids = [str(item.get("group_id")) for item in groups]
    if len(group_ids) != len(set(group_ids)):
        raise ScientificEvidenceError("groups must have unique group_id values")
    decisions, unresolved = validate_eligibility_decisions(
        eligibility_document, groups
    )
    clean_extractions: dict[str, dict[str, Any]] = {}
    for value in extractions:
        clean = validate_study_extraction(value)
        if clean["group_id"] not in set(group_ids) or clean["group_id"] in clean_extractions:
            raise ScientificEvidenceError("extraction references unknown or duplicate group")
        clean_extractions[clean["group_id"]] = clean
    duplicate_decisions_complete = False
    if duplicate_decision_report is not None:
        recorded_hash = duplicate_decision_report.get("report_sha256")
        material = dict(duplicate_decision_report)
        material.pop("report_sha256", None)
        duplicate_decisions_complete = (
            isinstance(recorded_hash, str)
            and recorded_hash == payload_hash(material)
            and duplicate_decision_report.get("complete") is True
            and duplicate_decision_report.get("unresolved_pairs") == 0
        )

    retrieval = abstract_retrieval_queue(groups, proposals)
    screening = _screening_rows(groups, proposals, features, decisions)
    priorities = _priority_rows(groups, features, decisions)
    matrix = _comparison_matrix(priorities, features, clean_extractions)
    proposal_counts = Counter(str(item.get("proposed_status")) for item in proposals)
    metadata_report = {
        "schema_version": FUNNEL_SCHEMA,
        "stage": "metadata_screening",
        "groups": len(groups),
        "proposal_counts": dict(sorted(proposal_counts.items())),
        "abstracts_available": sum(
            _clean_abstract(item.get("abstract")) is not None for item in groups
        ),
        "abstracts_missing": sum(
            _clean_abstract(item.get("abstract")) is None for item in groups
        ),
        "authority": "deterministic_triage_only",
        "effectiveness_conclusion_authority": False,
    }
    metadata_report["report_sha256"] = payload_hash(metadata_report)
    evidence_groups: dict[str, dict[str, Any]] = {}
    for row in matrix:
        key = f"{row['condition']}::{row['intervention']}"
        bucket = evidence_groups.setdefault(
            key,
            {
                "condition": row["condition"],
                "intervention": row["intervention"],
                "group_ids": [],
                "design_counts": Counter(),
                "extracted": 0,
            },
        )
        bucket["group_ids"].append(row["group_id"])
        bucket["design_counts"][row["design_bucket"]] += 1
        bucket["extracted"] += int(row["extraction_complete"])
    group_document = {
        "schema_version": FUNNEL_SCHEMA,
        "stage": "condition_intervention_grouping",
        "groups": [
            {
                **value,
                "group_ids": sorted(set(value["group_ids"])),
                "design_counts": dict(sorted(value["design_counts"].items())),
            }
            for _, value in sorted(evidence_groups.items())
        ],
        "authority": "organisation_only",
    }
    group_document["report_sha256"] = payload_hash(group_document)
    eligibility_template = {
        "schema_version": ELIGIBILITY_DECISIONS_SCHEMA,
        "campaign_id": eligibility_document.get("campaign_id")
        if eligibility_document
        else "replace-with-campaign-id",
        "instructions": (
            "Copy rows from ABSTRACT_SCREENING_QUEUE.csv. Record include/exclude only "
            "after checking title and abstract; bind each decision to abstract_sha256."
        ),
        "decisions": [],
    }
    fulltext_queue = [
        {
            **row,
            "selection_basis": "human_eligibility_include",
            "full_text_required_for_synthesis": True,
            "abstract_substitution_forbidden": True,
            "reference_coverage_check_required": row["design_bucket"]
            in {"randomized_controlled_trial", "controlled_trial", "clinical_trial"},
        }
        for row in priorities
    ]

    artifacts = {
        "metadata_screening": _write_json(
            output / "METADATA_SCREENING_REPORT.json", metadata_report
        ),
        "abstract_retrieval": _write_jsonl(output / "ABSTRACT_RETRIEVAL_QUEUE.jsonl", retrieval),
        "abstract_screening": _write_csv(
            output / "ABSTRACT_SCREENING_QUEUE.csv",
            screening,
            [
                "group_id", "title", "year", "design_bucket", "abstract_available",
                "abstract_sha256", "proposed_status", "proposal_reasons",
                "condition_concepts", "intervention_concepts", "outcome_concepts",
                "human_decision", "human_reason_codes", "reviewed_by", "reviewed_at",
                "abstract_is_eligibility_only", "effectiveness_authority",
            ],
        ),
        "priority_queue": _write_csv(
            output / "EVIDENCE_PRIORITY_QUEUE.csv",
            priorities,
            [
                "group_id", "priority", "design_bucket", "year", "title", "pmcid",
                "doi", "condition_concepts", "intervention_concepts",
                "review_coverage_status", "reason",
            ],
        ),
        "selective_fulltext_queue": _write_jsonl(
            output / "SELECTIVE_FULLTEXT_QUEUE.jsonl", fulltext_queue
        ),
        "evidence_groups": _write_json(
            output / "EVIDENCE_GROUPS.json", group_document
        ),
        "comparison_matrix": _write_jsonl(output / "EVIDENCE_COMPARISON_MATRIX.jsonl", matrix),
        "eligibility_decisions_template": _write_json(
            output / "ELIGIBILITY_DECISIONS_TEMPLATE.json", eligibility_template
        ),
    }
    matrix_sha = artifacts["comparison_matrix"]["sha256"]
    selected = {str(item["group_id"]) for item in priorities}
    extracted = set(clean_extractions)
    gates = {
        "duplicate_decisions_complete": duplicate_decisions_complete,
        "eligibility_decisions_complete": not unresolved,
        "included_full_text_extractions_complete": bool(selected)
        and selected == extracted,
        "risk_of_bias_complete": bool(selected)
        and all(
            clean_extractions[item]["risk_of_bias"] != "unclear"
            for item in selected & extracted
        )
        and selected == extracted,
        "effect_estimates_source_located": bool(selected)
        and all(
            clean_extractions[item]["effect_estimates"]
            and clean_extractions[item]["source_locators"]
            for item in selected & extracted
        )
        and selected == extracted,
        "harms_and_conflicts_extracted": bool(selected)
        and all(
            clean_extractions[item]["adverse_events"]
            and clean_extractions[item]["conflicts"]
            for item in selected & extracted
        )
        and selected == extracted,
    }
    signoff_valid = False
    if synthesis_signoff is not None:
        signoff_valid = (
            synthesis_signoff.get("schema_version") == SYNTHESIS_SIGNOFF_SCHEMA
            and synthesis_signoff.get("comparison_matrix_sha256") == matrix_sha
            and synthesis_signoff.get("human_verified") is True
            and isinstance(synthesis_signoff.get("reviewed_by"), str)
            and bool(synthesis_signoff["reviewed_by"].strip())
        )
    gates["human_synthesis_signoff"] = signoff_valid
    readiness = {
        "schema_version": FUNNEL_SCHEMA,
        "counts": {
            "groups": len(groups),
            "abstracts_missing_retrieval_queued": len(retrieval),
            "eligibility_decisions": len(decisions),
            "eligibility_unresolved": len(unresolved),
            "included_for_full_text": len(selected),
            "study_extractions": len(extracted),
            "comparison_rows": len(matrix),
        },
        "gates": gates,
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
        "scientific_synthesis_ready": all(gates.values()),
        "effectiveness_ranking_authorised": all(gates.values()),
        "recommendation_authority": False,
        "llm_calls_made": 0,
        "abstracts_are_effectiveness_evidence": False,
        "comparison_matrix_sha256": matrix_sha,
        "artifacts": artifacts,
    }
    readiness["report_sha256"] = payload_hash(readiness)
    _write_json(output / "REVIEW_FUNNEL_READINESS.json", readiness)
    return readiness


def build_review_funnel_files(
    *,
    groups: Path,
    proposals: Path,
    features: Path,
    output: Path,
    eligibility_decisions: Path | None = None,
    extractions: Path | None = None,
    duplicate_decision_report: Path | None = None,
    synthesis_signoff: Path | None = None,
) -> dict[str, Any]:
    return build_review_funnel(
        groups=_jsonl(groups),
        proposals=_jsonl(proposals),
        features=_jsonl(features),
        output=output,
        eligibility_document=_json(eligibility_decisions) if eligibility_decisions else None,
        extractions=_jsonl(extractions) if extractions else (),
        duplicate_decision_report=(
            _json(duplicate_decision_report) if duplicate_decision_report else None
        ),
        synthesis_signoff=_json(synthesis_signoff) if synthesis_signoff else None,
    )


AbstractTransport = Callable[[str, int], bytes]


def _strip_official_pubmed_doctype(data: bytes) -> bytes:
    lowered = data[:10_000].lower()
    if b"<!entity" in lowered:
        raise ScientificEvidenceError("XML entity declarations are forbidden")
    stripped, declarations = PUBMED_DOCTYPE_RE.subn(b"", data, count=1)
    if b"<!doctype" in stripped[:10_000].lower():
        raise ScientificEvidenceError("unexpected XML DTD declaration")
    if b"<!doctype" in lowered and declarations != 1:
        raise ScientificEvidenceError("PubMed XML has an unrecognised DTD declaration")
    return stripped


def _pubmed_transport(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Accept": "application/xml,text/xml",
            "User-Agent": "giga-scientific-evidence/4.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(50 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        raise ScientificEvidenceError(
            f"PubMed abstract request failed with HTTP {exc.code}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ScientificEvidenceError(
            f"PubMed abstract connection failed: {exc.reason}"
        ) from exc


def fetch_pubmed_abstracts(
    pmids: Sequence[str],
    *,
    timeout: int = 60,
    transport: AbstractTransport | None = None,
) -> list[dict[str, Any]]:
    identifiers = list(dict.fromkeys(str(item) for item in pmids))
    if not identifiers or len(identifiers) > 200:
        raise ScientificEvidenceError("PubMed abstract batch must contain 1 to 200 PMIDs")
    if any(not PMID_RE.fullmatch(item) for item in identifiers):
        raise ScientificEvidenceError("every PMID must contain 1 to 12 digits")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 1200:
        raise ScientificEvidenceError("timeout must be an integer from 1 to 1200")
    query = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "id": ",".join(identifiers),
            "retmode": "xml",
            "rettype": "abstract",
            "tool": "giga_scientific_evidence",
        }
    )
    url = f"{PUBMED_EFETCH}?{query}"
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "eutils.ncbi.nlm.nih.gov":
        raise ScientificEvidenceError("PubMed URL failed official-host validation")
    data = (transport or _pubmed_transport)(url, timeout)
    if not isinstance(data, bytes) or len(data) > 50 * 1024 * 1024:
        raise ScientificEvidenceError("PubMed response is invalid or too large")
    data = _strip_official_pubmed_doctype(data)
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ScientificEvidenceError("PubMed response is not valid XML") from exc
    results: list[dict[str, Any]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid_node = article.find(".//MedlineCitation/PMID")
        if pmid_node is None or not (pmid_node.text or "").strip():
            continue
        pmid = (pmid_node.text or "").strip()
        parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            text = " ".join(" ".join(node.itertext()).split())
            label = node.attrib.get("Label")
            if text:
                parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(parts) or None
        result = {
            "schema_version": ABSTRACT_RETRIEVAL_SCHEMA,
            "provider": "pubmed",
            "pmid": pmid,
            "abstract": abstract,
            "abstract_sha256": hashlib.sha256((abstract or "").encode("utf-8")).hexdigest(),
            "source_url": url,
            "authority": "provider_metadata_for_eligibility_only",
            "full_text_substitute": False,
        }
        result["retrieval_sha256"] = payload_hash(result)
        results.append(result)
    return sorted(results, key=lambda item: int(item["pmid"]))


def write_pubmed_abstracts(
    pmids: Sequence[str],
    destination: Path,
    *,
    timeout: int = 60,
    transport: AbstractTransport | None = None,
) -> dict[str, Any]:
    results = fetch_pubmed_abstracts(pmids, timeout=timeout, transport=transport)
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    content = "".join(canonical_json(item) + "\n" for item in results).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    os.chmod(destination, 0o600)
    return _artifact(destination, len(results))


def write_pubmed_abstract_queue(
    queue: Path,
    destination: Path,
    *,
    timeout: int = 60,
    transport: AbstractTransport | None = None,
) -> dict[str, Any]:
    requests = _jsonl(queue)
    pmids = sorted(
        {
            str(route["identifier"])
            for request in requests
            for route in request.get("routes", [])
            if isinstance(route, Mapping)
            and route.get("provider") == "pubmed"
            and PMID_RE.fullmatch(str(route.get("identifier") or ""))
        },
        key=int,
    )
    results: list[dict[str, Any]] = []
    for start in range(0, len(pmids), 200):
        results.extend(
            fetch_pubmed_abstracts(
                pmids[start : start + 200],
                timeout=timeout,
                transport=transport,
            )
        )
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    content = "".join(canonical_json(item) + "\n" for item in results).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    os.chmod(destination, 0o600)
    artifact = _artifact(destination, len(results))
    return {
        **artifact,
        "requested_pmids": len(pmids),
        "returned_abstract_records": len(results),
        "batches": (len(pmids) + 199) // 200,
        "missing_from_provider_response": len(pmids)
        - len({item["pmid"] for item in results}),
        "authority": "provider_metadata_for_eligibility_only",
    }
