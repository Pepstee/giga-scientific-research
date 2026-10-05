"""Hash-bound AI eligibility review and retained-abstract descriptive synthesis."""

from __future__ import annotations

import hashlib
import os
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .scientific_evidence import ScientificEvidenceError, payload_hash
from .scientific_review_funnel import (
    _artifact,
    _clean_abstract,
    _json,
    _jsonl,
    _sha,
    _text,
    _timestamp,
    _write_json,
)


INPUT_SCHEMA = "giga.scientific-automated-review.v1"
OUTPUT_SCHEMA = "giga.scientific-automated-review-output.v1"
CAMPAIGN_ID = "solar-cell-efficiency-neutral-review-20260930"
AUTHORITY = {
    "kind": "operator_workflow_amendment",
    "id": "scientific-ai-review-20261005",
    "date": "2026-10-05",
    "scope": "routine eligibility and descriptive synthesis may be AI-reviewed",
}
REVIEW_FIELDS = {
    "record_ref",
    "group_id",
    "evidence_sha256",
    "abstract_sha256",
    "decision",
    "reason_codes",
    "reasons",
    "uncertainty",
    "study_basis",
    "claims",
    "reviewer",
    "independent_review",
    "human_verified",
}


def _fields(value: Any, expected: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ScientificEvidenceError(
            f"{field} fields differ; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    return value


def _strings(value: Any, field: str, *, required: bool = False) -> list[str]:
    if not isinstance(value, list) or (required and not value):
        qualifier = "non-empty " if required else ""
        raise ScientificEvidenceError(f"{field} must be a {qualifier}text array")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ScientificEvidenceError(f"{field} must contain non-empty text values")
    return [item.strip() for item in value]


def _ai_reviewer(value: Any, field: str) -> dict[str, str]:
    reviewer = _fields(value, {"type", "model", "job_id", "raw_output_sha256"}, field)
    if reviewer.get("type") != "ai":
        raise ScientificEvidenceError(f"{field}.type must be ai")
    model = _text(reviewer.get("model"), f"{field}.model", maximum=200)
    if "glm" not in model.casefold():
        raise ScientificEvidenceError(f"{field}.model must identify the GLM reviewer")
    return {
        "type": "ai",
        "model": model,
        "job_id": _text(reviewer.get("job_id"), f"{field}.job_id", maximum=200),
        "raw_output_sha256": _sha(
            reviewer.get("raw_output_sha256"), f"{field}.raw_output_sha256"
        ),
    }


def _independent_review(
    value: Any,
    field: str,
    *,
    corrections_allowed: bool,
) -> dict[str, Any]:
    keys = {"type", "model", "reviewed_at", "verdict"}
    if corrections_allowed:
        keys.add("corrections")
    review = _fields(value, keys, field)
    if review.get("type") != "ai" or review.get("model") != "gpt-6-astra":
        raise ScientificEvidenceError(f"{field} must record the independent Astra AI review")
    verdict = _text(review.get("verdict"), f"{field}.verdict", maximum=80)
    allowed = {"accepted", "accepted_with_corrections"} if corrections_allowed else {"accepted"}
    if verdict not in allowed:
        raise ScientificEvidenceError(f"{field}.verdict is invalid")
    corrections = _strings(review.get("corrections"), f"{field}.corrections") if corrections_allowed else []
    if verdict == "accepted_with_corrections" and not corrections:
        raise ScientificEvidenceError(f"{field} must record accepted corrections")
    if verdict == "accepted" and corrections:
        raise ScientificEvidenceError(f"{field} accepted verdict cannot include corrections")
    return {
        "type": "ai",
        "model": "gpt-6-astra",
        "reviewed_at": _timestamp(review.get("reviewed_at"), f"{field}.reviewed_at"),
        "verdict": verdict,
        **({"corrections": corrections} if corrections_allowed else {}),
    }


def validate_automated_review(
    document: Mapping[str, Any],
    *,
    groups: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    source_groups_sha256: str,
    source_protocol_sha256: str,
) -> dict[str, Any]:
    top_keys = {
        "schema_version", "campaign_id", "review_scope", "authority",
        "source_groups_sha256", "source_protocol_sha256", "human_verified",
        "reviews", "synthesis", "report_sha256",
    }
    _fields(document, top_keys, "automated review")
    if document.get("schema_version") != INPUT_SCHEMA:
        raise ScientificEvidenceError("automated review has the wrong schema")
    if document.get("campaign_id") != CAMPAIGN_ID:
        raise ScientificEvidenceError("automated review campaign does not match this run")
    if document.get("review_scope") != "retained_abstracts":
        raise ScientificEvidenceError("automated review scope must be retained_abstracts")
    if document.get("authority") != AUTHORITY:
        raise ScientificEvidenceError("automated review authority does not match the operator amendment")
    if document.get("human_verified") is not False:
        raise ScientificEvidenceError("automated review must not claim human verification")
    if _sha(document.get("source_groups_sha256"), "source_groups_sha256") != source_groups_sha256:
        raise ScientificEvidenceError("automated review is stale: source groups hash changed")
    if _sha(document.get("source_protocol_sha256"), "source_protocol_sha256") != source_protocol_sha256:
        raise ScientificEvidenceError("automated review is stale: protocol hash changed")
    recorded_report_hash = _sha(document.get("report_sha256"), "report_sha256")
    report_material = dict(document)
    report_material.pop("report_sha256")
    if payload_hash(report_material) != recorded_report_hash:
        raise ScientificEvidenceError("automated review report hash mismatch")

    by_group: dict[str, Mapping[str, Any]] = {}
    group_order: list[str] = []
    for index, group in enumerate(groups):
        group_id = _sha(group.get("group_id"), f"groups[{index}].group_id")
        if group_id in by_group:
            raise ScientificEvidenceError("source groups must have unique group_id values")
        recorded_group_hash = _sha(
            group.get("group_payload_sha256"), f"groups[{index}].group_payload_sha256"
        )
        group_material = dict(group)
        group_material.pop("group_payload_sha256")
        group_material.pop("screening_priority", None)
        group_material.pop("priority_reason_codes", None)
        if payload_hash(group_material) != recorded_group_hash:
            raise ScientificEvidenceError(f"source group {group_id} payload hash mismatch")
        urls = group.get("urls")
        if not isinstance(urls, list) or not all(isinstance(url, str) and url.strip() for url in urls):
            raise ScientificEvidenceError(f"source group {group_id} has invalid retained URLs")
        for url in urls:
            parsed_url = urllib.parse.urlsplit(url)
            if (
                parsed_url.scheme not in {"http", "https"}
                or not parsed_url.netloc
                or parsed_url.username is not None
                or parsed_url.password is not None
                or any(char.isspace() for char in url)
                or any(char in url for char in "()<>")
            ):
                raise ScientificEvidenceError(f"source group {group_id} has an unsafe retained URL")
        by_group[group_id] = group
        group_order.append(group_id)

    expected_ids = set(by_group)

    def row_ids(rows: Sequence[Mapping[str, Any]], field: str) -> set[str]:
        found: set[str] = set()
        for index, row in enumerate(rows):
            group_id = _sha(row.get("group_id"), f"{field}[{index}].group_id")
            if group_id in found:
                raise ScientificEvidenceError(f"{field} contains duplicate group ids")
            found.add(group_id)
        return found

    if row_ids(proposals, "proposals") != expected_ids:
        raise ScientificEvidenceError("proposals do not exactly cover the source groups")
    if row_ids(features, "features") != expected_ids:
        raise ScientificEvidenceError("features do not exactly cover the source groups")

    raw_reviews = document.get("reviews")
    if not isinstance(raw_reviews, list) or len(raw_reviews) != len(expected_ids):
        raise ScientificEvidenceError("automated reviews must cover every source group exactly once")
    reviews: dict[str, dict[str, Any]] = {}
    refs: set[str] = set()
    for index, raw in enumerate(raw_reviews):
        review = _fields(raw, REVIEW_FIELDS, f"reviews[{index}]")
        ref = _text(review.get("record_ref"), f"reviews[{index}].record_ref", maximum=80)
        if ref in refs:
            raise ScientificEvidenceError("automated review record_ref values must be unique")
        refs.add(ref)
        group_id = _sha(review.get("group_id"), f"reviews[{index}].group_id")
        if group_id not in by_group or group_id in reviews:
            raise ScientificEvidenceError("automated review has an unknown or duplicate group id")
        group = by_group[group_id]
        evidence_hash = _sha(review.get("evidence_sha256"), f"reviews[{index}].evidence_sha256")
        if evidence_hash != group.get("group_payload_sha256"):
            raise ScientificEvidenceError(f"automated review evidence is stale for {group_id}")
        abstract = _clean_abstract(group.get("abstract"))
        expected_abstract_hash = hashlib.sha256((abstract or "").encode("utf-8")).hexdigest()
        abstract_hash = _sha(review.get("abstract_sha256"), f"reviews[{index}].abstract_sha256")
        if abstract_hash != expected_abstract_hash:
            raise ScientificEvidenceError(f"automated review abstract is stale for {group_id}")
        decision = _text(review.get("decision"), f"reviews[{index}].decision", maximum=40)
        if decision not in {"include", "exclude", "uncertain"}:
            raise ScientificEvidenceError("automated review decision is invalid")
        reason_codes = _strings(review.get("reason_codes"), f"reviews[{index}].reason_codes", required=True)
        reasons = _strings(review.get("reasons"), f"reviews[{index}].reasons", required=True)
        uncertainty = _strings(review.get("uncertainty"), f"reviews[{index}].uncertainty")
        if decision == "uncertain" and not uncertainty:
            raise ScientificEvidenceError(f"uncertain review {group_id} must state its uncertainty")
        study_basis = _text(review.get("study_basis"), f"reviews[{index}].study_basis", maximum=40)
        if study_basis not in {"experimental", "theoretical", "review", "unclear"}:
            raise ScientificEvidenceError("automated review study_basis is invalid")
        raw_claims = review.get("claims")
        if not isinstance(raw_claims, list):
            raise ScientificEvidenceError("automated review claims must be an array")
        source_urls = set(group.get("urls", [])) if isinstance(group.get("urls"), list) else set()
        claims: list[dict[str, str]] = []
        for claim_index, raw_claim in enumerate(raw_claims):
            claim = _fields(raw_claim, {"claim", "quote", "source_url"}, f"reviews[{index}].claims[{claim_index}]")
            claim_text = _text(claim.get("claim"), "claim.claim", maximum=2_000)
            quote = claim.get("quote")
            if not isinstance(quote, str) or not quote.strip() or abstract is None or quote not in abstract:
                raise ScientificEvidenceError(f"claim quote is not an exact retained abstract substring for {group_id}")
            url = _text(claim.get("source_url"), "claim.source_url", maximum=2_000)
            parsed = urllib.parse.urlsplit(url)
            if (
                url not in source_urls or parsed.scheme not in {"https", "http"}
                or not parsed.netloc or parsed.username is not None or parsed.password is not None
                or any(char.isspace() for char in url) or any(char in url for char in "()<>")
            ):
                raise ScientificEvidenceError(f"claim source URL is not a source URL for {group_id}")
            claims.append({"claim": claim_text, "quote": quote, "source_url": url})
        if decision == "include" and (abstract is None or not claims):
            raise ScientificEvidenceError(f"include review {group_id} requires abstract-backed claims")
        if abstract is None and (decision != "uncertain" or claims):
            raise ScientificEvidenceError(f"missing abstract {group_id} must remain uncertain without claims")
        reviewer = _ai_reviewer(review.get("reviewer"), f"reviews[{index}].reviewer")
        independent = _independent_review(
            review.get("independent_review"), f"reviews[{index}].independent_review",
            corrections_allowed=True,
        )
        if review.get("human_verified") is not False:
            raise ScientificEvidenceError(f"automated review for {group_id} cannot claim human verification")
        reviews[group_id] = {
            "record_ref": ref,
            "group_id": group_id,
            "evidence_sha256": evidence_hash,
            "abstract_sha256": abstract_hash,
            "decision": decision,
            "reason_codes": list(dict.fromkeys(reason_codes)),
            "reasons": reasons,
            "uncertainty": uncertainty,
            "study_basis": study_basis,
            "claims": claims,
            "reviewer": reviewer,
            "independent_review": independent,
            "human_verified": False,
        }
    if set(reviews) != expected_ids:
        raise ScientificEvidenceError("automated review coverage differs from the source groups")

    synthesis = _fields(
        document.get("synthesis"),
        {"title", "summary", "findings", "limitations", "reviewer", "independent_review"},
        "synthesis",
    )
    title = _text(synthesis.get("title"), "synthesis.title", maximum=500)
    summary = _text(synthesis.get("summary"), "synthesis.summary", maximum=10_000)
    limitations = _strings(synthesis.get("limitations"), "synthesis.limitations", required=True)
    raw_findings = synthesis.get("findings")
    if not isinstance(raw_findings, list) or not raw_findings:
        raise ScientificEvidenceError("descriptive synthesis must contain findings")
    included_ids = {group_id for group_id, item in reviews.items() if item["decision"] == "include"}
    findings: list[dict[str, Any]] = []
    for index, raw_finding in enumerate(raw_findings):
        finding = _fields(
            raw_finding, {"text", "supporting_group_ids", "limitations"},
            f"synthesis.findings[{index}]",
        )
        text = _text(finding.get("text"), f"synthesis.findings[{index}].text", maximum=5_000)
        supporting = finding.get("supporting_group_ids")
        if not isinstance(supporting, list) or not supporting:
            raise ScientificEvidenceError("each synthesis finding must name supporting groups")
        supporting_ids = [_sha(item, "supporting_group_id") for item in supporting]
        if len(supporting_ids) != len(set(supporting_ids)) or not set(supporting_ids) <= included_ids:
            raise ScientificEvidenceError("synthesis may cite only unique included groups")
        if any(not reviews[group_id]["claims"] for group_id in supporting_ids):
            raise ScientificEvidenceError("synthesis support lacks source-linked abstract claims")
        finding_limits = _strings(
            finding.get("limitations"), f"synthesis.findings[{index}].limitations", required=True
        )
        findings.append({"text": text, "supporting_group_ids": supporting_ids, "limitations": finding_limits})
    synthesis_reviewer = _ai_reviewer(synthesis.get("reviewer"), "synthesis.reviewer")
    synthesis_review = _independent_review(
        synthesis.get("independent_review"), "synthesis.independent_review",
        corrections_allowed=True,
    )
    return {
        "campaign_id": CAMPAIGN_ID,
        "authority": dict(AUTHORITY),
        "source_groups_sha256": source_groups_sha256,
        "source_protocol_sha256": source_protocol_sha256,
        "human_verified": False,
        "group_order": group_order,
        "groups": by_group,
        "reviews": reviews,
        "synthesis": {
            "title": title,
            "summary": summary,
            "findings": findings,
            "limitations": limitations,
            "reviewer": synthesis_reviewer,
            "independent_review": synthesis_review,
        },
    }


def _md(value: str) -> str:
    for old, new in (("\\", "\\\\"), ("`", "\\`"), ("*", "\\*"),
                     ("_", "\\_"), ("[", "\\["), ("]", "\\]"),
                     ("<", "&lt;"), (">", "&gt;"), ("|", "\\|")):
        value = value.replace(old, new)
    return " ".join(value.split())


def build_automated_review_files(
    *,
    groups: Path,
    proposals: Path,
    features: Path,
    output: Path,
    automated_review: Path,
    protocol: Path,
    duplicate_decision_report: Path,
    command_argv: Sequence[str],
    cwd: str | None = None,
) -> dict[str, Any]:
    groups_raw = groups.read_bytes()
    proposals_raw = proposals.read_bytes()
    features_raw = features.read_bytes()
    protocol_raw = protocol.read_bytes()
    duplicate_raw = duplicate_decision_report.read_bytes()
    review_raw = automated_review.read_bytes()
    group_rows = _jsonl(groups)
    proposal_rows = _jsonl(proposals)
    feature_rows = _jsonl(features)
    review_document = _json(automated_review)
    duplicate_document = _json(duplicate_decision_report)
    duplicate_material = dict(duplicate_document)
    duplicate_recorded_hash = duplicate_material.pop("report_sha256", None)
    zero_candidate_case = (
        duplicate_document.get("candidate_pairs") == 0
        and duplicate_document.get("decisions_recorded") == 0
        and duplicate_document.get("same_study") == 0
        and duplicate_document.get("different_study") == 0
        and duplicate_document.get("uncertain") == 0
        and duplicate_document.get("unresolved_pairs") == 0
        and duplicate_document.get("human_verified") is False
    )
    if (
        duplicate_recorded_hash != payload_hash(duplicate_material)
        or duplicate_document.get("complete") is not True
        or duplicate_document.get("unresolved_pairs") != 0
        or not (duplicate_document.get("human_verified") is True or zero_candidate_case)
    ):
        raise ScientificEvidenceError(
            "existing duplicate report is incomplete, ambiguous, or lacks required verification"
        )
    group_digest = hashlib.sha256(groups_raw).hexdigest()
    protocol_digest = hashlib.sha256(protocol_raw).hexdigest()
    validated = validate_automated_review(
        review_document,
        groups=group_rows,
        proposals=proposal_rows,
        features=feature_rows,
        source_groups_sha256=group_digest,
        source_protocol_sha256=protocol_digest,
    )
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ScientificEvidenceError("automated review output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)

    reviews = validated["reviews"]
    groups_by_id = validated["groups"]
    decision_counts = Counter(review["decision"] for review in reviews.values())
    missing = [
        group_id for group_id in validated["group_order"]
        if _clean_abstract(groups_by_id[group_id].get("abstract")) is None
    ]
    output_reviews: list[dict[str, Any]] = []
    for group_id in validated["group_order"]:
        group = groups_by_id[group_id]
        output_reviews.append({
            **reviews[group_id],
            "source_title": group.get("title"),
            "source_record_ids": group.get("source_record_ids", []),
            "source_urls": list(group.get("urls", [])),
        })
    output_document = {
        "schema_version": OUTPUT_SCHEMA,
        "input_report_sha256": review_document["report_sha256"],
        "campaign_id": CAMPAIGN_ID,
        "review_scope": "retained_abstracts",
        "authority": dict(AUTHORITY),
        "source_groups_sha256": group_digest,
        "source_protocol_sha256": protocol_digest,
        "duplicate_decision_report_sha256": hashlib.sha256(duplicate_raw).hexdigest(),
        "human_verified": False,
        "reviews": output_reviews,
        "synthesis": validated["synthesis"],
    }
    output_document["report_sha256"] = payload_hash(output_document)
    review_path = output / "AUTOMATED_ELIGIBILITY_REVIEW.json"
    _write_json(review_path, output_document)

    synthesis = validated["synthesis"]
    lines = [
        f"# {_md(synthesis['title'])}",
        "",
        "**Scope:** AI-reviewed descriptions of retained abstracts. This is not a full-text "
        "systematic review or an effectiveness ranking.",
        "",
        _md(synthesis["summary"]),
        "",
        "## Descriptive findings",
        "",
    ]
    for number, finding in enumerate(synthesis["findings"], 1):
        lines.extend([f"### Finding {number}", "", _md(finding["text"]), "", "Source-linked abstract support:"])
        for group_id in finding["supporting_group_ids"]:
            group = groups_by_id[group_id]
            for claim in reviews[group_id]["claims"]:
                title = _md(str(group.get("title") or group_id))
                lines.append(
                    f"- [{title} ({group_id[:8]})]({claim['source_url']}): “{_md(claim['quote'])}”"
                )
        lines.extend([
            "",
            "Limitations: " + "; ".join(_md(item) for item in finding["limitations"]),
            "",
        ])
    lines.extend([
        "## Record-level eligibility review",
        "",
        "Every deduplicated source group is represented. Uncertain and excluded records are "
        "not used as descriptive synthesis support.",
        "",
    ])
    for group_id in validated["group_order"]:
        group = groups_by_id[group_id]
        review = reviews[group_id]
        lines.extend([
            f"### {_md(str(group.get('title') or group_id))} ({group_id[:8]})",
            "",
            f"Decision: **{review['decision']}**; study basis: {_md(review['study_basis'])}.",
            "",
            "Reasons: " + "; ".join(_md(item) for item in review["reasons"]),
        ])
        lines.append("Retained source references:")
        for source_number, source_url in enumerate(group.get("urls", []), 1):
            lines.append(f"- [Source {source_number}]({source_url})")
        if review["uncertainty"]:
            lines.append("Uncertainty: " + "; ".join(_md(item) for item in review["uncertainty"]))
        if review["claims"]:
            lines.append("Source-linked claims:")
            for claim in review["claims"]:
                lines.append(
                    f"- {_md(claim['claim'])}: “{_md(claim['quote'])}” "
                    f"([source]({claim['source_url']}))"
                )
        else:
            lines.append("No abstract-supported claim recorded.")
        lines.append("")
    limitations = list(synthesis["limitations"])
    limitations.extend([
        f"{len(missing)} retained groups have no cleaned abstract and remain uncertain.",
        "This operation used abstracts only; it did not perform full-text extraction, risk-of-bias appraisal, or effect estimation.",
        "This descriptive synthesis does not authorise effectiveness ranking or recommendations.",
    ])
    lines.extend(["## Scope limitations", ""])
    lines.extend(f"- {_md(item)}" for item in dict.fromkeys(limitations))
    lines.extend([
        "",
        "**Review provenance:** AI eligibility and independent AI reviews are recorded in "
        "`AUTOMATED_ELIGIBILITY_REVIEW.json`. No human verification is claimed.",
        "",
    ])
    synthesis_path = output / "AUTOMATED_DESCRIPTIVE_SYNTHESIS.md"
    synthesis_path.write_text("\n".join(lines), encoding="utf-8")

    readiness = {
        "schema_version": OUTPUT_SCHEMA,
        "input_report_sha256": review_document["report_sha256"],
        "campaign_id": CAMPAIGN_ID,
        "review_scope": "retained_abstracts",
        "counts": {
            "groups": len(reviews),
            "include": decision_counts["include"],
            "exclude": decision_counts["exclude"],
            "uncertain": decision_counts["uncertain"],
            "abstract_missing": len(missing),
        },
        "automated_eligibility_review_complete": len(reviews) == len(group_rows),
        "automated_descriptive_synthesis_ready": True,
        "full_text_review_completed": False,
        "risk_of_bias_appraisal_completed": False,
        "effectiveness_ranking_authorised": False,
        "recommendation_authority": False,
        "human_verified": False,
        "structural_validation_only": True,
        "independent_ai_review_recorded": True,
        "limitations": list(dict.fromkeys(limitations)),
        "artifacts": {
            "eligibility_review": _artifact(review_path, len(reviews)),
            "descriptive_synthesis": _artifact(synthesis_path, len(synthesis["findings"])),
        },
    }
    readiness["report_sha256"] = payload_hash(readiness)
    readiness_path = output / "AUTOMATED_REVIEW_READINESS.json"
    _write_json(readiness_path, readiness)
    run_receipt = {
        "schema_version": OUTPUT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "command_argv": list(command_argv),
        "cwd": cwd or os.getcwd(),
        "inputs": {
            "groups_sha256": group_digest,
            "proposals_sha256": hashlib.sha256(proposals_raw).hexdigest(),
            "features_sha256": hashlib.sha256(features_raw).hexdigest(),
            "protocol_sha256": protocol_digest,
            "duplicate_decision_report_sha256": hashlib.sha256(duplicate_raw).hexdigest(),
            "automated_review_sha256": hashlib.sha256(review_raw).hexdigest(),
        },
        "outputs": {
            "eligibility_review": _artifact(review_path, len(reviews)),
            "descriptive_synthesis": _artifact(synthesis_path, len(synthesis["findings"])),
            "readiness": _artifact(readiness_path, 1),
        },
        "provider_calls": 0,
        "acquisition_performed": False,
    }
    run_path = output / "AUTOMATED_REVIEW_RUN.json"
    _write_json(run_path, run_receipt)
    return {
        **readiness,
        "artifacts": run_receipt["outputs"],
        "run_receipt": _artifact(run_path, 1),
    }
