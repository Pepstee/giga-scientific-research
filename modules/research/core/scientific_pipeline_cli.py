"""Command line interface for the deterministic scientific-review pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .scientific_evidence import ScientificEvidenceError
from .scientific_fulltext import (
    extract_pmc_xml,
    fetch_europe_pmc_xml,
    plan_full_text,
    write_fulltext_plan,
)
from .scientific_query_compiler import (
    compile_manifest,
    embed_compiled_queries,
    load_ontology,
    load_search_spec,
)
from .scientific_readiness import assess_screening_readiness_files
from .scientific_review_state import (
    DEFAULT_REVIEW_DATABASE,
    ScientificReviewStore,
    ingest_screening_snapshot,
)
from .scientific_review_funnel import (
    apply_abstract_enrichments_files,
    build_review_funnel_files,
    write_pubmed_abstracts,
    write_pubmed_abstract_queue,
)
from .scientific_automated_review import build_automated_review_files
from .scientific_saturation import evaluate_saturation
from .scientific_screening import (
    compile_screening,
    validate_exclusion_audit,
    write_screening,
)


DEFAULT_PROVIDERS = ("europe_pmc", "pubmed")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScientificEvidenceError(f"cannot read JSON object from {path}") from exc
    if not isinstance(value, dict):
        raise ScientificEvidenceError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _negative_control_campaign(
    campaign: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    controls = [
        query for query in manifest["queries"] if query["negative_control"]
    ]
    return {
        "schema_version": "giga.scientific-search-campaign.v1",
        "campaign_id": f"{campaign['campaign_id']}-negative-controls",
        "protocol": campaign["protocol"],
        "created_at": campaign["created_at"],
        "execution_state": "compiled_negative_controls_not_executed",
        "defaults": {
            "endpoint": "search/papers",
            "corpus": "pubmed",
            "search_mode": "semantic",
            "max_results": 10,
            "filters": {"retracted": "exclude_retracted"},
        },
        "queries": [
            {
                "id": control["id"],
                "phase": 99,
                "question": control["human_question"],
                "provider_queries": {
                    provider: expression
                    for provider, expression in control["provider_queries"].items()
                    if provider != "clinical_trials"
                },
            }
            for control in controls
        ],
        "completion_rule": (
            "Every retrieved negative-control record must be proposed for exclusion; "
            "any failure blocks search promotion."
        ),
    }


def _compile_search(args: argparse.Namespace) -> dict[str, Any]:
    campaign = _json(args.campaign)
    ontology = load_ontology(args.ontology)
    specification = load_search_spec(args.specification, ontology=ontology)
    manifest = compile_manifest(
        campaign,
        ontology,
        specification,
        providers=args.provider or DEFAULT_PROVIDERS,
    )
    compiled_campaign = embed_compiled_queries(campaign, manifest)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "COMPILED_SEARCH_MANIFEST.json", manifest)
    _write_json(output / "COMPILED_CAMPAIGN.json", compiled_campaign)
    _write_json(
        output / "NEGATIVE_CONTROL_CAMPAIGN.json",
        _negative_control_campaign(campaign, manifest),
    )
    return {
        "mode": "compile_search",
        "campaign_id": campaign["campaign_id"],
        "compiled_queries": sum(
            not item["negative_control"] for item in manifest["queries"]
        ),
        "negative_controls": sum(
            item["negative_control"] for item in manifest["queries"]
        ),
        "providers": manifest["providers"],
        "manifest_sha256": manifest["manifest_sha256"],
        "output": str(output),
        "api_calls_made": 0,
        "scientific_conclusion_authority": False,
    }


def _screen(args: argparse.Namespace) -> dict[str, Any]:
    campaign = _json(args.campaign)
    manifest = _json(args.manifest)
    ontology = load_ontology(args.ontology)
    specification = load_search_spec(args.specification, ontology=ontology)
    duplicate_document = (
        _json(args.duplicate_decisions) if args.duplicate_decisions else None
    )
    build = compile_screening(
        args.database.expanduser().resolve(),
        campaign,
        manifest,
        ontology,
        specification,
        duplicate_document,
    )
    output = args.output.expanduser().resolve()
    summary = write_screening(build, output)
    fulltext = plan_full_text(build.groups, build.proposals)
    fulltext_artifact = write_fulltext_plan(
        fulltext,
        output / "FULLTEXT_PLAN.jsonl",
    )
    summary["artifacts"]["fulltext_plan"] = fulltext_artifact
    summary["full_text_plans"] = len(fulltext)
    _write_json(output / "SCREENING_SUMMARY.json", summary)
    state_result = None
    if not args.no_state_ledger:
        store = ScientificReviewStore(args.review_database)
        if duplicate_document:
            for decision in duplicate_document["decisions"]:
                store.record_duplicate_decision(
                    campaign_id=campaign["campaign_id"],
                    pair_sha256=decision["pair_sha256"],
                    decision=decision["decision"],
                    decided_by=decision["decided_by"],
                    evidence=decision["evidence"],
                    decided_at=decision["decided_at"],
                )
        state_result = ingest_screening_snapshot(
            store,
            campaign_id=campaign["campaign_id"],
            groups=build.groups,
            proposals=build.proposals,
            snapshot_sha256=summary["snapshot_payload_sha256"],
            recorded_at=args.recorded_at or campaign["created_at"],
        )
    return {
        "mode": "screen",
        **summary,
        "state_ledger": state_result,
        "api_calls_made": 0,
    }


def _fetch_pmc(args: argparse.Namespace) -> dict[str, Any]:
    receipt = fetch_europe_pmc_xml(
        args.pmcid,
        args.destination,
        timeout=args.timeout,
    )
    if args.receipt:
        _write_json(args.receipt, receipt)
    return receipt


def _extract_pmc(args: argparse.Namespace) -> dict[str, Any]:
    extraction = extract_pmc_xml(args.xml)
    if args.output:
        _write_json(args.output, extraction)
    return extraction


def _validate_exclusion_audit(args: argparse.Namespace) -> dict[str, Any]:
    summary = _json(args.screening_summary)
    artifact = summary.get("artifacts", {}).get("exclusion_audit")
    if not isinstance(artifact, dict):
        raise ScientificEvidenceError(
            "screening summary has no exclusion_audit artifact"
        )
    expected_path = (
        args.screening_summary.expanduser().resolve().parent / artifact["path"]
    )
    if hashlib.sha256(expected_path.read_bytes()).hexdigest() != artifact["sha256"]:
        raise ScientificEvidenceError(
            "deterministic exclusion-audit sample hash mismatch"
        )
    with expected_path.open(encoding="utf-8", newline="") as handle:
        expected = list(csv.DictReader(handle))
    report = validate_exclusion_audit(
        args.completed_audit,
        expected,
        maximum_false_exclusion_rate=args.maximum_false_exclusion_rate,
    )
    if args.output:
        _write_json(args.output, report)
    return report


def _saturation(args: argparse.Namespace) -> dict[str, Any]:
    report = evaluate_saturation(
        args.screening_summary,
        maximum_new_candidate_rate=args.maximum_new_candidate_rate,
    )
    if args.output:
        _write_json(args.output, report)
    return report


def _readiness(args: argparse.Namespace) -> dict[str, Any]:
    report = assess_screening_readiness_files(
        main_summary=args.main_summary,
        negative_control_summary=args.negative_control_summary,
        review_database=args.review_database,
        exclusion_audit_report=args.exclusion_audit_report,
        saturation_report=args.saturation_report,
    )
    if args.output:
        _write_json(args.output, report)
    return report


def _review_funnel(
    args: argparse.Namespace,
    command_argv: list[str],
) -> dict[str, Any]:
    if args.review_mode == "automated":
        if args.automated_review is None or args.protocol is None:
            raise ScientificEvidenceError(
                "automated review requires --automated-review and --protocol"
            )
        if args.duplicate_decision_report is None:
            raise ScientificEvidenceError(
                "automated review requires the existing --duplicate-decision-report"
            )
        if any(
            value is not None
            for value in (
                args.eligibility_decisions,
                args.extractions,
                args.synthesis_signoff,
            )
        ):
            raise ScientificEvidenceError(
                "automated review cannot be combined with human/full-text gate inputs"
            )
        return build_automated_review_files(
            groups=args.groups,
            proposals=args.proposals,
            features=args.features,
            output=args.output,
            automated_review=args.automated_review,
            protocol=args.protocol,
            duplicate_decision_report=args.duplicate_decision_report,
            command_argv=command_argv,
        )
    if args.automated_review is not None or args.protocol is not None:
        raise ScientificEvidenceError(
            "--automated-review and --protocol require --review-mode automated"
        )
    return build_review_funnel_files(
        groups=args.groups,
        proposals=args.proposals,
        features=args.features,
        output=args.output,
        eligibility_decisions=args.eligibility_decisions,
        extractions=args.extractions,
        duplicate_decision_report=args.duplicate_decision_report,
        synthesis_signoff=args.synthesis_signoff,
    )


def _fetch_abstracts(args: argparse.Namespace) -> dict[str, Any]:
    identifiers: list[str] = []
    for value in args.pmid:
        identifiers.extend(part.strip() for part in value.split(",") if part.strip())
    return write_pubmed_abstracts(
        identifiers,
        args.destination,
        timeout=args.timeout,
    )


def _merge_abstracts(args: argparse.Namespace) -> dict[str, Any]:
    return apply_abstract_enrichments_files(
        groups=args.groups,
        retrievals=args.retrievals,
        ontology=args.ontology,
        specification=args.specification,
        output=args.output,
    )


def _fetch_abstract_queue(args: argparse.Namespace) -> dict[str, Any]:
    return write_pubmed_abstract_queue(
        args.queue,
        args.destination,
        timeout=args.timeout,
    )


def _state(args: argparse.Namespace) -> dict[str, Any]:
    store = ScientificReviewStore(args.review_database)
    if args.state_command == "audit":
        return store.audit()
    if args.state_command == "status":
        return store.counts(args.campaign_id)
    if args.state_command == "decide":
        current = store.current_states(args.campaign_id).get(args.group_id)
        if current not in {
            "awaiting_review",
            "proposed_include",
            "proposed_exclude",
        }:
            raise ScientificEvidenceError(
                "human eligibility decisions require an awaiting/proposed state"
            )
        evidence = _json(args.evidence)
        event_id, created = store.transition(
            campaign_id=args.campaign_id,
            group_id=args.group_id,
            from_state=current,
            to_state="included" if args.decision == "include" else "excluded",
            actor_kind="human",
            reason_codes=[args.reason_code],
            evidence=evidence,
            recorded_at=args.recorded_at,
        )
        return {
            "event_id": event_id,
            "created": created,
            "from_state": current,
            "to_state": "included"
            if args.decision == "include"
            else "excluded",
            "authority": "eligibility_only_not_scientific_conclusion",
        }
    if args.state_command == "llm-proposal":
        proposal = _json(args.proposal)
        metadata = _json(args.source_metadata)
        proposal_id, created = store.record_llm_proposal(
            proposal,
            source_metadata=metadata,
        )
        return {
            "proposal_id": proposal_id,
            "created": created,
            "state_changed": False,
            "human_confirmation_required": True,
        }
    if args.state_command == "fulltext-plan":
        evidence = _json(args.evidence)
        event_id, created = store.transition(
            campaign_id=args.campaign_id,
            group_id=args.group_id,
            from_state="included",
            to_state="fulltext_planned",
            actor_kind="system",
            reason_codes=["FULLTEXT_ROUTE_SELECTED"],
            evidence=evidence,
            recorded_at=args.recorded_at,
        )
        return {
            "event_id": event_id,
            "created": created,
            "to_state": "fulltext_planned",
        }
    if args.state_command == "fulltext-result":
        evidence = _json(args.evidence)
        target = (
            "fulltext_obtained"
            if args.result == "obtained"
            else "fulltext_unavailable"
        )
        event_id, created = store.transition(
            campaign_id=args.campaign_id,
            group_id=args.group_id,
            from_state="fulltext_planned",
            to_state=target,
            actor_kind="system",
            reason_codes=[
                "FULLTEXT_OBTAINED_LAWFULLY"
                if args.result == "obtained"
                else "FULLTEXT_UNAVAILABLE_AFTER_LAWFUL_ROUTES"
            ],
            evidence=evidence,
            recorded_at=args.recorded_at,
        )
        return {"event_id": event_id, "created": created, "to_state": target}
    if args.state_command == "appraised":
        evidence = _json(args.evidence)
        event_id, created = store.transition(
            campaign_id=args.campaign_id,
            group_id=args.group_id,
            from_state="fulltext_obtained",
            to_state="appraised",
            actor_kind="human",
            reason_codes=["HUMAN_FULLTEXT_APPRAISAL_RECORDED"],
            evidence=evidence,
            recorded_at=args.recorded_at,
        )
        return {
            "event_id": event_id,
            "created": created,
            "to_state": "appraised",
            "scientific_conclusion_authority": False,
        }
    if args.state_command == "duplicate-decision":
        evidence = _json(args.evidence)
        decision_id, created = store.record_duplicate_decision(
            campaign_id=args.campaign_id,
            pair_sha256=args.pair_sha256,
            decision=args.decision,
            decided_by=args.decided_by,
            evidence=evidence,
            decided_at=args.decided_at,
        )
        return {
            "decision_id": decision_id,
            "created": created,
            "decision": args.decision,
            "screening_snapshot_recompile_required": True,
        }
    raise ScientificEvidenceError("unknown state command")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser(
        "compile-search",
        help="compile human questions into provider-specific queries without network",
    )
    compile_parser.add_argument("campaign", type=Path)
    compile_parser.add_argument("ontology", type=Path)
    compile_parser.add_argument("specification", type=Path)
    compile_parser.add_argument("output", type=Path)
    compile_parser.add_argument(
        "--provider",
        action="append",
        choices=[
            "clinical_trials",
            "europe_pmc",
            "openalex",
            "pubmed",
            "semantic_scholar",
        ],
    )

    screen = subparsers.add_parser(
        "screen",
        help="normalise, deduplicate, triage and queue an acquisition ledger",
    )
    screen.add_argument("database", type=Path)
    screen.add_argument("campaign", type=Path)
    screen.add_argument("manifest", type=Path)
    screen.add_argument("ontology", type=Path)
    screen.add_argument("specification", type=Path)
    screen.add_argument("output", type=Path)
    screen.add_argument(
        "--review-database",
        type=Path,
        default=DEFAULT_REVIEW_DATABASE,
    )
    screen.add_argument("--recorded-at")
    screen.add_argument("--no-state-ledger", action="store_true")
    screen.add_argument("--duplicate-decisions", type=Path)

    fetch = subparsers.add_parser(
        "fetch-pmc",
        help="explicitly fetch one full-text XML file from official Europe PMC",
    )
    fetch.add_argument("pmcid")
    fetch.add_argument("destination", type=Path)
    fetch.add_argument("--receipt", type=Path)
    fetch.add_argument("--timeout", type=int, default=60)

    extract = subparsers.add_parser(
        "extract-pmc",
        help="deterministically extract source-located text from PMC XML",
    )
    extract.add_argument("xml", type=Path)
    extract.add_argument("--output", type=Path)

    audit_exclusions = subparsers.add_parser(
        "validate-exclusion-audit",
        help="validate a human-completed copy of the deterministic exclusion sample",
    )
    audit_exclusions.add_argument("screening_summary", type=Path)
    audit_exclusions.add_argument("completed_audit", type=Path)
    audit_exclusions.add_argument(
        "--maximum-false-exclusion-rate",
        type=float,
        default=0.0,
    )
    audit_exclusions.add_argument("--output", type=Path)

    saturation = subparsers.add_parser(
        "saturation",
        help="evaluate two consecutive search amendments across three snapshots",
    )
    saturation.add_argument(
        "screening_summary",
        type=Path,
        nargs="+",
    )
    saturation.add_argument(
        "--maximum-new-candidate-rate",
        type=float,
        default=0.01,
    )
    saturation.add_argument("--output", type=Path)

    readiness = subparsers.add_parser(
        "readiness",
        help="evaluate fail-closed deterministic screening promotion gates",
    )
    readiness.add_argument("main_summary", type=Path)
    readiness.add_argument("negative_control_summary", type=Path)
    readiness.add_argument(
        "--review-database",
        type=Path,
        default=DEFAULT_REVIEW_DATABASE,
    )
    readiness.add_argument("--exclusion-audit-report", type=Path)
    readiness.add_argument("--saturation-report", type=Path)
    readiness.add_argument("--output", type=Path)

    funnel = subparsers.add_parser(
        "review-funnel",
        help=(
            "build metadata, abstract, priority, full-text, grouping and synthesis "
            "review artifacts without LLM calls"
        ),
    )
    funnel.add_argument("groups", type=Path)
    funnel.add_argument("proposals", type=Path)
    funnel.add_argument("features", type=Path)
    funnel.add_argument("output", type=Path)
    funnel.add_argument("--review-mode", choices=["human", "automated"], default="human")
    funnel.add_argument("--automated-review", type=Path)
    funnel.add_argument("--protocol", type=Path)
    funnel.add_argument("--eligibility-decisions", type=Path)
    funnel.add_argument("--extractions", type=Path)
    funnel.add_argument("--duplicate-decision-report", type=Path)
    funnel.add_argument("--synthesis-signoff", type=Path)

    abstracts = subparsers.add_parser(
        "fetch-pubmed-abstracts",
        help="explicitly retrieve one official PubMed abstract batch (maximum 200)",
    )
    abstracts.add_argument("pmid", nargs="+")
    abstracts.add_argument("destination", type=Path)
    abstracts.add_argument("--timeout", type=int, default=60)

    merge_abstracts = subparsers.add_parser(
        "merge-abstracts",
        help=(
            "identifier-match retrieved abstracts, preserve conflicts, and recompute "
            "features and eligibility proposals"
        ),
    )
    merge_abstracts.add_argument("groups", type=Path)
    merge_abstracts.add_argument("retrievals", type=Path)
    merge_abstracts.add_argument("ontology", type=Path)
    merge_abstracts.add_argument("specification", type=Path)
    merge_abstracts.add_argument("output", type=Path)

    fetch_queue = subparsers.add_parser(
        "fetch-abstract-queue",
        help="explicitly fetch all PubMed routes in a review queue in batches of 200",
    )
    fetch_queue.add_argument("queue", type=Path)
    fetch_queue.add_argument("destination", type=Path)
    fetch_queue.add_argument("--timeout", type=int, default=60)

    state = subparsers.add_parser(
        "state",
        help="inspect or append authorised review-state records",
    )
    state.add_argument(
        "--review-database",
        type=Path,
        default=DEFAULT_REVIEW_DATABASE,
    )
    state_sub = state.add_subparsers(dest="state_command", required=True)
    audit = state_sub.add_parser("audit")
    audit.set_defaults()
    status = state_sub.add_parser("status")
    status.add_argument("campaign_id")
    decide = state_sub.add_parser("decide")
    decide.add_argument("campaign_id")
    decide.add_argument("group_id")
    decide.add_argument("decision", choices=["include", "exclude"])
    decide.add_argument("reason_code")
    decide.add_argument("evidence", type=Path)
    decide.add_argument("--recorded-at")
    llm = state_sub.add_parser("llm-proposal")
    llm.add_argument("proposal", type=Path)
    llm.add_argument(
        "source_metadata",
        type=Path,
        help=(
            "full source group JSON from DEDUP_GROUPS.jsonl or "
            "ENRICHED_GROUPS.jsonl, including group_id and "
            "group_payload_sha256"
        ),
    )
    fulltext_plan = state_sub.add_parser("fulltext-plan")
    fulltext_plan.add_argument("campaign_id")
    fulltext_plan.add_argument("group_id")
    fulltext_plan.add_argument("evidence", type=Path)
    fulltext_plan.add_argument("--recorded-at")
    fulltext_result = state_sub.add_parser("fulltext-result")
    fulltext_result.add_argument("campaign_id")
    fulltext_result.add_argument("group_id")
    fulltext_result.add_argument("result", choices=["obtained", "unavailable"])
    fulltext_result.add_argument("evidence", type=Path)
    fulltext_result.add_argument("--recorded-at")
    appraised = state_sub.add_parser("appraised")
    appraised.add_argument("campaign_id")
    appraised.add_argument("group_id")
    appraised.add_argument("evidence", type=Path)
    appraised.add_argument("--recorded-at")
    duplicate = state_sub.add_parser("duplicate-decision")
    duplicate.add_argument("campaign_id")
    duplicate.add_argument("pair_sha256")
    duplicate.add_argument(
        "decision",
        choices=["same_study", "different_study", "uncertain"],
    )
    duplicate.add_argument("decided_by")
    duplicate.add_argument("evidence", type=Path)
    duplicate.add_argument("--decided-at")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    cli_args = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(cli_args)
    command_argv = [
        sys.executable,
        "-B",
        "-m",
        "modules.research.core.scientific_pipeline_cli",
        *cli_args,
    ]
    try:
        if args.command == "compile-search":
            result = _compile_search(args)
        elif args.command == "screen":
            result = _screen(args)
        elif args.command == "fetch-pmc":
            result = _fetch_pmc(args)
        elif args.command == "extract-pmc":
            result = _extract_pmc(args)
        elif args.command == "validate-exclusion-audit":
            result = _validate_exclusion_audit(args)
        elif args.command == "saturation":
            result = _saturation(args)
        elif args.command == "readiness":
            result = _readiness(args)
        elif args.command == "review-funnel":
            result = _review_funnel(args, command_argv)
        elif args.command == "fetch-pubmed-abstracts":
            result = _fetch_abstracts(args)
        elif args.command == "merge-abstracts":
            result = _merge_abstracts(args)
        elif args.command == "fetch-abstract-queue":
            result = _fetch_abstract_queue(args)
        else:
            result = _state(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (KeyError, OSError, ScientificEvidenceError) as exc:
        print(f"scientific-pipeline: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
