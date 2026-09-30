"""Command line interface for the GIGA scientific evidence module."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .scientific_evidence import (
    APPRAISAL_SCHEMA,
    DEFAULT_DATABASE,
    ElicitAPIError,
    ElicitClient,
    MAX_CUMULATIVE_RECORDS,
    ScientificEvidenceError,
    ScientificEvidenceStore,
)
from .scientific_providers import (
    DEFAULT_FREE_PAPER_PROVIDERS,
    DOI_LOOKUP_PROVIDERS,
    PAPER_SEARCH_PROVIDERS,
    ClinicalTrialsClient,
    ScientificProviderAPIError,
    ScientificProviderConfigurationError,
    create_doi_client,
    create_paper_client,
    paper_search_request,
)


def _json_file(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScientificEvidenceError(f"cannot read JSON object from {path}") from exc
    if not isinstance(value, dict):
        raise ScientificEvidenceError(f"{path} must contain a JSON object")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="giga-scientific-evidence",
        description=(
            "Acquire scientific evidence into an append-only A0 ledger. "
            "Search output never becomes advice automatically."
        ),
    )
    result.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    result.add_argument(
        "--aggregate-database",
        action="append",
        type=Path,
        default=[],
        help="include another existing acquisition ledger in the cumulative 20-record cap",
    )
    sub = result.add_subparsers(dest="command", required=True)

    search = sub.add_parser(
        "search-papers", help="search papers and preserve the raw response"
    )
    search.add_argument("query")
    search.add_argument("--max-results", type=int, default=10)
    search.add_argument(
        "--provider",
        action="append",
        choices=sorted(PAPER_SEARCH_PROVIDERS | {"elicit"}),
        dest="providers",
        help=(
            "repeat to search multiple providers; default: Europe PMC and PubMed"
        ),
    )
    search.add_argument("--corpus", choices=("elicit", "pubmed"), default="elicit")
    search.add_argument(
        "--search-mode", choices=("semantic", "keyword"), default="semantic"
    )
    search.add_argument("--type", action="append", dest="types")
    search.add_argument("--min-year", type=int)
    search.add_argument("--max-year", type=int)
    search.add_argument("--max-quartile", type=int)
    search.add_argument("--has-pdf", action="store_true")
    search.add_argument("--pubmed-only", action="store_true")

    trials = sub.add_parser(
        "search-trials",
        help="search ClinicalTrials.gov directly and preserve results",
    )
    trials.add_argument("query")
    trials.add_argument("--max-results", type=int, default=10)

    report = sub.add_parser(
        "start-report", help="explicitly start a private Elicit research report"
    )
    report.add_argument("question")
    report.add_argument("--max-search-papers", type=int, default=50)
    report.add_argument("--max-extract-papers", type=int, default=10)

    review = sub.add_parser(
        "start-review",
        help="explicitly start a private systematic review from a reviewed JSON protocol",
    )
    review.add_argument("protocol", type=Path)

    session = sub.add_parser("session", help="read current report/review status")
    session.add_argument("session_id")
    session.add_argument("--systematic", action="store_true")

    imported = sub.add_parser(
        "import-response",
        help="import an already-captured provider JSON response offline",
    )
    imported.add_argument(
        "--endpoint",
        required=True,
        choices=(
            "search/papers",
            "search/trials",
            "reports",
            "systematic-reviews",
            "session",
            "lookup/doi",
        ),
    )
    imported.add_argument(
        "--provider",
        choices=(
            "clinical_trials",
            "crossref",
            "elicit",
            "europe_pmc",
            "openalex",
            "pubmed",
            "semantic_scholar",
            "unpaywall",
        ),
        default="elicit",
    )
    imported.add_argument("--request", type=Path, required=True)
    imported.add_argument("--response", type=Path, required=True)
    imported.add_argument("--observed-at")

    lookup = sub.add_parser(
        "lookup-doi",
        help="verify a DOI and locate lawful open-access copies",
    )
    lookup.add_argument("doi")
    lookup.add_argument(
        "--provider",
        action="append",
        choices=sorted(DOI_LOOKUP_PROVIDERS),
        dest="providers",
        help=(
            "repeat to select resolvers; default: Crossref plus Unpaywall when "
            "a contact email is configured"
        ),
    )

    document = sub.add_parser(
        "import-document",
        help="hash and register a scientific document already held lawfully",
    )
    document.add_argument("file", type=Path)
    document.add_argument("--title", required=True)
    document.add_argument("--year", type=int)
    document.add_argument("--doi")
    document.add_argument("--pmid")

    appraisal = sub.add_parser(
        "appraise", help=f"record a strict {APPRAISAL_SCHEMA} human appraisal"
    )
    appraisal.add_argument("file", type=Path)

    candidate = sub.add_parser(
        "candidate", help="evaluate an evidence set for A0 plan readiness"
    )
    candidate.add_argument("--question", required=True)
    candidate.add_argument("--intervention", required=True)
    candidate.add_argument(
        "--risk-class",
        choices=("low_risk_lifestyle", "health_information", "clinical"),
        required=True,
    )
    candidate.add_argument("--source", action="append", required=True)
    candidate.add_argument("--contradictions-addressed", action="store_true")
    candidate.add_argument("--clinician-reviewed", action="store_true")

    sources = sub.add_parser("sources", help="list locally preserved source records")
    sources.add_argument("--limit", type=int, default=100)
    source = sub.add_parser("source", help="show one raw, immutable source record")
    source.add_argument("source_record_id")
    sub.add_parser("status", help="show ledger counts")
    sub.add_parser("audit", help="verify hashes, chain integrity, and permissions")
    return result


def _record(
    store: ScientificEvidenceStore,
    provider: str,
    endpoint: str,
    request: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    aggregate_databases: Sequence[Path] = (),
) -> dict[str, Any]:
    acquisition_id, created, source_ids = store.record_acquisition(
        provider=provider,
        endpoint=endpoint,
        request=request,
        response=response,
        aggregate_databases=aggregate_databases,
        max_cumulative_records=MAX_CUMULATIVE_RECORDS,
    )
    return {
        "acquisition_id": acquisition_id,
        "created": created,
        "source_record_ids": source_ids,
        "source_count": len(source_ids),
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    store = ScientificEvidenceStore(args.database)
    aggregate_databases = tuple(args.aggregate_database)
    try:
        if args.command in {"start-report", "start-review", "session"}:
            budget = store.require_acquisition_capacity(aggregate_databases)
            if budget["remaining_records"] < 1:
                raise ScientificEvidenceError(
                    "cumulative acquisition record cap "
                    f"{MAX_CUMULATIVE_RECORDS} is exhausted; "
                    "further remote acquisition refused"
                )
        if args.command == "search-papers":
            filters: dict[str, Any] = {}
            if args.types:
                filters["typeTags"] = args.types
            for key, value in (
                ("minYear", args.min_year),
                ("maxYear", args.max_year),
                ("maxQuartile", args.max_quartile),
            ):
                if value is not None:
                    filters[key] = value
            if args.has_pdf:
                filters["hasPdf"] = True
            if args.pubmed_only:
                filters["pubmedOnly"] = True
            providers = tuple(args.providers or DEFAULT_FREE_PAPER_PROVIDERS)
            budget = store.require_acquisition_capacity(aggregate_databases)
            remaining = budget["remaining_records"]
            clients = {
                provider: (
                    ElicitClient() if provider == "elicit" else create_paper_client(provider)
                )
                for provider in providers
            }
            acquisitions: list[dict[str, Any]] = []
            providers_left = len(providers)
            for provider in providers:
                quota = (
                    (remaining + providers_left - 1) // providers_left
                    if remaining and providers_left
                    else 0
                )
                max_results = min(args.max_results, quota)
                providers_left -= 1
                if max_results < 1:
                    acquisitions.append(
                        {
                            "provider": provider,
                            "status": "skipped_record_budget_exhausted",
                            "source_count": 0,
                        }
                    )
                    continue
                if provider == "elicit":
                    expected_request = ElicitClient.paper_search_request(
                        args.query,
                        max_results=max_results,
                        corpus=args.corpus,
                        search_mode=args.search_mode,
                        filters=filters or None,
                    )
                else:
                    expected_request = paper_search_request(
                        provider,
                        args.query,
                        max_results=max_results,
                        filters=filters or None,
                    )
                prior = store.acquisitions_for_request(
                    provider=provider,
                    endpoint="search/papers",
                    request=expected_request,
                    aggregate_databases=aggregate_databases,
                    match_max_results=False,
                )
                if prior:
                    acquisitions.append(
                        {
                            "provider": provider,
                            "status": "skipped_already_acquired",
                            "acquisition_ids": prior,
                            "source_count": 0,
                        }
                    )
                    continue
                if provider == "elicit":
                    request, response = clients[provider].search_papers(
                        args.query,
                        max_results=max_results,
                        corpus=args.corpus,
                        search_mode=args.search_mode,
                        filters=filters or None,
                    )
                else:
                    request, response = clients[provider].search_papers(
                        args.query,
                        max_results=max_results,
                        filters=filters or None,
                    )
                acquisitions.append(
                    {
                        "provider": provider,
                        **_record(
                            store,
                            provider,
                            "search/papers",
                            request,
                            response,
                            aggregate_databases=aggregate_databases,
                        ),
                    }
                )
                remaining = store.acquisition_budget(
                    aggregate_databases
                )["remaining_records"]
            output = {"acquisitions": acquisitions}
        elif args.command == "search-trials":
            budget = store.require_acquisition_capacity(aggregate_databases)
            max_results = min(args.max_results, budget["remaining_records"])
            if max_results < 1:
                raise ScientificEvidenceError(
                    "cumulative acquisition record cap "
                    f"{MAX_CUMULATIVE_RECORDS} is exhausted; further acquisition refused"
                )
            request, response = ClinicalTrialsClient().search_trials(
                args.query, max_results=max_results
            )
            output = _record(
                store,
                "clinical_trials",
                "search/trials",
                request,
                response,
                aggregate_databases=aggregate_databases,
            )
        elif args.command == "start-report":
            request, response = ElicitClient().create_report(
                args.question,
                max_search_papers=args.max_search_papers,
                max_extract_papers=args.max_extract_papers,
                is_public=False,
            )
            output = _record(
                store,
                "elicit",
                "reports",
                request,
                response,
                aggregate_databases=aggregate_databases,
            )
        elif args.command == "start-review":
            request, response = ElicitClient().create_systematic_review(
                _json_file(args.protocol)
            )
            output = _record(
                store,
                "elicit",
                "systematic-reviews",
                request,
                response,
                aggregate_databases=aggregate_databases,
            )
        elif args.command == "session":
            response = ElicitClient().get_session(
                args.session_id, systematic=args.systematic
            )
            output = _record(
                store,
                "elicit",
                "session",
                {"sessionId": args.session_id, "systematic": args.systematic},
                response,
                aggregate_databases=aggregate_databases,
            )
        elif args.command == "import-response":
            acquisition_id, created, source_ids = store.record_acquisition(
                provider=args.provider,
                endpoint=args.endpoint,
                request=_json_file(args.request),
                response=_json_file(args.response),
                observed_at=args.observed_at,
                aggregate_databases=aggregate_databases,
                max_cumulative_records=MAX_CUMULATIVE_RECORDS,
            )
            output = {
                "acquisition_id": acquisition_id,
                "created": created,
                "source_record_ids": source_ids,
            }
        elif args.command == "lookup-doi":
            budget = store.require_acquisition_capacity(aggregate_databases)
            remaining = budget["remaining_records"]
            if args.providers:
                providers = tuple(args.providers)
                omitted: list[str] = []
            else:
                providers = ("crossref",)
                if os.environ.get("UNPAYWALL_EMAIL") or os.environ.get(
                    "SCIENTIFIC_CONTACT_EMAIL"
                ):
                    providers += ("unpaywall",)
                    omitted = []
                else:
                    omitted = ["unpaywall: contact email not configured"]
            clients = {
                provider: create_doi_client(provider) for provider in providers
            }
            acquisitions = []
            for provider in providers:
                if remaining < 1:
                    acquisitions.append(
                        {
                            "provider": provider,
                            "status": "skipped_record_budget_exhausted",
                            "source_count": 0,
                        }
                    )
                    continue
                request, response = clients[provider].lookup_doi(args.doi)
                acquisitions.append(
                    {
                        "provider": provider,
                        **_record(
                            store,
                            provider,
                            "lookup/doi",
                            request,
                            response,
                            aggregate_databases=aggregate_databases,
                        ),
                    }
                )
                remaining = store.acquisition_budget(
                    aggregate_databases
                )["remaining_records"]
            output = {
                "acquisitions": acquisitions,
                "providers_not_run": omitted,
            }
        elif args.command == "import-document":
            path = args.file.expanduser().resolve()
            if not path.is_file():
                raise ScientificEvidenceError(f"document does not exist: {path}")
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
            document_sha256 = digest.hexdigest()
            request = {
                "provider": "local_file",
                "filename": path.name,
                "sizeBytes": size,
                "documentSha256": document_sha256,
            }
            response = {
                "papers": [
                    {
                        "title": args.title,
                        "year": args.year,
                        "doi": args.doi,
                        "pmid": args.pmid,
                        "localDocument": {
                            "filename": path.name,
                            "sizeBytes": size,
                            "sha256": document_sha256,
                        },
                        "sourceProvider": "local_file",
                    }
                ]
            }
            output = _record(
                store,
                "local_file",
                "import/local-document",
                request,
                response,
                aggregate_databases=aggregate_databases,
            )
        elif args.command == "appraise":
            appraisal_id, created, tier = store.record_appraisal(_json_file(args.file))
            output = {
                "appraisal_id": appraisal_id,
                "created": created,
                "evidence_tier": tier,
            }
        elif args.command == "candidate":
            candidate_id, candidate = store.create_action_candidate(
                question=args.question,
                intervention=args.intervention,
                risk_class=args.risk_class,
                source_record_ids=args.source,
                contradictions_addressed=args.contradictions_addressed,
                clinician_reviewed=args.clinician_reviewed,
            )
            output = {"candidate_id": candidate_id, "candidate": candidate}
        elif args.command == "sources":
            output = {"sources": store.list_sources(limit=args.limit)}
        elif args.command == "source":
            output = store.get_source(args.source_record_id)
        elif args.command == "status":
            output = {"database": str(store.database), "counts": store.counts()}
        elif args.command == "audit":
            output = store.audit()
            print(json.dumps(output, indent=2, ensure_ascii=False))
            return 0 if output["ok"] else 1
        else:
            raise AssertionError(args.command)
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0
    except (
        ScientificEvidenceError,
        ElicitAPIError,
        ScientificProviderAPIError,
        ScientificProviderConfigurationError,
        KeyError,
    ) as exc:
        print(f"giga-scientific-evidence: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
