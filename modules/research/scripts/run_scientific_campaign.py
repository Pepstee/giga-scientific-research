#!/usr/bin/env python3
"""Plan or explicitly execute a scientific campaign through lawful public APIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modules.research.core.scientific_campaign import (  # noqa: E402
    execute_provider_campaign,
    load_campaign,
    provider_campaign_plan,
    select_queries,
)
from modules.research.core.scientific_evidence import (  # noqa: E402
    DEFAULT_DATABASE,
    ScientificEvidenceError,
    ScientificEvidenceStore,
)
from modules.research.core.scientific_providers import (  # noqa: E402
    DEFAULT_FREE_PAPER_PROVIDERS,
    PAPER_SEARCH_PROVIDERS,
    ClinicalTrialsClient,
    ScientificProviderAPIError,
    ScientificProviderConfigurationError,
    create_paper_client,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--aggregate-database",
        action="append",
        type=Path,
        default=[],
        help="include another existing acquisition ledger in the cumulative 20-record cap",
    )
    parser.add_argument("--phase", type=int, action="append", dest="phases")
    parser.add_argument("--query-id", action="append", dest="query_ids")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument(
        "--paper-provider",
        action="append",
        choices=sorted(PAPER_SEARCH_PROVIDERS),
        dest="paper_providers",
        help=(
            "repeat to select paper providers; default: Europe PMC and PubMed"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="make live public-API calls; otherwise perform a network-free dry run",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="repeat provider requests already represented in the ledger",
    )
    args = parser.parse_args(argv)
    providers = tuple(args.paper_providers or DEFAULT_FREE_PAPER_PROVIDERS)
    try:
        campaign = load_campaign(args.campaign)
        selected = select_queries(
            campaign,
            phases=args.phases,
            query_ids=args.query_ids,
            max_requests=args.max_requests,
        )
        store = ScientificEvidenceStore(args.database)
        if args.execute:
            paper_clients = {
                provider: create_paper_client(provider) for provider in providers
            }
            output = execute_provider_campaign(
                campaign,
                selected,
                store,
                paper_clients,
                ClinicalTrialsClient(),
                paper_providers=providers,
                refresh=args.refresh,
                aggregate_databases=args.aggregate_database,
            )
        else:
            if args.refresh:
                raise ScientificEvidenceError("--refresh requires --execute")
            output = {
                "mode": "dry_run",
                "api_calls_made": 0,
                "paper_providers": list(providers),
                **provider_campaign_plan(
                    campaign,
                    selected,
                    store,
                    paper_providers=providers,
                    aggregate_databases=args.aggregate_database,
                ),
            }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0
    except (
        KeyError,
        ScientificEvidenceError,
        ScientificProviderAPIError,
        ScientificProviderConfigurationError,
    ) as exc:
        print(f"scientific-campaign: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
