#!/usr/bin/env python3
"""Validate, plan, or explicitly execute a machine-readable Elicit campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modules.research.core.scientific_campaign import (  # noqa: E402
    campaign_plan,
    execute_campaign,
    load_campaign,
    select_queries,
)
from modules.research.core.scientific_evidence import (  # noqa: E402
    DEFAULT_DATABASE,
    ElicitAPIError,
    ElicitClient,
    ScientificEvidenceError,
    ScientificEvidenceStore,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--phase", type=int, action="append", dest="phases")
    parser.add_argument("--query-id", action="append", dest="query_ids")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="make live API calls; without this flag the command is a quota-free dry run",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="repeat requests already represented in the local ledger",
    )
    args = parser.parse_args(argv)
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
            output = execute_campaign(
                campaign,
                selected,
                store,
                ElicitClient(),
                refresh=args.refresh,
            )
        else:
            if args.refresh:
                raise ScientificEvidenceError("--refresh requires --execute")
            output = {
                "mode": "dry_run",
                "api_calls_made": 0,
                **campaign_plan(campaign, selected, store),
            }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0
    except (ScientificEvidenceError, ElicitAPIError, KeyError) as exc:
        print(f"elicit-campaign: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
