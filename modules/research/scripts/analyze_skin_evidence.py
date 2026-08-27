#!/usr/bin/env python3
"""Deterministically deduplicate and queue the skin-evidence campaign records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "giga.skin-evidence-acquisition-summary.v1"
PROVIDER_PREFERENCE = {
    "europe_pmc": 0,
    "pubmed": 1,
    "clinical_trials": 2,
    "semantic_scholar": 3,
    "openalex": 4,
    "crossref": 5,
    "unpaywall": 6,
    "elicit": 7,
    "local_file": 8,
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def payload_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalise_title(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    decomposed = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", decomposed))


def identity_tokens(record: Mapping[str, Any]) -> list[str]:
    tokens: list[str] = []
    for field, prefix in (("doi", "doi"), ("pmid", "pmid"), ("nct_id", "nct")):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            tokens.append(f"{prefix}:{value.strip().casefold()}")
    title = record.get("title")
    year = record.get("year")
    if isinstance(title, str) and title.strip():
        normalised = normalise_title(title)
        if len(normalised) >= 20:
            tokens.append(f"title-year:{normalised}:{year or 'unknown'}")
    return tokens


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def deduplicate(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    union = UnionFind(len(records))
    seen: dict[str, int] = {}
    for index, record in enumerate(records):
        for token in identity_tokens(record):
            if token in seen:
                union.union(index, seen[token])
            else:
                seen[token] = index
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[union.find(index)].append(record)
    return list(grouped.values())


def _publication_types(raw: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    direct = raw.get("publicationTypes")
    if isinstance(direct, list):
        values.extend(str(value) for value in direct if isinstance(value, str))
    provider = raw.get("providerRecord")
    if isinstance(provider, Mapping):
        pub_types = provider.get("pubTypeList")
        if isinstance(pub_types, Mapping):
            nested = pub_types.get("pubType")
            if isinstance(nested, list):
                values.extend(str(value) for value in nested if isinstance(value, str))
        pubtype = provider.get("pubtype")
        if isinstance(pubtype, list):
            values.extend(str(value) for value in pubtype if isinstance(value, str))
    return sorted(set(values))


def _urls(raw: Mapping[str, Any]) -> list[str]:
    value = raw.get("urls")
    if not isinstance(value, list):
        return []
    return sorted({url for url in value if isinstance(url, str) and url.strip()})


def _design_bucket(types: Iterable[str], *, is_trial: bool) -> str:
    joined = " ".join(types).casefold()
    if "guideline" in joined:
        return "guideline"
    if "systematic review" in joined or "meta-analysis" in joined:
        return "systematic_review_or_meta_analysis"
    if "randomized controlled trial" in joined:
        return "randomized_controlled_trial"
    if "controlled clinical trial" in joined or "clinical trial" in joined:
        return "clinical_trial"
    if "review" in joined:
        return "nonsystematic_review"
    if is_trial:
        return "trial_registry_record"
    return "primary_or_other"


def _screening_priority(bucket: str, phases: Iterable[int]) -> int:
    if bucket in {"guideline", "systematic_review_or_meta_analysis"}:
        return 1
    if bucket in {
        "randomized_controlled_trial",
        "clinical_trial",
        "trial_registry_record",
    }:
        return 2
    return 3 if 2 in set(phases) else 4


def _evidence_lanes(query_ids: Iterable[str]) -> list[str]:
    lanes: set[str] = set()
    for query_id in query_ids:
        lowered = query_id.casefold()
        if "photoaging" in lowered or "collagen" in lowered:
            lanes.add("photoaging_texture_collagen")
        if "atrophic" in lowered or "acne-scar" in lowered:
            lanes.add("atrophic_scars")
        if any(term in lowered for term in ("pathologic", "keloid", "hypertrophic")):
            lanes.add("hypertrophic_scars_keloids")
        if query_id.startswith("P3-"):
            lanes.add("misconceptions_misuse")
        if query_id.startswith("P4-"):
            lanes.add("trial_registry")
    return sorted(lanes or {"unclassified"})


def load_records(database: Path, campaign: Path) -> tuple[list[dict[str, Any]], dict]:
    campaign_payload = json.loads(campaign.read_text(encoding="utf-8"))
    question_map = {
        item["question"]: {"id": item["id"], "phase": int(item["phase"])}
        for item in campaign_payload["queries"]
    }
    uri = f"{database.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT s.*, a.provider, a.endpoint, a.request_json, a.observed_at
        FROM source_records AS s
        JOIN acquisitions AS a ON a.id = s.acquisition_id
        ORDER BY a.observed_at, a.provider, s.rank
        """
    ).fetchall()
    ledger = connection.execute(
        """
        SELECT COUNT(*) AS entries,
               (SELECT entry_sha256 FROM ledger_entries
                ORDER BY sequence DESC LIMIT 1) AS head
        FROM ledger_entries
        """
    ).fetchone()
    connection.close()
    records: list[dict[str, Any]] = []
    for row in rows:
        raw = json.loads(row["raw_json"])
        request = json.loads(row["request_json"])
        query = request.get("query")
        campaign_item = question_map.get(query, {})
        records.append(
            {
                "source_record_id": row["id"],
                "provider": row["provider"],
                "kind": row["kind"],
                "identity_key": row["identity_key"],
                "title": row["title"],
                "year": row["year"],
                "doi": row["doi"],
                "pmid": row["pmid"],
                "nct_id": row["nct_id"],
                "raw": raw,
                "query_id": campaign_item.get("id", "unmapped"),
                "phase": campaign_item.get("phase", 0),
            }
        )
    return records, {
        "entries": int(ledger["entries"]),
        "head_sha256": ledger["head"],
    }


def project_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        group,
        key=lambda item: (
            PROVIDER_PREFERENCE.get(item["provider"], 99),
            not bool(item["raw"].get("abstract")),
            item["source_record_id"],
        ),
    )
    preferred = ordered[0]

    def first(field: str) -> Any:
        for item in ordered:
            value = item.get(field)
            if value not in (None, ""):
                return value
        return None

    providers = sorted({item["provider"] for item in group})
    query_ids = sorted({item["query_id"] for item in group})
    phases = sorted({int(item["phase"]) for item in group if item["phase"]})
    types = sorted(
        {
            publication_type
            for item in group
            for publication_type in _publication_types(item["raw"])
        }
    )
    urls = sorted({url for item in group for url in _urls(item["raw"])})
    pmcids = sorted(
        {
            str(item["raw"]["pmcid"])
            for item in group
            if item["raw"].get("pmcid")
        }
    )
    open_access = bool(
        pmcids
        or any(
            item["raw"].get("isOpenAccess") in {True, "Y", "yes", "true"}
            for item in group
        )
        or any(url.casefold().endswith(".pdf") for url in urls)
    )
    bucket = _design_bucket(
        types,
        is_trial=any(item["kind"] == "trial" for item in group),
    )
    canonical_identity = (
        f"doi:{first('doi').casefold()}"
        if first("doi")
        else f"pmid:{first('pmid')}"
        if first("pmid")
        else f"nct:{first('nct_id').casefold()}"
        if first("nct_id")
        else f"title-year:{normalise_title(first('title'))}:{first('year') or 'unknown'}"
    )
    return {
        "group_id": payload_hash(
            {
                "canonical_identity": canonical_identity,
                "source_record_ids": sorted(
                    item["source_record_id"] for item in group
                ),
            }
        ),
        "canonical_identity": canonical_identity,
        "title": first("title"),
        "year": first("year"),
        "doi": first("doi"),
        "pmid": first("pmid"),
        "nct_id": first("nct_id"),
        "providers": providers,
        "query_ids": query_ids,
        "phases": phases,
        "evidence_lanes": _evidence_lanes(query_ids),
        "publication_types": types,
        "design_bucket": bucket,
        "screening_priority": _screening_priority(bucket, phases),
        "open_access_candidate": open_access,
        "pmcids": pmcids,
        "urls": urls,
        "abstract_available": any(bool(item["raw"].get("abstract")) for item in group),
        "source_record_ids": sorted(item["source_record_id"] for item in group),
        "retrieval_record_count": len(group),
        "preferred_source_record_id": preferred["source_record_id"],
    }


def _csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "|".join(str(value) for value in row[field])
                    if isinstance(row.get(field), list)
                    else row.get(field)
                    for field in fields
                }
            )


def build_outputs(database: Path, campaign: Path, output: Path) -> dict[str, Any]:
    records, ledger = load_records(database, campaign)
    groups = [project_group(group) for group in deduplicate(records)]
    groups.sort(
        key=lambda item: (
            item["screening_priority"],
            -(item["year"] or 0),
            item["title"].casefold(),
        )
    )
    by_provider = Counter(record["provider"] for record in records)
    by_design = Counter(group["design_bucket"] for group in groups)
    by_lane = Counter(
        lane for group in groups for lane in group["evidence_lanes"]
    )
    full_text = [
        {
            **group,
            "retrieval_action": (
                "retrieve_europe_pmc_or_existing_oa"
                if group["open_access_candidate"]
                else "query_unpaywall_then_library_or_author"
                if group["doi"]
                else "resolve_identifier_then_library_or_author"
            ),
        }
        for group in groups
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": json.loads(campaign.read_text(encoding="utf-8"))[
            "campaign_id"
        ],
        "database": str(database.resolve()),
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "ledger_entries": ledger["entries"],
        "ledger_head_sha256": ledger["head_sha256"],
        "retrieval_records": len(records),
        "unique_candidate_groups": len(groups),
        "duplicate_retrieval_records": len(records) - len(groups),
        "providers": dict(sorted(by_provider.items())),
        "design_buckets": dict(sorted(by_design.items())),
        "evidence_lanes": dict(sorted(by_lane.items())),
        "open_access_candidates": sum(
            group["open_access_candidate"] for group in groups
        ),
        "doi_resolution_candidates": sum(
            bool(group["doi"]) and not group["open_access_candidate"]
            for group in groups
        ),
        "identifier_resolution_needed": sum(
            not group["doi"] and not group["open_access_candidate"]
            for group in groups
        ),
        "screening_queue_payload_sha256": payload_hash(groups),
        "full_text_queue_payload_sha256": payload_hash(full_text),
        "authority": "retrieval_and_screening_queue_only",
        "scientific_ranking_complete": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    common_fields = [
        "group_id",
        "canonical_identity",
        "screening_priority",
        "design_bucket",
        "evidence_lanes",
        "title",
        "year",
        "doi",
        "pmid",
        "nct_id",
        "providers",
        "query_ids",
        "publication_types",
        "abstract_available",
        "open_access_candidate",
        "pmcids",
        "urls",
        "retrieval_record_count",
        "source_record_ids",
    ]
    _csv(output / "SCREENING_QUEUE.csv", groups, common_fields)
    _csv(
        output / "FULL_TEXT_QUEUE.csv",
        full_text,
        [*common_fields, "retrieval_action"],
    )
    summary["screening_queue_file_sha256"] = hashlib.sha256(
        (output / "SCREENING_QUEUE.csv").read_bytes()
    ).hexdigest()
    summary["full_text_queue_file_sha256"] = hashlib.sha256(
        (output / "FULL_TEXT_QUEUE.csv").read_bytes()
    ).hexdigest()
    (output / "ACQUISITION_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = f"""# Skin evidence acquisition report

Status: **retrieval and deterministic deduplication complete; eligibility screening,
full-text appraisal and treatment ranking remain incomplete**.

- Retrieval records: **{summary['retrieval_records']}**
- Unique candidate groups: **{summary['unique_candidate_groups']}**
- Duplicate retrieval records collapsed: **{summary['duplicate_retrieval_records']}**
- Europe PMC records: **{by_provider.get('europe_pmc', 0)}**
- PubMed records: **{by_provider.get('pubmed', 0)}**
- ClinicalTrials.gov records: **{by_provider.get('clinical_trials', 0)}**
- Open-access candidates: **{summary['open_access_candidates']}**
- DOI candidates requiring Unpaywall/library resolution:
  **{summary['doi_resolution_candidates']}**
- Candidates requiring identifier resolution:
  **{summary['identifier_resolution_needed']}**

The queue order is a screening priority, not an effectiveness ranking. Search retrieval,
citation count, publication type and open-access availability do not establish benefit,
safety or applicability. No treatment recommendation is authorised.
"""
    (output / "ACQUISITION_REPORT.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = build_outputs(args.database, args.campaign, args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
