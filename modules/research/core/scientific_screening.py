"""Deterministic canonicalisation, deduplication and initial scientific screening.

This module deliberately stops before scientific appraisal.  It may prioritise records and
propose conservative eligibility dispositions, but effectiveness conclusions still require
verified full text and the appraisal contract in :mod:`scientific_evidence`.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .scientific_evidence import ScientificEvidenceError, canonical_json, payload_hash


SCREENING_SCHEMA = "giga.scientific-screening-snapshot.v1"
CANONICAL_RECORD_SCHEMA = "giga.scientific-canonical-record.v1"
ELIGIBILITY_SCHEMA = "giga.scientific-eligibility-proposal.v1"
LLM_REVIEW_SCHEMA = "giga.scientific-llm-review-queue.v1"
PRISMA_SCHEMA = "giga.scientific-prisma-flow.v1"
NEGATIVE_CONTROL_SCHEMA = "giga.scientific-negative-control-report.v1"
EXCLUSION_AUDIT_SCHEMA = "giga.scientific-exclusion-audit-report.v1"
DUPLICATE_DECISION_SCHEMA = "giga.scientific-duplicate-decisions.v1"
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
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
PMID_PATTERN = re.compile(r"^\d{1,12}$")
PMCID_PATTERN = re.compile(r"^PMC\d+$", re.IGNORECASE)
NCT_PATTERN = re.compile(r"^NCT\d{8}$", re.IGNORECASE)
SPACE = re.compile(r"\s+")
TITLE_WORD = re.compile(r"[a-z0-9]+")
HUMAN_TERMS = re.compile(
    r"\b(human|patient|participant|volunteer|adult|women|men|clinical)\b",
    re.IGNORECASE,
)
EXPLICIT_ANIMAL_TERMS = re.compile(
    r"\b(mouse|mice|murine|rat|rats|rabbit|rabbits|animal model|"
    r"animal study|animal studies|porcine (?:skin|model|wound|subjects?))\b",
    re.IGNORECASE,
)
LAB_ONLY_TERMS = re.compile(
    r"\b(in vitro|ex vivo|cell line|fibroblast culture)\b",
    re.IGNORECASE,
)
RETRACTION_TERMS = re.compile(
    r"\b(retracted publication|retraction of|this article has been retracted)\b",
    re.IGNORECASE,
)
DOMAIN_ANCHOR = re.compile(
    r"\b(photoag\w*|photodamag\w*|anti[ -]?ag(?:e|ing)|anti[ -]?ageing|"
    r"skin|cutaneous|dermal|face|facial|rejuvenation|cosmetic dermatology|"
    r"aesthetic dermatology|"
    r"wrinkles?|scars?|scarring|keloids?|facial pores?|"
    r"postinflammatory hyperpigmentation|collagen biostimulators?)\b",
    re.IGNORECASE,
)
EDITORIAL_TYPES = {
    "comment",
    "editorial",
    "letter",
    "news",
    "newspaper article",
}
SCALE_PATTERNS = {
    "ECCA": re.compile(r"\bECCA\b", re.IGNORECASE),
    "Goodman-Baron": re.compile(r"\bGoodman(?: and| &)? Baron\b", re.IGNORECASE),
    "POSAS": re.compile(
        r"\b(?:POSAS|Patient and Observer Scar Assessment Scale)\b",
        re.IGNORECASE,
    ),
    "Vancouver Scar Scale": re.compile(
        r"\b(?:Vancouver Scar Scale|VSS)\b", re.IGNORECASE
    ),
    "Fitzpatrick wrinkle scale": re.compile(
        r"\bFitzpatrick wrinkle(?: and elastosis)? scale\b", re.IGNORECASE
    ),
}
SAMPLE_PATTERNS = (
    re.compile(r"\b(?:n|N)\s*=\s*(\d{1,5})\b"),
    re.compile(
        r"\b(\d{1,5})\s+(?:patients|participants|subjects|volunteers)\b",
        re.IGNORECASE,
    ),
)
FOLLOWUP_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)"
    r"(?:\s+(?:of\s+)?follow[ -]?up)?\b",
    re.IGNORECASE,
)
PHOTOTYPE_PATTERN = re.compile(
    r"\bFitzpatrick(?: skin)?(?: type| phototype)?s?\s*"
    r"([IVXLC]+(?:\s*[-–]\s*[IVXLC]+)?)",
    re.IGNORECASE,
)
FUNDING_PATTERN = re.compile(
    r"\b(fund(?:ed|ing)|sponsor(?:ed|ship)?|industry|manufacturer|"
    r"conflict(?:s)? of interest|competing interest)\b",
    re.IGNORECASE,
)


def normalise_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    result = html.unescape(value)
    result = unicodedata.normalize("NFKC", result)
    return SPACE.sub(" ", result).strip()


def normalise_title(value: Any) -> str:
    text = normalise_text(value) or ""
    decomposed = unicodedata.normalize("NFKD", text).casefold()
    decomposed = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(TITLE_WORD.findall(decomposed))


def normalise_doi(value: Any) -> str | None:
    text = normalise_text(value)
    if text is None:
        return None
    lowered = text.casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :]
            break
    result = text.strip().rstrip(".,;").casefold()
    return result if DOI_PATTERN.fullmatch(result) else None


def normalise_identifier(
    value: Any,
    *,
    pattern: re.Pattern[str],
    upper: bool = False,
) -> str | None:
    text = normalise_text(value)
    if text is None:
        return None
    result = text.upper() if upper else text
    return result if pattern.fullmatch(result) else None


def _normalise_year(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value if 1600 <= value <= 2200 else None
    text = normalise_text(value)
    if text:
        match = re.search(r"\b(1[6-9]\d{2}|20\d{2}|21\d{2}|2200)\b", text)
        if match:
            return int(match.group(1))
    return None


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        text = normalise_text(value)
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                candidate = (
                    item.get("name")
                    or item.get("display_name")
                    or " ".join(
                        part
                        for part in (
                            str(item.get("given") or "").strip(),
                            str(item.get("family") or "").strip(),
                        )
                        if part
                    )
                )
            else:
                candidate = item
            text = normalise_text(candidate)
            if text:
                result.append(text)
        return list(dict.fromkeys(result))
    return []


def _publication_types(raw: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("publicationTypes", "pubType", "pubtype"):
        values.extend(_strings(raw.get(key)))
    provider = raw.get("providerRecord")
    if isinstance(provider, Mapping):
        for key in ("publicationTypes", "pubType", "pubtype"):
            values.extend(_strings(provider.get(key)))
        nested = provider.get("pubTypeList")
        if isinstance(nested, Mapping):
            values.extend(_strings(nested.get("pubType")))
    return sorted(set(values), key=str.casefold)


def _authors(raw: Mapping[str, Any]) -> list[str]:
    values = _strings(raw.get("authors"))
    if values:
        return values
    provider = raw.get("providerRecord")
    if isinstance(provider, Mapping):
        values = _strings(provider.get("authors"))
        if values:
            return values
        author_string = normalise_text(provider.get("authorString"))
        if author_string:
            return [
                item.strip()
                for item in author_string.split(",")
                if item.strip()
            ]
    return []


def _urls(raw: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for value in _strings(raw.get("urls")):
        if value.startswith("http://doi.org/"):
            value = "https://doi.org/" + value.removeprefix("http://doi.org/")
        if value.startswith("https://"):
            result.append(value)
    return sorted(set(result))


def _pmcid(raw: Mapping[str, Any]) -> str | None:
    direct = normalise_identifier(raw.get("pmcid"), pattern=PMCID_PATTERN, upper=True)
    if direct:
        return direct
    provider = raw.get("providerRecord")
    if isinstance(provider, Mapping):
        return normalise_identifier(
            provider.get("pmcid"),
            pattern=PMCID_PATTERN,
            upper=True,
        )
    return None


def _retracted(raw: Mapping[str, Any], title: str, types: Sequence[str]) -> bool:
    flags = (
        raw.get("isRetracted"),
        raw.get("retracted"),
        (raw.get("providerRecord") or {}).get("isRetracted")
        if isinstance(raw.get("providerRecord"), Mapping)
        else None,
    )
    if any(flag is True or str(flag).strip().upper() == "Y" for flag in flags):
        return True
    return bool(
        RETRACTION_TERMS.search(title)
        or any("retracted publication" in value.casefold() for value in types)
    )


def _query_lookup(
    campaign: Mapping[str, Any],
    compiled_manifest: Mapping[str, Any],
) -> dict[tuple[str | None, str, int | None], tuple[str, bool, bool]]:
    candidates: dict[
        tuple[str | None, str, int | None],
        set[tuple[str, bool, bool]],
    ] = defaultdict(set)
    campaign_queries = {
        str(query.get("id")): query
        for query in campaign.get("queries", [])
        if isinstance(query, Mapping)
    }
    default_max = int(campaign.get("defaults", {}).get("max_results", 50))
    for query in campaign.get("queries", []):
        if isinstance(query, Mapping):
            maximum = int(query.get("max_results", default_max))
            candidates[(None, str(query.get("question")), maximum)].add(
                (
                    str(query.get("id")),
                    bool(query.get("negative_control")),
                    False,
                )
            )
    for query in compiled_manifest.get("queries", []):
        if not isinstance(query, Mapping):
            continue
        campaign_query = campaign_queries.get(str(query["id"]))
        if campaign_query is None:
            continue
        maximum = int(campaign_query.get("max_results", default_max))
        for provider, expression in query.get("provider_queries", {}).items():
            candidates[(str(provider), str(expression), maximum)].add(
                (
                    str(query["id"]),
                    bool(query.get("negative_control")),
                    True,
                )
            )
    lookup: dict[
        tuple[str | None, str, int | None],
        tuple[str, bool, bool],
    ] = {}
    by_expression: dict[
        tuple[str | None, str],
        set[tuple[str, bool, bool]],
    ] = defaultdict(set)
    for key, values in candidates.items():
        if len(values) != 1:
            raise ScientificEvidenceError(
                "campaign query provenance is ambiguous for "
                f"provider={key[0]!r}, max_results={key[2]!r}, "
                f"expression={key[1]!r}"
            )
        value = next(iter(values))
        lookup[key] = value
        by_expression[(key[0], key[1])].add(value)
    for (provider, expression), values in by_expression.items():
        if len(values) == 1:
            lookup[(provider, expression, None)] = next(iter(values))
    return lookup


def load_canonical_records(
    database: Path,
    campaign: Mapping[str, Any],
    compiled_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not database.is_file():
        raise ScientificEvidenceError(f"evidence database does not exist: {database}")
    lookup = _query_lookup(campaign, compiled_manifest)
    connection = sqlite3.connect(
        f"{database.resolve().as_uri()}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT s.*,a.provider,a.endpoint,a.request_json,a.observed_at
        FROM source_records AS s
        JOIN acquisitions AS a ON a.id=s.acquisition_id
        ORDER BY a.observed_at,a.provider,s.rank,s.id
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
    unmapped: Counter[str] = Counter()
    for row in rows:
        raw = json.loads(row["raw_json"])
        request = json.loads(row["request_json"])
        provider = str(row["provider"])
        expression = str(request.get("query") or "")
        request_maximum = request.get("maxResults")
        maximum = int(request_maximum) if request_maximum is not None else None
        query_id, negative_control, compiled_query = lookup.get(
            (provider, expression, maximum),
            lookup.get(
                (None, expression, maximum),
                lookup.get(
                    (provider, expression, None),
                    lookup.get(
                        (None, expression, None),
                        ("unmapped", False, False),
                    ),
                ),
            ),
        )
        if query_id == "unmapped":
            unmapped[expression] += 1
        title = normalise_text(row["title"]) or ""
        abstract = normalise_text(raw.get("abstract"))
        types = _publication_types(raw)
        doi = normalise_doi(row["doi"] or raw.get("doi"))
        pmid = normalise_identifier(
            row["pmid"] or raw.get("pmid"),
            pattern=PMID_PATTERN,
        )
        nct = normalise_identifier(
            row["nct_id"] or raw.get("nctId"),
            pattern=NCT_PATTERN,
            upper=True,
        )
        pmcid = _pmcid(raw)
        record = {
            "schema_version": CANONICAL_RECORD_SCHEMA,
            "source_record_id": row["id"],
            "acquisition_id": row["acquisition_id"],
            "provider": provider,
            "endpoint": row["endpoint"],
            "kind": row["kind"],
            "query_id": query_id,
            "negative_control": negative_control,
            "compiled_query_retrieval": compiled_query,
            "title": title,
            "normalised_title": normalise_title(title),
            "abstract": abstract,
            "year": _normalise_year(row["year"] or raw.get("year")),
            "doi": doi,
            "pmid": pmid,
            "pmcid": pmcid,
            "nct_id": nct,
            "authors": _authors(raw),
            "venue": normalise_text(raw.get("venue")),
            "publication_types": types,
            "urls": _urls(raw),
            "open_access_candidate": bool(
                pmcid
                or raw.get("isOpenAccess") is True
                or str(raw.get("isOpenAccess")).upper() == "Y"
                or any(url.casefold().endswith(".pdf") for url in _urls(raw))
            ),
            "retracted": _retracted(raw, title, types),
            "retrieved_at": row["retrieved_at"],
            "raw_sha256": row["raw_sha256"],
        }
        record["canonical_record_sha256"] = payload_hash(record)
        records.append(record)
    return records, {
        "ledger_entries": int(ledger["entries"]),
        "ledger_head_sha256": ledger["head"],
        "unmapped_records": sum(unmapped.values()),
        "unmapped_queries": dict(sorted(unmapped.items())),
    }


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


def _exact_tokens(record: Mapping[str, Any]) -> list[str]:
    result = [
        f"{field}:{record[field]}"
        for field in ("doi", "pmid", "pmcid", "nct_id")
        if record.get(field)
    ]
    if len(record["normalised_title"]) >= 20:
        result.append(
            f"title-year:{record['normalised_title']}:{record.get('year') or 'unknown'}"
        )
    return result


def exact_deduplicate(
    records: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    union = UnionFind(len(records))
    seen: dict[str, int] = {}
    for index, record in enumerate(records):
        for token in _exact_tokens(record):
            if token in seen:
                union.union(index, seen[token])
            else:
                seen[token] = index
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[union.find(index)].append(record)
    return [
        sorted(group, key=lambda item: item["source_record_id"])
        for _, group in sorted(grouped.items())
    ]


def _first_author(record: Mapping[str, Any]) -> str | None:
    authors = record.get("authors", [])
    if not authors:
        return None
    return normalise_title(authors[0])


def project_group(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        records,
        key=lambda item: (
            PROVIDER_PREFERENCE.get(item["provider"], 99),
            not bool(item.get("abstract")),
            item["source_record_id"],
        ),
    )

    def first(field: str) -> Any:
        for record in ordered:
            if record.get(field) not in (None, "", []):
                return record[field]
        return None

    identifiers = {
        field: sorted(
            {
                str(record[field])
                for record in records
                if record.get(field) not in (None, "")
            }
        )
        for field in ("doi", "pmid", "pmcid", "nct_id")
    }
    canonical_identity = next(
        (
            f"{field}:{values[0]}"
            for field, values in identifiers.items()
            if values
        ),
        f"title-year:{normalise_title(first('title'))}:{first('year') or 'unknown'}",
    )
    source_ids = sorted(record["source_record_id"] for record in records)
    group = {
        "group_id": payload_hash(
            {
                "canonical_identity": canonical_identity,
                "source_record_ids": source_ids,
            }
        ),
        "canonical_identity": canonical_identity,
        "title": first("title"),
        "normalised_title": normalise_title(first("title")),
        "abstract": first("abstract"),
        "year": first("year"),
        "doi": first("doi"),
        "pmid": first("pmid"),
        "pmcid": first("pmcid"),
        "nct_id": first("nct_id"),
        "authors": first("authors") or [],
        "providers": sorted({record["provider"] for record in records}),
        "query_ids": sorted({record["query_id"] for record in records}),
        "negative_control": all(record["negative_control"] for record in records),
        "compiled_query_retrieval": any(
            bool(record.get("compiled_query_retrieval")) for record in records
        ),
        "publication_types": sorted(
            {
                item
                for record in records
                for item in record["publication_types"]
            },
            key=str.casefold,
        ),
        "urls": sorted(
            {url for record in records for url in record["urls"]}
        ),
        "open_access_candidate": any(
            record["open_access_candidate"] for record in records
        ),
        "retracted": any(record["retracted"] for record in records),
        "source_record_ids": source_ids,
        "retrieval_record_count": len(records),
        "preferred_source_record_id": ordered[0]["source_record_id"],
        "identifier_sets": identifiers,
    }
    group["group_payload_sha256"] = payload_hash(group)
    return group


def possible_duplicate_pairs(
    groups: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.94,
) -> list[dict[str, Any]]:
    if not 0.8 <= threshold <= 1:
        raise ScientificEvidenceError("possible-duplicate threshold must be 0.8 to 1")
    by_year: dict[int | None, list[Mapping[str, Any]]] = defaultdict(list)
    for group in groups:
        by_year[group.get("year")].append(group)
    candidates: list[dict[str, Any]] = []
    for year, bucket in sorted(by_year.items(), key=lambda pair: pair[0] or 0):
        for left_index, left in enumerate(bucket):
            left_title = str(left["normalised_title"])
            if len(left_title) < 20:
                continue
            for right in bucket[left_index + 1 :]:
                right_title = str(right["normalised_title"])
                if len(right_title) < 20:
                    continue
                similarity = SequenceMatcher(
                    None,
                    left_title,
                    right_title,
                    autojunk=False,
                ).ratio()
                if similarity < threshold:
                    continue
                left_author = _first_author(left)
                right_author = _first_author(right)
                author_compatible = (
                    not left_author
                    or not right_author
                    or left_author == right_author
                )
                if not author_compatible:
                    continue
                pair = {
                    "left_group_id": left["group_id"],
                    "right_group_id": right["group_id"],
                    "year": year,
                    "title_similarity": round(similarity, 6),
                    "first_author_compatible": author_compatible,
                    "disposition": "manual_review_required",
                    "auto_merged": False,
                }
                pair["pair_sha256"] = payload_hash(pair)
                candidates.append(pair)
    return sorted(candidates, key=lambda item: item["pair_sha256"])


def apply_duplicate_decisions(
    record_groups: Sequence[Sequence[dict[str, Any]]],
    projected_groups: Sequence[dict[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    decisions_document: Mapping[str, Any] | None,
    *,
    campaign_id: str,
) -> tuple[
    list[list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    pair_by_id = {item["pair_sha256"]: item for item in pairs}
    if decisions_document is None:
        duplicate_report = {
            "schema_version": DUPLICATE_DECISION_SCHEMA,
            "candidate_pairs": len(pairs),
            "decisions_recorded": 0,
            "same_study": 0,
            "different_study": 0,
            "uncertain": 0,
            "unresolved_pairs": len(pairs),
            "complete": not pairs,
            "human_verified": False,
        }
        duplicate_report["report_sha256"] = payload_hash(duplicate_report)
        return (
            [list(group) for group in record_groups],
            [dict(pair) for pair in pairs],
            duplicate_report,
        )
    required = {"schema_version", "campaign_id", "pair_set_sha256", "decisions"}
    if not isinstance(decisions_document, Mapping) or set(decisions_document) != required:
        raise ScientificEvidenceError(
            "invalid duplicate decision fields; "
            f"missing={sorted(required - set(decisions_document))}, "
            f"extra={sorted(set(decisions_document) - required)}"
        )
    if decisions_document["schema_version"] != DUPLICATE_DECISION_SCHEMA:
        raise ScientificEvidenceError(
            f"duplicate decision schema_version must be {DUPLICATE_DECISION_SCHEMA}"
        )
    if decisions_document["campaign_id"] != campaign_id:
        raise ScientificEvidenceError("duplicate decision campaign_id mismatch")
    expected_pair_hash = payload_hash(list(pairs))
    if decisions_document["pair_set_sha256"] != expected_pair_hash:
        raise ScientificEvidenceError("duplicate decision pair_set_sha256 mismatch")
    decisions = decisions_document["decisions"]
    if not isinstance(decisions, list):
        raise ScientificEvidenceError("duplicate decisions must be an array")
    seen: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for index, decision in enumerate(decisions):
        if not isinstance(decision, Mapping) or set(decision) != {
            "pair_sha256",
            "decision",
            "decided_by",
            "decided_at",
            "evidence",
        }:
            raise ScientificEvidenceError(
                f"duplicate decisions[{index}] has invalid fields"
            )
        pair_id = decision["pair_sha256"]
        if pair_id not in pair_by_id:
            raise ScientificEvidenceError(
                f"duplicate decision references unknown pair: {pair_id}"
            )
        if pair_id in seen:
            raise ScientificEvidenceError(
                f"duplicate decision repeats pair: {pair_id}"
            )
        seen.add(pair_id)
        disposition = decision["decision"]
        if disposition not in {"same_study", "different_study", "uncertain"}:
            raise ScientificEvidenceError(
                "duplicate decision must be same_study, different_study or uncertain"
            )
        if (
            not isinstance(decision["decided_by"], str)
            or not decision["decided_by"].strip()
            or not isinstance(decision["decided_at"], str)
            or not decision["decided_at"].strip()
            or not isinstance(decision["evidence"], Mapping)
            or not decision["evidence"]
        ):
            raise ScientificEvidenceError(
                "duplicate decisions require decided_by, decided_at and evidence"
            )
        cleaned.append(dict(decision))
    group_index = {
        group["group_id"]: index for index, group in enumerate(projected_groups)
    }
    union = UnionFind(len(record_groups))
    for decision in cleaned:
        if decision["decision"] != "same_study":
            continue
        pair = pair_by_id[decision["pair_sha256"]]
        union.union(
            group_index[pair["left_group_id"]],
            group_index[pair["right_group_id"]],
        )
    merged: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, records in enumerate(record_groups):
        merged[union.find(index)].extend(records)
    final_record_groups = [
        sorted(records, key=lambda item: item["source_record_id"])
        for _, records in sorted(merged.items())
    ]
    resolved = {
        decision["pair_sha256"]
        for decision in cleaned
        if decision["decision"] in {"same_study", "different_study"}
    }
    uncertain = {
        decision["pair_sha256"]
        for decision in cleaned
        if decision["decision"] == "uncertain"
    }
    unresolved = [
        dict(pair)
        for pair in pairs
        if pair["pair_sha256"] not in resolved
        or pair["pair_sha256"] in uncertain
    ]
    counts = Counter(decision["decision"] for decision in cleaned)
    report = {
        "schema_version": DUPLICATE_DECISION_SCHEMA,
        "candidate_pairs": len(pairs),
        "pair_set_sha256": expected_pair_hash,
        "decisions_recorded": len(cleaned),
        "same_study": counts["same_study"],
        "different_study": counts["different_study"],
        "uncertain": counts["uncertain"],
        "unresolved_pairs": len(unresolved),
        "complete": not unresolved,
        "human_verified": bool(cleaned),
    }
    report["report_sha256"] = payload_hash(report)
    return final_record_groups, unresolved, report


def _design_bucket(types: Sequence[str], title: str, abstract: str | None) -> str:
    joined = " ".join([*types, title, abstract or ""]).casefold()
    if "guideline" in joined or "practice guideline" in joined:
        return "guideline"
    if "meta-analysis" in joined or "meta analysis" in joined:
        return "meta_analysis"
    if "systematic review" in joined:
        return "systematic_review"
    if (
        "randomized controlled trial" in joined
        or "randomised controlled trial" in joined
        or "randomized trial" in joined
        or "randomised trial" in joined
    ):
        return "randomized_controlled_trial"
    if "controlled clinical trial" in joined or "controlled trial" in joined:
        return "controlled_trial"
    if "clinical trial" in joined:
        return "clinical_trial"
    if "cohort" in joined or "longitudinal" in joined:
        return "cohort_or_longitudinal"
    if "case-control" in joined or "case control" in joined:
        return "case_control"
    if "case series" in joined:
        return "case_series"
    if "review" in joined:
        return "nonsystematic_review"
    if "protocol" in joined:
        return "protocol"
    return "primary_or_other"


def _term_evidence(
    text_fields: Mapping[str, str | None],
    terms: Iterable[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for concept_kind, concept_id, term in terms:
        pattern = _phrase_pattern(term)
        for field, text in text_fields.items():
            if not text:
                continue
            for match in pattern.finditer(text):
                matches.append(
                    {
                        "rule_id": f"ontology.{concept_kind}.{concept_id}",
                        "concept_kind": concept_kind,
                        "concept_id": concept_id,
                        "term": term,
                        "field": field,
                        "start": match.start(),
                        "end": match.end(),
                        "text": match.group(0),
                    }
                )
    return matches


def _phrase_pattern(term: str) -> re.Pattern[str]:
    suffix = ""
    if term[-1:].isalpha() and not term.casefold().endswith(
        ("s", "y", "ing", "ed")
    ):
        suffix = "s?"
    return re.compile(
        r"(?<!\w)" + re.escape(term) + suffix + r"(?!\w)",
        re.IGNORECASE,
    )


def _all_ontology_terms(
    ontology: Mapping[str, Any],
    *,
    excluded_concepts: Mapping[str, set[str]] | None = None,
) -> list[tuple[str, str, str]]:
    excluded = excluded_concepts or {}
    return [
        (kind, concept_id, term)
        for kind in ("conditions", "interventions", "outcomes")
        for concept_id, concept in ontology[kind].items()
        if concept_id not in excluded.get(kind, set())
        for term in concept["terms"]
    ]


def _control_only_concepts(
    specification: Mapping[str, Any],
) -> dict[str, set[str]]:
    field_by_kind = {
        "conditions": "condition_concepts",
        "interventions": "intervention_concepts",
        "outcomes": "outcome_concepts",
    }
    result: dict[str, set[str]] = {}
    for kind, field in field_by_kind.items():
        positive = {
            concept
            for query in specification["queries"]
            for concept in query[field]
        }
        controls = {
            concept
            for query in specification["negative_controls"]
            for concept in query[field]
        }
        result[kind] = controls - positive
    return result


def extract_features(
    group: Mapping[str, Any],
    ontology: Mapping[str, Any],
    specification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    text_fields = {
        "title": group.get("title"),
        "abstract": group.get("abstract"),
    }
    excluded = (
        _control_only_concepts(specification)
        if specification is not None and not group.get("negative_control")
        else None
    )
    term_matches = _term_evidence(
        text_fields,
        _all_ontology_terms(ontology, excluded_concepts=excluded),
    )
    concepts: dict[str, list[str]] = {}
    for kind in ("conditions", "interventions", "outcomes"):
        concepts[kind] = sorted(
            {
                match["concept_id"]
                for match in term_matches
                if match["concept_kind"] == kind
            }
        )
    combined = " ".join(
        value for value in text_fields.values() if isinstance(value, str)
    )
    scales = [
        {
            "name": name,
            "text": match.group(0),
            "start": match.start(),
            "end": match.end(),
        }
        for name, pattern in SCALE_PATTERNS.items()
        for match in pattern.finditer(combined)
    ]
    samples = sorted(
        {
            int(match.group(1))
            for pattern in SAMPLE_PATTERNS
            for match in pattern.finditer(combined)
            if 1 <= int(match.group(1)) <= 100_000
        }
    )
    follow_up = [
        {
            "value": float(match.group(1)),
            "unit": match.group(2).casefold(),
            "text": match.group(0),
        }
        for match in FOLLOWUP_PATTERN.finditer(combined)
    ]
    phototypes = [
        {"value": match.group(1), "text": match.group(0)}
        for match in PHOTOTYPE_PATTERN.finditer(combined)
    ]
    funding_mentions = [
        {"text": match.group(0), "start": match.start(), "end": match.end()}
        for match in FUNDING_PATTERN.finditer(combined)
    ]
    return {
        "group_id": group["group_id"],
        "design_bucket": _design_bucket(
            group["publication_types"],
            group["title"],
            group.get("abstract"),
        ),
        "concepts": concepts,
        "term_evidence": term_matches,
        "outcome_scales": scales,
        "sample_size_candidates": samples,
        "follow_up_candidates": follow_up,
        "phototype_candidates": phototypes,
        "funding_or_conflict_mentions": funding_mentions,
        "extraction_authority": "metadata_triage_only",
        "effect_estimate_extracted": False,
    }


def _spec_by_id(specification: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        item["id"]: item
        for item in [
            *specification["queries"],
            *specification["negative_controls"],
        ]
    }


def propose_eligibility(
    group: Mapping[str, Any],
    features: Mapping[str, Any],
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    title = group["title"]
    abstract = group.get("abstract")
    combined = " ".join(value for value in (title, abstract) if value)
    types = {value.casefold() for value in group["publication_types"]}
    design = features["design_bucket"]
    matched = features["concepts"]
    spec_map = _spec_by_id(specification)
    query_evaluations: list[dict[str, Any]] = []
    for query_id in group["query_ids"]:
        query = spec_map.get(query_id)
        if query is None:
            continue
        condition_matches = sorted(
            set(query["condition_concepts"]) & set(matched["conditions"])
        )
        intervention_matches = sorted(
            set(query["intervention_concepts"]) & set(matched["interventions"])
        )
        outcome_matches = sorted(
            set(query["outcome_concepts"]) & set(matched["outcomes"])
        )
        extra_groups = [
            [
                term
                for term in group_terms
                if _phrase_pattern(term).search(combined)
            ]
            for group_terms in query["extra_required_groups"]
        ]
        satisfied = bool(condition_matches and intervention_matches)
        if query["require_outcomes"]:
            satisfied = satisfied and bool(outcome_matches)
        satisfied = satisfied and all(extra_groups)
        query_evaluations.append(
            {
                "query_id": query_id,
                "condition_matches": condition_matches,
                "intervention_matches": intervention_matches,
                "outcome_matches": outcome_matches,
                "extra_group_matches": extra_groups,
                "criteria_satisfied": satisfied,
                "negative_control": query["negative_control"],
            }
        )
    reasons: list[str] = []
    status: str
    if group["retracted"]:
        status = "auto_exclude"
        reasons.append("EX_RETRACTED")
    elif EXPLICIT_ANIMAL_TERMS.search(title) and not HUMAN_TERMS.search(combined):
        status = "auto_exclude"
        reasons.append("EX_NONHUMAN_ONLY")
    elif (
        LAB_ONLY_TERMS.search(title)
        and not HUMAN_TERMS.search(combined)
        and not re.search(r"\bin vivo\b", title, re.IGNORECASE)
    ):
        status = "auto_exclude"
        reasons.append("EX_LAB_ONLY")
    elif any(item in types for item in EDITORIAL_TYPES):
        # Provider publication types can conflict: PubMed may label a short
        # clinical study or systematic review as a letter.  The type is useful
        # prioritisation evidence, but never sufficient for automatic
        # exclusion.
        status = "manual_review"
        reasons.append("MR_NONRESEARCH_PUBLICATION_TYPE")
    elif not DOMAIN_ANCHOR.search(combined) and group["negative_control"]:
        status = "auto_exclude"
        reasons.append("EX_NO_SKIN_SCAR_DOMAIN_ANCHOR")
    elif not DOMAIN_ANCHOR.search(combined) and group["compiled_query_retrieval"]:
        status = "manual_review"
        reasons.append("MR_COMPILED_QUERY_MATCH_WITHOUT_METADATA_ANCHOR")
        if abstract is None:
            reasons.append("MR_ABSTRACT_UNAVAILABLE")
    elif not DOMAIN_ANCHOR.search(combined):
        status = "auto_exclude"
        reasons.append("EX_NO_SKIN_SCAR_DOMAIN_ANCHOR")
    elif any(
        evaluation["criteria_satisfied"] and not evaluation["negative_control"]
        for evaluation in query_evaluations
    ):
        status = "auto_include"
        reasons.append("IN_QUERY_CONCEPTS_SATISFIED")
    else:
        status = "manual_review"
        reasons.append("MR_INCOMPLETE_OR_AMBIGUOUS_CONCEPT_MATCH")
        if abstract is None:
            reasons.append("MR_ABSTRACT_UNAVAILABLE")
        if design == "protocol":
            reasons.append("MR_PROTOCOL_OR_REGISTRY_ROLE")
    proposal = {
        "schema_version": ELIGIBILITY_SCHEMA,
        "group_id": group["group_id"],
        "proposed_status": status,
        "reason_codes": reasons,
        "query_evaluations": query_evaluations,
        "design_bucket": design,
        "title": title,
        "year": group.get("year"),
        "human_verified": False,
        "final_eligibility_decision": None,
        "recommendation_authority": False,
    }
    proposal["proposal_sha256"] = payload_hash(proposal)
    return proposal


def _priority(
    group: Mapping[str, Any],
    proposal: Mapping[str, Any],
    features: Mapping[str, Any],
) -> tuple[int, list[str]]:
    if proposal["proposed_status"] == "auto_exclude":
        return 90, ["P90_PROPOSED_EXCLUSION_AUDIT_ONLY"]
    if proposal["proposed_status"] == "manual_review":
        return 5, ["P05_AMBIGUOUS_ELIGIBILITY"]
    design = features["design_bucket"]
    if design in {"guideline", "meta_analysis", "systematic_review"}:
        return 10, ["P10_SYNTHESIS_OR_GUIDELINE"]
    if design in {
        "randomized_controlled_trial",
        "controlled_trial",
        "clinical_trial",
    }:
        return 20, ["P20_CONTROLLED_OR_CLINICAL_TRIAL"]
    if group.get("nct_id") or design == "protocol":
        return 30, ["P30_REGISTRY_OR_PROTOCOL"]
    if design in {"cohort_or_longitudinal", "case_control"}:
        return 40, ["P40_OBSERVATIONAL_SAFETY_OR_DURABILITY"]
    return 50, ["P50_OTHER_ELIGIBLE_CANDIDATE"]


def deterministic_exclusion_sample(
    proposals: Sequence[Mapping[str, Any]],
    *,
    per_reason: int = 5,
) -> list[dict[str, Any]]:
    if isinstance(per_reason, bool) or not 1 <= per_reason <= 100:
        raise ScientificEvidenceError("per_reason must be an integer from 1 to 100")
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        if proposal["proposed_status"] != "auto_exclude":
            continue
        for reason in proposal["reason_codes"]:
            buckets[reason].append(proposal)
    sampled: dict[str, dict[str, Any]] = {}
    for reason, items in sorted(buckets.items()):
        ordered = sorted(
            items,
            key=lambda item: hashlib.sha256(
                f"{reason}\0{item['group_id']}".encode("utf-8")
            ).hexdigest(),
        )
        for proposal in ordered[:per_reason]:
            sampled.setdefault(
                proposal["group_id"],
                {
                    "group_id": proposal["group_id"],
                    "title": proposal["title"],
                    "year": proposal["year"],
                    "sampled_reason_codes": [],
                    "human_audit_decision": "",
                    "human_audit_notes": "",
                    "audit_complete": False,
                },
            )["sampled_reason_codes"].append(reason)
    return [
        {**item, "sampled_reason_codes": sorted(item["sampled_reason_codes"])}
        for _, item in sorted(sampled.items())
    ]


def negative_control_report(
    groups: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    control_ids = {
        group["group_id"] for group in groups if group["negative_control"]
    }
    proposal_by_group = {item["group_id"]: item for item in proposals}
    failures = [
        {
            "group_id": group_id,
            "proposed_status": proposal_by_group[group_id]["proposed_status"],
            "title": proposal_by_group[group_id]["title"],
        }
        for group_id in sorted(control_ids)
        if proposal_by_group[group_id]["proposed_status"] != "auto_exclude"
    ]
    return {
        "schema_version": NEGATIVE_CONTROL_SCHEMA,
        "controls_retrieved": len(control_ids),
        "controls_proposed_for_exclusion": len(control_ids) - len(failures),
        "failures": failures,
        "status": (
            "not_run"
            if not control_ids
            else "pass"
            if not failures
            else "fail"
        ),
        "promotion_gate_passed": bool(control_ids) and not failures,
    }


def validate_exclusion_audit(
    path: Path,
    expected_sample: Sequence[Mapping[str, Any]],
    *,
    maximum_false_exclusion_rate: float = 0.0,
) -> dict[str, Any]:
    if not 0 <= maximum_false_exclusion_rate <= 1:
        raise ScientificEvidenceError(
            "maximum_false_exclusion_rate must be between 0 and 1"
        )
    expected = {item["group_id"]: item for item in expected_sample}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if {row.get("group_id") for row in rows} != set(expected):
        raise ScientificEvidenceError(
            "exclusion audit rows do not exactly match the deterministic sample"
        )
    decisions = Counter()
    incomplete: list[str] = []
    for row in rows:
        group_id = row["group_id"]
        if row.get("title") != str(expected[group_id]["title"]):
            raise ScientificEvidenceError(
                f"exclusion audit title mismatch for {group_id}"
            )
        decision = (row.get("human_audit_decision") or "").strip()
        complete = (row.get("audit_complete") or "").strip().casefold() == "true"
        if not complete or decision not in {
            "correct_exclusion",
            "false_exclusion",
            "uncertain",
        }:
            incomplete.append(group_id)
            continue
        decisions[decision] += 1
    completed = sum(decisions.values())
    false_rate = (
        decisions["false_exclusion"] / completed if completed else None
    )
    report = {
        "schema_version": EXCLUSION_AUDIT_SCHEMA,
        "sample_size": len(expected),
        "completed": completed,
        "incomplete_group_ids": sorted(incomplete),
        "decisions": dict(sorted(decisions.items())),
        "false_exclusion_rate": false_rate,
        "maximum_false_exclusion_rate": maximum_false_exclusion_rate,
        "status": (
            "incomplete"
            if incomplete
            else "pass"
            if false_rate is not None
            and false_rate <= maximum_false_exclusion_rate
            and decisions["uncertain"] == 0
            else "fail"
        ),
    }
    report["promotion_gate_passed"] = report["status"] == "pass"
    report["report_sha256"] = payload_hash(report)
    return report


def prisma_flow(
    *,
    retrieval_records: int,
    groups: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    possible_duplicates: Sequence[Mapping[str, Any]],
    exclusion_audit: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts = Counter(proposal["proposed_status"] for proposal in proposals)
    return {
        "schema_version": PRISMA_SCHEMA,
        "records_identified": retrieval_records,
        "duplicate_retrieval_records_removed": retrieval_records - len(groups),
        "unique_candidate_groups": len(groups),
        "possible_duplicate_pairs_awaiting_review": len(possible_duplicates),
        "records_screened_by_deterministic_rules": len(proposals),
        "records_proposed_for_inclusion": counts["auto_include"],
        "records_proposed_for_exclusion": counts["auto_exclude"],
        "records_awaiting_eligibility_review": counts["manual_review"],
        "exclusion_audit_sample_size": len(exclusion_audit),
        "exclusion_audit_completed": sum(
            bool(item["audit_complete"]) for item in exclusion_audit
        ),
        "full_texts_sought": 0,
        "full_texts_not_retrieved": 0,
        "full_texts_assessed": 0,
        "studies_included_in_synthesis": 0,
        "screening_complete": False,
        "scientific_synthesis_complete": False,
        "authority": "screening_accounting_only",
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: canonical_json(row[field])
                    if isinstance(row.get(field), (list, dict))
                    else row.get(field)
                    for field in fields
                }
            )


@dataclass(frozen=True)
class ScreeningBuild:
    summary: dict[str, Any]
    records: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    features: list[dict[str, Any]]
    proposals: list[dict[str, Any]]
    possible_duplicates: list[dict[str, Any]]
    exclusion_audit: list[dict[str, Any]]
    prisma: dict[str, Any]
    negative_controls: dict[str, Any]
    duplicate_decisions: dict[str, Any]


def compile_screening(
    database: Path,
    campaign: Mapping[str, Any],
    compiled_manifest: Mapping[str, Any],
    ontology: Mapping[str, Any],
    specification: Mapping[str, Any],
    duplicate_decisions: Mapping[str, Any] | None = None,
) -> ScreeningBuild:
    records, ledger = load_canonical_records(
        database,
        campaign,
        compiled_manifest,
    )
    exact_record_groups = exact_deduplicate(records)
    paired_groups = sorted(
        (
            (project_group(group), group)
            for group in exact_record_groups
        ),
        key=lambda pair: pair[0]["group_id"],
    )
    initial_groups = [pair[0] for pair in paired_groups]
    exact_record_groups = [pair[1] for pair in paired_groups]
    initial_possible = possible_duplicate_pairs(initial_groups)
    final_record_groups, possible, duplicate_report = apply_duplicate_decisions(
        exact_record_groups,
        initial_groups,
        initial_possible,
        duplicate_decisions,
        campaign_id=campaign["campaign_id"],
    )
    groups = [project_group(group) for group in final_record_groups]
    groups.sort(key=lambda item: item["group_id"])
    if duplicate_report["same_study"]:
        new_possible = possible_duplicate_pairs(groups)
        decided_different_pairs = {
            decision["pair_sha256"]
            for decision in duplicate_decisions["decisions"]
            if decision["decision"] == "different_study"
        }
        possible = [
            pair
            for pair in new_possible
            if pair["pair_sha256"] not in decided_different_pairs
        ]
        duplicate_report["unresolved_pairs"] = len(possible)
        duplicate_report["complete"] = not possible
        duplicate_report["report_sha256"] = payload_hash(
            {
                key: value
                for key, value in duplicate_report.items()
                if key != "report_sha256"
            }
        )
    features = [
        extract_features(group, ontology, specification) for group in groups
    ]
    features_by_group = {item["group_id"]: item for item in features}
    proposals = [
        propose_eligibility(
            group,
            features_by_group[group["group_id"]],
            specification,
        )
        for group in groups
    ]
    proposal_by_group = {item["group_id"]: item for item in proposals}
    for group in groups:
        priority, reasons = _priority(
            group,
            proposal_by_group[group["group_id"]],
            features_by_group[group["group_id"]],
        )
        group["screening_priority"] = priority
        group["priority_reason_codes"] = reasons
    groups.sort(
        key=lambda item: (
            item["screening_priority"],
            -(item.get("year") or 0),
            item["normalised_title"],
            item["group_id"],
        )
    )
    proposals.sort(
        key=lambda item: (
            next(
                group["screening_priority"]
                for group in groups
                if group["group_id"] == item["group_id"]
            ),
            item["group_id"],
        )
    )
    exclusion_audit = deterministic_exclusion_sample(proposals)
    prisma = prisma_flow(
        retrieval_records=len(records),
        groups=groups,
        proposals=proposals,
        possible_duplicates=possible,
        exclusion_audit=exclusion_audit,
    )
    negative_controls = negative_control_report(groups, proposals)
    status_counts = Counter(item["proposed_status"] for item in proposals)
    summary = {
        "schema_version": SCREENING_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "database": str(database.resolve()),
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        **ledger,
        "retrieval_records": len(records),
        "unique_candidate_groups": len(groups),
        "retrieval_records_collapsed_total": len(records) - len(groups),
        "exact_duplicate_records_collapsed_before_human_merges": (
            len(records) - len(initial_groups)
        ),
        "human_confirmed_duplicate_group_merges": duplicate_report["same_study"],
        "possible_duplicate_pairs": len(possible),
        "eligibility_proposals": dict(sorted(status_counts.items())),
        "exclusion_audit_sample_size": len(exclusion_audit),
        "screening_complete": False,
        "full_text_appraisal_complete": False,
        "effectiveness_ranking_authorised": False,
        "authority": "deterministic_initial_processing_only",
        "negative_controls": negative_controls,
        "duplicate_decisions": duplicate_report,
    }
    return ScreeningBuild(
        summary=summary,
        records=records,
        groups=groups,
        features=features,
        proposals=proposals,
        possible_duplicates=possible,
        exclusion_audit=exclusion_audit,
        prisma=prisma,
        negative_controls=negative_controls,
        duplicate_decisions=duplicate_report,
    )


def write_screening(build: ScreeningBuild, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {
        "canonical_records": output / "CANONICAL_RECORDS.jsonl",
        "dedup_groups": output / "DEDUP_GROUPS.jsonl",
        "features": output / "FEATURES.jsonl",
        "eligibility": output / "ELIGIBILITY_PROPOSALS.jsonl",
        "possible_duplicates": output / "POSSIBLE_DUPLICATES.csv",
        "screening_queue": output / "SCREENING_QUEUE.csv",
        "llm_review_queue": output / "LLM_REVIEW_QUEUE.jsonl",
        "exclusion_audit": output / "EXCLUSION_AUDIT.csv",
        "prisma": output / "PRISMA.json",
        "negative_controls": output / "NEGATIVE_CONTROLS.json",
        "duplicate_decisions": output / "DUPLICATE_DECISION_REPORT.json",
        "duplicate_decision_template": output / "DUPLICATE_DECISION_TEMPLATE.json",
    }
    _write_jsonl(files["canonical_records"], build.records)
    _write_jsonl(files["dedup_groups"], build.groups)
    _write_jsonl(files["features"], build.features)
    _write_jsonl(files["eligibility"], build.proposals)
    _write_csv(
        files["possible_duplicates"],
        build.possible_duplicates,
        [
            "pair_sha256",
            "left_group_id",
            "right_group_id",
            "year",
            "title_similarity",
            "first_author_compatible",
            "disposition",
            "auto_merged",
        ],
    )
    proposal_by_group = {item["group_id"]: item for item in build.proposals}
    screening_rows = [
        {
            "group_id": group["group_id"],
            "screening_priority": group["screening_priority"],
            "priority_reason_codes": group["priority_reason_codes"],
            "proposed_status": proposal_by_group[group["group_id"]][
                "proposed_status"
            ],
            "eligibility_reason_codes": proposal_by_group[group["group_id"]][
                "reason_codes"
            ],
            "title": group["title"],
            "year": group["year"],
            "doi": group["doi"],
            "pmid": group["pmid"],
            "pmcid": group["pmcid"],
            "nct_id": group["nct_id"],
            "providers": group["providers"],
            "query_ids": group["query_ids"],
            "open_access_candidate": group["open_access_candidate"],
            "source_record_ids": group["source_record_ids"],
        }
        for group in build.groups
    ]
    _write_csv(
        files["screening_queue"],
        screening_rows,
        list(screening_rows[0]) if screening_rows else ["group_id"],
    )
    feature_by_group = {item["group_id"]: item for item in build.features}
    llm_rows = [
        {
            "schema_version": LLM_REVIEW_SCHEMA,
            "group_id": proposal["group_id"],
            "title": proposal["title"],
            "year": proposal["year"],
            "abstract": next(
                group.get("abstract")
                for group in build.groups
                if group["group_id"] == proposal["group_id"]
            ),
            "publication_types": next(
                group["publication_types"]
                for group in build.groups
                if group["group_id"] == proposal["group_id"]
            ),
            "query_evaluations": proposal["query_evaluations"],
            "deterministic_features": feature_by_group[proposal["group_id"]],
            "requested_output": {
                "decision": "include|exclude|uncertain",
                "reason_code": "controlled non-empty identifier",
                "rationale": "brief source-bound rationale",
                "evidence_spans": [
                    {
                        "field": "title|abstract",
                        "exact_text": "verbatim span from supplied metadata",
                    }
                ],
            },
            "proposal_only": True,
            "human_confirmation_required": True,
        }
        for proposal in build.proposals
        if proposal["proposed_status"] == "manual_review"
    ]
    _write_jsonl(files["llm_review_queue"], llm_rows)
    _write_csv(
        files["exclusion_audit"],
        build.exclusion_audit,
        [
            "group_id",
            "title",
            "year",
            "sampled_reason_codes",
            "human_audit_decision",
            "human_audit_notes",
            "audit_complete",
        ],
    )
    files["prisma"].write_text(
        json.dumps(build.prisma, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    files["negative_controls"].write_text(
        json.dumps(build.negative_controls, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    files["duplicate_decisions"].write_text(
        json.dumps(build.duplicate_decisions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    files["duplicate_decision_template"].write_text(
        json.dumps(
            {
                "schema_version": DUPLICATE_DECISION_SCHEMA,
                "campaign_id": build.summary["campaign_id"],
                "pair_set_sha256": build.duplicate_decisions.get(
                    "pair_set_sha256",
                    payload_hash(build.possible_duplicates),
                ),
                "decisions": [],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = dict(build.summary)
    summary["artifacts"] = {
        name: {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for name, path in sorted(files.items())
    }
    summary["snapshot_payload_sha256"] = payload_hash(
        {
            "records": build.records,
            "groups": build.groups,
            "features": build.features,
            "proposals": build.proposals,
            "possible_duplicates": build.possible_duplicates,
            "exclusion_audit": build.exclusion_audit,
            "prisma": build.prisma,
            "negative_controls": build.negative_controls,
            "duplicate_decisions": build.duplicate_decisions,
        }
    )
    summary_path = output / "SCREENING_SUMMARY.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
