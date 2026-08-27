"""Deterministic provider-specific query compilation for scientific campaigns.

Human research questions are protocol text, not executable search syntax.  This module
compiles versioned ontology concepts into provider-specific Boolean expressions and embeds
those expressions in a campaign without granting the compiler scientific authority.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .scientific_evidence import ScientificEvidenceError, payload_hash


ONTOLOGY_SCHEMA = "giga.scientific-search-ontology.v1"
SEARCH_SPEC_SCHEMA = "giga.scientific-search-specification.v1"
COMPILED_SCHEMA = "giga.scientific-compiled-search-manifest.v1"
SUPPORTED_PROVIDERS = {
    "clinical_trials",
    "europe_pmc",
    "openalex",
    "pubmed",
    "semantic_scholar",
}
CONCEPT_KINDS = {"conditions", "interventions", "outcomes"}
CONCEPT_ID = re.compile(r"^[a-z][a-z0-9_]{1,79}$")
QUERY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,159}$")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScientificEvidenceError(f"cannot read JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ScientificEvidenceError(f"{path} must contain a JSON object")
    return value


def _text(value: Any, field: str, *, maximum: int = 10_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScientificEvidenceError(f"{field} must be non-empty text")
    result = value.strip()
    if len(result) > maximum:
        raise ScientificEvidenceError(f"{field} exceeds {maximum} characters")
    return result


def _text_list(value: Any, field: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise ScientificEvidenceError(f"{field} must be an array")
    result = [_text(item, f"{field}[]", maximum=500) for item in value]
    if not allow_empty and not result:
        raise ScientificEvidenceError(f"{field} must not be empty")
    if len(result) != len(set(result)):
        raise ScientificEvidenceError(f"{field} must not contain duplicates")
    return result


def _identifier(value: Any, field: str, pattern: re.Pattern[str]) -> str:
    result = _text(value, field, maximum=160)
    if not pattern.fullmatch(result):
        raise ScientificEvidenceError(f"{field} has an invalid identifier")
    return result


def validate_ontology(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "ontology_id",
        "version",
        "conditions",
        "interventions",
        "outcomes",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ScientificEvidenceError(
            "invalid ontology fields; "
            f"missing={sorted(required - set(value))}, "
            f"extra={sorted(set(value) - required)}"
        )
    if value["schema_version"] != ONTOLOGY_SCHEMA:
        raise ScientificEvidenceError(
            f"ontology schema_version must be {ONTOLOGY_SCHEMA}"
        )
    clean: dict[str, Any] = {
        "schema_version": ONTOLOGY_SCHEMA,
        "ontology_id": _identifier(
            value["ontology_id"], "ontology_id", QUERY_ID
        ),
        "version": _text(value["version"], "version", maximum=100),
    }
    all_ids: set[str] = set()
    for kind in sorted(CONCEPT_KINDS):
        raw_concepts = value[kind]
        if not isinstance(raw_concepts, Mapping) or not raw_concepts:
            raise ScientificEvidenceError(f"{kind} must be a non-empty object")
        concepts: dict[str, Any] = {}
        for raw_id, raw_concept in sorted(raw_concepts.items()):
            identifier = _identifier(raw_id, f"{kind} concept", CONCEPT_ID)
            if identifier in all_ids:
                raise ScientificEvidenceError(
                    f"duplicate ontology concept identifier: {identifier}"
                )
            all_ids.add(identifier)
            if not isinstance(raw_concept, Mapping):
                raise ScientificEvidenceError(
                    f"{kind}.{identifier} must be an object"
                )
            allowed = {"terms", "subject_headings"}
            if not set(raw_concept).issubset(allowed) or "terms" not in raw_concept:
                raise ScientificEvidenceError(
                    f"{kind}.{identifier} must contain terms and optional "
                    "subject_headings only"
                )
            concepts[identifier] = {
                "terms": _text_list(
                    raw_concept["terms"], f"{kind}.{identifier}.terms"
                ),
                "subject_headings": _text_list(
                    raw_concept.get("subject_headings", []),
                    f"{kind}.{identifier}.subject_headings",
                    allow_empty=True,
                ),
            }
        clean[kind] = concepts
    return clean


def load_ontology(path: Path) -> dict[str, Any]:
    return validate_ontology(_read_json(path))


def validate_search_spec(
    value: Mapping[str, Any],
    *,
    ontology: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "campaign_id",
        "ontology_id",
        "ontology_version",
        "queries",
        "negative_controls",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ScientificEvidenceError(
            "invalid search specification fields; "
            f"missing={sorted(required - set(value))}, "
            f"extra={sorted(set(value) - required)}"
        )
    if value["schema_version"] != SEARCH_SPEC_SCHEMA:
        raise ScientificEvidenceError(
            f"search specification schema_version must be {SEARCH_SPEC_SCHEMA}"
        )
    if value["ontology_id"] != ontology["ontology_id"]:
        raise ScientificEvidenceError("search specification ontology_id mismatch")
    if value["ontology_version"] != ontology["version"]:
        raise ScientificEvidenceError("search specification ontology_version mismatch")
    known = {
        identifier
        for kind in CONCEPT_KINDS
        for identifier in ontology[kind]
    }

    def clean_query(raw: Any, field: str, *, negative: bool) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ScientificEvidenceError(f"{field} must be an object")
        allowed = {
            "id",
            "condition_concepts",
            "intervention_concepts",
            "outcome_concepts",
            "require_outcomes",
            "extra_required_groups",
            "excluded_terms",
            "expected_disposition",
        }
        if not set(raw).issubset(allowed):
            raise ScientificEvidenceError(
                f"{field} has unsupported fields: {sorted(set(raw) - allowed)}"
            )
        identifier = _identifier(raw.get("id"), f"{field}.id", QUERY_ID)
        concepts: dict[str, list[str]] = {}
        for name, kind in (
            ("condition_concepts", "conditions"),
            ("intervention_concepts", "interventions"),
            ("outcome_concepts", "outcomes"),
        ):
            items = _text_list(
                raw.get(name, []),
                f"{field}.{name}",
                allow_empty=name == "outcome_concepts",
            )
            wrong = [
                item
                for item in items
                if item not in known or item not in ontology[kind]
            ]
            if wrong:
                raise ScientificEvidenceError(
                    f"{field}.{name} has unknown/wrong-kind concepts: {wrong}"
                )
            concepts[name] = items
        groups = raw.get("extra_required_groups", [])
        if not isinstance(groups, list):
            raise ScientificEvidenceError(
                f"{field}.extra_required_groups must be an array"
            )
        clean_groups = [
            _text_list(group, f"{field}.extra_required_groups[]")
            for group in groups
        ]
        expected = raw.get(
            "expected_disposition",
            "auto_exclude" if negative else "eligible_or_manual_review",
        )
        allowed_expected = (
            {"auto_exclude"} if negative else {"eligible_or_manual_review"}
        )
        if expected not in allowed_expected:
            raise ScientificEvidenceError(
                f"{field}.expected_disposition must be one of "
                f"{sorted(allowed_expected)}"
            )
        require_outcomes = raw.get("require_outcomes", False)
        if not isinstance(require_outcomes, bool):
            raise ScientificEvidenceError(
                f"{field}.require_outcomes must be boolean"
            )
        if require_outcomes and not concepts["outcome_concepts"]:
            raise ScientificEvidenceError(
                f"{field} requires outcomes but declares none"
            )
        return {
            "id": identifier,
            **concepts,
            "require_outcomes": require_outcomes,
            "extra_required_groups": clean_groups,
            "excluded_terms": _text_list(
                raw.get("excluded_terms", []),
                f"{field}.excluded_terms",
                allow_empty=True,
            ),
            "expected_disposition": expected,
            "negative_control": negative,
        }

    queries = value["queries"]
    controls = value["negative_controls"]
    if not isinstance(queries, list) or not queries:
        raise ScientificEvidenceError("queries must be a non-empty array")
    if not isinstance(controls, list) or not controls:
        raise ScientificEvidenceError("negative_controls must be a non-empty array")
    clean_queries = [
        clean_query(raw, f"queries[{index}]", negative=False)
        for index, raw in enumerate(queries)
    ]
    clean_controls = [
        clean_query(raw, f"negative_controls[{index}]", negative=True)
        for index, raw in enumerate(controls)
    ]
    identifiers = [
        query["id"] for query in [*clean_queries, *clean_controls]
    ]
    if len(identifiers) != len(set(identifiers)):
        raise ScientificEvidenceError(
            "query and negative-control identifiers must be unique"
        )
    return {
        "schema_version": SEARCH_SPEC_SCHEMA,
        "campaign_id": _identifier(
            value["campaign_id"], "campaign_id", QUERY_ID
        ),
        "ontology_id": ontology["ontology_id"],
        "ontology_version": ontology["version"],
        "queries": clean_queries,
        "negative_controls": clean_controls,
    }


def load_search_spec(
    path: Path,
    *,
    ontology: Mapping[str, Any],
) -> dict[str, Any]:
    return validate_search_spec(_read_json(path), ontology=ontology)


def _escape_phrase(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _pubmed_term(value: str) -> str:
    escaped = _escape_phrase(value)
    return f'"{escaped}"[tiab]'


def _europe_term(value: str) -> str:
    escaped = _escape_phrase(value)
    return f'TITLE_ABS:"{escaped}"'


def _generic_term(value: str) -> str:
    return f'"{_escape_phrase(value)}"'


def _concept_terms(
    ontology: Mapping[str, Any],
    identifiers: Sequence[str],
) -> tuple[list[str], list[str]]:
    terms: list[str] = []
    headings: list[str] = []
    for kind in CONCEPT_KINDS:
        for identifier in identifiers:
            concept = ontology[kind].get(identifier)
            if concept is None:
                continue
            terms.extend(concept["terms"])
            headings.extend(concept["subject_headings"])
    return list(dict.fromkeys(terms)), list(dict.fromkeys(headings))


def _or_group(values: Sequence[str]) -> str:
    if not values:
        raise ScientificEvidenceError("compiled search group must not be empty")
    return "(" + " OR ".join(values) + ")"


def compile_provider_query(
    provider: str,
    query: Mapping[str, Any],
    ontology: Mapping[str, Any],
) -> str:
    if provider not in SUPPORTED_PROVIDERS:
        raise ScientificEvidenceError(f"unsupported search provider: {provider}")
    identifier_groups = [
        query["condition_concepts"],
        query["intervention_concepts"],
    ]
    if query["require_outcomes"]:
        identifier_groups.append(query["outcome_concepts"])
    term_groups: list[tuple[list[str], list[str]]] = [
        _concept_terms(ontology, identifiers)
        for identifiers in identifier_groups
    ]
    term_groups.extend((list(group), []) for group in query["extra_required_groups"])
    excluded = query["excluded_terms"]

    if provider == "pubmed":
        compiled_groups: list[str] = []
        for terms, headings in term_groups:
            values = [
                *(f'"{_escape_phrase(value)}"[Mesh]' for value in headings),
                *(_pubmed_term(value) for value in terms),
            ]
            compiled_groups.append(_or_group(list(dict.fromkeys(values))))
        expression = " AND ".join(compiled_groups)
        if excluded:
            expression += " NOT " + _or_group([_pubmed_term(term) for term in excluded])
        return expression

    if provider == "europe_pmc":
        expression = " AND ".join(
            _or_group([_europe_term(term) for term in terms])
            for terms, _ in term_groups
        )
        if excluded:
            expression += " NOT " + _or_group(
                [_europe_term(term) for term in excluded]
            )
        return expression

    if provider == "clinical_trials":
        # ClinicalTrials.gov query.term accepts Boolean expressions but not PubMed fields.
        expression = " AND ".join(
            _or_group([_generic_term(term) for term in terms])
            for terms, _ in term_groups
        )
        if excluded:
            expression += " NOT " + _or_group(
                [_generic_term(term) for term in excluded]
            )
        return expression

    # Semantic Scholar and OpenAlex support simpler search syntax.  Preserve explicit
    # concept grouping without provider-specific field tags.
    expression = " AND ".join(
        _or_group([_generic_term(term) for term in terms])
        for terms, _ in term_groups
    )
    if excluded:
        expression += " NOT " + _or_group(
            [_generic_term(term) for term in excluded]
        )
    return expression


def compile_manifest(
    campaign: Mapping[str, Any],
    ontology: Mapping[str, Any],
    specification: Mapping[str, Any],
    *,
    providers: Sequence[str],
) -> dict[str, Any]:
    if campaign.get("campaign_id") != specification["campaign_id"]:
        raise ScientificEvidenceError("campaign_id mismatch between campaign and search spec")
    requested = list(providers)
    if not requested or len(requested) != len(set(requested)):
        raise ScientificEvidenceError(
            "providers must be a non-empty list without duplicates"
        )
    unknown = sorted(set(requested) - SUPPORTED_PROVIDERS)
    if unknown:
        raise ScientificEvidenceError(f"unsupported provider(s): {unknown}")
    campaign_queries = campaign.get("queries")
    if not isinstance(campaign_queries, list):
        raise ScientificEvidenceError("campaign queries must be an array")
    questions = {
        query.get("id"): query.get("question")
        for query in campaign_queries
        if isinstance(query, Mapping)
    }
    endpoints = {
        query.get("id"): query.get(
            "endpoint",
            campaign.get("defaults", {}).get("endpoint", "search/papers"),
        )
        for query in campaign_queries
        if isinstance(query, Mapping)
    }
    specs = [*specification["queries"], *specification["negative_controls"]]
    missing = sorted(
        query["id"]
        for query in specification["queries"]
        if query["id"] not in questions
    )
    if missing:
        raise ScientificEvidenceError(
            f"search specification references unknown campaign queries: {missing}"
        )
    compiled: list[dict[str, Any]] = []
    for query in specs:
        applicable = [
            "clinical_trials"
            if endpoints.get(query["id"]) == "search/trials"
            else provider
            for provider in requested
        ]
        applicable = list(dict.fromkeys(applicable))
        provider_queries = {
            provider: compile_provider_query(provider, query, ontology)
            for provider in applicable
        }
        payload = {
            "id": query["id"],
            "human_question": questions.get(query["id"])
            or f"Negative control: {query['id']}",
            "negative_control": query["negative_control"],
            "expected_disposition": query["expected_disposition"],
            "concepts": {
                "conditions": query["condition_concepts"],
                "interventions": query["intervention_concepts"],
                "outcomes": query["outcome_concepts"],
            },
            "require_outcomes": query["require_outcomes"],
            "provider_queries": dict(sorted(provider_queries.items())),
        }
        payload["compiled_query_sha256"] = payload_hash(payload["provider_queries"])
        compiled.append(payload)
    manifest = {
        "schema_version": COMPILED_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "campaign_source_sha256": payload_hash(campaign),
        "ontology": {
            "id": ontology["ontology_id"],
            "version": ontology["version"],
            "sha256": payload_hash(ontology),
        },
        "search_specification_sha256": payload_hash(specification),
        "providers": sorted(set(requested)),
        "queries": compiled,
        "authority": "search_execution_only",
        "scientific_conclusion_authority": False,
    }
    manifest["manifest_sha256"] = payload_hash(manifest)
    return manifest


def embed_compiled_queries(
    campaign: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if campaign.get("campaign_id") != manifest.get("campaign_id"):
        raise ScientificEvidenceError("compiled manifest campaign_id mismatch")
    by_id = {query["id"]: query for query in manifest["queries"]}
    result = json.loads(json.dumps(campaign))
    for query in result["queries"]:
        compiled = by_id.get(query["id"])
        if compiled is None:
            raise ScientificEvidenceError(
                f"compiled manifest is missing campaign query {query['id']}"
            )
        query["provider_queries"] = compiled["provider_queries"]
    result["execution_state"] = "compiled_provider_queries_ready"
    return result
