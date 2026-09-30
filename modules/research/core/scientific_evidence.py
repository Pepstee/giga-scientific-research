"""Scientific evidence acquisition and deterministic readiness gates.

Acquisition providers are not authorities.  This module preserves raw provider responses in
an owner-only, append-only ledger and requires separately recorded human appraisal before any
evidence set can become an A0 action-planning candidate.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


LEDGER_SCHEMA = "giga.scientific-ledger.v1"
APPRAISAL_SCHEMA = "giga.scientific-appraisal.v1"
CANDIDATE_SCHEMA = "giga.scientific-action-candidate.v1"
MAX_CUMULATIVE_RECORDS = 20
DEFAULT_DATABASE = Path.home() / ".giga" / "scientific-evidence.sqlite3"
DEFAULT_ELICIT_BASE_URL = "https://elicit.com/api/v2"
SHA256_PATTERN = frozenset("0123456789abcdef")

PAPER_TYPE_TAGS = {
    "Review",
    "Meta-Analysis",
    "Systematic Review",
    "RCT",
    "Longitudinal",
}
PAPER_CORPORA = {"elicit", "pubmed"}
SEARCH_MODES = {"semantic", "keyword"}
STUDY_DESIGNS = {
    "clinical_guideline",
    "systematic_review",
    "meta_analysis",
    "randomized_controlled_trial",
    "nonrandomized_controlled",
    "longitudinal",
    "cohort",
    "case_control",
    "cross_sectional",
    "case_series",
    "mechanistic_human",
    "animal",
    "in_vitro",
    "expert_opinion",
    "other",
}
RISK_OF_BIAS = {"low", "some_concerns", "high", "critical", "unclear"}
APPLICABILITY = {"direct", "partial", "indirect", "unknown"}
RETRACTION_STATES = {"clear", "corrected", "retracted", "unknown"}
OUTCOME_DIRECTIONS = {"beneficial", "neutral", "harmful", "mixed", "unknown"}
RISK_CLASSES = {"low_risk_lifestyle", "health_information", "clinical"}
ACQUISITION_PROVIDERS = {
    "clinical_trials",
    "crossref",
    "elicit",
    "europe_pmc",
    "local_file",
    "openalex",
    "pubmed",
    "semantic_scholar",
    "unpaywall",
}
ACQUISITION_ENDPOINTS = {
    "import/local-document",
    "lookup/doi",
    "reports",
    "search/papers",
    "search/trials",
    "session",
    "systematic-reviews",
}

APPRAISAL_FIELDS = {
    "schema_version",
    "source_record_id",
    "study_identity",
    "study_design",
    "human_verified",
    "full_text_verified",
    "risk_of_bias",
    "population_applicability",
    "harms_assessed",
    "conflicts_assessed",
    "retraction_status",
    "effect_estimate",
    "confidence_interval",
    "outcome_direction",
    "supporting_quote",
    "source_locator",
    "assessed_by",
    "assessed_at",
    "notes",
}

_DESIGN_TIER = {
    "clinical_guideline": 1,
    "systematic_review": 1,
    "meta_analysis": 1,
    "randomized_controlled_trial": 2,
    "nonrandomized_controlled": 3,
    "longitudinal": 3,
    "cohort": 3,
    "case_control": 3,
    "cross_sectional": 4,
    "case_series": 4,
    "mechanistic_human": 4,
    "animal": 5,
    "in_vitro": 5,
    "expert_opinion": 5,
    "other": 5,
}


class ScientificEvidenceError(ValueError):
    """Base error for invalid evidence or ledger operations."""


class ElicitConfigurationError(ScientificEvidenceError):
    """Raised when Elicit credentials or client settings are unavailable."""


class ElicitAPIError(ScientificEvidenceError):
    """A redacted Elicit API failure safe to surface to callers."""

    def __init__(self, status: int | None, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message
        prefix = f"Elicit API {status}" if status is not None else "Elicit connection"
        super().__init__(f"{prefix} [{code}]: {message}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _timestamp(value: str | None, *, field: str) -> str:
    candidate = value or utc_now()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ScientificEvidenceError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ScientificEvidenceError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _text(value: Any, field: str, *, maximum: int = 50_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScientificEvidenceError(f"{field} must be non-empty text")
    result = value.strip()
    if len(result) > maximum:
        raise ScientificEvidenceError(f"{field} exceeds {maximum} characters")
    return result


def _optional_text(value: Any, field: str, *, maximum: int = 50_000) -> str | None:
    if value is None:
        return None
    return _text(value, field, maximum=maximum)


def _integer(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ScientificEvidenceError(
            f"{field} must be an integer from {minimum} to {maximum}"
        )
    return value


def _enum(value: Any, field: str, allowed: set[str]) -> str:
    if value not in allowed:
        raise ScientificEvidenceError(f"{field} must be one of {sorted(allowed)}")
    return str(value)


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ScientificEvidenceError(f"{field} must be an array")
    result = [_text(item, f"{field}[]", maximum=10_000) for item in value]
    if len(result) != len(set(result)):
        raise ScientificEvidenceError(f"{field} must not contain duplicates")
    return result


def _sha256(value: str, field: str) -> str:
    if len(value) != 64 or any(character not in SHA256_PATTERN for character in value):
        raise ScientificEvidenceError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _reject_secret_material(value: Any, *, path: str = "request") -> None:
    """Reject credential-shaped request fields before immutable persistence."""
    sensitive_names = {
        "authorization",
        "api_key",
        "apikey",
        "access_token",
        "bearer_token",
        "password",
        "secret",
    }
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalised = str(key).strip().lower().replace("-", "_")
            if normalised in sensitive_names:
                raise ScientificEvidenceError(
                    f"{path} contains forbidden credential field {key!r}"
                )
            _reject_secret_material(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_secret_material(nested, path=f"{path}[{index}]")


Transport = Callable[
    [str, str, Mapping[str, Any] | None, Mapping[str, str], int],
    Mapping[str, Any],
]


def _default_transport(
    method: str,
    url: str,
    body: Mapping[str, Any] | None,
    headers: Mapping[str, str],
    timeout: int,
) -> Mapping[str, Any]:
    data = canonical_json(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url, data=data, method=method, headers=dict(headers)
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        try:
            parsed = json.loads(raw)
            error = parsed.get("error", {}) if isinstance(parsed, dict) else {}
            code = str(error.get("code") or f"http_{exc.code}")
            message = str(error.get("message") or "request failed")
        except (json.JSONDecodeError, AttributeError):
            code, message = f"http_{exc.code}", "request failed"
        raise ElicitAPIError(exc.code, code, message[:1_000]) from exc
    except urllib.error.URLError as exc:
        raise ElicitAPIError(None, "connection_error", str(exc.reason)[:1_000]) from exc
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ElicitAPIError(
            None, "invalid_json", "provider returned invalid JSON"
        ) from exc
    if not isinstance(result, dict):
        raise ElicitAPIError(
            None, "invalid_response", "provider response must be an object"
        )
    return result


class ElicitClient:
    """Minimal v2 Elicit client with strict requests and injectable offline transport."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_ELICIT_BASE_URL,
        timeout: int = 60,
        transport: Transport | None = None,
    ):
        key = api_key if api_key is not None else os.environ.get("ELICIT_API_KEY")
        if not isinstance(key, str) or not key.strip():
            raise ElicitConfigurationError(
                "ELICIT_API_KEY is not configured; create one at https://elicit.com/settings"
            )
        parsed_base = urllib.parse.urlsplit(base_url)
        if (
            parsed_base.scheme != "https"
            or parsed_base.hostname != "elicit.com"
            or parsed_base.port not in {None, 443}
            or parsed_base.path.rstrip("/") != "/api/v2"
            or parsed_base.query
            or parsed_base.fragment
        ):
            raise ElicitConfigurationError(
                "Elicit base URL must be the official https://elicit.com/api/v2 endpoint"
            )
        self._api_key = key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = _integer(timeout, "timeout", minimum=1, maximum=1_200)
        self._transport = transport or _default_transport

    def _request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if not path.startswith("/") or "://" in path:
            raise ScientificEvidenceError("Elicit path must be a relative API path")
        url = self.base_url + path
        if query:
            pairs = {key: value for key, value in query.items() if value is not None}
            if pairs:
                url += "?" + urllib.parse.urlencode(pairs)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "giga-scientific-evidence/1.0",
        }
        try:
            result = self._transport(method, url, body, headers, self.timeout)
        except ElicitAPIError as exc:
            redacted = exc.message.replace(self._api_key, "[REDACTED]")
            raise ElicitAPIError(exc.status, exc.code, redacted) from exc
        if not isinstance(result, Mapping):
            raise ElicitAPIError(
                None, "invalid_response", "provider response must be an object"
            )
        return dict(result)

    @staticmethod
    def paper_search_request(
        query: str,
        *,
        max_results: int = 10,
        corpus: str = "elicit",
        search_mode: str = "semantic",
        filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        research_query = _text(query, "query", maximum=10_000)
        maximum = _integer(max_results, "max_results", minimum=1, maximum=10_000)
        corpus_value = _enum(corpus, "corpus", PAPER_CORPORA)
        mode = _enum(search_mode, "search_mode", SEARCH_MODES)
        clean_filters = _paper_filters(filters)
        if mode == "keyword" and clean_filters:
            raise ScientificEvidenceError(
                "keyword search cannot be combined with filters"
            )
        request: dict[str, Any] = {
            "query": research_query,
            "maxResults": maximum,
            "corpus": corpus_value,
            "searchMode": mode,
        }
        if clean_filters:
            request["filters"] = clean_filters
        return request

    def search_papers(
        self,
        query: str,
        *,
        max_results: int = 10,
        corpus: str = "elicit",
        search_mode: str = "semantic",
        filters: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        request = self.paper_search_request(
            query,
            max_results=max_results,
            corpus=corpus,
            search_mode=search_mode,
            filters=filters,
        )
        response = self._request("POST", "/search/papers", request)
        _validate_search_response(response, "papers")
        return request, response

    @staticmethod
    def trial_search_request(
        query: str,
        *,
        max_results: int = 10,
        trial_filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "query": _text(query, "query", maximum=10_000),
            "maxResults": _integer(
                max_results, "max_results", minimum=1, maximum=10_000
            ),
        }
        if trial_filters:
            if not isinstance(trial_filters, Mapping):
                raise ScientificEvidenceError("trial_filters must be an object")
            request["trialFilters"] = dict(trial_filters)
        return request

    def search_trials(
        self,
        query: str,
        *,
        max_results: int = 10,
        trial_filters: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        request = self.trial_search_request(
            query, max_results=max_results, trial_filters=trial_filters
        )
        response = self._request("POST", "/search/trials", request)
        _validate_search_response(response, "trials")
        return request, response

    def create_report(
        self,
        research_question: str,
        *,
        max_search_papers: int = 50,
        max_extract_papers: int = 10,
        is_public: bool = False,
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        if is_public is not False:
            raise ScientificEvidenceError("Elicit reports must remain private")
        request = {
            "researchQuestion": _text(
                research_question, "research_question", maximum=20_000
            ),
            "maxSearchPapers": _integer(
                max_search_papers, "max_search_papers", minimum=1, maximum=10_000
            ),
            "maxExtractPapers": _integer(
                max_extract_papers, "max_extract_papers", minimum=1, maximum=10_000
            ),
            "isPublic": is_public,
        }
        response = self._request("POST", "/sessions/reports", request)
        _validate_session_response(response)
        return request, response

    def create_systematic_review(
        self, payload: Mapping[str, Any]
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        if not isinstance(payload, Mapping):
            raise ScientificEvidenceError("systematic review payload must be an object")
        request = dict(payload)
        request["researchQuestion"] = _text(
            request.get("researchQuestion"), "researchQuestion", maximum=20_000
        )
        if request.get("isPublic", False) is not False:
            raise ScientificEvidenceError("systematic reviews must default to private")
        request["isPublic"] = False
        response = self._request("POST", "/sessions/systematic-reviews", request)
        _validate_session_response(response)
        return request, response

    def get_session(
        self, session_id: str, *, systematic: bool = False
    ) -> Mapping[str, Any]:
        identifier = _text(session_id, "session_id", maximum=500)
        kind = "systematic-reviews" if systematic else "reports"
        return self._request(
            "GET", f"/sessions/{kind}/{urllib.parse.quote(identifier, safe='')}"
        )

    def resume_session(self, session_id: str) -> Mapping[str, Any]:
        identifier = _text(session_id, "session_id", maximum=500)
        return self._request(
            "POST", f"/sessions/{urllib.parse.quote(identifier, safe='')}/resume"
        )


def _paper_filters(filters: Mapping[str, Any] | None) -> dict[str, Any]:
    if filters is None:
        return {"retracted": "exclude_retracted"}
    if not isinstance(filters, Mapping):
        raise ScientificEvidenceError("filters must be an object")
    allowed = {
        "excludeKeywords",
        "hasPdf",
        "includeKeywords",
        "maxQuartile",
        "maxYear",
        "minYear",
        "pubmedOnly",
        "retracted",
        "typeTags",
    }
    if not set(filters).issubset(allowed):
        raise ScientificEvidenceError(
            f"unsupported paper filter(s): {sorted(set(filters) - allowed)}"
        )
    result = dict(filters)
    result.setdefault("retracted", "exclude_retracted")
    if result["retracted"] not in {
        "exclude_retracted",
        "include_retracted",
        "only_retracted",
    }:
        raise ScientificEvidenceError("invalid retracted filter")
    if "typeTags" in result:
        tags = _string_list(result["typeTags"], "typeTags")
        unknown = set(tags) - PAPER_TYPE_TAGS
        if unknown:
            raise ScientificEvidenceError(f"unsupported typeTags: {sorted(unknown)}")
        result["typeTags"] = tags
    for name in ("minYear", "maxYear"):
        if name in result:
            result[name] = _integer(result[name], name, minimum=1600, maximum=2200)
    if "maxQuartile" in result:
        result["maxQuartile"] = _integer(
            result["maxQuartile"], "maxQuartile", minimum=1, maximum=4
        )
    for name in ("hasPdf", "pubmedOnly"):
        if name in result and not isinstance(result[name], bool):
            raise ScientificEvidenceError(f"{name} must be boolean")
    for name in ("includeKeywords", "excludeKeywords"):
        if name in result:
            result[name] = _string_list(result[name], name)
    return result


def _validate_search_response(response: Mapping[str, Any], field: str) -> None:
    records = response.get(field)
    if not isinstance(records, list):
        raise ElicitAPIError(
            None, "invalid_response", f"response must contain {field} array"
        )
    for record in records:
        if not isinstance(record, Mapping):
            raise ElicitAPIError(
                None, "invalid_response", f"{field} entries must be objects"
            )
        _text(record.get("title"), f"{field}.title", maximum=20_000)


def _validate_session_response(response: Mapping[str, Any]) -> None:
    _text(response.get("sessionId"), "sessionId", maximum=500)
    status = response.get("status")
    if status is not None and status not in {
        "processing",
        "completed",
        "failed",
        "pausedForInsufficientQuota",
    }:
        raise ElicitAPIError(None, "invalid_response", "unknown Elicit session status")


def _source_identity(kind: str, record: Mapping[str, Any]) -> str:
    for field, prefix in (("doi", "doi"), ("pmid", "pmid"), ("nctId", "nct")):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return f"{prefix}:{value.strip().lower()}"
    return f"{kind}:fallback:{payload_hash({'title': record.get('title'), 'year': record.get('year')})}"


def _normalise_year(value: Any) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1600 <= value <= 2200
    ):
        raise ScientificEvidenceError(
            "source year must be an integer from 1600 to 2200"
        )
    return value


class ScientificEvidenceStore:
    """Owner-only append-only scientific acquisition and appraisal ledger."""

    def __init__(self, database: Path = DEFAULT_DATABASE):
        self.database = database.expanduser().resolve()
        parent_existed = self.database.parent.exists()
        self.database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if (
            not parent_existed
            or self.database.parent == DEFAULT_DATABASE.parent.resolve()
        ):
            os.chmod(self.database.parent, 0o700)
        self._initialise()
        os.chmod(self.database, 0o600)

    def _acquisition_database_paths(
        self, aggregate_databases: Sequence[Path] = ()
    ) -> tuple[Path, ...]:
        paths: list[Path] = []
        seen: set[Path] = set()
        for value in (self.database, *aggregate_databases):
            try:
                path = Path(value).expanduser().resolve(strict=True)
            except OSError as exc:
                raise ScientificEvidenceError(
                    f"aggregate acquisition ledger is unavailable: {value}"
                ) from exc
            if not path.is_file():
                raise ScientificEvidenceError(
                    f"aggregate acquisition ledger is not a file: {path}"
                )
            if path not in seen:
                seen.add(path)
                paths.append(path)
        return tuple(paths)

    def _acquisition_state(
        self, aggregate_databases: Sequence[Path] = ()
    ) -> tuple[int, set[str]]:
        source_records = 0
        identities: set[str] = set()
        for path in self._acquisition_database_paths(aggregate_databases):
            try:
                uri = f"{path.as_uri()}?mode=ro"
                with sqlite3.connect(uri, uri=True, timeout=5) as connection:
                    rows = connection.execute(
                        "SELECT identity_key FROM source_records"
                    ).fetchall()
            except sqlite3.Error as exc:
                raise ScientificEvidenceError(
                    f"cannot read aggregate acquisition ledger: {path}"
                ) from exc
            source_records += len(rows)
            identities.update(row[0] for row in rows)
        return source_records, identities

    def acquisition_budget(
        self, aggregate_databases: Sequence[Path] = ()
    ) -> dict[str, int]:
        source_records, identities = self._acquisition_state(aggregate_databases)
        return {
            "record_limit": MAX_CUMULATIVE_RECORDS,
            "source_records": source_records,
            "unique_identities": len(identities),
            "remaining_records": max(0, MAX_CUMULATIVE_RECORDS - source_records),
        }

    def require_acquisition_capacity(
        self, aggregate_databases: Sequence[Path] = ()
    ) -> dict[str, int]:
        budget = self.acquisition_budget(aggregate_databases)
        if (
            budget["source_records"] > MAX_CUMULATIVE_RECORDS
            or budget["unique_identities"] > MAX_CUMULATIVE_RECORDS
        ):
            raise ScientificEvidenceError(
                "cumulative acquisition record cap "
                f"{MAX_CUMULATIVE_RECORDS} exceeded: existing aggregate has "
                f"{budget['source_records']} source records / "
                f"{budget['unique_identities']} unique identities "
                f"({max(0, budget['source_records'] - MAX_CUMULATIVE_RECORDS)} "
                "records and "
                f"{max(0, budget['unique_identities'] - MAX_CUMULATIVE_RECORDS)} "
                "identities over the cap); further acquisition refused"
            )
        return budget

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    schema_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS acquisitions (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    response_sha256 TEXT NOT NULL,
                    source_count INTEGER NOT NULL CHECK(source_count >= 0),
                    observed_at TEXT NOT NULL,
                    UNIQUE(endpoint, request_sha256, response_sha256)
                );
                CREATE TABLE IF NOT EXISTS source_records (
                    id TEXT PRIMARY KEY,
                    acquisition_id TEXT NOT NULL REFERENCES acquisitions(id),
                    rank INTEGER NOT NULL CHECK(rank > 0),
                    kind TEXT NOT NULL CHECK(kind IN ('paper','trial')),
                    identity_key TEXT NOT NULL,
                    title TEXT NOT NULL,
                    year INTEGER,
                    doi TEXT,
                    pmid TEXT,
                    nct_id TEXT,
                    raw_json TEXT NOT NULL,
                    raw_sha256 TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    UNIQUE(acquisition_id, rank)
                );
                CREATE INDEX IF NOT EXISTS idx_source_identity
                    ON source_records(identity_key, retrieved_at);
                CREATE TABLE IF NOT EXISTS appraisals (
                    id TEXT PRIMARY KEY,
                    source_record_id TEXT NOT NULL REFERENCES source_records(id),
                    appraisal_json TEXT NOT NULL,
                    appraisal_sha256 TEXT NOT NULL,
                    evidence_tier TEXT NOT NULL CHECK(evidence_tier IN
                        ('T1','T2','T3','T4','T5','U')),
                    recorded_at TEXT NOT NULL,
                    UNIQUE(source_record_id, appraisal_sha256)
                );
                CREATE TABLE IF NOT EXISTS action_candidates (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    intervention TEXT NOT NULL,
                    risk_class TEXT NOT NULL CHECK(risk_class IN
                        ('low_risk_lifestyle','health_information','clinical')),
                    source_record_ids_json TEXT NOT NULL,
                    readiness_json TEXT NOT NULL,
                    authority_level TEXT NOT NULL CHECK(authority_level = 'A0'),
                    recommendation_authority INTEGER NOT NULL
                        CHECK(recommendation_authority = 0),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ledger_entries (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    previous_entry_sha256 TEXT,
                    entry_sha256 TEXT NOT NULL UNIQUE
                );
                """
            )
            current = connection.execute(
                "SELECT schema_version FROM schema_metadata WHERE singleton = 1"
            ).fetchone()
            if current is None:
                connection.execute(
                    "INSERT INTO schema_metadata VALUES (1, ?, ?)",
                    (LEDGER_SCHEMA, utc_now()),
                )
            elif current["schema_version"] != LEDGER_SCHEMA:
                raise ScientificEvidenceError(
                    f"unsupported scientific ledger {current['schema_version']}; "
                    f"expected {LEDGER_SCHEMA}"
                )
            for table in (
                "acquisitions",
                "source_records",
                "appraisals",
                "action_candidates",
                "ledger_entries",
            ):
                connection.execute(
                    f"""CREATE TRIGGER IF NOT EXISTS no_update_{table}
                        BEFORE UPDATE ON {table}
                        BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END"""
                )
                connection.execute(
                    f"""CREATE TRIGGER IF NOT EXISTS no_delete_{table}
                        BEFORE DELETE ON {table}
                        BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END"""
                )

    @staticmethod
    def _append_ledger(
        connection: sqlite3.Connection,
        *,
        kind: str,
        payload: Mapping[str, Any],
        recorded_at: str,
    ) -> str:
        raw = canonical_json(payload)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        previous = connection.execute(
            "SELECT entry_sha256 FROM ledger_entries ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous_digest = previous["entry_sha256"] if previous else None
        entry_id = str(uuid.uuid4())
        material = {
            "id": entry_id,
            "kind": kind,
            "recorded_at": recorded_at,
            "payload_sha256": digest,
            "previous_entry_sha256": previous_digest,
        }
        entry_digest = payload_hash(material)
        connection.execute(
            """INSERT INTO ledger_entries
               (id,kind,recorded_at,payload_json,payload_sha256,
                previous_entry_sha256,entry_sha256)
               VALUES (?,?,?,?,?,?,?)""",
            (entry_id, kind, recorded_at, raw, digest, previous_digest, entry_digest),
        )
        return entry_digest

    def record_acquisition(
        self,
        *,
        provider: str = "elicit",
        endpoint: str,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
        observed_at: str | None = None,
        aggregate_databases: Sequence[Path] = (),
        max_cumulative_records: int = MAX_CUMULATIVE_RECORDS,
    ) -> tuple[str, bool, list[str]]:
        limit = _integer(
            max_cumulative_records,
            "max_cumulative_records",
            minimum=0,
            maximum=MAX_CUMULATIVE_RECORDS,
        )
        provider_value = _enum(
            provider, "provider", ACQUISITION_PROVIDERS
        )
        endpoint_value = _enum(endpoint, "endpoint", ACQUISITION_ENDPOINTS)
        if not isinstance(request, Mapping) or not isinstance(response, Mapping):
            raise ScientificEvidenceError("request and response must be objects")
        _reject_secret_material(request)
        if endpoint_value in {
            "import/local-document",
            "lookup/doi",
            "search/papers",
        }:
            _validate_search_response(response, "papers")
            records, kind = response["papers"], "paper"
        elif endpoint_value == "search/trials":
            _validate_search_response(response, "trials")
            records, kind = response["trials"], "trial"
        else:
            _validate_session_response(response)
            records, kind = [], "paper"
        observed = _timestamp(observed_at, field="observed_at")
        request_data, response_data = dict(request), dict(response)
        request_digest, response_digest = (
            payload_hash(request_data),
            payload_hash(response_data),
        )
        acquisition_id = payload_hash(
            {
                "provider": provider_value,
                "endpoint": endpoint_value,
                "request_sha256": request_digest,
                "response_sha256": response_digest,
            }
        )
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM acquisitions WHERE id = ?", (acquisition_id,)
            ).fetchone()
            if existing:
                source_ids = [
                    row["id"]
                    for row in connection.execute(
                        "SELECT id FROM source_records WHERE acquisition_id = ? ORDER BY rank",
                        (acquisition_id,),
                    )
                ]
                return acquisition_id, False, source_ids
            existing_records, existing_identities = self._acquisition_state(
                aggregate_databases
            )
            added_identities = {
                _source_identity(kind, record) for record in records
            } - existing_identities
            projected_records = existing_records + len(records)
            projected_identities = len(existing_identities) + len(added_identities)
            if records and (
                projected_records > limit or projected_identities > limit
            ):
                raise ScientificEvidenceError(
                    f"cumulative acquisition record cap {limit} would be exceeded: "
                    f"existing aggregate has {existing_records} source records / "
                    f"{len(existing_identities)} unique identities; response adds "
                    f"{len(records)} records / {len(added_identities)} new identities, "
                    f"projecting {projected_records} records / "
                    f"{projected_identities} identities; acquisition not stored"
                )
            connection.execute(
                """INSERT INTO acquisitions
                   (id,provider,endpoint,request_json,request_sha256,response_json,
                    response_sha256,source_count,observed_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    acquisition_id,
                    provider_value,
                    endpoint_value,
                    canonical_json(request_data),
                    request_digest,
                    canonical_json(response_data),
                    response_digest,
                    len(records),
                    observed,
                ),
            )
            source_ids: list[str] = []
            for rank, raw_record in enumerate(records, 1):
                record = dict(raw_record)
                raw_digest = payload_hash(record)
                source_id = payload_hash(
                    {
                        "acquisition_id": acquisition_id,
                        "rank": rank,
                        "raw_sha256": raw_digest,
                    }
                )
                source_ids.append(source_id)
                connection.execute(
                    """INSERT INTO source_records
                       (id,acquisition_id,rank,kind,identity_key,title,year,doi,pmid,nct_id,
                        raw_json,raw_sha256,retrieved_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        source_id,
                        acquisition_id,
                        rank,
                        kind,
                        _source_identity(kind, record),
                        _text(record.get("title"), "source.title", maximum=20_000),
                        _normalise_year(record.get("year"))
                        if kind == "paper"
                        else None,
                        _optional_text(record.get("doi"), "source.doi", maximum=1_000),
                        _optional_text(
                            record.get("pmid"), "source.pmid", maximum=1_000
                        ),
                        _optional_text(
                            record.get("nctId"), "source.nctId", maximum=1_000
                        ),
                        canonical_json(record),
                        raw_digest,
                        observed,
                    ),
                )
            self._append_ledger(
                connection,
                kind=f"{provider_value}.acquisition.recorded",
                recorded_at=observed,
                payload={
                    "acquisition_id": acquisition_id,
                    "provider": provider_value,
                    "endpoint": endpoint_value,
                    "request_sha256": request_digest,
                    "response_sha256": response_digest,
                    "source_record_ids": source_ids,
                    "authority": "third_party_claim",
                },
            )
        return acquisition_id, True, source_ids

    def get_source(self, source_record_id: str) -> dict[str, Any]:
        identifier = _sha256(source_record_id, "source_record_id")
        with self.connect() as connection:
            row = connection.execute(
                """SELECT source_records.*, acquisitions.provider
                   FROM source_records
                   JOIN acquisitions ON acquisitions.id = source_records.acquisition_id
                   WHERE source_records.id = ?""",
                (identifier,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown source record {identifier}")
        result = dict(row)
        result["raw"] = json.loads(result.pop("raw_json"))
        return result

    def acquisitions_for_request(
        self,
        *,
        endpoint: str,
        request: Mapping[str, Any],
        provider: str = "elicit",
        aggregate_databases: Sequence[Path] = (),
        match_max_results: bool = True,
    ) -> list[str]:
        provider_value = _enum(provider, "provider", ACQUISITION_PROVIDERS)
        endpoint_value = _enum(endpoint, "endpoint", ACQUISITION_ENDPOINTS)
        if not isinstance(request, Mapping):
            raise ScientificEvidenceError("request must be an object")
        _reject_secret_material(request)
        digest = payload_hash(dict(request))
        comparable_request = dict(request)
        comparable_request.pop("maxResults", None)
        acquisition_ids: list[str] = []
        for path in self._acquisition_database_paths(aggregate_databases):
            try:
                uri = f"{path.as_uri()}?mode=ro"
                with sqlite3.connect(uri, uri=True, timeout=5) as connection:
                    if match_max_results:
                        rows = connection.execute(
                            """SELECT id FROM acquisitions
                               WHERE provider = ? AND endpoint = ? AND request_sha256 = ?
                               ORDER BY observed_at, id""",
                            (provider_value, endpoint_value, digest),
                        ).fetchall()
                        acquisition_ids.extend(row[0] for row in rows)
                    else:
                        rows = connection.execute(
                            """SELECT id, request_json FROM acquisitions
                               WHERE provider = ? AND endpoint = ?
                               ORDER BY observed_at, id""",
                            (provider_value, endpoint_value),
                        ).fetchall()
                        for row in rows:
                            try:
                                previous_request = json.loads(row[1])
                            except (json.JSONDecodeError, TypeError):
                                continue
                            previous_request.pop("maxResults", None)
                            if previous_request == comparable_request:
                                acquisition_ids.append(row[0])
            except sqlite3.Error as exc:
                raise ScientificEvidenceError(
                    f"cannot read aggregate acquisition ledger: {path}"
                ) from exc
        return list(dict.fromkeys(acquisition_ids))

    def list_sources(self, *, limit: int = 100) -> list[dict[str, Any]]:
        maximum = _integer(limit, "limit", minimum=1, maximum=10_000)
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT source_records.id,source_records.kind,
                          source_records.identity_key,source_records.title,
                          source_records.year,source_records.doi,source_records.pmid,
                          source_records.nct_id,source_records.retrieved_at,
                          acquisitions.provider
                   FROM source_records
                   JOIN acquisitions ON acquisitions.id = source_records.acquisition_id
                   ORDER BY source_records.retrieved_at DESC, source_records.rank LIMIT ?""",
                (maximum,),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_appraisal(self, appraisal: Mapping[str, Any]) -> tuple[str, bool, str]:
        clean = validate_appraisal(appraisal)
        source_id = clean["source_record_id"]
        recorded = clean["assessed_at"]
        digest = payload_hash(clean)
        appraisal_id = payload_hash(
            {"source_record_id": source_id, "appraisal_sha256": digest}
        )
        tier = evidence_tier(clean["study_design"], clean["risk_of_bias"])
        with self.connect() as connection:
            if not connection.execute(
                "SELECT 1 FROM source_records WHERE id = ?", (source_id,)
            ).fetchone():
                raise KeyError(f"unknown source record {source_id}")
            existing = connection.execute(
                "SELECT id FROM appraisals WHERE id = ?", (appraisal_id,)
            ).fetchone()
            if existing:
                return appraisal_id, False, tier
            connection.execute(
                """INSERT INTO appraisals
                   (id,source_record_id,appraisal_json,appraisal_sha256,
                    evidence_tier,recorded_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    appraisal_id,
                    source_id,
                    canonical_json(clean),
                    digest,
                    tier,
                    recorded,
                ),
            )
            self._append_ledger(
                connection,
                kind="scientific.appraisal.recorded",
                recorded_at=recorded,
                payload={
                    "appraisal_id": appraisal_id,
                    "source_record_id": source_id,
                    "appraisal_sha256": digest,
                    "evidence_tier": tier,
                    "human_verified": clean["human_verified"],
                },
            )
        return appraisal_id, True, tier

    def latest_appraisal(self, source_record_id: str) -> dict[str, Any] | None:
        identifier = _sha256(source_record_id, "source_record_id")
        with self.connect() as connection:
            row = connection.execute(
                """SELECT appraisal_json,evidence_tier,id
                   FROM appraisals WHERE source_record_id = ?
                   ORDER BY recorded_at DESC, rowid DESC LIMIT 1""",
                (identifier,),
            ).fetchone()
        if row is None:
            return None
        result = json.loads(row["appraisal_json"])
        result["evidence_tier"] = row["evidence_tier"]
        result["appraisal_id"] = row["id"]
        return result

    def create_action_candidate(
        self,
        *,
        question: str,
        intervention: str,
        risk_class: str,
        source_record_ids: Sequence[str],
        contradictions_addressed: bool = False,
        clinician_reviewed: bool = False,
        created_at: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        question_value = _text(question, "question", maximum=20_000)
        intervention_value = _text(intervention, "intervention", maximum=20_000)
        risk = _enum(risk_class, "risk_class", RISK_CLASSES)
        if not isinstance(contradictions_addressed, bool) or not isinstance(
            clinician_reviewed, bool
        ):
            raise ScientificEvidenceError(
                "contradictions_addressed and clinician_reviewed must be boolean"
            )
        source_ids = [
            _sha256(value, "source_record_ids[]") for value in source_record_ids
        ]
        if not source_ids or len(source_ids) != len(set(source_ids)):
            raise ScientificEvidenceError(
                "source_record_ids must be a non-empty list of unique source records"
            )
        appraisals: list[dict[str, Any] | None] = [
            self.latest_appraisal(identifier) for identifier in source_ids
        ]
        readiness = evaluate_readiness(
            appraisals,
            risk_class=risk,
            contradictions_addressed=contradictions_addressed,
            clinician_reviewed=clinician_reviewed,
        )
        created = _timestamp(created_at, field="created_at")
        payload = {
            "schema_version": CANDIDATE_SCHEMA,
            "question": question_value,
            "intervention": intervention_value,
            "risk_class": risk,
            "source_record_ids": source_ids,
            "readiness": readiness,
            "authority_level": "A0",
            "recommendation_authority": False,
            "created_at": created,
        }
        candidate_id = payload_hash(payload)
        with self.connect() as connection:
            missing = [
                identifier
                for identifier in source_ids
                if not connection.execute(
                    "SELECT 1 FROM source_records WHERE id = ?", (identifier,)
                ).fetchone()
            ]
            if missing:
                raise KeyError(f"unknown source record(s): {', '.join(missing)}")
            connection.execute(
                """INSERT OR IGNORE INTO action_candidates
                   (id,question,intervention,risk_class,source_record_ids_json,
                    readiness_json,authority_level,recommendation_authority,created_at)
                   VALUES (?,?,?,?,?,?,'A0',0,?)""",
                (
                    candidate_id,
                    question_value,
                    intervention_value,
                    risk,
                    canonical_json(source_ids),
                    canonical_json(readiness),
                    created,
                ),
            )
            if connection.execute("SELECT changes()").fetchone()[0]:
                self._append_ledger(
                    connection,
                    kind="scientific.action-candidate.recorded",
                    recorded_at=created,
                    payload={
                        "candidate_id": candidate_id,
                        "status": readiness["status"],
                        "authority_level": "A0",
                        "recommendation_authority": False,
                    },
                )
        return candidate_id, payload

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            return {
                table: int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
                for table in (
                    "acquisitions",
                    "source_records",
                    "appraisals",
                    "action_candidates",
                    "ledger_entries",
                )
            }

    def audit(self) -> dict[str, Any]:
        errors: list[str] = []
        previous_digest: str | None = None
        with self.connect() as connection:
            metadata = connection.execute(
                "SELECT schema_version FROM schema_metadata WHERE singleton = 1"
            ).fetchone()
            if metadata is None or metadata["schema_version"] != LEDGER_SCHEMA:
                errors.append("scientific ledger schema metadata mismatch")
            protected_tables = (
                "acquisitions",
                "source_records",
                "appraisals",
                "action_candidates",
                "ledger_entries",
            )
            expected_triggers = {
                f"no_{operation}_{table}"
                for table in protected_tables
                for operation in ("update", "delete")
            }
            actual_triggers = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                )
            }
            missing_triggers = sorted(expected_triggers - actual_triggers)
            if missing_triggers:
                errors.append(
                    "append-only trigger(s) missing: " + ", ".join(missing_triggers)
                )
            for row in connection.execute(
                "SELECT * FROM ledger_entries ORDER BY sequence"
            ):
                try:
                    payload = json.loads(row["payload_json"])
                except (json.JSONDecodeError, TypeError):
                    errors.append(f"ledger sequence {row['sequence']} has invalid JSON")
                    payload = None
                if (
                    payload is not None
                    and payload_hash(payload) != row["payload_sha256"]
                ):
                    errors.append(
                        f"ledger sequence {row['sequence']} payload hash mismatch"
                    )
                material = {
                    "id": row["id"],
                    "kind": row["kind"],
                    "recorded_at": row["recorded_at"],
                    "payload_sha256": row["payload_sha256"],
                    "previous_entry_sha256": row["previous_entry_sha256"],
                }
                if row["previous_entry_sha256"] != previous_digest:
                    errors.append(f"ledger sequence {row['sequence']} chain mismatch")
                if payload_hash(material) != row["entry_sha256"]:
                    errors.append(
                        f"ledger sequence {row['sequence']} entry hash mismatch"
                    )
                previous_digest = row["entry_sha256"]
            for row in connection.execute("SELECT * FROM acquisitions"):
                decoded: dict[str, Any] = {}
                for value_field, digest_field in (
                    ("request_json", "request_sha256"),
                    ("response_json", "response_sha256"),
                ):
                    try:
                        value = json.loads(row[value_field])
                    except (json.JSONDecodeError, TypeError):
                        errors.append(
                            f"acquisition {row['id']} has invalid {value_field}"
                        )
                        continue
                    decoded[value_field] = value
                    if payload_hash(value) != row[digest_field]:
                        errors.append(
                            f"acquisition {row['id']} {digest_field} mismatch"
                        )
                if "request_json" in decoded:
                    try:
                        _reject_secret_material(decoded["request_json"])
                    except ScientificEvidenceError:
                        errors.append(
                            f"acquisition {row['id']} contains credential material"
                        )
                expected_acquisition_id = payload_hash(
                    {
                        "provider": row["provider"],
                        "endpoint": row["endpoint"],
                        "request_sha256": row["request_sha256"],
                        "response_sha256": row["response_sha256"],
                    }
                )
                if expected_acquisition_id != row["id"]:
                    errors.append(f"acquisition {row['id']} identity mismatch")
                source_count = connection.execute(
                    "SELECT COUNT(*) FROM source_records WHERE acquisition_id = ?",
                    (row["id"],),
                ).fetchone()[0]
                if source_count != row["source_count"]:
                    errors.append(f"acquisition {row['id']} source count mismatch")
            for row in connection.execute("SELECT * FROM source_records"):
                try:
                    value = json.loads(row["raw_json"])
                except (json.JSONDecodeError, TypeError):
                    errors.append(f"source {row['id']} has invalid raw JSON")
                    continue
                raw_digest = payload_hash(value)
                if raw_digest != row["raw_sha256"]:
                    errors.append(f"source {row['id']} raw hash mismatch")
                expected_source_id = payload_hash(
                    {
                        "acquisition_id": row["acquisition_id"],
                        "rank": row["rank"],
                        "raw_sha256": raw_digest,
                    }
                )
                if expected_source_id != row["id"]:
                    errors.append(f"source {row['id']} identity mismatch")
                try:
                    projected = {
                        "identity_key": _source_identity(row["kind"], value),
                        "title": _text(
                            value.get("title"), "source.title", maximum=20_000
                        ),
                        "year": (
                            _normalise_year(value.get("year"))
                            if row["kind"] == "paper"
                            else None
                        ),
                        "doi": _optional_text(
                            value.get("doi"), "source.doi", maximum=1_000
                        ),
                        "pmid": _optional_text(
                            value.get("pmid"), "source.pmid", maximum=1_000
                        ),
                        "nct_id": _optional_text(
                            value.get("nctId"), "source.nctId", maximum=1_000
                        ),
                    }
                except (ScientificEvidenceError, AttributeError, TypeError):
                    errors.append(f"source {row['id']} has invalid projected fields")
                else:
                    if any(
                        row[field] != expected for field, expected in projected.items()
                    ):
                        errors.append(f"source {row['id']} projection mismatch")
            for row in connection.execute("SELECT * FROM appraisals"):
                try:
                    value = json.loads(row["appraisal_json"])
                except (json.JSONDecodeError, TypeError):
                    errors.append(f"appraisal {row['id']} has invalid JSON")
                    continue
                appraisal_digest = payload_hash(value)
                if appraisal_digest != row["appraisal_sha256"]:
                    errors.append(f"appraisal {row['id']} hash mismatch")
                expected_appraisal_id = payload_hash(
                    {
                        "source_record_id": row["source_record_id"],
                        "appraisal_sha256": appraisal_digest,
                    }
                )
                if expected_appraisal_id != row["id"]:
                    errors.append(f"appraisal {row['id']} identity mismatch")
                try:
                    clean = validate_appraisal(value)
                    expected_tier = evidence_tier(
                        clean["study_design"], clean["risk_of_bias"]
                    )
                except (ScientificEvidenceError, TypeError):
                    errors.append(f"appraisal {row['id']} violates its contract")
                else:
                    if clean["source_record_id"] != row["source_record_id"]:
                        errors.append(f"appraisal {row['id']} source mismatch")
                    if expected_tier != row["evidence_tier"]:
                        errors.append(f"appraisal {row['id']} tier mismatch")
            for row in connection.execute("SELECT * FROM action_candidates"):
                try:
                    source_ids = json.loads(row["source_record_ids_json"])
                    readiness = json.loads(row["readiness_json"])
                except (json.JSONDecodeError, TypeError):
                    errors.append(f"action candidate {row['id']} has invalid JSON")
                    continue
                if (
                    row["authority_level"] != "A0"
                    or row["recommendation_authority"] != 0
                ):
                    errors.append(f"action candidate {row['id']} exceeds A0 authority")
                candidate_payload = {
                    "schema_version": CANDIDATE_SCHEMA,
                    "question": row["question"],
                    "intervention": row["intervention"],
                    "risk_class": row["risk_class"],
                    "source_record_ids": source_ids,
                    "readiness": readiness,
                    "authority_level": "A0",
                    "recommendation_authority": False,
                    "created_at": row["created_at"],
                }
                if payload_hash(candidate_payload) != row["id"]:
                    errors.append(f"action candidate {row['id']} identity mismatch")
        mode = self.database.stat().st_mode & 0o777
        if mode != 0o600:
            errors.append(f"database permissions are {oct(mode)}, expected 0o600")
        if self.database.parent == DEFAULT_DATABASE.parent.resolve():
            parent_mode = self.database.parent.stat().st_mode & 0o777
            if parent_mode != 0o700:
                errors.append(
                    f"default evidence directory permissions are {oct(parent_mode)}, "
                    "expected 0o700"
                )
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self.database}{suffix}")
            if sidecar.exists() and sidecar.stat().st_mode & 0o077:
                errors.append(f"database sidecar is not owner-only: {sidecar}")
        return {
            "schema_version": LEDGER_SCHEMA,
            "ok": not errors,
            "database": str(self.database),
            "counts": self.counts(),
            "head_sha256": previous_digest,
            "errors": errors,
        }


def validate_appraisal(appraisal: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(appraisal, Mapping) or set(appraisal) != APPRAISAL_FIELDS:
        missing = sorted(APPRAISAL_FIELDS - set(appraisal))
        extra = sorted(set(appraisal) - APPRAISAL_FIELDS)
        raise ScientificEvidenceError(
            f"invalid appraisal fields; missing={missing}, extra={extra}"
        )
    clean = dict(appraisal)
    if clean["schema_version"] != APPRAISAL_SCHEMA:
        raise ScientificEvidenceError(f"schema_version must be {APPRAISAL_SCHEMA}")
    clean["source_record_id"] = _sha256(clean["source_record_id"], "source_record_id")
    clean["study_identity"] = _text(
        clean["study_identity"], "study_identity", maximum=2_000
    )
    clean["study_design"] = _enum(clean["study_design"], "study_design", STUDY_DESIGNS)
    for field in (
        "human_verified",
        "full_text_verified",
        "harms_assessed",
        "conflicts_assessed",
    ):
        if not isinstance(clean[field], bool):
            raise ScientificEvidenceError(f"{field} must be boolean")
    clean["risk_of_bias"] = _enum(clean["risk_of_bias"], "risk_of_bias", RISK_OF_BIAS)
    clean["population_applicability"] = _enum(
        clean["population_applicability"],
        "population_applicability",
        APPLICABILITY,
    )
    clean["retraction_status"] = _enum(
        clean["retraction_status"], "retraction_status", RETRACTION_STATES
    )
    clean["outcome_direction"] = _enum(
        clean["outcome_direction"], "outcome_direction", OUTCOME_DIRECTIONS
    )
    for field in (
        "effect_estimate",
        "confidence_interval",
        "supporting_quote",
        "source_locator",
    ):
        clean[field] = _optional_text(clean[field], field)
    clean["assessed_by"] = _text(clean["assessed_by"], "assessed_by", maximum=500)
    clean["assessed_at"] = _timestamp(clean["assessed_at"], field="assessed_at")
    clean["notes"] = _string_list(clean["notes"], "notes")
    if clean["human_verified"] and not clean["supporting_quote"]:
        raise ScientificEvidenceError(
            "human_verified appraisal requires a supporting_quote"
        )
    if clean["full_text_verified"] and not clean["source_locator"]:
        raise ScientificEvidenceError(
            "full_text_verified appraisal requires a source_locator"
        )
    return clean


def evidence_tier(study_design: str, risk_of_bias: str) -> str:
    design = _enum(study_design, "study_design", STUDY_DESIGNS)
    risk = _enum(risk_of_bias, "risk_of_bias", RISK_OF_BIAS)
    if risk == "critical":
        return "U"
    numeric = _DESIGN_TIER[design]
    if risk == "high":
        numeric = min(5, numeric + 1)
    elif risk == "unclear" and numeric < 5:
        numeric += 1
    return f"T{numeric}"


def evaluate_readiness(
    appraisals: Sequence[Mapping[str, Any] | None],
    *,
    risk_class: str,
    contradictions_addressed: bool,
    clinician_reviewed: bool,
) -> dict[str, Any]:
    risk = _enum(risk_class, "risk_class", RISK_CLASSES)
    present = [dict(value) for value in appraisals if value is not None]
    study_ids = {value.get("study_identity") for value in present}
    tiers = [value.get("evidence_tier") for value in present]
    directions = {
        value.get("outcome_direction")
        for value in present
        if value.get("outcome_direction") not in {None, "unknown"}
    }
    contradictions_present = "mixed" in directions or (
        "beneficial" in directions
        and ("neutral" in directions or "harmful" in directions)
    )
    gates = {
        "all_sources_appraised": len(present) == len(appraisals),
        "at_least_two_independent_studies": len(study_ids) >= 2,
        "all_human_verified": bool(present)
        and all(value.get("human_verified") is True for value in present),
        "all_full_text_verified": bool(present)
        and all(value.get("full_text_verified") is True for value in present),
        "all_retraction_checked_clear": bool(present)
        and all(value.get("retraction_status") == "clear" for value in present),
        "all_harms_assessed": bool(present)
        and all(value.get("harms_assessed") is True for value in present),
        "all_conflicts_assessed": bool(present)
        and all(value.get("conflicts_assessed") is True for value in present),
        "applicability_established": bool(present)
        and all(
            value.get("population_applicability") in {"direct", "partial"}
            for value in present
        ),
        "effect_and_uncertainty_extracted": bool(present)
        and all(
            bool(value.get("effect_estimate"))
            and bool(value.get("confidence_interval"))
            for value in present
        ),
        "controlled_or_synthesis_evidence": any(tier in {"T1", "T2"} for tier in tiers),
        "contradictions_resolved": not contradictions_present
        or contradictions_addressed is True,
        "clinical_review_complete": risk != "clinical" or clinician_reviewed is True,
    }
    failed = [name for name, passed in gates.items() if not passed]
    ready = not failed
    return {
        "status": "ready_for_plan" if ready else "blocked",
        "ready_for_plan": ready,
        "ready_for_execution": False,
        "recommendation_authority": False,
        "gates": gates,
        "failed_gates": failed,
        "evidence_summary": {
            "source_records": len(appraisals),
            "appraised_records": len(present),
            "independent_studies": len(study_ids),
            "tiers": sorted(str(tier) for tier in tiers if tier),
            "outcome_directions": sorted(str(direction) for direction in directions),
            "contradictions_present": contradictions_present,
        },
        "boundary": (
            "A0 planning candidate only; this module never grants recommendation "
            "or execution authority."
        ),
    }
