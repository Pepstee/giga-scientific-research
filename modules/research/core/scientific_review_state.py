"""Append-only state ledger for deterministic scientific-review processing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .scientific_evidence import ScientificEvidenceError, canonical_json, payload_hash


REVIEW_LEDGER_SCHEMA = "giga.scientific-review-ledger.v1"
LLM_PROPOSAL_SCHEMA = "giga.scientific-llm-eligibility-proposal.v1"
DEFAULT_REVIEW_DATABASE = Path.home() / ".giga" / "scientific-review.sqlite3"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]{2,79}$")
STATES = {
    "discovered",
    "triaged",
    "proposed_include",
    "proposed_exclude",
    "awaiting_review",
    "included",
    "excluded",
    "fulltext_planned",
    "fulltext_obtained",
    "fulltext_unavailable",
    "appraised",
}
TRANSITIONS = {
    (None, "discovered"),
    ("discovered", "triaged"),
    ("triaged", "proposed_include"),
    ("triaged", "proposed_exclude"),
    ("triaged", "awaiting_review"),
    ("proposed_include", "included"),
    ("proposed_exclude", "excluded"),
    ("awaiting_review", "included"),
    ("awaiting_review", "excluded"),
    ("included", "fulltext_planned"),
    ("fulltext_planned", "fulltext_obtained"),
    ("fulltext_planned", "fulltext_unavailable"),
    ("fulltext_obtained", "appraised"),
}
ACTOR_TRANSITIONS = {
    "deterministic": {
        (None, "discovered"),
        ("discovered", "triaged"),
        ("triaged", "proposed_include"),
        ("triaged", "proposed_exclude"),
        ("triaged", "awaiting_review"),
    },
    "human": {
        ("proposed_include", "included"),
        ("proposed_exclude", "excluded"),
        ("awaiting_review", "included"),
        ("awaiting_review", "excluded"),
        ("fulltext_obtained", "appraised"),
    },
    "system": {
        ("included", "fulltext_planned"),
        ("fulltext_planned", "fulltext_obtained"),
        ("fulltext_planned", "fulltext_unavailable"),
    },
}


def _timestamp(value: str | None, field: str) -> str:
    candidate = value or datetime.now(timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ScientificEvidenceError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ScientificEvidenceError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _text(value: Any, field: str, *, maximum: int = 20_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScientificEvidenceError(f"{field} must be non-empty text")
    result = value.strip()
    if len(result) > maximum:
        raise ScientificEvidenceError(f"{field} exceeds {maximum} characters")
    return result


def _sha(value: Any, field: str) -> str:
    result = _text(value, field, maximum=64)
    if not SHA256.fullmatch(result):
        raise ScientificEvidenceError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _verified_group_payload_sha256(
    source_metadata: Mapping[str, Any],
    *,
    expected_group_id: str,
    recorded_sha256: str,
) -> str:
    source_group_id = _sha(
        source_metadata.get("group_id"), "source_metadata.group_id"
    )
    if source_group_id != expected_group_id:
        raise ScientificEvidenceError(
            "source_metadata group_id does not match the awaiting_review group"
        )
    supplied_sha256 = _sha(
        source_metadata.get("group_payload_sha256"),
        "source_metadata.group_payload_sha256",
    )
    if supplied_sha256 != recorded_sha256:
        raise ScientificEvidenceError(
            "source_metadata group payload digest does not match the recorded group evidence"
        )

    material = dict(source_metadata)
    material.pop("group_payload_sha256", None)
    derived_fields = (
        "screening_priority",
        "priority_reason_codes",
    )
    for mask in range(1 << len(derived_fields)):
        candidate = dict(material)
        for index, field in enumerate(derived_fields):
            if mask & (1 << index):
                candidate.pop(field, None)
        try:
            candidate_sha256 = payload_hash(candidate)
        except (TypeError, ValueError):
            continue
        if candidate_sha256 == recorded_sha256:
            return recorded_sha256
    raise ScientificEvidenceError(
        "source_metadata must include the full source group payload matching "
        "the recorded group evidence; title/abstract alone are insufficient"
    )


def _reason_codes(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ScientificEvidenceError("reason_codes must be a non-empty array")
    result = [_text(item, "reason_codes[]", maximum=80) for item in value]
    if len(result) != len(set(result)):
        raise ScientificEvidenceError("reason_codes must not contain duplicates")
    invalid = [item for item in result if not REASON_CODE.fullmatch(item)]
    if invalid:
        raise ScientificEvidenceError(f"invalid reason code(s): {invalid}")
    return result


class ScientificReviewStore:
    """Owner-only append-only state and LLM-proposal ledger."""

    def __init__(self, database: Path = DEFAULT_REVIEW_DATABASE):
        self.database = database.expanduser().resolve()
        parent_existed = self.database.parent.exists()
        self.database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if (
            not parent_existed
            or self.database.parent == DEFAULT_REVIEW_DATABASE.parent.resolve()
        ):
            os.chmod(self.database.parent, 0o700)
        self._initialise()
        os.chmod(self.database, 0o600)

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
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    schema_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    campaign_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    from_state TEXT,
                    to_state TEXT NOT NULL,
                    actor_kind TEXT NOT NULL,
                    reason_codes_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    evidence_sha256 TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    previous_event_sha256 TEXT,
                    event_sha256 TEXT NOT NULL UNIQUE
                );
                CREATE INDEX IF NOT EXISTS idx_review_events_group
                    ON review_events(campaign_id,group_id,sequence);
                CREATE TABLE IF NOT EXISTS llm_proposals (
                    id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    proposal_json TEXT NOT NULL,
                    proposal_sha256 TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    human_confirmed INTEGER NOT NULL CHECK(human_confirmed=0),
                    state_authority INTEGER NOT NULL CHECK(state_authority=0),
                    UNIQUE(campaign_id,group_id,proposal_sha256)
                );
                CREATE TABLE IF NOT EXISTS duplicate_decisions (
                    id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL,
                    pair_sha256 TEXT NOT NULL,
                    decision TEXT NOT NULL CHECK(decision IN
                        ('same_study','different_study','uncertain')),
                    decided_by TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    evidence_sha256 TEXT NOT NULL,
                    decided_at TEXT NOT NULL,
                    state_authority INTEGER NOT NULL CHECK(state_authority=0),
                    UNIQUE(campaign_id,pair_sha256)
                );
                """
            )
            current = connection.execute(
                "SELECT schema_version FROM schema_metadata WHERE singleton=1"
            ).fetchone()
            if current is None:
                connection.execute(
                    "INSERT INTO schema_metadata VALUES (1,?,?)",
                    (
                        REVIEW_LEDGER_SCHEMA,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            elif current["schema_version"] != REVIEW_LEDGER_SCHEMA:
                raise ScientificEvidenceError(
                    f"unsupported review ledger {current['schema_version']}; "
                    f"expected {REVIEW_LEDGER_SCHEMA}"
                )
            for table in ("review_events", "llm_proposals", "duplicate_decisions"):
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
    def _current_state(
        connection: sqlite3.Connection,
        campaign_id: str,
        group_id: str,
    ) -> str | None:
        row = connection.execute(
            """
            SELECT to_state FROM review_events
            WHERE campaign_id=? AND group_id=?
            ORDER BY sequence DESC LIMIT 1
            """,
            (campaign_id, group_id),
        ).fetchone()
        return row["to_state"] if row else None

    def transition(
        self,
        *,
        campaign_id: str,
        group_id: str,
        from_state: str | None,
        to_state: str,
        actor_kind: str,
        reason_codes: Sequence[str],
        evidence: Mapping[str, Any],
        recorded_at: str | None = None,
    ) -> tuple[str, bool]:
        campaign = _text(campaign_id, "campaign_id", maximum=200)
        group = _sha(group_id, "group_id")
        if from_state is not None and from_state not in STATES:
            raise ScientificEvidenceError("from_state is invalid")
        if to_state not in STATES:
            raise ScientificEvidenceError("to_state is invalid")
        transition = (from_state, to_state)
        if transition not in TRANSITIONS:
            raise ScientificEvidenceError(f"illegal review transition: {transition}")
        if actor_kind not in ACTOR_TRANSITIONS:
            raise ScientificEvidenceError(
                f"actor_kind must be one of {sorted(ACTOR_TRANSITIONS)}"
            )
        if transition not in ACTOR_TRANSITIONS[actor_kind]:
            raise ScientificEvidenceError(
                f"{actor_kind} cannot perform review transition {transition}"
            )
        reasons = _reason_codes(list(reason_codes))
        if not isinstance(evidence, Mapping) or not evidence:
            raise ScientificEvidenceError("transition evidence must be a non-empty object")
        evidence_value = dict(evidence)
        evidence_digest = payload_hash(evidence_value)
        identity_material = {
            "campaign_id": campaign,
            "group_id": group,
            "from_state": from_state,
            "to_state": to_state,
            "actor_kind": actor_kind,
            "reason_codes": reasons,
            "evidence_sha256": evidence_digest,
        }
        event_id = payload_hash(identity_material)
        timestamp = _timestamp(recorded_at, "recorded_at")
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM review_events WHERE id=?",
                (event_id,),
            ).fetchone()
            if existing:
                return event_id, False
            current = self._current_state(connection, campaign, group)
            if current != from_state:
                raise ScientificEvidenceError(
                    f"review state mismatch for {group}: "
                    f"expected {from_state!r}, current {current!r}"
                )
            previous = connection.execute(
                "SELECT event_sha256 FROM review_events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous_digest = previous["event_sha256"] if previous else None
            event_material = {
                "id": event_id,
                **identity_material,
                "recorded_at": timestamp,
                "previous_event_sha256": previous_digest,
            }
            event_digest = payload_hash(event_material)
            connection.execute(
                """
                INSERT INTO review_events
                (id,campaign_id,group_id,from_state,to_state,actor_kind,
                 reason_codes_json,evidence_json,evidence_sha256,recorded_at,
                 previous_event_sha256,event_sha256)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event_id,
                    campaign,
                    group,
                    from_state,
                    to_state,
                    actor_kind,
                    canonical_json(reasons),
                    canonical_json(evidence_value),
                    evidence_digest,
                    timestamp,
                    previous_digest,
                    event_digest,
                ),
            )
        return event_id, True

    def record_llm_proposal(
        self,
        proposal: Mapping[str, Any],
        *,
        source_metadata: Mapping[str, Any],
    ) -> tuple[str, bool]:
        required = {
            "schema_version",
            "campaign_id",
            "group_id",
            "decision",
            "reason_code",
            "rationale",
            "evidence_spans",
            "model",
            "proposed_at",
        }
        if not isinstance(proposal, Mapping) or set(proposal) != required:
            raise ScientificEvidenceError(
                "invalid LLM proposal fields; "
                f"missing={sorted(required - set(proposal))}, "
                f"extra={sorted(set(proposal) - required)}"
            )
        if proposal["schema_version"] != LLM_PROPOSAL_SCHEMA:
            raise ScientificEvidenceError(
                f"schema_version must be {LLM_PROPOSAL_SCHEMA}"
            )
        campaign = _text(proposal["campaign_id"], "campaign_id", maximum=200)
        group = _sha(proposal["group_id"], "group_id")
        if not isinstance(source_metadata, Mapping):
            raise ScientificEvidenceError(
                "source_metadata must be the full source group payload"
            )
        decision = proposal["decision"]
        if decision not in {"include", "exclude", "uncertain"}:
            raise ScientificEvidenceError(
                "LLM decision must be include, exclude or uncertain"
            )
        reason = _text(proposal["reason_code"], "reason_code", maximum=80)
        if not REASON_CODE.fullmatch(reason):
            raise ScientificEvidenceError("LLM reason_code is invalid")
        spans = proposal["evidence_spans"]
        if not isinstance(spans, list) or not spans:
            raise ScientificEvidenceError(
                "LLM proposal requires at least one evidence span"
            )
        clean_spans: list[dict[str, str]] = []
        for index, span in enumerate(spans):
            if not isinstance(span, Mapping) or set(span) != {
                "field",
                "exact_text",
            }:
                raise ScientificEvidenceError(
                    f"evidence_spans[{index}] has invalid fields"
                )
            field = span["field"]
            if field not in {"title", "abstract"}:
                raise ScientificEvidenceError(
                    "LLM evidence span field must be title or abstract"
                )
            exact = _text(
                span["exact_text"],
                f"evidence_spans[{index}].exact_text",
                maximum=5_000,
            )
            source = source_metadata.get(field)
            if not isinstance(source, str) or exact not in source:
                raise ScientificEvidenceError(
                    f"LLM evidence span is not verbatim in source {field}"
                )
            clean_spans.append({"field": field, "exact_text": exact})
        rationale = _text(proposal["rationale"], "rationale", maximum=10_000)
        model = _text(proposal["model"], "model", maximum=200)
        proposed_at = _timestamp(proposal["proposed_at"], "proposed_at")
        with self.connect() as connection:
            current = self._current_state(connection, campaign, group)
            if current != "awaiting_review":
                raise ScientificEvidenceError(
                    "LLM proposals are accepted only for awaiting_review records"
                )
            triaged = connection.execute(
                """
                SELECT evidence_json,evidence_sha256 FROM review_events
                WHERE campaign_id=? AND group_id=? AND to_state='triaged'
                ORDER BY sequence DESC LIMIT 1
                """,
                (campaign, group),
            ).fetchone()
            if triaged is None:
                raise ScientificEvidenceError(
                    "awaiting_review record has no stored source group binding; "
                    "a verifiable screening group payload is required"
                )
            try:
                triaged_evidence = json.loads(triaged["evidence_json"])
            except (json.JSONDecodeError, TypeError) as exc:
                raise ScientificEvidenceError(
                    "stored source group binding is invalid"
                ) from exc
            if (
                not isinstance(triaged_evidence, Mapping)
                or payload_hash(triaged_evidence) != triaged["evidence_sha256"]
            ):
                raise ScientificEvidenceError(
                    "stored source group binding failed its evidence hash check"
                )
            recorded_group_sha256 = triaged_evidence.get(
                "group_payload_sha256"
            )
            if recorded_group_sha256 is None:
                raise ScientificEvidenceError(
                    "awaiting_review record has no stored source group digest; "
                    "re-ingest a verifiable screening snapshot before proposing"
                )
            recorded_group_sha256 = _sha(
                recorded_group_sha256, "recorded group_payload_sha256"
            )
            verified_source_sha256 = _verified_group_payload_sha256(
                source_metadata,
                expected_group_id=group,
                recorded_sha256=recorded_group_sha256,
            )
            clean = {
                "schema_version": LLM_PROPOSAL_SCHEMA,
                "campaign_id": campaign,
                "group_id": group,
                "decision": decision,
                "reason_code": reason,
                "rationale": rationale,
                "evidence_spans": clean_spans,
                "model": model,
                "proposed_at": proposed_at,
                "source_group_payload_sha256": verified_source_sha256,
                "proposal_only": True,
                "human_confirmation_required": True,
                "state_authority": False,
            }
            digest = payload_hash(clean)
            proposal_id = payload_hash(
                {
                    "campaign_id": campaign,
                    "group_id": group,
                    "proposal_sha256": digest,
                }
            )
            existing = connection.execute(
                "SELECT id FROM llm_proposals WHERE id=?",
                (proposal_id,),
            ).fetchone()
            if existing:
                return proposal_id, False
            connection.execute(
                """
                INSERT INTO llm_proposals
                (id,campaign_id,group_id,proposal_json,proposal_sha256,
                 recorded_at,human_confirmed,state_authority)
                VALUES (?,?,?,?,?,?,0,0)
                """,
                (
                    proposal_id,
                    campaign,
                    group,
                    canonical_json(clean),
                    digest,
                    clean["proposed_at"],
                ),
            )
        return proposal_id, True

    def record_duplicate_decision(
        self,
        *,
        campaign_id: str,
        pair_sha256: str,
        decision: str,
        decided_by: str,
        evidence: Mapping[str, Any],
        decided_at: str | None = None,
    ) -> tuple[str, bool]:
        campaign = _text(campaign_id, "campaign_id", maximum=200)
        pair = _sha(pair_sha256, "pair_sha256")
        if decision not in {"same_study", "different_study", "uncertain"}:
            raise ScientificEvidenceError(
                "duplicate decision must be same_study, different_study or uncertain"
            )
        actor = _text(decided_by, "decided_by", maximum=200)
        if not isinstance(evidence, Mapping) or not evidence:
            raise ScientificEvidenceError(
                "duplicate decision evidence must be a non-empty object"
            )
        evidence_value = dict(evidence)
        evidence_digest = payload_hash(evidence_value)
        timestamp = _timestamp(decided_at, "decided_at")
        identity = {
            "campaign_id": campaign,
            "pair_sha256": pair,
            "decision": decision,
            "decided_by": actor,
            "evidence_sha256": evidence_digest,
            "decided_at": timestamp,
        }
        identifier = payload_hash(identity)
        with self.connect() as connection:
            existing_pair = connection.execute(
                """
                SELECT id,decision FROM duplicate_decisions
                WHERE campaign_id=? AND pair_sha256=?
                """,
                (campaign, pair),
            ).fetchone()
            if existing_pair:
                if existing_pair["id"] != identifier:
                    raise ScientificEvidenceError(
                        "duplicate pair already has a different immutable decision"
                    )
                return identifier, False
            connection.execute(
                """
                INSERT INTO duplicate_decisions
                (id,campaign_id,pair_sha256,decision,decided_by,evidence_json,
                 evidence_sha256,decided_at,state_authority)
                VALUES (?,?,?,?,?,?,?,?,0)
                """,
                (
                    identifier,
                    campaign,
                    pair,
                    decision,
                    actor,
                    canonical_json(evidence_value),
                    evidence_digest,
                    timestamp,
                ),
            )
        return identifier, True

    def current_states(self, campaign_id: str) -> dict[str, str]:
        campaign = _text(campaign_id, "campaign_id", maximum=200)
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT e.group_id,e.to_state
                FROM review_events AS e
                JOIN (
                    SELECT group_id,MAX(sequence) AS sequence
                    FROM review_events WHERE campaign_id=?
                    GROUP BY group_id
                ) AS latest ON latest.sequence=e.sequence
                WHERE e.campaign_id=?
                ORDER BY e.group_id
                """,
                (campaign, campaign),
            ).fetchall()
        return {row["group_id"]: row["to_state"] for row in rows}

    def counts(self, campaign_id: str) -> dict[str, Any]:
        states = self.current_states(campaign_id)
        with self.connect() as connection:
            events = connection.execute(
                "SELECT COUNT(*) FROM review_events WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()[0]
            proposals = connection.execute(
                "SELECT COUNT(*) FROM llm_proposals WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()[0]
            duplicate_decisions = connection.execute(
                "SELECT COUNT(*) FROM duplicate_decisions WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()[0]
        state_counts: dict[str, int] = {}
        for state in sorted(STATES):
            count = sum(value == state for value in states.values())
            if count:
                state_counts[state] = count
        return {
            "campaign_id": campaign_id,
            "groups": len(states),
            "events": int(events),
            "llm_proposals": int(proposals),
            "duplicate_decisions": int(duplicate_decisions),
            "states": state_counts,
        }

    def audit(self) -> dict[str, Any]:
        errors: list[str] = []
        previous: str | None = None
        current_states: dict[tuple[str, str], str] = {}
        with self.connect() as connection:
            metadata = connection.execute(
                "SELECT schema_version FROM schema_metadata WHERE singleton=1"
            ).fetchone()
            if metadata is None or metadata["schema_version"] != REVIEW_LEDGER_SCHEMA:
                errors.append("review ledger schema metadata mismatch")
            expected_triggers = {
                f"no_{operation}_{table}"
                for operation in ("update", "delete")
                for table in (
                    "review_events",
                    "llm_proposals",
                    "duplicate_decisions",
                )
            }
            actual_triggers = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                )
            }
            missing = sorted(expected_triggers - actual_triggers)
            if missing:
                errors.append("append-only trigger(s) missing: " + ", ".join(missing))
            for row in connection.execute(
                "SELECT * FROM review_events ORDER BY sequence"
            ):
                key = (row["campaign_id"], row["group_id"])
                if current_states.get(key) != row["from_state"]:
                    errors.append(f"review event {row['id']} state-chain mismatch")
                transition = (row["from_state"], row["to_state"])
                if transition not in TRANSITIONS:
                    errors.append(f"review event {row['id']} has illegal transition")
                if transition not in ACTOR_TRANSITIONS.get(row["actor_kind"], set()):
                    errors.append(f"review event {row['id']} actor boundary mismatch")
                try:
                    reasons = json.loads(row["reason_codes_json"])
                    evidence = json.loads(row["evidence_json"])
                except (json.JSONDecodeError, TypeError):
                    errors.append(f"review event {row['id']} has invalid JSON")
                    continue
                try:
                    _reason_codes(reasons)
                except ScientificEvidenceError:
                    errors.append(f"review event {row['id']} has invalid reason codes")
                evidence_digest = payload_hash(evidence)
                if evidence_digest != row["evidence_sha256"]:
                    errors.append(f"review event {row['id']} evidence hash mismatch")
                identity_material = {
                    "campaign_id": row["campaign_id"],
                    "group_id": row["group_id"],
                    "from_state": row["from_state"],
                    "to_state": row["to_state"],
                    "actor_kind": row["actor_kind"],
                    "reason_codes": reasons,
                    "evidence_sha256": row["evidence_sha256"],
                }
                if payload_hash(identity_material) != row["id"]:
                    errors.append(f"review event {row['id']} identity mismatch")
                event_material = {
                    "id": row["id"],
                    **identity_material,
                    "recorded_at": row["recorded_at"],
                    "previous_event_sha256": row["previous_event_sha256"],
                }
                if row["previous_event_sha256"] != previous:
                    errors.append(f"review event {row['id']} ledger-chain mismatch")
                if payload_hash(event_material) != row["event_sha256"]:
                    errors.append(f"review event {row['id']} event hash mismatch")
                previous = row["event_sha256"]
                current_states[key] = row["to_state"]
            for row in connection.execute("SELECT * FROM llm_proposals"):
                try:
                    proposal = json.loads(row["proposal_json"])
                except (json.JSONDecodeError, TypeError):
                    errors.append(f"LLM proposal {row['id']} has invalid JSON")
                    continue
                digest = payload_hash(proposal)
                if digest != row["proposal_sha256"]:
                    errors.append(f"LLM proposal {row['id']} hash mismatch")
                expected = payload_hash(
                    {
                        "campaign_id": row["campaign_id"],
                        "group_id": row["group_id"],
                        "proposal_sha256": digest,
                    }
                )
                if expected != row["id"]:
                    errors.append(f"LLM proposal {row['id']} identity mismatch")
                if (
                    row["human_confirmed"] != 0
                    or row["state_authority"] != 0
                    or proposal.get("state_authority") is not False
                ):
                    errors.append(f"LLM proposal {row['id']} gained authority")
            for row in connection.execute("SELECT * FROM duplicate_decisions"):
                try:
                    evidence = json.loads(row["evidence_json"])
                except (json.JSONDecodeError, TypeError):
                    errors.append(
                        f"duplicate decision {row['id']} has invalid JSON"
                    )
                    continue
                digest = payload_hash(evidence)
                if digest != row["evidence_sha256"]:
                    errors.append(
                        f"duplicate decision {row['id']} evidence hash mismatch"
                    )
                identity = {
                    "campaign_id": row["campaign_id"],
                    "pair_sha256": row["pair_sha256"],
                    "decision": row["decision"],
                    "decided_by": row["decided_by"],
                    "evidence_sha256": digest,
                    "decided_at": row["decided_at"],
                }
                if payload_hash(identity) != row["id"]:
                    errors.append(
                        f"duplicate decision {row['id']} identity mismatch"
                    )
                if row["state_authority"] != 0:
                    errors.append(
                        f"duplicate decision {row['id']} gained state authority"
                    )
        mode = self.database.stat().st_mode & 0o777
        if mode != 0o600:
            errors.append(f"review database permissions are {oct(mode)}, expected 0o600")
        return {
            "schema_version": REVIEW_LEDGER_SCHEMA,
            "ok": not errors,
            "errors": errors,
            "event_count": len(current_states)
            and sum(1 for _ in self._event_rows())
            or 0,
            "ledger_head_sha256": previous,
            "database_sha256": hashlib.sha256(self.database.read_bytes()).hexdigest(),
        }

    def _event_rows(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM review_events ORDER BY sequence"
                )
            ]


def ingest_screening_snapshot(
    store: ScientificReviewStore,
    *,
    campaign_id: str,
    groups: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    snapshot_sha256: str,
    recorded_at: str,
) -> dict[str, Any]:
    snapshot = _sha(snapshot_sha256, "snapshot_sha256")
    proposal_by_group = {item["group_id"]: item for item in proposals}
    created = 0
    replayed = 0
    for group in groups:
        group_id = group["group_id"]
        transitions = [
            (
                None,
                "discovered",
                ["DISCOVERED_FROM_ACQUISITION_LEDGER"],
                {
                    "snapshot_sha256": snapshot,
                    "source_record_ids": group["source_record_ids"],
                },
            ),
            (
                "discovered",
                "triaged",
                ["DETERMINISTIC_METADATA_TRIAGE"],
                {
                    "snapshot_sha256": snapshot,
                    "group_payload_sha256": group["group_payload_sha256"],
                },
            ),
        ]
        proposal = proposal_by_group[group_id]
        target = {
            "auto_include": "proposed_include",
            "auto_exclude": "proposed_exclude",
            "manual_review": "awaiting_review",
        }[proposal["proposed_status"]]
        transitions.append(
            (
                "triaged",
                target,
                proposal["reason_codes"],
                {
                    "snapshot_sha256": snapshot,
                    "proposal_sha256": proposal["proposal_sha256"],
                    "final_decision": False,
                },
            )
        )
        for from_state, to_state, reasons, evidence in transitions:
            _, was_created = store.transition(
                campaign_id=campaign_id,
                group_id=group_id,
                from_state=from_state,
                to_state=to_state,
                actor_kind="deterministic",
                reason_codes=reasons,
                evidence=evidence,
                recorded_at=recorded_at,
            )
            if was_created:
                created += 1
            else:
                replayed += 1
    return {
        "campaign_id": campaign_id,
        "groups": len(groups),
        "events_created": created,
        "events_replayed": replayed,
        "audit": store.audit(),
    }
