from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from modules.research.core.scientific_automated_review import (
    AUTHORITY,
    CAMPAIGN_ID,
    validate_automated_review,
)
from modules.research.core.scientific_evidence import ScientificEvidenceError, payload_hash
from modules.research.core.scientific_review_funnel import (
    ABSTRACT_RETRIEVAL_SCHEMA,
    apply_abstract_enrichments,
)


def _group(marker: str, abstract: str | None) -> dict:
    value = {
        "group_id": marker * 64,
        "title": f"Retained study {marker}",
        "abstract": abstract,
        "urls": [f"https://example.org/{marker}"],
        "source_record_ids": [f"public-{marker}"],
    }
    value["group_payload_sha256"] = payload_hash(value)
    value["screening_priority"] = 50
    value["priority_reason_codes"] = ["SYNTHETIC_PRIORITY_FIXTURE"]
    return value


def _review(group: dict, *, decision: str) -> dict:
    abstract = " ".join((group.get("abstract") or "").split())
    claim = []
    if decision == "include":
        quote = "The retained abstract reports improved conversion efficiency."
        claim = [{
            "claim": "The abstract reports improved conversion efficiency.",
            "quote": quote,
            "source_url": group["urls"][0],
        }]
    return {
        "record_ref": f"G{group['group_id'][0].upper()}",
        "group_id": group["group_id"],
        "evidence_sha256": group["group_payload_sha256"],
        "abstract_sha256": hashlib.sha256(abstract.encode()).hexdigest(),
        "decision": decision,
        "reason_codes": ["ABSTRACT_SCOPE"],
        "reasons": ["The retained abstract was reviewed within the stated scope."],
        "uncertainty": ["The evidence is abstract-only."] if decision == "uncertain" else [],
        "study_basis": "experimental" if decision == "include" else "unclear",
        "claims": claim,
        "reviewer": {
            "type": "ai",
            "model": "GLM-5.3",
            "job_id": "synthetic-fixture-only",
            "raw_output_sha256": hashlib.sha256(b"synthetic review output").hexdigest(),
        },
        "independent_review": {
            "type": "ai",
            "model": "gpt-6-astra",
            "reviewed_at": "2026-10-05T12:00:00+00:00",
            "verdict": "accepted",
            "corrections": [],
        },
        "human_verified": False,
    }


def _fixture(tmp_path: Path) -> dict[str, Path | dict | list]:
    included = _group(
        "a", "The retained abstract reports improved conversion efficiency."
    )
    missing = _group("b", None)
    groups = [included, missing]
    paths = {
        "groups": tmp_path / "groups.jsonl",
        "proposals": tmp_path / "proposals.jsonl",
        "features": tmp_path / "features.jsonl",
        "protocol": tmp_path / "protocol.md",
        "review": tmp_path / "review.json",
        "duplicates": tmp_path / "duplicates.json",
    }
    for key, values in (
        ("groups", groups),
        ("proposals", [{"group_id": item["group_id"]} for item in groups]),
        ("features", [{"group_id": item["group_id"]} for item in groups]),
    ):
        paths[key].write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in values),
            encoding="utf-8",
        )
    paths["protocol"].write_text("Retained abstract scope; synthetic test fixture.\n", encoding="utf-8")
    document = {
        "schema_version": "giga.scientific-automated-review.v1",
        "campaign_id": CAMPAIGN_ID,
        "review_scope": "retained_abstracts",
        "authority": copy.deepcopy(AUTHORITY),
        "source_groups_sha256": hashlib.sha256(paths["groups"].read_bytes()).hexdigest(),
        "source_protocol_sha256": hashlib.sha256(paths["protocol"].read_bytes()).hexdigest(),
        "human_verified": False,
        "reviews": [_review(included, decision="include"), _review(missing, decision="uncertain")],
        "synthesis": {
            "title": "Synthetic descriptive summary",
            "summary": "This fixture exercises the abstract-only renderer.",
            "findings": [{
                "text": "One retained abstract reports a conversion-efficiency result.",
                "supporting_group_ids": [included["group_id"]],
                "limitations": ["One abstract only."],
            }],
            "limitations": ["Synthetic test content."],
            "reviewer": {
                "type": "ai",
                "model": "GLM-5.3",
                "job_id": "synthetic-synthesis-fixture",
                "raw_output_sha256": hashlib.sha256(b"synthetic synthesis").hexdigest(),
            },
            "independent_review": {
                "type": "ai",
                "model": "gpt-6-astra",
                "reviewed_at": "2026-10-05T12:01:00+00:00",
                "verdict": "accepted_with_corrections",
                "corrections": ["Keep the single-study limitation visible."],
            },
        },
    }
    document["report_sha256"] = payload_hash(document)
    paths["review"].write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    duplicate = {
        "schema_version": "giga.scientific-duplicate-decisions.v1",
        "candidate_pairs": 0,
        "complete": True,
        "decisions_recorded": 0,
        "same_study": 0,
        "different_study": 0,
        "uncertain": 0,
        "unresolved_pairs": 0,
        "human_verified": False,
    }
    duplicate["report_sha256"] = payload_hash(duplicate)
    paths["duplicates"].write_text(json.dumps(duplicate, indent=2) + "\n", encoding="utf-8")
    paths["groups_value"] = groups
    paths["document"] = document
    return paths


def _rehash(document: dict) -> None:
    document.pop("report_sha256", None)
    document["report_sha256"] = payload_hash(document)


def _validate(paths: dict, document: dict) -> dict:
    return validate_automated_review(
        document,
        groups=paths["groups_value"],
        proposals=[json.loads(line) for line in paths["proposals"].read_text().splitlines()],
        features=[json.loads(line) for line in paths["features"].read_text().splitlines()],
        source_groups_sha256=hashlib.sha256(paths["groups"].read_bytes()).hexdigest(),
        source_protocol_sha256=hashlib.sha256(paths["protocol"].read_bytes()).hexdigest(),
    )


def _rehash_group(group: dict) -> None:
    material = dict(group)
    material.pop("group_payload_sha256", None)
    group["group_payload_sha256"] = payload_hash(material)


def _resync(paths: dict, document: dict, *, build: dict | None = None) -> None:
    groups = paths["groups_value"]
    paths["groups"].write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in groups),
        encoding="utf-8",
    )
    if build is not None:
        for key in ("proposals", "features"):
            paths[key].write_text(
                "".join(
                    json.dumps(item, ensure_ascii=False) + "\n" for item in build[key]
                ),
                encoding="utf-8",
            )
    document["source_groups_sha256"] = hashlib.sha256(
        paths["groups"].read_bytes()
    ).hexdigest()
    document["reviews"] = [
        _review(groups[0], decision="include"),
        _review(groups[1], decision="uncertain"),
    ]
    _rehash(document)
    paths["document"] = document
    paths["review"].write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _enriched_fixture(tmp_path: Path) -> dict:
    paths = _fixture(tmp_path)
    groups = copy.deepcopy(paths["groups_value"])
    for group, pmid in zip(groups, ("87654321", "12345678"), strict=True):
        group["pmid"] = pmid
        group["publication_types"] = ["Journal Article"]
        group["query_ids"] = ["Q1"]
        group["negative_control"] = False
        group["compiled_query_retrieval"] = True
        group["retracted"] = False
        material = dict(group)
        material.pop("group_payload_sha256", None)
        material.pop("screening_priority", None)
        material.pop("priority_reason_codes", None)
        group["group_payload_sha256"] = payload_hash(material)

    abstract = "Surface treatment improved conversion efficiency in a perovskite solar cell."
    retrieval = {
        "schema_version": ABSTRACT_RETRIEVAL_SCHEMA,
        "provider": "synthetic-provider",
        "pmid": "12345678",
        "abstract": abstract,
        "retrieval_sha256": hashlib.sha256(b"synthetic retrieval receipt").hexdigest(),
    }
    ontology = {
        "conditions": {
            "perovskite_solar_cell": {
                "terms": ["perovskite solar cell"],
                "subject_headings": [],
            }
        },
        "interventions": {
            "surface_treatment": {"terms": ["surface treatment"], "subject_headings": []}
        },
        "outcomes": {
            "conversion_efficiency": {
                "terms": ["conversion efficiency"],
                "subject_headings": [],
            }
        },
    }
    specification = {
        "queries": [
            {
                "id": "Q1",
                "condition_concepts": ["perovskite_solar_cell"],
                "intervention_concepts": ["surface_treatment"],
                "outcome_concepts": ["conversion_efficiency"],
                "require_outcomes": False,
                "extra_required_groups": [],
                "negative_control": False,
            }
        ],
        "negative_controls": [],
    }
    build = apply_abstract_enrichments(groups, [retrieval], ontology, specification)
    assert build["report"]["applied"] == 1
    paths["groups_value"] = build["groups"]
    _resync(paths, paths["document"], build=build)
    return paths


def test_automated_review_accepts_legacy_and_producer_enriched_priority_groups(
    tmp_path: Path,
) -> None:
    paths = _enriched_fixture(tmp_path)
    groups = paths["groups_value"]
    assert "abstract_enrichment" not in groups[0]
    assert "abstract_enrichment" in groups[1]
    enriched_material = dict(groups[1])
    enriched_material.pop("group_payload_sha256")
    assert groups[1]["group_payload_sha256"] == payload_hash(enriched_material)
    assert _validate(paths, paths["document"]) is not None


def test_automated_review_rejects_stale_enrichment_abstract_hash(tmp_path: Path) -> None:
    paths = _enriched_fixture(tmp_path)
    enriched = paths["groups_value"][1]
    enriched["abstract_enrichment"]["abstract_sha256"] = hashlib.sha256(
        b"stale abstract"
    ).hexdigest()
    _rehash_group(enriched)
    _resync(paths, paths["document"])
    with pytest.raises(ScientificEvidenceError, match="abstract_enrichment"):
        _validate(paths, paths["document"])


@pytest.mark.parametrize("malformed", [None, {"provider": "synthetic-provider"}])
def test_automated_review_rejects_null_or_malformed_enrichment_without_fallback(
    tmp_path: Path, malformed: dict | None
) -> None:
    paths = _enriched_fixture(tmp_path)
    enriched = paths["groups_value"][1]
    enriched["abstract_enrichment"] = malformed
    _rehash_group(enriched)
    _resync(paths, paths["document"])
    with pytest.raises(ScientificEvidenceError, match="abstract_enrichment"):
        _validate(paths, paths["document"])


@pytest.mark.parametrize(
    "bad_metadata",
    ["missing_provider", "retrieval_hash", "invalid_pmid", "pmid", "authority"],
)
def test_automated_review_requires_complete_bound_enrichment_metadata(
    tmp_path: Path, bad_metadata: str
) -> None:
    paths = _enriched_fixture(tmp_path)
    enriched = paths["groups_value"][1]
    metadata = dict(enriched["abstract_enrichment"])
    if bad_metadata == "missing_provider":
        metadata.pop("provider")
    elif bad_metadata == "retrieval_hash":
        metadata["retrieval_sha256"] = "not-a-sha256"
    elif bad_metadata == "invalid_pmid":
        metadata["pmid"] = "not-a-pmid"
    elif bad_metadata == "pmid":
        metadata["pmid"] = "87654321"
    else:
        metadata["authority"] = "full_text_or_effectiveness"
    enriched["abstract_enrichment"] = metadata
    _rehash_group(enriched)
    _resync(paths, paths["document"])
    with pytest.raises(ScientificEvidenceError, match="abstract_enrichment"):
        _validate(paths, paths["document"])


def test_automated_review_rejects_enriched_hash_omitting_priority_fields(
    tmp_path: Path,
) -> None:
    paths = _enriched_fixture(tmp_path)
    enriched = paths["groups_value"][1]
    material = dict(enriched)
    material.pop("group_payload_sha256")
    material.pop("screening_priority", None)
    material.pop("priority_reason_codes", None)
    enriched["group_payload_sha256"] = payload_hash(material)
    _resync(paths, paths["document"])
    with pytest.raises(ScientificEvidenceError, match="payload hash mismatch"):
        _validate(paths, paths["document"])


def test_real_cli_accepts_complete_ai_provenance_and_keeps_scope_flags_false(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "automated-output"
    from modules.research.core.scientific_pipeline_cli import main

    argv = [
        "review-funnel", str(paths["groups"]), str(paths["proposals"]),
        str(paths["features"]), str(output), "--review-mode", "automated",
        "--automated-review", str(paths["review"]), "--protocol", str(paths["protocol"]),
        "--duplicate-decision-report", str(paths["duplicates"]),
    ]
    assert main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    readiness = json.loads((output / "AUTOMATED_REVIEW_READINESS.json").read_text())
    review_output = json.loads((output / "AUTOMATED_ELIGIBILITY_REVIEW.json").read_text())
    rows = review_output["reviews"]
    assert result["run_receipt"]["path"] == "AUTOMATED_REVIEW_RUN.json"
    run_receipt = json.loads((output / "AUTOMATED_REVIEW_RUN.json").read_text())
    assert run_receipt["provider_calls"] == 0
    assert run_receipt["acquisition_performed"] is False
    assert readiness["automated_descriptive_synthesis_ready"] is True
    assert readiness["human_verified"] is False
    assert readiness["full_text_review_completed"] is False
    assert readiness["risk_of_bias_appraisal_completed"] is False
    assert readiness["effectiveness_ranking_authorised"] is False
    assert readiness["recommendation_authority"] is False
    assert rows[1]["decision"] == "uncertain"
    assert rows[1]["claims"] == []
    assert rows[1]["source_urls"] == ["https://example.org/b"]
    assert all(row["human_verified"] is False for row in rows)
    assert "[Source 1](https://example.org/b)" in (output / "AUTOMATED_DESCRIPTIVE_SYNTHESIS.md").read_text()
    assert "accepted_with_corrections" in review_output["synthesis"]["independent_review"]["verdict"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda doc: doc["reviews"][0].update(human_verified=True), "cannot claim human"),
        (lambda doc: doc["reviews"][0].update(evidence_sha256="0" * 64), "stale"),
        (lambda doc: doc["reviews"][0]["claims"][0].update(quote="not in the abstract"), "exact retained abstract substring"),
    ],
)
def test_automated_review_rejects_false_human_stale_evidence_and_bad_citation(
    tmp_path: Path, change, message: str
) -> None:
    paths = _fixture(tmp_path)
    document = copy.deepcopy(paths["document"])
    change(document)
    _rehash(document)
    with pytest.raises(ScientificEvidenceError, match=message):
        _validate(paths, document)


def test_automated_review_rejects_wrong_source_protocol_hash(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    document = copy.deepcopy(paths["document"])
    document["source_protocol_sha256"] = "0" * 64
    _rehash(document)
    with pytest.raises(ScientificEvidenceError, match="protocol hash changed"):
        _validate(paths, document)


def test_nonzero_duplicate_candidate_without_verification_is_refused(tmp_path: Path) -> None:
    from modules.research.core.scientific_automated_review import build_automated_review_files

    paths = _fixture(tmp_path)
    duplicate = {
        "schema_version": "giga.scientific-duplicate-decisions.v1",
        "candidate_pairs": 1,
        "complete": True,
        "decisions_recorded": 1,
        "same_study": 0,
        "different_study": 0,
        "uncertain": 1,
        "unresolved_pairs": 0,
        "human_verified": False,
    }
    duplicate["report_sha256"] = payload_hash(duplicate)
    bad_duplicate_path = tmp_path / "unverified-duplicates.json"
    bad_duplicate_path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ScientificEvidenceError, match="lacks required verification"):
        build_automated_review_files(
            groups=paths["groups"],
            proposals=paths["proposals"],
            features=paths["features"],
            output=tmp_path / "refused-output",
            automated_review=paths["review"],
            protocol=paths["protocol"],
            duplicate_decision_report=bad_duplicate_path,
            command_argv=["synthetic-test"],
        )


def test_automated_cli_refuses_human_synthesis_signoff_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "must-not-be-created"
    from modules.research.core.scientific_pipeline_cli import main

    assert main([
        "review-funnel", str(paths["groups"]), str(paths["proposals"]),
        str(paths["features"]), str(output), "--review-mode", "automated",
        "--automated-review", str(paths["review"]), "--protocol", str(paths["protocol"]),
        "--duplicate-decision-report", str(paths["duplicates"]),
        "--synthesis-signoff", str(paths["review"]),
    ]) == 2
    assert "cannot be combined with human/full-text gate inputs" in capsys.readouterr().err
    assert not output.exists()


def test_automated_cli_preserves_nonempty_output_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "existing-output"
    output.mkdir()
    sentinel = output / "prior-receipt.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    from modules.research.core.scientific_pipeline_cli import main

    assert main([
        "review-funnel", str(paths["groups"]), str(paths["proposals"]),
        str(paths["features"]), str(output), "--review-mode", "automated",
        "--automated-review", str(paths["review"]), "--protocol", str(paths["protocol"]),
        "--duplicate-decision-report", str(paths["duplicates"]),
    ]) == 2
    assert "output directory is not empty" in capsys.readouterr().err
    assert sentinel.read_text(encoding="utf-8") == "preserve"
