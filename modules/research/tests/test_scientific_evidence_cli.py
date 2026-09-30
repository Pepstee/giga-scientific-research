from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from modules.research.core import scientific_evidence_cli as cli_module
from modules.research.core.scientific_evidence import ScientificEvidenceStore


FIXED_TIME = "2026-09-30T12:00:00+00:00"


def _fill_ledger(
    store: ScientificEvidenceStore, *, tag: str, dois: list[str]
) -> None:
    store.record_acquisition(
        provider="openalex",
        endpoint="search/papers",
        request={"query": f"synthetic-{tag}", "maxResults": len(dois)},
        response={
            "papers": [
                {"title": f"Synthetic Paper {tag} {index}", "doi": doi}
                for index, doi in enumerate(dois)
            ]
        },
        observed_at=FIXED_TIME,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _settle_synthetic_ledger(store: ScientificEvidenceStore) -> None:
    with store.connect() as connection:
        result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    assert result is not None
    assert result[0] == 0
    assert result[1] == result[2] or result[1] == -1


def test_cli_aggregate_cap_refuses_before_provider_construction(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    store_a = ScientificEvidenceStore(tmp_path / "aggregate_a.db")
    store_b = ScientificEvidenceStore(tmp_path / "aggregate_b.db")
    current_store = ScientificEvidenceStore(tmp_path / "current.db")

    first_dois = [f"10.1000/synthetic-{index:03d}" for index in range(19)]
    second_dois = first_dois[:3] + [
        f"10.1000/synthetic-{index:03d}" for index in range(19, 34)
    ]
    _fill_ledger(store_a, tag="A", dois=first_dois)
    _fill_ledger(store_b, tag="B", dois=second_dois)

    counts_a = store_a.counts()
    counts_b = store_b.counts()
    assert counts_a["source_records"] == 19
    assert counts_b["source_records"] == 18
    assert store_a.acquisition_budget()["unique_identities"] == 19
    assert store_b.acquisition_budget()["unique_identities"] == 18

    aggregate = current_store.acquisition_budget(
        (store_a.database, store_b.database)
    )
    assert aggregate["source_records"] == 37
    assert aggregate["unique_identities"] == 34

    hash_a = _sha256(store_a.database)
    hash_b = _sha256(store_b.database)
    hash_current = _sha256(current_store.database)
    factory_calls: list[str] = []

    def unexpected_provider_factory(provider: str):
        factory_calls.append(provider)
        raise AssertionError("provider construction must follow the cap check")

    monkeypatch.setattr(
        cli_module, "create_paper_client", unexpected_provider_factory
    )

    exit_code = cli_module.main(
        [
            "--database",
            str(current_store.database),
            "--aggregate-database",
            str(store_a.database),
            "--aggregate-database",
            str(store_b.database),
            "search-papers",
            "synthetic neutral query",
            "--provider",
            "openalex",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "existing aggregate has 37 source records / 34 unique identities" in captured.err
    assert factory_calls == []
    assert store_a.counts() == counts_a
    assert store_b.counts() == counts_b
    assert current_store.counts()["source_records"] == 0
    assert current_store.counts()["acquisitions"] == 0
    assert _sha256(store_a.database) == hash_a
    assert _sha256(store_b.database) == hash_b
    assert _sha256(current_store.database) == hash_current
    assert all(
        store.audit()["ok"] is True
        for store in (store_a, store_b, current_store)
    )


def test_cli_search_papers_at_cap_skips_elicit_without_client_construction(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    store_a, store_b = _ledgers_at_cap(tmp_path)
    current_store = ScientificEvidenceStore(tmp_path / "current.db")
    budget = current_store.acquisition_budget((store_a.database, store_b.database))
    assert budget["source_records"] == 20
    assert budget["unique_identities"] == 20

    # Settle only these freshly created synthetic fixtures before byte snapshots.
    for store in (store_a, store_b, current_store):
        _settle_synthetic_ledger(store)

    counts_a = store_a.counts()
    counts_b = store_b.counts()
    counts_current = current_store.counts()
    hash_a = _sha256(store_a.database)
    hash_b = _sha256(store_b.database)
    hash_current = _sha256(current_store.database)
    client_constructions: list[str] = []

    def track_elicit_construction(*args: object, **kwargs: object) -> object:
        client_constructions.append("elicit")
        return object()

    monkeypatch.setattr(cli_module, "ElicitClient", track_elicit_construction)
    monkeypatch.delenv("ELICIT_API_KEY", raising=False)

    exit_code = cli_module.main(
        [
            "--database",
            str(current_store.database),
            "--aggregate-database",
            str(store_a.database),
            "--aggregate-database",
            str(store_b.database),
            "search-papers",
            "synthetic exact-cap zero-budget query",
            "--provider",
            "elicit",
            "--max-results",
            "1",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "acquisitions": [
            {
                "provider": "elicit",
                "status": "skipped_record_budget_exhausted",
                "source_count": 0,
            }
        ]
    }
    assert client_constructions == []
    assert store_a.counts() == counts_a
    assert store_b.counts() == counts_b
    assert current_store.counts() == counts_current
    assert _sha256(store_a.database) == hash_a
    assert _sha256(store_b.database) == hash_b
    assert _sha256(current_store.database) == hash_current
    assert all(store.audit()["ok"] is True for store in (store_a, store_b, current_store))


def _ledgers_over_cap(
    tmp_path: Path,
) -> tuple[ScientificEvidenceStore, ScientificEvidenceStore]:
    store_a = ScientificEvidenceStore(tmp_path / "aggregate_a.db")
    store_b = ScientificEvidenceStore(tmp_path / "aggregate_b.db")
    first_dois = [f"10.1000/synthetic-{index:03d}" for index in range(19)]
    second_dois = first_dois[:3] + [
        f"10.1000/synthetic-{index:03d}" for index in range(19, 34)
    ]
    _fill_ledger(store_a, tag="A", dois=first_dois)
    _fill_ledger(store_b, tag="B", dois=second_dois)
    return store_a, store_b


def _ledgers_at_cap(
    tmp_path: Path,
) -> tuple[ScientificEvidenceStore, ScientificEvidenceStore]:
    store_a = ScientificEvidenceStore(tmp_path / "aggregate_a.db")
    store_b = ScientificEvidenceStore(tmp_path / "aggregate_b.db")
    first_dois = [f"10.1000/synthetic-{index:03d}" for index in range(11)]
    second_dois = [
        f"10.1000/synthetic-{index:03d}" for index in range(11, 20)
    ]
    _fill_ledger(store_a, tag="A", dois=first_dois)
    _fill_ledger(store_b, tag="B", dois=second_dois)
    return store_a, store_b


@pytest.mark.parametrize(
    (
        "ledger_builder",
        "expected_rows",
        "expected_identities",
        "expected_refusal",
    ),
    [
        pytest.param(
            _ledgers_over_cap,
            37,
            34,
            "existing aggregate has 37 source records / 34 unique identities",
            id="over-cap",
        ),
        pytest.param(
            _ledgers_at_cap,
            20,
            20,
            "cumulative acquisition record cap 20 is exhausted",
            id="at-cap",
        ),
    ],
)
@pytest.mark.parametrize("command", ["start-report", "start-review", "session"])
def test_cli_remote_commands_refuse_without_record_capacity(
    tmp_path: Path,
    monkeypatch,
    capsys,
    command: str,
    ledger_builder,
    expected_rows: int,
    expected_identities: int,
    expected_refusal: str,
) -> None:
    store_a, store_b = ledger_builder(tmp_path)
    current_store = ScientificEvidenceStore(tmp_path / "current.db")
    aggregate = current_store.acquisition_budget(
        (store_a.database, store_b.database)
    )
    assert aggregate["source_records"] == expected_rows
    assert aggregate["unique_identities"] == expected_identities

    counts_a = store_a.counts()
    counts_b = store_b.counts()
    hash_a = _sha256(store_a.database)
    hash_b = _sha256(store_b.database)
    hash_current = _sha256(current_store.database)
    elicit_constructions: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def unexpected_elicit_client(*args: object, **kwargs: object) -> None:
        elicit_constructions.append((args, kwargs))
        raise AssertionError("ElicitClient construction must follow the cap check")

    monkeypatch.setattr(cli_module, "ElicitClient", unexpected_elicit_client)

    if command == "start-report":
        command_argv = ["start-report", "synthetic neutral question"]
    elif command == "start-review":
        protocol = tmp_path / "protocol.json"
        protocol.write_text(
            json.dumps({"researchQuestion": "synthetic neutral question"}),
            encoding="utf-8",
        )
        command_argv = ["start-review", str(protocol)]
    else:
        command_argv = ["session", "synthetic-session-id"]

    exit_code = cli_module.main(
        [
            "--database",
            str(current_store.database),
            "--aggregate-database",
            str(store_a.database),
            "--aggregate-database",
            str(store_b.database),
            *command_argv,
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert expected_refusal in captured.err
    assert elicit_constructions == []
    assert store_a.counts() == counts_a
    assert store_b.counts() == counts_b
    assert current_store.counts()["source_records"] == 0
    assert current_store.counts()["acquisitions"] == 0
    assert _sha256(store_a.database) == hash_a
    assert _sha256(store_b.database) == hash_b
    assert _sha256(current_store.database) == hash_current
    assert all(
        store.audit()["ok"] is True
        for store in (store_a, store_b, current_store)
    )
