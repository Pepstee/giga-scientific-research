from __future__ import annotations

import hashlib
from pathlib import Path

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
