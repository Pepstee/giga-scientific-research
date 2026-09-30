from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

import pytest

from modules.research.core.scientific_evidence import (
    ScientificEvidenceStore,
)
from modules.research.core.scientific_evidence_cli import main as cli_main
from modules.research.core.scientific_providers import (
    ClinicalTrialsClient,
    CrossrefClient,
    EuropePMCClient,
    OpenAlexClient,
    PubMedClient,
    ScientificProviderConfigurationError,
    SemanticScholarClient,
    UnpaywallClient,
)


def test_europe_pmc_projection_preserves_raw_response_and_filters_retracted() -> None:
    captured: dict = {}
    raw = {
        "resultList": {
            "result": [
                {
                    "title": "Open clinical study",
                    "authorString": "A Researcher, B Researcher",
                    "pubYear": "2025",
                    "doi": "10.1000/europe",
                    "pmid": "100",
                    "abstractText": "Abstract.",
                    "journalTitle": "Journal",
                    "isRetracted": "N",
                },
                {
                    "title": "Retracted study",
                    "pubYear": "2024",
                    "isRetracted": "Y",
                },
            ]
        }
    }

    def transport(method, url, body, headers, timeout):
        captured.update(url=url, headers=headers)
        return raw

    request, response = EuropePMCClient(transport=transport).search_papers(
        "skin texture",
        max_results=10,
    )
    assert captured["url"].startswith(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
    )
    assert request["provider"] == "europe_pmc"
    assert [paper["title"] for paper in response["papers"]] == [
        "Open clinical study"
    ]
    assert response["providerRaw"] == raw


def test_pubmed_uses_official_endpoints_and_never_persists_key_or_email() -> None:
    urls: list[str] = []

    def transport(method, url, body, headers, timeout):
        urls.append(url)
        if "/esearch.fcgi?" in url:
            return {"esearchresult": {"idlist": ["123"]}}
        return {
            "result": {
                "uids": ["123"],
                "123": {
                    "uid": "123",
                    "title": "PubMed result",
                    "pubdate": "2026 Jan",
                    "authors": [{"name": "A Author"}],
                    "articleids": [
                        {"idtype": "doi", "value": "10.1000/pubmed"}
                    ],
                    "fulljournalname": "Medical Journal",
                },
            }
        }

    request, response = PubMedClient(
        api_key="private-key",
        contact_email="person@example.com",
        transport=transport,
    ).search_papers("scar treatment")
    assert len(urls) == 2
    assert all(url.startswith("https://eutils.ncbi.nlm.nih.gov/") for url in urls)
    assert "private-key" not in json.dumps(request)
    assert "person@example.com" not in json.dumps(request)
    decoded_term = urllib.parse.parse_qs(urllib.parse.urlsplit(urls[0]).query)["term"][0]
    assert "hasretractionin" in decoded_term
    assert response["papers"][0]["pmid"] == "123"
    assert response["papers"][0]["doi"] == "10.1000/pubmed"


def test_semantic_scholar_and_openalex_normalise_citation_graph_records() -> None:
    semantic_raw = {
        "total": 1,
        "offset": 0,
        "data": [
            {
                "paperId": "s2",
                "title": "Semantic result",
                "year": 2025,
                "authors": [{"name": "S Author"}],
                "externalIds": {"DOI": "10.1000/s2", "PubMed": "456"},
                "citationCount": 9,
                "url": "https://www.semanticscholar.org/paper/s2",
            }
        ],
    }
    request, response = SemanticScholarClient(
        api_key="s2-secret",
        transport=lambda *args: semantic_raw,
    ).search_papers("collagen", max_results=1)
    assert "s2-secret" not in json.dumps(request)
    assert response["papers"][0]["citedByCount"] == 9

    openalex_raw = {
        "meta": {"count": 1},
        "results": [
            {
                "id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.1000/openalex",
                "title": "OpenAlex result",
                "publication_year": 2024,
                "authorships": [
                    {"author": {"display_name": "O Author"}}
                ],
                "abstract_inverted_index": {
                    "A": [0],
                    "result": [1],
                },
                "primary_location": {
                    "source": {"display_name": "Open Journal"}
                },
                "cited_by_count": 11,
                "is_retracted": False,
            }
        ],
    }
    request, response = OpenAlexClient(
        api_key="oa-secret",
        transport=lambda *args: openalex_raw,
    ).search_papers("photoaging", max_results=1)
    assert "oa-secret" not in json.dumps(request)
    assert response["papers"][0]["abstract"] == "A result"
    assert response["papers"][0]["doi"] == "10.1000/openalex"


def test_clinical_trials_search_projects_nct_identity_and_preserves_study() -> None:
    raw = {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT00000001",
                        "briefTitle": "Registered scar trial",
                    }
                }
            }
        ]
    }
    request, response = ClinicalTrialsClient(
        transport=lambda *args: raw
    ).search_trials("keloid")
    assert request["provider"] == "clinical_trials"
    assert response["trials"][0]["nctId"] == "NCT00000001"
    assert response["providerRaw"] == raw


def test_crossref_and_unpaywall_resolve_only_lawful_metadata_and_locations() -> None:
    crossref_raw = {
        "status": "ok",
        "message": {
            "title": ["Resolved work"],
            "DOI": "10.1000/resolved",
            "published": {"date-parts": [[2025, 1, 1]]},
            "author": [{"given": "A", "family": "Author"}],
            "container-title": ["Journal"],
            "URL": "https://doi.org/10.1000/resolved",
        },
    }
    request, response = CrossrefClient(
        contact_email="person@example.com",
        transport=lambda *args: crossref_raw,
    ).lookup_doi("https://doi.org/10.1000/resolved")
    assert "person@example.com" not in json.dumps(request)
    assert response["papers"][0]["year"] == 2025

    unpaywall_raw = {
        "title": "Resolved work",
        "doi": "10.1000/resolved",
        "year": 2025,
        "journal_name": "Journal",
        "is_oa": True,
        "oa_status": "gold",
        "oa_locations": [
            {
                "url_for_landing_page": "https://repository.example/work",
                "url_for_pdf": "https://repository.example/work.pdf",
            }
        ],
    }
    request, response = UnpaywallClient(
        contact_email="person@example.com",
        transport=lambda *args: unpaywall_raw,
    ).lookup_doi("10.1000/resolved")
    assert "person@example.com" not in json.dumps(request)
    assert response["papers"][0]["isOpenAccess"] is True
    assert response["papers"][0]["urls"] == [
        "https://repository.example/work",
        "https://repository.example/work.pdf",
    ]


def test_clients_reject_nonofficial_hosts_and_missing_required_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    monkeypatch.delenv("SCIENTIFIC_CONTACT_EMAIL", raising=False)
    with pytest.raises(ScientificProviderConfigurationError, match="official"):
        EuropePMCClient(base_url="https://example.test/api")
    assert OpenAlexClient(api_key="").api_key == ""
    with pytest.raises(ScientificProviderConfigurationError, match="UNPAYWALL_EMAIL"):
        UnpaywallClient(contact_email="")


def test_store_distinguishes_provider_provenance_for_same_query(tmp_path: Path) -> None:
    store = ScientificEvidenceStore(tmp_path / "private" / "evidence.sqlite3")
    request = {
        "provider": "europe_pmc",
        "endpoint": "search/papers",
        "query": "test",
        "maxResults": 1,
    }
    _, _, source_ids = store.record_acquisition(
        provider="europe_pmc",
        endpoint="search/papers",
        request=request,
        response={"papers": [{"title": "One", "doi": "10.1000/one"}]},
    )
    assert store.get_source(source_ids[0])["provider"] == "europe_pmc"
    assert store.acquisitions_for_request(
        provider="pubmed",
        endpoint="search/papers",
        request=request,
    ) == []
    assert store.audit()["ok"] is True


def test_cli_imports_lawfully_held_document_by_hash(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document = tmp_path / "paper.pdf"
    document.write_bytes(b"%PDF-1.4 lawful local test")
    database = tmp_path / "scientific.sqlite3"
    assert (
        cli_main(
            [
                "--database",
                str(database),
                "import-document",
                str(document),
                "--title",
                "Lawfully held paper",
                "--year",
                "2024",
                "--doi",
                "10.1000/local",
            ]
        )
        == 0
    )
    imported = json.loads(capsys.readouterr().out)
    source_id = imported["source_record_ids"][0]
    store = ScientificEvidenceStore(database)
    source = store.get_source(source_id)
    assert source["provider"] == "local_file"
    assert source["raw"]["localDocument"]["filename"] == "paper.pdf"
    assert "path" not in source["raw"]["localDocument"]
