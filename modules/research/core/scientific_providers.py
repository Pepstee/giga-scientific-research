"""Lawful public scientific-discovery and open-access provider adapters.

Every adapter returns a provider-neutral projection for the evidence ledger while retaining
the complete JSON response under ``providerRaw``. API keys and contact email addresses are
used only for transport and are deliberately excluded from the persisted request envelope.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Mapping, Sequence

from .scientific_evidence import ScientificEvidenceError


PAPER_SEARCH_PROVIDERS = {
    "europe_pmc",
    "openalex",
    "pubmed",
    "semantic_scholar",
}
TRIAL_SEARCH_PROVIDERS = {"clinical_trials"}
DOI_LOOKUP_PROVIDERS = {"crossref", "unpaywall"}
DEFAULT_FREE_PAPER_PROVIDERS = (
    "europe_pmc",
    "pubmed",
)
PROVIDER_FILTERS = {
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
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(1[6-9]\d{2}|20\d{2}|21\d{2}|2200)\b")


class ScientificProviderConfigurationError(ScientificEvidenceError):
    """Raised when a public provider cannot be configured safely."""


class ScientificProviderAPIError(ScientificEvidenceError):
    """A provider-neutral API error safe to persist or surface."""

    def __init__(self, provider: str, status: int | None, code: str, message: str):
        self.provider = provider
        self.status = status
        self.code = code
        self.message = message
        prefix = f"{provider} API {status}" if status is not None else provider
        super().__init__(f"{prefix} [{code}]: {message}")


ProviderTransport = Callable[
    [str, str, Mapping[str, Any] | None, Mapping[str, str], int],
    Mapping[str, Any],
]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_transport(
    method: str,
    url: str,
    body: Mapping[str, Any] | None,
    headers: Mapping[str, str],
    timeout: int,
) -> Mapping[str, Any]:
    data = _canonical_json(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers=dict(headers),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, Mapping):
                message = str(
                    parsed.get("message")
                    or parsed.get("error")
                    or parsed.get("detail")
                    or "request failed"
                )
            else:
                message = "request failed"
        except json.JSONDecodeError:
            message = "request failed"
        raise ScientificProviderAPIError(
            "scientific provider",
            exc.code,
            f"http_{exc.code}",
            message[:1_000],
        ) from exc
    except urllib.error.URLError as exc:
        raise ScientificProviderAPIError(
            "scientific provider",
            None,
            "connection_error",
            str(exc.reason)[:1_000],
        ) from exc
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ScientificProviderAPIError(
            "scientific provider",
            None,
            "invalid_json",
            "provider returned invalid JSON",
        ) from exc
    if not isinstance(result, Mapping):
        raise ScientificProviderAPIError(
            "scientific provider",
            None,
            "invalid_response",
            "provider response must be an object",
        )
    return dict(result)


def _text(value: Any, field: str, *, maximum: int = 20_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScientificEvidenceError(f"{field} must be non-empty text")
    result = value.strip()
    if len(result) > maximum:
        raise ScientificEvidenceError(f"{field} exceeds {maximum} characters")
    return result


def _max_results(value: Any, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ScientificEvidenceError(
            f"max_results must be an integer from 1 to {maximum}"
        )
    return value


def _filters(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {"retracted": "exclude_retracted"}
    if not isinstance(value, Mapping):
        raise ScientificEvidenceError("filters must be an object")
    unknown = set(value) - PROVIDER_FILTERS
    if unknown:
        raise ScientificEvidenceError(f"unsupported filter(s): {sorted(unknown)}")
    result = dict(value)
    result.setdefault("retracted", "exclude_retracted")
    return result


def _normalise_year(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value if 1600 <= value <= 2200 else None
    if isinstance(value, str):
        match = YEAR_PATTERN.search(value)
        if match:
            return int(match.group(1))
    return None


def _normalise_doi(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    result = value.strip()
    lowered = result.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if lowered.startswith(prefix):
            result = result[len(prefix) :]
            break
    result = result.strip()
    return result if DOI_PATTERN.match(result) else None


def _doi(value: Any) -> str:
    result = _normalise_doi(value)
    if result is None:
        raise ScientificEvidenceError("doi must be a valid DOI")
    return result


def _title(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def _author_names(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result: list[str] = []
    for author in value:
        if isinstance(author, str) and author.strip():
            result.append(author.strip())
        elif isinstance(author, Mapping):
            name = author.get("name") or author.get("display_name")
            if isinstance(name, str) and name.strip():
                result.append(name.strip())
    return result


def _type_tags(filters: Mapping[str, Any]) -> list[str]:
    value = filters.get("typeTags", [])
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _request_envelope(
    provider: str,
    endpoint: str,
    query: str,
    max_results: int,
    filters: Mapping[str, Any],
    *,
    contact_email_provided: bool = False,
    api_key_provided: bool = False,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "endpoint": endpoint,
        "query": query,
        "maxResults": max_results,
        "filters": dict(filters),
        "contactEmailProvided": contact_email_provided,
        "apiKeyProvided": api_key_provided,
    }


class _OfficialJSONClient:
    provider = "scientific_provider"
    official_base_url = ""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: int = 60,
        transport: ProviderTransport | None = None,
    ):
        selected = (base_url or self.official_base_url).rstrip("/")
        actual = urllib.parse.urlsplit(selected)
        expected = urllib.parse.urlsplit(self.official_base_url.rstrip("/"))
        if (
            actual.scheme != "https"
            or actual.hostname != expected.hostname
            or actual.port not in {None, 443}
            or actual.username is not None
            or actual.password is not None
            or actual.path.rstrip("/") != expected.path.rstrip("/")
            or actual.query
            or actual.fragment
        ):
            raise ScientificProviderConfigurationError(
                f"{self.provider} base URL must be the official "
                f"{self.official_base_url} endpoint"
            )
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 1200:
            raise ScientificEvidenceError("timeout must be an integer from 1 to 1200")
        self.base_url = selected
        self.timeout = timeout
        self._transport = transport or _json_transport

    def _get(
        self,
        path: str,
        params: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
        secrets: Sequence[str] = (),
    ) -> Mapping[str, Any]:
        if not path.startswith("/") or "://" in path:
            raise ScientificEvidenceError("provider path must be a relative API path")
        query = urllib.parse.urlencode(
            [(key, value) for key, value in params.items() if value is not None],
            doseq=True,
        )
        url = self.base_url + path + ("?" + query if query else "")
        request_headers = {
            "Accept": "application/json",
            "User-Agent": "giga-scientific-evidence/2.0",
            **dict(headers or {}),
        }
        try:
            result = self._transport(
                "GET",
                url,
                None,
                request_headers,
                self.timeout,
            )
        except ScientificProviderAPIError as exc:
            message = exc.message
            for secret in secrets:
                if secret:
                    message = message.replace(secret, "[REDACTED]")
            raise ScientificProviderAPIError(
                self.provider,
                exc.status,
                exc.code,
                message,
            ) from exc
        if not isinstance(result, Mapping):
            raise ScientificProviderAPIError(
                self.provider,
                None,
                "invalid_response",
                "provider response must be an object",
            )
        return dict(result)


class EuropePMCClient(_OfficialJSONClient):
    provider = "europe_pmc"
    official_base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest"

    @staticmethod
    def paper_search_request(
        query: str,
        *,
        max_results: int = 100,
        filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _request_envelope(
            "europe_pmc",
            "search/papers",
            _text(query, "query"),
            _max_results(max_results, maximum=1000),
            _filters(filters),
        )

    def search_papers(
        self,
        query: str,
        *,
        max_results: int = 100,
        filters: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        request = self.paper_search_request(
            query,
            max_results=max_results,
            filters=filters,
        )
        params = {
            "query": request["query"],
            "format": "json",
            "resultType": "core",
            "pageSize": request["maxResults"],
        }
        raw = self._get("/search", params)
        result_list = raw.get("resultList", {})
        records = result_list.get("result", []) if isinstance(result_list, Mapping) else []
        if not isinstance(records, list):
            raise ScientificProviderAPIError(
                self.provider,
                None,
                "invalid_response",
                "resultList.result must be an array",
            )
        papers: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            if (
                request["filters"].get("retracted") == "exclude_retracted"
                and str(record.get("isRetracted", "")).upper() == "Y"
            ):
                continue
            title = _title(record.get("title"))
            if title is None:
                continue
            links = record.get("fullTextUrlList", {})
            urls: list[str] = []
            if isinstance(links, Mapping):
                for link in links.get("fullTextUrl", []) or []:
                    if isinstance(link, Mapping) and isinstance(link.get("url"), str):
                        urls.append(link["url"])
            papers.append(
                {
                    "title": title,
                    "authors": [
                        item.strip()
                        for item in str(record.get("authorString") or "").split(",")
                        if item.strip()
                    ],
                    "year": _normalise_year(record.get("pubYear")),
                    "doi": _normalise_doi(record.get("doi")),
                    "pmid": str(record["pmid"]) if record.get("pmid") else None,
                    "pmcid": record.get("pmcid"),
                    "abstract": record.get("abstractText"),
                    "venue": record.get("journalTitle"),
                    "citedByCount": record.get("citedByCount"),
                    "urls": urls,
                    "isOpenAccess": record.get("isOpenAccess"),
                    "sourceProvider": self.provider,
                    "providerRecord": dict(record),
                }
            )
        return request, {"papers": papers, "providerRaw": raw}


class PubMedClient(_OfficialJSONClient):
    provider = "pubmed"
    official_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        contact_email: str | None = None,
        base_url: str | None = None,
        timeout: int = 60,
        transport: ProviderTransport | None = None,
    ):
        super().__init__(base_url=base_url, timeout=timeout, transport=transport)
        self.api_key = (api_key or os.environ.get("NCBI_API_KEY") or "").strip()
        self.contact_email = (
            contact_email
            or os.environ.get("SCIENTIFIC_CONTACT_EMAIL")
            or os.environ.get("NCBI_EMAIL")
            or ""
        ).strip()

    @staticmethod
    def paper_search_request(
        query: str,
        *,
        max_results: int = 100,
        filters: Mapping[str, Any] | None = None,
        contact_email_provided: bool = False,
        api_key_provided: bool = False,
    ) -> dict[str, Any]:
        return _request_envelope(
            "pubmed",
            "search/papers",
            _text(query, "query"),
            _max_results(max_results, maximum=1000),
            _filters(filters),
            contact_email_provided=contact_email_provided,
            api_key_provided=api_key_provided,
        )

    @staticmethod
    def _effective_query(query: str, filters: Mapping[str, Any]) -> str:
        parts = [f"({query})"]
        if filters.get("retracted") == "exclude_retracted":
            parts.append("NOT (hasretractionin OR retracted publication[pt])")
        mapped = {
            "Review": "review[pt]",
            "Meta-Analysis": "meta-analysis[pt]",
            "Systematic Review": "systematic[sb]",
            "RCT": "randomized controlled trial[pt]",
            "Longitudinal": "longitudinal studies[mh]",
        }
        tags = [mapped[tag] for tag in _type_tags(filters) if tag in mapped]
        if tags:
            parts.append("AND (" + " OR ".join(tags) + ")")
        if filters.get("minYear") or filters.get("maxYear"):
            minimum = filters.get("minYear", 1600)
            maximum = filters.get("maxYear", 2200)
            parts.append(f"AND {minimum}:{maximum}[dp]")
        return " ".join(parts)

    def search_papers(
        self,
        query: str,
        *,
        max_results: int = 100,
        filters: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        request = self.paper_search_request(
            query,
            max_results=max_results,
            filters=filters,
        )
        shared = {
            "db": "pubmed",
            "retmode": "json",
            "tool": "giga_scientific_evidence",
            "email": self.contact_email or None,
            "api_key": self.api_key or None,
        }
        raw_search = self._get(
            "/esearch.fcgi",
            {
                **shared,
                "term": self._effective_query(
                    request["query"],
                    request["filters"],
                ),
                "retmax": request["maxResults"],
                "sort": "relevance",
            },
            secrets=(self.api_key, self.contact_email),
        )
        search_result = raw_search.get("esearchresult", {})
        identifiers = (
            search_result.get("idlist", []) if isinstance(search_result, Mapping) else []
        )
        if not isinstance(identifiers, list):
            raise ScientificProviderAPIError(
                self.provider,
                None,
                "invalid_response",
                "esearchresult.idlist must be an array",
            )
        summary_pages: list[Mapping[str, Any]] = []
        papers: list[dict[str, Any]] = []
        for start in range(0, len(identifiers), 100):
            batch = identifiers[start : start + 100]
            summary = self._get(
                "/esummary.fcgi",
                {
                    **shared,
                    "id": ",".join(str(item) for item in batch),
                    "version": "2.0",
                },
                secrets=(self.api_key, self.contact_email),
            )
            summary_pages.append(summary)
            result = summary.get("result", {})
            if not isinstance(result, Mapping):
                continue
            for pmid in batch:
                record = result.get(str(pmid))
                if not isinstance(record, Mapping):
                    continue
                title = _title(record.get("title"))
                if title is None:
                    continue
                article_ids = record.get("articleids", [])
                identifier_map: dict[str, str] = {}
                if isinstance(article_ids, list):
                    for item in article_ids:
                        if isinstance(item, Mapping):
                            kind = str(item.get("idtype") or "")
                            value = item.get("value")
                            if kind and isinstance(value, str):
                                identifier_map[kind] = value
                papers.append(
                    {
                        "title": title,
                        "authors": _author_names(record.get("authors")),
                        "year": _normalise_year(record.get("pubdate")),
                        "doi": _normalise_doi(identifier_map.get("doi")),
                        "pmid": str(pmid),
                        "pmcid": identifier_map.get("pmc"),
                        "venue": record.get("fulljournalname"),
                        "publicationTypes": record.get("pubtype", []),
                        "urls": [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"],
                        "sourceProvider": self.provider,
                        "providerRecord": dict(record),
                    }
                )
        return request, {
            "papers": papers,
            "providerRaw": {
                "search": raw_search,
                "summaries": summary_pages,
            },
        }


class SemanticScholarClient(_OfficialJSONClient):
    provider = "semantic_scholar"
    official_base_url = "https://api.semanticscholar.org/graph/v1"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: int = 60,
        transport: ProviderTransport | None = None,
    ):
        super().__init__(base_url=base_url, timeout=timeout, transport=transport)
        self.api_key = (
            api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or ""
        ).strip()

    @staticmethod
    def paper_search_request(
        query: str,
        *,
        max_results: int = 100,
        filters: Mapping[str, Any] | None = None,
        api_key_provided: bool = False,
    ) -> dict[str, Any]:
        return _request_envelope(
            "semantic_scholar",
            "search/papers",
            _text(query, "query"),
            _max_results(max_results, maximum=1000),
            _filters(filters),
            api_key_provided=api_key_provided,
        )

    def search_papers(
        self,
        query: str,
        *,
        max_results: int = 100,
        filters: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        request = self.paper_search_request(
            query,
            max_results=max_results,
            filters=filters,
        )
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        fields = (
            "paperId,title,year,authors,abstract,venue,citationCount,externalIds,"
            "url,openAccessPdf,publicationTypes,publicationDate"
        )
        raw_pages: list[Mapping[str, Any]] = []
        papers: list[dict[str, Any]] = []
        offset = 0
        while len(papers) < request["maxResults"]:
            page_limit = min(100, request["maxResults"] - len(papers))
            raw = self._get(
                "/paper/search",
                {
                    "query": request["query"],
                    "offset": offset,
                    "limit": page_limit,
                    "fields": fields,
                },
                headers=headers,
                secrets=(self.api_key,),
            )
            raw_pages.append(raw)
            records = raw.get("data", [])
            if not isinstance(records, list):
                raise ScientificProviderAPIError(
                    self.provider,
                    None,
                    "invalid_response",
                    "data must be an array",
                )
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                title = _title(record.get("title"))
                if title is None:
                    continue
                external = record.get("externalIds", {})
                external = external if isinstance(external, Mapping) else {}
                open_pdf = record.get("openAccessPdf")
                pdf_url = (
                    open_pdf.get("url")
                    if isinstance(open_pdf, Mapping)
                    else None
                )
                papers.append(
                    {
                        "title": title,
                        "authors": _author_names(record.get("authors")),
                        "year": _normalise_year(record.get("year")),
                        "doi": _normalise_doi(external.get("DOI")),
                        "pmid": str(external["PubMed"])
                        if external.get("PubMed")
                        else None,
                        "abstract": record.get("abstract"),
                        "venue": record.get("venue"),
                        "citedByCount": record.get("citationCount"),
                        "publicationTypes": record.get("publicationTypes"),
                        "urls": [
                            value
                            for value in (record.get("url"), pdf_url)
                            if isinstance(value, str)
                        ],
                        "sourceProvider": self.provider,
                        "providerRecord": dict(record),
                    }
                )
            if not records or raw.get("next") is None:
                break
            offset = int(raw["next"])
        return request, {"papers": papers, "providerRaw": {"pages": raw_pages}}


def _openalex_abstract(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    positions: list[tuple[int, str]] = []
    for word, indices in value.items():
        if not isinstance(word, str) or not isinstance(indices, list):
            continue
        for index in indices:
            if isinstance(index, int) and not isinstance(index, bool):
                positions.append((index, word))
    if not positions:
        return None
    return " ".join(word for _, word in sorted(positions))


class OpenAlexClient(_OfficialJSONClient):
    provider = "openalex"
    official_base_url = "https://api.openalex.org"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: int = 60,
        transport: ProviderTransport | None = None,
    ):
        super().__init__(base_url=base_url, timeout=timeout, transport=transport)
        self.api_key = (api_key or os.environ.get("OPENALEX_API_KEY") or "").strip()

    @staticmethod
    def paper_search_request(
        query: str,
        *,
        max_results: int = 100,
        filters: Mapping[str, Any] | None = None,
        api_key_provided: bool = False,
    ) -> dict[str, Any]:
        return _request_envelope(
            "openalex",
            "search/papers",
            _text(query, "query"),
            _max_results(max_results, maximum=1000),
            _filters(filters),
            api_key_provided=api_key_provided,
        )

    def search_papers(
        self,
        query: str,
        *,
        max_results: int = 100,
        filters: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        request = self.paper_search_request(
            query,
            max_results=max_results,
            filters=filters,
        )
        raw_pages: list[Mapping[str, Any]] = []
        papers: list[dict[str, Any]] = []
        page = 1
        select = (
            "id,doi,title,publication_year,authorships,abstract_inverted_index,"
            "primary_location,cited_by_count,open_access,best_oa_location,"
            "is_retracted,type,ids"
        )
        provider_filter = (
            "is_retracted:false"
            if request["filters"].get("retracted") == "exclude_retracted"
            else None
        )
        while len(papers) < request["maxResults"]:
            page_size = min(100, request["maxResults"] - len(papers))
            raw = self._get(
                "/works",
                {
                    "search": request["query"],
                    "per_page": page_size,
                    "page": page,
                    "select": select,
                    "filter": provider_filter,
                    "api_key": self.api_key or None,
                },
                secrets=(self.api_key,),
            )
            raw_pages.append(raw)
            records = raw.get("results", [])
            if not isinstance(records, list):
                raise ScientificProviderAPIError(
                    self.provider,
                    None,
                    "invalid_response",
                    "results must be an array",
                )
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                title = _title(record.get("title"))
                if title is None:
                    continue
                authors: list[str] = []
                for authorship in record.get("authorships", []) or []:
                    if isinstance(authorship, Mapping):
                        author = authorship.get("author", {})
                        if isinstance(author, Mapping) and isinstance(
                            author.get("display_name"),
                            str,
                        ):
                            authors.append(author["display_name"])
                location = record.get("primary_location", {})
                source = location.get("source", {}) if isinstance(location, Mapping) else {}
                best = record.get("best_oa_location", {})
                urls = [
                    value
                    for value in (
                        record.get("id"),
                        best.get("landing_page_url") if isinstance(best, Mapping) else None,
                        best.get("pdf_url") if isinstance(best, Mapping) else None,
                    )
                    if isinstance(value, str)
                ]
                external_ids = record.get("ids", {})
                pmid = (
                    external_ids.get("pmid")
                    if isinstance(external_ids, Mapping)
                    else None
                )
                if isinstance(pmid, str) and "/" in pmid:
                    pmid = pmid.rsplit("/", 1)[-1]
                papers.append(
                    {
                        "title": title,
                        "authors": authors,
                        "year": _normalise_year(record.get("publication_year")),
                        "doi": _normalise_doi(record.get("doi")),
                        "pmid": pmid,
                        "abstract": _openalex_abstract(
                            record.get("abstract_inverted_index")
                        ),
                        "venue": source.get("display_name")
                        if isinstance(source, Mapping)
                        else None,
                        "citedByCount": record.get("cited_by_count"),
                        "isRetracted": record.get("is_retracted"),
                        "openAccess": record.get("open_access"),
                        "urls": urls,
                        "sourceProvider": self.provider,
                        "providerRecord": dict(record),
                    }
                )
            if not records:
                break
            page += 1
        return request, {"papers": papers, "providerRaw": {"pages": raw_pages}}


class ClinicalTrialsClient(_OfficialJSONClient):
    provider = "clinical_trials"
    official_base_url = "https://clinicaltrials.gov/api/v2"

    @staticmethod
    def trial_search_request(
        query: str,
        *,
        max_results: int = 100,
    ) -> dict[str, Any]:
        return _request_envelope(
            "clinical_trials",
            "search/trials",
            _text(query, "query"),
            _max_results(max_results, maximum=1000),
            {},
        )

    def search_trials(
        self,
        query: str,
        *,
        max_results: int = 100,
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        request = self.trial_search_request(query, max_results=max_results)
        raw = self._get(
            "/studies",
            {
                "query.term": request["query"],
                "pageSize": request["maxResults"],
                "format": "json",
            },
        )
        records = raw.get("studies", [])
        if not isinstance(records, list):
            raise ScientificProviderAPIError(
                self.provider,
                None,
                "invalid_response",
                "studies must be an array",
            )
        trials: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            protocol = record.get("protocolSection", {})
            if not isinstance(protocol, Mapping):
                continue
            identification = protocol.get("identificationModule", {})
            if not isinstance(identification, Mapping):
                continue
            nct_id = identification.get("nctId")
            title = _title(
                identification.get("briefTitle")
                or identification.get("officialTitle")
            )
            if title is None:
                continue
            trials.append(
                {
                    "title": title,
                    "nctId": str(nct_id) if nct_id else None,
                    "urls": [
                        f"https://clinicaltrials.gov/study/{nct_id}"
                    ]
                    if nct_id
                    else [],
                    "sourceProvider": self.provider,
                    "providerRecord": dict(record),
                }
            )
        return request, {"trials": trials, "providerRaw": raw}


class CrossrefClient(_OfficialJSONClient):
    provider = "crossref"
    official_base_url = "https://api.crossref.org"

    def __init__(
        self,
        *,
        contact_email: str | None = None,
        base_url: str | None = None,
        timeout: int = 60,
        transport: ProviderTransport | None = None,
    ):
        super().__init__(base_url=base_url, timeout=timeout, transport=transport)
        self.contact_email = (
            contact_email
            or os.environ.get("SCIENTIFIC_CONTACT_EMAIL")
            or os.environ.get("CROSSREF_EMAIL")
            or ""
        ).strip()

    def lookup_doi(
        self,
        doi: str,
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        identifier = _doi(doi)
        request = {
            "provider": self.provider,
            "endpoint": "lookup/doi",
            "doi": identifier,
            "contactEmailProvided": bool(self.contact_email),
        }
        raw = self._get(
            f"/works/{urllib.parse.quote(identifier, safe='/')}",
            {"mailto": self.contact_email or None},
            secrets=(self.contact_email,),
        )
        message = raw.get("message", {})
        if not isinstance(message, Mapping):
            raise ScientificProviderAPIError(
                self.provider,
                None,
                "invalid_response",
                "message must be an object",
            )
        date_parts = message.get("published", {})
        date_parts = date_parts.get("date-parts", []) if isinstance(date_parts, Mapping) else []
        year = None
        if (
            isinstance(date_parts, list)
            and date_parts
            and isinstance(date_parts[0], list)
            and date_parts[0]
        ):
            year = _normalise_year(date_parts[0][0])
        title = _title(message.get("title"))
        if title is None:
            raise ScientificProviderAPIError(
                self.provider,
                None,
                "invalid_response",
                "Crossref work has no title",
            )
        urls = [message["URL"]] if isinstance(message.get("URL"), str) else []
        for link in message.get("link", []) or []:
            if isinstance(link, Mapping) and isinstance(link.get("URL"), str):
                urls.append(link["URL"])
        paper = {
            "title": title,
            "authors": [
                " ".join(
                    value
                    for value in (
                        str(author.get("given") or "").strip(),
                        str(author.get("family") or "").strip(),
                    )
                    if value
                )
                for author in message.get("author", []) or []
                if isinstance(author, Mapping)
            ],
            "year": year,
            "doi": _normalise_doi(message.get("DOI")) or identifier,
            "abstract": message.get("abstract"),
            "venue": _title(message.get("container-title")),
            "citedByCount": message.get("is-referenced-by-count"),
            "urls": urls,
            "sourceProvider": self.provider,
            "providerRecord": dict(message),
        }
        return request, {"papers": [paper], "providerRaw": raw}


class UnpaywallClient(_OfficialJSONClient):
    provider = "unpaywall"
    official_base_url = "https://api.unpaywall.org/v2"

    def __init__(
        self,
        contact_email: str | None = None,
        *,
        base_url: str | None = None,
        timeout: int = 60,
        transport: ProviderTransport | None = None,
    ):
        super().__init__(base_url=base_url, timeout=timeout, transport=transport)
        self.contact_email = (
            contact_email
            or os.environ.get("UNPAYWALL_EMAIL")
            or os.environ.get("SCIENTIFIC_CONTACT_EMAIL")
            or ""
        ).strip()
        if not self.contact_email:
            raise ScientificProviderConfigurationError(
                "UNPAYWALL_EMAIL or SCIENTIFIC_CONTACT_EMAIL is required by "
                "the Unpaywall API"
            )

    def lookup_doi(
        self,
        doi: str,
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        identifier = _doi(doi)
        request = {
            "provider": self.provider,
            "endpoint": "lookup/doi",
            "doi": identifier,
            "contactEmailProvided": True,
        }
        raw = self._get(
            f"/{urllib.parse.quote(identifier, safe='/')}",
            {"email": self.contact_email},
            secrets=(self.contact_email,),
        )
        title = _title(raw.get("title"))
        if title is None:
            raise ScientificProviderAPIError(
                self.provider,
                None,
                "invalid_response",
                "Unpaywall work has no title",
            )
        locations = raw.get("oa_locations", [])
        urls: list[str] = []
        if isinstance(locations, list):
            for location in locations:
                if not isinstance(location, Mapping):
                    continue
                for key in ("url", "url_for_landing_page", "url_for_pdf"):
                    value = location.get(key)
                    if isinstance(value, str) and value not in urls:
                        urls.append(value)
        paper = {
            "title": title,
            "authors": [],
            "year": _normalise_year(raw.get("year")),
            "doi": _normalise_doi(raw.get("doi")) or identifier,
            "venue": raw.get("journal_name"),
            "isOpenAccess": raw.get("is_oa"),
            "oaStatus": raw.get("oa_status"),
            "urls": urls,
            "sourceProvider": self.provider,
            "providerRecord": dict(raw),
        }
        return request, {"papers": [paper], "providerRaw": raw}


def paper_search_request(
    provider: str,
    query: str,
    *,
    max_results: int,
    filters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if provider == "europe_pmc":
        return EuropePMCClient.paper_search_request(
            query,
            max_results=max_results,
            filters=filters,
        )
    if provider == "pubmed":
        return PubMedClient.paper_search_request(
            query,
            max_results=max_results,
            filters=filters,
        )
    if provider == "semantic_scholar":
        return SemanticScholarClient.paper_search_request(
            query,
            max_results=max_results,
            filters=filters,
        )
    if provider == "openalex":
        return OpenAlexClient.paper_search_request(
            query,
            max_results=max_results,
            filters=filters,
        )
    raise ScientificEvidenceError(f"unsupported paper provider: {provider}")


def create_paper_client(provider: str) -> Any:
    if provider == "europe_pmc":
        return EuropePMCClient()
    if provider == "pubmed":
        return PubMedClient()
    if provider == "semantic_scholar":
        return SemanticScholarClient()
    if provider == "openalex":
        return OpenAlexClient()
    raise ScientificEvidenceError(f"unsupported paper provider: {provider}")


def create_doi_client(provider: str) -> Any:
    if provider == "crossref":
        return CrossrefClient()
    if provider == "unpaywall":
        return UnpaywallClient()
    raise ScientificEvidenceError(f"unsupported DOI provider: {provider}")
