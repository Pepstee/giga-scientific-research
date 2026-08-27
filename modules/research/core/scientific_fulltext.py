"""Lawful, deterministic full-text planning and official PMC retrieval."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .scientific_evidence import ScientificEvidenceError, canonical_json, payload_hash


FULLTEXT_PLAN_SCHEMA = "giga.scientific-fulltext-plan.v1"
FULLTEXT_RECEIPT_SCHEMA = "giga.scientific-fulltext-receipt.v1"
PMC_EXTRACTION_SCHEMA = "giga.scientific-pmc-extraction.v1"
PMC_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PMCID_PATTERN = re.compile(r"^PMC\d+$", re.IGNORECASE)
MAX_FULLTEXT_BYTES = 100 * 1024 * 1024


BinaryTransport = Callable[[str, int], tuple[bytes, Mapping[str, str]]]


def _binary_transport(url: str, timeout: int) -> tuple[bytes, Mapping[str, str]]:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Accept": "application/xml,text/xml",
            "User-Agent": "giga-scientific-evidence/3.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > MAX_FULLTEXT_BYTES:
                raise ScientificEvidenceError("PMC full text exceeds size limit")
            data = response.read(MAX_FULLTEXT_BYTES + 1)
            headers = dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        raise ScientificEvidenceError(
            f"Europe PMC full-text request failed with HTTP {exc.code}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ScientificEvidenceError(
            f"Europe PMC full-text connection failed: {exc.reason}"
        ) from exc
    if len(data) > MAX_FULLTEXT_BYTES:
        raise ScientificEvidenceError("PMC full text exceeds size limit")
    return data, headers


def _https_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return urllib.parse.urlunsplit(parsed)


def _route(
    route_type: str,
    *,
    priority: int,
    url: str | None,
    automated: bool,
    notes: str,
) -> dict[str, Any]:
    result = {
        "route_type": route_type,
        "priority": priority,
        "url": url,
        "automated": automated,
        "notes": notes,
    }
    result["route_id"] = payload_hash(result)
    return result


def plan_full_text(
    groups: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    proposal_by_group = {item["group_id"]: item for item in proposals}
    plans: list[dict[str, Any]] = []
    for group in groups:
        proposal = proposal_by_group[group["group_id"]]
        if proposal["proposed_status"] == "auto_exclude":
            continue
        routes: list[dict[str, Any]] = []
        pmcid = group.get("pmcid")
        if isinstance(pmcid, str) and PMCID_PATTERN.fullmatch(pmcid):
            routes.append(
                _route(
                    "europe_pmc_fulltext_xml",
                    priority=10,
                    url=f"{PMC_BASE_URL}/{pmcid.upper()}/fullTextXML",
                    automated=True,
                    notes="Official Europe PMC XML; explicit execution required.",
                )
            )
        for url in group.get("urls", []):
            safe = _https_url(url)
            if safe is None:
                continue
            if safe.startswith("https://doi.org/"):
                continue
            routes.append(
                _route(
                    "existing_lawful_oa_location",
                    priority=20,
                    url=safe,
                    automated=False,
                    notes="Provider-declared location; licence and file identity require verification.",
                )
            )
        doi = group.get("doi")
        if doi:
            routes.extend(
                [
                    _route(
                        "unpaywall_lookup",
                        priority=30,
                        url=None,
                        automated=False,
                        notes="Query Unpaywall with configured contact email; do not persist email.",
                    ),
                    _route(
                        "doi_publisher_location",
                        priority=40,
                        url=f"https://doi.org/{doi}",
                        automated=False,
                        notes="Publisher landing page; access rights are not assumed.",
                    ),
                ]
            )
        routes.extend(
            [
                _route(
                    "institutional_or_public_library",
                    priority=50,
                    url=None,
                    automated=False,
                    notes="Use lawful institutional or public-library access.",
                ),
                _route(
                    "author_manuscript_request",
                    priority=60,
                    url=None,
                    automated=False,
                    notes="Search an author repository or request a manuscript from the author.",
                ),
            ]
        )
        unique = {
            (route["route_type"], route["url"]): route for route in routes
        }
        ordered = sorted(
            unique.values(),
            key=lambda item: (item["priority"], item["route_type"], item["url"] or ""),
        )
        plan = {
            "schema_version": FULLTEXT_PLAN_SCHEMA,
            "group_id": group["group_id"],
            "title": group["title"],
            "doi": doi,
            "pmcid": pmcid,
            "proposed_eligibility": proposal["proposed_status"],
            "routes": ordered,
            "full_text_status": "not_obtained",
            "abstract_substitution_forbidden": True,
            "paywall_circumvention_forbidden": True,
        }
        plan["plan_sha256"] = payload_hash(plan)
        plans.append(plan)
    return sorted(plans, key=lambda item: item["group_id"])


def fetch_europe_pmc_xml(
    pmcid: str,
    destination: Path,
    *,
    timeout: int = 60,
    transport: BinaryTransport | None = None,
) -> dict[str, Any]:
    identifier = pmcid.upper()
    if not PMCID_PATTERN.fullmatch(identifier):
        raise ScientificEvidenceError("pmcid must match PMC followed by digits")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 1200:
        raise ScientificEvidenceError("timeout must be an integer from 1 to 1200")
    url = f"{PMC_BASE_URL}/{identifier}/fullTextXML"
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "www.ebi.ac.uk"
        or parsed.path
        != f"/europepmc/webservices/rest/{identifier}/fullTextXML"
    ):
        raise ScientificEvidenceError("Europe PMC URL failed official-host validation")
    data, headers = (transport or _binary_transport)(url, timeout)
    if not isinstance(data, bytes):
        raise ScientificEvidenceError("full-text transport must return bytes")
    if len(data) > MAX_FULLTEXT_BYTES:
        raise ScientificEvidenceError("PMC full text exceeds size limit")
    lowered = data[:2_000].lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise ScientificEvidenceError("DTD/entity declarations are forbidden")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ScientificEvidenceError("Europe PMC response is not valid XML") from exc
    tag = root.tag.rsplit("}", 1)[-1].casefold()
    if tag not in {"article", "pmc-articleset"}:
        raise ScientificEvidenceError("Europe PMC XML has an unexpected root element")
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    os.chmod(destination, 0o600)
    receipt = {
        "schema_version": FULLTEXT_RECEIPT_SCHEMA,
        "provider": "europe_pmc",
        "pmcid": identifier,
        "source_url": url,
        "destination_filename": destination.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "content_type": headers.get("Content-Type") or headers.get("content-type"),
        "lawful_route": "official_open_repository",
        "full_text_status": "obtained_unappraised",
        "human_verified": False,
    }
    receipt["receipt_sha256"] = payload_hash(receipt)
    return receipt


def _element_text(element: ET.Element) -> str:
    return " ".join(" ".join(element.itertext()).split())


def extract_pmc_xml(xml_path: Path) -> dict[str, Any]:
    path = xml_path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ScientificEvidenceError("PMC XML path must be a regular, non-symlink file")
    data = path.read_bytes()
    if len(data) > MAX_FULLTEXT_BYTES:
        raise ScientificEvidenceError("PMC XML exceeds size limit")
    lowered = data[:2_000].lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise ScientificEvidenceError("DTD/entity declarations are forbidden")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ScientificEvidenceError("PMC document is not valid XML") from exc
    sections: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, element in enumerate(root.iter()):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag not in {"article-title", "abstract", "sec", "p", "table-wrap"}:
            continue
        text = _element_text(element)
        if not text or text in seen:
            continue
        seen.add(text)
        section = {
            "locator": f"xml:{tag}[{index}]",
            "kind": tag,
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        sections.append(section)
    extraction = {
        "schema_version": PMC_EXTRACTION_SCHEMA,
        "source_filename": path.name,
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "sections": sections,
        "section_count": len(sections),
        "full_text_verified": False,
        "extraction_requires_human_verification": True,
        "effect_estimate_authority": False,
    }
    extraction["extraction_sha256"] = payload_hash(extraction)
    return extraction


def write_fulltext_plan(
    plans: Sequence[Mapping[str, Any]],
    path: Path,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for plan in plans:
            handle.write(canonical_json(plan) + "\n")
    result = {
        "path": path.name,
        "records": len(plans),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return result
