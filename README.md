# GIGA Scientific Research

Deterministic scientific acquisition and review infrastructure extracted from the GIGA control
plane. It collects public research records, preserves an append-only evidence trail, compiles and
executes bounded search campaigns, deduplicates and screens records, manages human review state,
routes lawful full-text retrieval, measures search saturation, and fails closed before synthesis or
ranking when required evidence is incomplete.

## What it provides

- provider adapters for Elicit, PubMed, Europe PMC, ClinicalTrials.gov, Semantic Scholar,
  OpenAlex, Crossref, Unpaywall, and local files;
- deterministic query compilation and resumable multi-provider acquisition campaigns;
- owner-only SQLite evidence and review ledgers with hash-chain audits;
- exact and near-duplicate grouping, screening queues, PRISMA-style counts, and negative controls;
- metadata, abstract, full-text extraction, appraisal, comparison, and synthesis gates;
- explicit A0 candidate outputs: model-generated proposals never become medical advice or
  autonomous action authority.

The repository intentionally contains no GIGA personal memory, private research database, raw
campaign corpus, credentials, identity data, or autonomous-action authority.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest

giga-scientific-evidence --help
giga-scientific-pipeline --help
giga-scientific-campaign --help
```

A neutral four-phase campaign is provided at
[`examples/systematic-review-campaign.json`](examples/systematic-review-campaign.json).

## Repository layout

- `modules/research/core/` — canonical research implementation;
- `modules/research/scripts/` — operator-facing commands;
- `modules/research/tests/` — focused deterministic tests;
- `modules/platform/contracts/` — closed scientific evidence schemas;
- `scripts/` — compatibility launchers retained from the original GIGA module;
- `examples/` — public, synthetic examples only.

## Safety boundary

The module can acquire and structure scientific evidence. It cannot authorize medical treatment,
publish findings, access personal memory, or act autonomously. Full-text extraction, appraisal,
comparison, and synthesis remain gated by explicit completeness and human-signoff contracts.

See [PROVENANCE.md](PROVENANCE.md) for the exact source history.
