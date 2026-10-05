# GIGA Scientific Research

Deterministic scientific acquisition and review infrastructure extracted from the GIGA control
plane. It collects public research records, preserves an append-only evidence trail, compiles and
executes bounded search campaigns, deduplicates and screens records, manages human review state and
a bounded automated abstract-review mode, routes lawful full-text retrieval, measures search
saturation, and fails closed before ranking when required evidence is incomplete.

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
publish findings, access personal memory, or act autonomously. Automated abstract review can emit
AI-attributed, source-linked eligibility and descriptive synthesis for an existing retained public
snapshot; its output is never a human decision. It does not acquire records or authorize full-text
extraction, risk-of-bias appraisal, effect estimation, ranking, or recommendations. Those steps
remain gated by their completeness and human-signoff contracts.

### Bounded automated abstract review

An operator-authorized run can use `--review-mode automated` with a hash-bound automated review
input, the matching protocol, and the duplicate-decision report. Supply the retained groups,
proposals, and feature files. Choose a new output directory for each run. The retained 18-group
ArtVault run receipt is
`/srv/artvault/sandboxes/giga-scientific-research-cycle/run/review-funnel/review-inventory-20260930t1612z/automated-review-20261005/AUTOMATED_REVIEW_RUN.json`;
its `cwd`, `command_argv`, and input/output hashes bind the actual invocation. To reproduce it,
reuse those inputs and working directory, set the `RESEARCH_*` input variables to the recorded paths,
and choose a fresh `RESEARCH_OUTPUT_DIR`.

```bash
RESEARCH_OUTPUT_DIR="/path/to/a/new/automated-review-output"
test ! -e "$RESEARCH_OUTPUT_DIR"
python -m modules.research.core.scientific_pipeline_cli review-funnel \
  "$RESEARCH_GROUPS" "$RESEARCH_PROPOSALS" "$RESEARCH_FEATURES" \
  "$RESEARCH_OUTPUT_DIR" \
  --review-mode automated \
  --automated-review "$RESEARCH_AUTOMATED_REVIEW_INPUT" \
  --protocol "$RESEARCH_PROTOCOL" \
  --duplicate-decision-report "$RESEARCH_DUPLICATE_REPORT"
```

See [PROVENANCE.md](PROVENANCE.md) for the exact source history.
