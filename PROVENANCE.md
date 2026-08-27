# Provenance

This standalone repository reuses the canonical GIGA Scientific Research module created by CSQ11.

- Source repository: `Pepstee/giga-user`
- Source branch: `codex/csq-cleanup-sequence`
- Module-introduction commit: `6f728169e526fa218a10561e4610f3fe72b04ac3`
- Verified source worktree head at extraction: `e361d7fc025c11b1b21c0fc33bc14a58c9a1de44`
- Original canonical root: `modules/research/`
- Original completion receipt: `architecture/modularization/receipts/CSQ11_RESEARCH_MODULE.v1.json`

The twelve core Python files were copied directly; three redundant trailing blank lines were
normalized so the standalone Git diff-hygiene gate passes. The standalone packaging, neutral
example, documentation, and module registry were added or adapted for an independent release
lifecycle. Private/local campaign databases and the GIGA `research/` evidence tree were
deliberately excluded.
