# Scientific Research module

This module owns the deterministic scientific-research pipeline: provider
adapters, query compilation, acquisition campaigns, screening, human-review
state, lawful full-text routing, saturation checks and fail-closed readiness
gates.

The canonical implementation is under `core/`; operator-facing commands are
under `scripts/`; focused tests are under `tests/`. The former root commands
remain thin compatibility launchers during the cleanup sequence.

Provider records and model output remain claims or proposals until the relevant
deterministic and human-review gates pass. The standalone repository deliberately
excludes the original protected `research/` evidence tree and all private/local
databases. This module has neither personal-memory authority nor autonomous-action
authority.
