# PATCH-NOTES — provenance & namespace defaulting

## Problem

Production showed 39 of 41 `memories` rows stamped `source="user"`, though the
vast majority were machine-derived captures (deployment states, scan results,
park notes, inferred context), not operator statements. Separately, automatic
captures landed in a junk `default` namespace and a vague `agent-memory` bucket.

Provenance integrity is the store's core selling point: automatic captures must
not claim the operator said them.

## Root cause

The public capture paths (`PalmPilot.remember`, `VaultMemoryProvider.store`,
`MCPPalmServer.tool_remember`, `mempalm remember`) already defaulted
`source="inferred"` — the "user" stamp came from the deployed Hermes bridge
(`plugins/memory/palmpilot/__init__.py` in the Hermes install), which hardcoded
`source="user"` on every `mem_add`. The namespace defaults in the same paths
hardcoded `"default"` (and the bridge passed `"default"` / `"agent-memory"`).

## Change

- `source` defaults to `"inferred"` on every capture path; `"user"` is stamped
  only via an explicit `source="user"` (operator-attributed). The Hermes bridge
  now passes the caller's optional `source` through instead of hardcoding
  `"user"`, and its `mem_add` schema documents the provenance contract.
- Added `agent_palm.extract.classify_namespace()` — a deterministic, no-ML
  keyword/allowlist classifier into a small canonical set (`infra`, `home`,
  `people`, `projects`, `estate-events`, fallback `agent-memory`). Word-boundary
  matching avoids substring false positives (e.g. `port` in `report`).
- Every capture path now derives the namespace via the classifier when none is
  given, instead of defaulting to `"default"`. Explicit namespaces are always
  honoured.

## Behavioural change

| Case | Before | After |
|------|--------|-------|
| automatic capture, no source | (bridge hardcoded) `user` | `inferred` |
| operator-dictated capture | n/a (no signal) | `source="user"` via explicit param |
| capture, no namespace | `default` / `agent-memory` | classified (`infra`/`home`/...) |
| capture, explicit namespace | honoured | honoured (unchanged) |

## Tests

`tests/test_provenance.py` (13 tests) locks in the classifier, the
`source="inferred"` default across `PalmPilot` / Hermes provider / MCP, the
explicit `source="user"` passthrough, and classified namespace defaulting.
Full suite: 31 → 44 passed.
