# Compliance Checks

This directory holds output from `mcp__project-compliance__full_compliance_check` calls. One file per phase gate plus a final pre-submission check.

## Files

- `phase-0-init-check.json` — initial check before any planning artifacts written; informed PROJECT.md, REQUIREMENTS.md, ROADMAP.md
- `phase-N-check.md` (created post-phase-N) — gate check for phase N
- `final-check.md` (created pre-submission) — sanity check on shipped artifacts

## Protocol

After each phase:

1. Call `mcp__project-compliance__full_compliance_check` with current state of the project (code shipped, dashboard status, AI usage, etc.)
2. Save output to `phase-N-check.md`
3. Walk through the report's three sections (proposal coverage, CIS 2450, NETS 1500)
4. Items marked MISSING or 🟡 partial that fall in the *current* phase's REQ-IDs = **P0 blocker** — fix before advancing
5. Items missing from *future* phases = expected, no action
6. Commit with message: `chore(compliance): phase N gate passed` (or `chore(compliance): phase N P0 fixed: <REQ-ID>` for fixes)

## Why

CIS 2450 + NETS 1500 are graded against external rubrics. Without a programmatic check, we drift. The MCP tool is the single source of truth. See `.planning/PROJECT.md` § Context for grounding details.
