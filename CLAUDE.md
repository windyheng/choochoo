# CLAUDE.md

Guidance for AI coding agents in this repo. See `CONTRIBUTING.md` for branch/PR
rules and `docs/specs/2026-08-26-aigc-detection-design.md` for the design.

## Agent skills

### Issue tracker

Issues and specs live as GitHub issues in `windyheng/choochoo`, managed via the
`gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: a root `CONTEXT.md` plus `docs/adr/`, created lazily by the
domain-modeling skill when terms or decisions actually get resolved. See
`docs/agents/domain.md`.
