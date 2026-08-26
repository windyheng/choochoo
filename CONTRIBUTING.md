# Contributing

## Branches

- `main` is protected — no direct pushes, PR + 1 review required.
- Branch naming: `feat/<role>-<short-desc>`, e.g. `feat/clip-branch-embedding-cache`,
  `feat/data-transforms`, `feat/infra-checkpoint-resume`.

## Before you start (Day 1)

1. Read [`docs/specs/2026-08-26-aigc-detection-design.md`](docs/specs/2026-08-26-aigc-detection-design.md) in full.
2. Read [`docs/interfaces.md`](docs/interfaces.md) — the tensor shape/format
   contract between branches. If you own a branch (CLIP, SRM, or the fusion
   head), agree on this with the other branch owners *before* writing code
   against placeholder assumptions. This is the #1 thing that stalls Day 2
   integration if skipped.
3. Claim your role in the README's team table.

## PRs

- Keep PRs scoped to one component (e.g. don't mix `transforms.py` changes
  with `train.py` changes).
- Every PR touching `data/transforms.py` or `infer.py` must keep the existing
  `tests/` passing — these two files have correctness contracts other
  people's code depends on.
- Use the PR template checklist.

## Data & secrets

- Never commit anything under `data/raw/`, `data/cache/`, `checkpoints/`, or
  `.env` — they're gitignored for a reason (dataset licenses, model size,
  secrets).
- Shared dataset cache + checkpoints live on the shared Google Drive folder
  (link in README once created) — Colab notebooks should mount it, not
  re-download from scratch each session.
