# Changelog

One entry per PRD step (newest first). Dates are YYYY-MM-DD.

## 2026-10-08 · Step 0: repository and tooling setup

- Monorepo layout from PRD §6 (files at the repo root), placeholder READMEs per package.
- Root README: purpose, architecture, prerequisites, quick start, team conventions (branch/PR workflow).
- `.gitignore`, `.gitattributes` (LF line endings).
- Pre-commit: Ruff + Black (Python), ESLint 10 + Prettier 3 (JS/TS, pinned in root `package.json`), general file checks incl. 1 MB max file size.
- `docker-compose.yml`: PostgreSQL 18, Redis 8, MinIO (`pgsty/minio` fork) + one-shot bucket creation.
- `.env.example` with every variable documented.
- `docs/decisions.md` started (PRD §5 decisions + Step 0 deviations: MinIO image, TypeScript pin).
- GitHub Actions CI: pre-commit lint job + one test job per package (skip until code exists).
