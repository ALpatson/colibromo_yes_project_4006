# Colibrimo Immersive Renovation Tour

Student project **JUNIA YES 26-27-R-4006** for the client Colibrimo (2026-10-05 → 2026-12-18).

**Goal:** film a room with a phone, rebuild it in 3D (Gaussian Splatting), and walk around it on a phone. Later stages add a renovated "after" version of the same room and a before/after switch.

**Current scope: Stage 1** (capture → 3D reconstruction → mobile viewer). No renovation generation yet.
The full requirements are in [PRD-colibrimo-stage1.md](PRD-colibrimo-stage1.md) (living document).

> **Status:** Step 0 (repo and tooling) done. Step 1 (reconstruction pipeline, [docs/pipeline.md](docs/pipeline.md)) in progress. See [CHANGELOG.md](CHANGELOG.md).

## Architecture

```
┌────────────────────┐  upload video   ┌───────────────────────┐  enqueue job  ┌──────────────────────────┐
│ Mobile app         │ ──────────────▶ │ Backend API            │ ────────────▶ │ GPU worker               │
│ (React Native/Expo)│                 │ (Django + DRF)         │               │ (Celery + gsplat)        │
│ - capture guide    │ ◀────────────── │ - scenes, jobs, files  │ ◀──────────── │ frames → COLMAP → train  │
│ - job status       │  status / URLs  │ - PostgreSQL           │  status/logs  │ → export → clean →       │
│ - WebView viewer   │                 │ - Redis (queue)        │               │ compress → metrics       │
└────────┬───────────┘                 └──────────┬────────────┘               └────────────┬─────────────┘
         │ loads viewer + splat                   │ presigned URLs                          │ writes artifacts
         ▼                                        ▼                                         ▼
┌────────────────────┐                 ┌──────────────────────────────────────────────────────────────────┐
│ Web viewer         │ ◀────────────── │ Object storage (MinIO locally, S3-compatible)                    │
│ (Vite + TS + Spark)│   splat file    │ videos/ frames/ checkpoints/ exports/ (ply, compressed) metrics/ │
└────────────────────┘                 └──────────────────────────────────────────────────────────────────┘
```

More detail: [docs/architecture.md](docs/architecture.md). Technical choices and why: [docs/decisions.md](docs/decisions.md).

## Repository layout

| Folder | What | Built in |
|---|---|---|
| [pipeline/](pipeline/) | Reconstruction pipeline `cpipe` (Python CLI + library: FFmpeg, COLMAP, gsplat). Guide: [docs/pipeline.md](docs/pipeline.md) | Steps 1–3 |
| [backend/](backend/) | Django + DRF API, Celery jobs | Step 5 |
| [viewer/](viewer/) | Web viewer (Vite + TypeScript + three.js + Spark) | Step 4 |
| [mobile/](mobile/) | Expo React Native app | Step 6 |
| [docs/](docs/) | Architecture, decisions, capture guide, benchmarks, API | all |
| `data/` | Local videos and run outputs. **Gitignored** (personal data) | n/a |

## Prerequisites

| Tool | Version | Used for |
|---|---|---|
| Git | any recent | |
| Docker + Docker Compose v2 | Docker Desktop or [Rancher Desktop](https://rancherdesktop.io/) (dockerd/moby engine) | local services |
| Python | 3.11 | pipeline, backend, pre-commit |
| Node.js | 22 LTS | viewer, mobile, ESLint/Prettier |
| NVIDIA GPU + CUDA (Linux) | ≥ 16 GB VRAM recommended | reconstruction only (Step 1+), not needed on dev laptops |

## Quick start

```bash
git clone https://github.com/ALpatson/colibromo_yes_project_4006.git
cd colibromo_yes_project_4006

# 1. Local services: PostgreSQL, Redis, MinIO (+ bucket)
cp .env.example .env          # optional: edit ports, passwords, storage folder
docker compose up -d
docker compose ps             # postgres, redis, minio should be "healthy"
```

Then:

- **MinIO console:** http://localhost:9001 (login: `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` from `.env`, defaults `colibrimo` / `colibrimo-dev-password`). The `colibrimo` bucket is created automatically.
- **PostgreSQL:** `localhost:5432`, database/user `colibrimo`.
- **Redis:** `localhost:6379`.

Stop with `docker compose down`. Add `-v` to also delete the Postgres and Redis data. MinIO files live in the folder set by `MINIO_DATA_DIR` (default `./data/minio`), so delete that folder to wipe them.

### Dev tooling (lint and format before each commit)

```bash
# Windows (PowerShell)                         # macOS / Linux
python -m venv .venv                           python3.11 -m venv .venv
.venv\Scripts\activate                         source .venv/bin/activate
pip install pre-commit                         pip install pre-commit

npm install            # ESLint + Prettier, pinned in package.json
pre-commit install     # run hooks automatically on `git commit`
pre-commit run --all-files
```

Windows: if `activate` fails with "running scripts is disabled on this system", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

Tip: pre-commit caches its hook environments in `~/.cache/pre-commit` (on Windows that's on `C:`). To keep the cache on another drive, set `PRE_COMMIT_HOME`, for example `setx PRE_COMMIT_HOME D:\Colibromo_yes_prooject_4006\.venv\pre-commit-cache`.

Hooks: Ruff (lint) + Black (format) for Python, ESLint + Prettier for JS/TS, and general checks. These include blocking files over 1 MB, so videos and splats can't be committed by mistake.

## Team conventions

**Branches and pull requests**
- `main` is protected: no direct pushes; it always passes CI.
- One branch per PRD step or task: `step-<n>-<short-name>` (e.g. `step-1-pipeline`), or `feat/…`, `fix/…`, `docs/…` for smaller work.
- Open a pull request into `main`. It needs **green CI and 1 review** from a teammate.
- **Squash-merge**, then delete the branch.

**Commits:** short imperative subject, e.g. `Add frame extraction step`. Run `pre-commit` before pushing (automatic after `pre-commit install`).

**Code:** Python is formatted with Black and linted with Ruff. TS/JS uses Prettier and ESLint. Public functions get docstrings. Every package has a README.

**Secrets:** only in `.env` (gitignored). Add new variables to `.env.example` with a comment.

**Data and privacy:** room videos are personal data. Never commit them, never upload them to third-party services without client approval. Keep them in `data/` or in project storage.

**Decisions:** log any notable technical choice (or any deviation from the PRD) in [docs/decisions.md](docs/decisions.md). Record each finished step in [CHANGELOG.md](CHANGELOG.md).
