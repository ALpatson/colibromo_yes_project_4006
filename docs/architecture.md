# Architecture

> Placeholder (Step 0). Filled in as components are built; finalised in Step 7.

See the overview diagram in the [root README](../README.md#architecture) and PRD §4.

## Components

| Component | Folder | Status |
|---|---|---|
| Reconstruction pipeline (CLI + library) | `pipeline/` | not started (Step 1) |
| Backend API + Celery jobs | `backend/` | not started (Step 5) |
| Web viewer | `viewer/` | not started (Step 4) |
| Mobile app | `mobile/` | not started (Step 6) |
| PostgreSQL, Redis, MinIO (S3) | `docker-compose.yml` | running locally (Step 0) |

## Local services (Step 0)

| Service | Image | Host port(s) | Data |
|---|---|---|---|
| postgres | `postgres:18.6` | 5432 | named volume `postgres-data` |
| redis | `redis:8.10.2` | 6379 | named volume `redis-data` |
| minio | `pgsty/minio` (MinIO fork, see decisions D-002) | 9000 (S3 API), 9001 (console) | host folder `MINIO_DATA_DIR` (default `./data/minio`) |
| minio-init | `pgsty/mc` | — | one-shot: creates bucket `S3_BUCKET` |

## Data model

See PRD §7 (Scene, Capture, SceneVariant, ReconstructionJob). Implemented in Step 5.
