# Decision log

Short, ADR-style. Newest decisions at the bottom. Never delete an entry: mark it **Superseded by D-xxx** instead.

Each entry: **Context** (why a decision was needed), **Decision**, **Consequences**.

---

## D-001 · Initial tech stack (PRD §5) · 2026-10-08

**Context:** Stage 1 must go from phone video → Gaussian Splatting scene → phone viewer, with a backend for uploads and long GPU jobs. The team knows Django; the client wants a mobile experience.

**Decision:**

| Layer | Choice | Notes |
|---|---|---|
| 3D reconstruction | Nerfstudio `splatfacto` (gsplat) | Camera poses via COLMAP through `ns-process-data`; GLOMAP may be evaluated later. |
| GPU runtime | Linux + NVIDIA GPU (≥ 16 GB VRAM), CUDA, PyTorch, Docker (CUDA base image) | Machine not chosen yet (PRD §11 Q1). |
| Frame extraction | FFmpeg | |
| Post-processing | Python (numpy, plyfile) | |
| Compression | SPZ preferred; `.ksplat` / `.splat` fallbacks | Final choice in Step 3, based on what the viewer loads reliably. |
| Web viewer | Vite + TypeScript + three.js + Spark | Fallback: GaussianSplats3D or PlayCanvas. |
| Backend | Django + Django REST Framework | |
| Jobs | Celery + Redis | Training takes 10–60 min. |
| Database | PostgreSQL (SQLite only for local dev) | |
| Storage | MinIO locally, any S3-compatible service later | Presigned URLs. See D-002 for the image. |
| Mobile | React Native + Expo (dev build), TypeScript | `react-native-vision-camera`, `react-native-webview`. |
| Orchestration | Docker Compose | |
| AI APIs | OpenRouter, **not used in Stage 1** | Reserved for Stage 3. |
| Tooling | Git + GitHub, pre-commit, Ruff + Black, ESLint + Prettier, pytest, vitest/jest | |

**Consequences:** two separate machines: dev laptops (no GPU) and a GPU host for the worker. The pipeline must be runnable in Docker and as a library.

---

## D-002 · Use `pgsty/minio` instead of the official MinIO image · 2026-10-08

**Context:** The PRD specifies MinIO for local S3-compatible storage. On 2026-10-08 the official `minio/minio` and `minio/mc` images could not be pulled anonymously from Docker Hub or Quay (registry returns 401 Unauthorized; for comparison, `redis` returns 200). MinIO Inc. stopped publishing free community images in 2025.

**Options considered:** `pgsty/minio` (community fork of the same MinIO server, "Silo", by Pigsty, AGPLv3); RustFS (different S3-compatible server); SeaweedFS (S3 gateway, no console).

**Decision:** use `pgsty/minio:RELEASE.2026-08-04T00-00-00Z` and `pgsty/mc:RELEASE.2026-09-16T00-00-00Z`, pinned. Same server, CLI and environment variables as MinIO, so the rest of the PRD stays valid.

**Consequences:**
- Third-party maintainer: pin tags and re-check before upgrading.
- All app code must talk plain S3 (boto3) and never MinIO-specific APIs, so storage can be swapped (RustFS, SeaweedFS, AWS S3, …) by changing configuration only.
- Storage data uses a bind mount (`MINIO_DATA_DIR`, default `./data/minio`), so each developer chooses which disk holds the videos. Postgres and Redis use named Docker volumes (small, and Postgres has permission issues on Windows bind mounts).

---

## D-003 · Pin TypeScript 6.0 for linting · 2026-10-08

**Context:** TypeScript 7.0 is the latest release, but `typescript-eslint` 8.71.1 (latest) only supports `typescript >=4.8.4 <6.1.0`.

**Decision:** pin `typescript@6.0.3` in the root tooling `package.json`. The viewer and mobile packages should use the same major until typescript-eslint supports 7.x.

**Consequences:** revisit when typescript-eslint adds TS 7 support.

---

## D-004 · JS/TS lint tools run from a root `package.json` · 2026-10-08

**Context:** pre-commit's isolated Node environments cannot reliably resolve ESLint flat-config imports (`typescript-eslint`, `@eslint/js`) from the repo root, and editors (VS Code ESLint/Prettier extensions) need the tools in `node_modules`.

**Decision:** ESLint and Prettier are pinned in a private root `package.json` (+ lockfile). The pre-commit hooks call them via `npx --no-install` (`language: system`). Developers run `npm install` once at the root.

**Consequences:** one extra setup command; editor and pre-commit use the same versions. Prettier formats only JS/TS/JSON/CSS/HTML; Markdown and YAML are left as written (so the PRD isn't reformatted).
