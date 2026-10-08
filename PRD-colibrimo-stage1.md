PRD: Colibrimo Immersive Renovation Tour, Stage 1 (Capture → 3D Reconstruction → Mobile Viewer)

| | |
|---|---|
| **Project** | JUNIA YES 26-27-R-4006, client Colibrimo |
| **Document version** | v0.1 (living document, expected to change) |
| **Last updated** | 2026-10-08 |
| **Project dates** | 2026-10-05 → 2026-12-18 |
| **Scope of this version** | Stage 1 only: turn a phone video of a room into a 3D Gaussian Splatting scene and view it on a phone. No renovation generation yet. |

---

## 0. How to use this document (instructions for Claude Code)

This PRD is implemented **one step at a time, in the order the user asks for**. Rules:

1. **Only implement the step the user names** (e.g. "Do Step 2"). Do not start the next step on your own.
2. Before coding a step, **re-read its section**, list your plan in a few bullets, and flag anything ambiguous. Ask before guessing on decisions that are expensive to undo.
3. At the end of each step:
   - all the step's **acceptance criteria** are met and checked (run the code and tests, don't just write them),
   - `docs/` and the relevant `README.md` are updated,
   - a short entry is added to `CHANGELOG.md`,
   - you give the user a summary: what was built, how to run it, what is untested or still uncertain, and suggested next step.
4. **Do not invent CLI flags, package names or library APIs.** Fast-moving tools are involved (Nerfstudio, gsplat, Spark, Expo). Check installed versions (`--help`, package docs, `node_modules` types) before using a flag or API. If something in this PRD contradicts the real tool, follow the real tool and note the difference in `docs/decisions.md`.
5. Keep secrets in `.env` files (never committed). Provide `.env.example`.
6. Prefer simple, readable code over clever code. This is a 10-week student project that must be **documented and handed over**.
7. If a requirement in this PRD turns out to be wrong or impossible, **stop and tell the user** instead of silently working around it.

---

## 1. Context

### 1.1 Client
Colibrimo is a SaaS platform for real estate agents and homeowners planning renovations. It already provides instant budget estimates, before/after photos aligned with the estimate, and connections with tradespeople.

### 1.2 Overall project goal (all stages)
Build a pipeline that:
1. **Captures** a home with a phone and reconstructs it in 3D (Gaussian Splatting).
2. **Generates** a renovated version of the 3D scene from a list of planned works, **geometrically aligned** with the original.
3. **Displays** an immersive before/after tour on a phone, with a switch between states inside the tour.

Final deliverables: working prototype, feasibility report on automatic renovation generation, live demo on a real home, documented source code.

### 1.3 Why this stage first
The 3D reconstruction is the foundation. If the "before" scene is poor, nothing built on top will work. Stage 1 proves we can reliably go from **phone video → good 3D room → smooth viewing on a phone**, before investing in renovation generation.

---

## 2. Stage 1 goals and non-goals

### 2.1 Goals
- G1. Turn a 1–3 minute phone video of a room into a Gaussian Splatting scene with a reproducible, scripted pipeline.
- G2. Measure reconstruction quality, time and size objectively, so the team can make a go/no-go decision.
- G3. Clean and compress the scene so it loads and runs smoothly on a mid-range phone.
- G4. View the scene in a web viewer and inside a React Native app.
- G5. Expose the pipeline through a backend API with background jobs, so the app can upload a video and get a 3D scene back.
- G6. Design data models so a future "after" (renovated) version of the same scene can be added **without refactoring**.

### 2.2 Non-goals (for now)
- Renovation generation of any kind (AI editing, segmentation, OpenRouter calls).
- Before/after toggle UI (data model must allow it; UI comes later).
- User accounts, authentication, payments, multi-tenant features.
- Production deployment, scaling, monitoring.
- Whole-house / multi-room stitching (one room per scene for now).

---

## 3. Users and core user story

**Primary user (for the demo):** a real estate agent or homeowner with a regular smartphone.
**Internal user (now):** the student team, testing captures and tuning the pipeline.

> As a user, I film a room with my phone by following on-screen guidance, upload the video, wait while it is processed, and then walk around a realistic 3D version of the room on my phone.

---

## 4. System overview

```
┌────────────────────┐  upload video   ┌───────────────────────┐  enqueue job  ┌──────────────────────────┐
│ Mobile app         │ ──────────────▶ │ Backend API            │ ────────────▶ │ GPU worker               │
│ (React Native/Expo)│                 │ (Django + DRF)         │               │ (Celery + Nerfstudio)    │
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

---

## 5. Tech stack (current decisions, may change)

| Layer | Choice | Notes |
|---|---|---|
| 3D reconstruction | **Nerfstudio `splatfacto`** (built on **gsplat**) | Camera poses via **COLMAP** through `ns-process-data`. GLOMAP may be evaluated later for speed. |
| GPU runtime | Linux, NVIDIA GPU (≥ 16 GB VRAM recommended), CUDA, PyTorch | School cluster or rented cloud GPU. Pipeline packaged in **Docker** (CUDA base image). |
| Frame extraction | **FFmpeg** | Also used for blur filtering / frame sampling if needed. |
| Post-processing | Python (numpy, plyfile) | Crop, floater removal, opacity filtering, SH degree reduction. |
| Compression | **SPZ** preferred; `.ksplat` / `.splat` as fallbacks | Pick whichever the chosen viewer loads reliably. Record the decision. |
| Web viewer | **Vite + TypeScript + three.js + Spark** | Fallback: GaussianSplats3D or PlayCanvas. |
| Backend API | **Django + Django REST Framework** | Team's main backend stack. |
| Job queue | **Celery + Redis** | Training jobs take 10–60 min. |
| Database | **PostgreSQL** | SQLite acceptable for local dev only. |
| Storage | **MinIO** (local), any S3-compatible service later | Presigned URLs for upload/download. |
| Mobile app | **React Native with Expo** (dev build), TypeScript | `react-native-vision-camera` for capture, `react-native-webview` for the viewer. |
| Local orchestration | **Docker Compose** | api, worker, postgres, redis, minio, viewer. |
| AI APIs | **OpenRouter** | **Not used in Stage 1.** Reserved for Stage 3 (parsing works lists, image edits). |
| Tooling | Git + GitHub, pre-commit, ruff + black (Python), ESLint + Prettier (TS), pytest, vitest/jest | |

---

## 6. Repository structure (target)

```
colibrimo-tour/
├── README.md                  # project overview + quick start
├── CHANGELOG.md
├── .env.example
├── docker-compose.yml
├── docs/
│   ├── architecture.md
│   ├── decisions.md           # architecture decision log (ADR-style, short)
│   ├── capture-guide.md       # how to film a room
│   ├── benchmarks.md          # reconstruction results per test room
│   └── api.md                 # API reference
├── pipeline/                  # GPU reconstruction pipeline (Python)
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── colibrimo_pipeline/
│   │   ├── cli.py             # entry point: `cpipe run ...`
│   │   ├── config.py          # typed config (pydantic) + YAML presets
│   │   ├── steps/
│   │   │   ├── extract_frames.py
│   │   │   ├── poses.py       # COLMAP via ns-process-data
│   │   │   ├── train.py       # splatfacto
│   │   │   ├── export.py
│   │   │   ├── postprocess.py # crop, floaters, opacity, SH
│   │   │   ├── compress.py
│   │   │   └── evaluate.py    # metrics
│   │   ├── manifest.py        # run manifest (inputs, versions, timings, outputs)
│   │   └── utils/
│   ├── presets/               # fast.yaml, balanced.yaml, quality.yaml
│   └── tests/
├── backend/                   # Django project
│   ├── manage.py
│   ├── config/                # settings, urls, celery app
│   ├── scenes/                # app: Scene, SceneVariant, Capture, Job models + API
│   └── tests/
├── viewer/                    # Vite + TS web viewer
│   ├── src/
│   └── tests/
├── mobile/                    # Expo React Native app
│   ├── app/
│   └── src/
└── data/                      # gitignored: sample videos, local outputs
```

---

## 7. Data model (designed for future before/after)

- **Scene**: one physical room. `id (uuid)`, `name`, `room_type` (living_room, bathroom, kitchen, bedroom, other), `created_at`, `notes`.
- **Capture**: a raw input for a scene. `id`, `scene_id`, `video_file` (storage key), `device_model`, `duration_s`, `resolution`, `fps`, `uploaded_at`.
- **SceneVariant**: a 3D version of a scene. `id`, `scene_id`, `kind` (`before` now; `after` reserved for later), `source_variant_id` (nullable; the `before` an `after` was derived from), `splat_file_raw` (ply key), `splat_file_web` (compressed key), `num_gaussians`, `file_size_bytes`, `metrics` (JSON), `created_at`.
  - **Invariant (future):** every `after` variant shares the **same coordinate frame** as its `before` variant. Stage 1 must store the transform/normalisation applied during processing (`transform` JSON on the variant) so later variants can be aligned.
- **ReconstructionJob**: `id`, `capture_id`, `variant_id` (nullable until done), `preset` (fast/balanced/quality), `status` (`queued`, `extracting_frames`, `estimating_poses`, `training`, `exporting`, `postprocessing`, `compressing`, `evaluating`, `succeeded`, `failed`, `cancelled`), `progress` (0–100), `error_message`, `log_file`, `started_at`, `finished_at`, `timings` (JSON per step), `pipeline_version`.

---

## 8. Implementation steps

Each step is a self-contained unit. **Implement only when the user asks.** Steps are listed in the recommended order, but the user may reorder them.

---

### Step 0: Repository and tooling setup

**Goal:** a clean monorepo the whole team can clone and run.

**Requirements**
- R0.1 Create the folder structure in §6 with placeholder READMEs.
- R0.2 Root `README.md`: project purpose, architecture diagram (from §4), prerequisites, quick start, team conventions.
- R0.3 `.gitignore` covering Python, Node, Expo, Docker, `data/`, outputs, `.env`.
- R0.4 Pre-commit hooks: ruff + black for Python, ESLint + Prettier for TS.
- R0.5 `docker-compose.yml` with `postgres`, `redis`, `minio` (+ bucket auto-creation) running; other services added in later steps.
- R0.6 `.env.example` with all variables documented.
- R0.7 `docs/decisions.md` started with the decisions in §5.
- R0.8 GitHub Actions CI: lint + unit tests for each package (jobs can be no-ops until code exists).

**Acceptance criteria**
- [ ] `docker compose up` starts postgres, redis, minio without errors; MinIO console reachable; bucket exists.
- [ ] `pre-commit run --all-files` passes.
- [ ] CI workflow runs green on push.

---

### Step 1: Reconstruction pipeline as a CLI (core of Stage 1)

**Goal:** one command turns a video into a trained Gaussian Splat, reproducibly, on a GPU machine. No API yet.

**Requirements**
- R1.1 CLI entry point, e.g.:
```bash
  cpipe run --video data/videos/room1.mp4 --output data/runs/room1 --preset balanced
```
- R1.2 Pipeline steps, each a separate module with clear inputs/outputs:
  1. **Validate input**: file exists, readable by FFmpeg, duration 20 s–5 min, resolution ≥ 1080p (warn, don't fail, if lower). Record metadata (ffprobe).
  2. **Extract frames**: target frame count from preset (e.g. fast 150, balanced 300, quality 500). Optionally drop the blurriest frames (variance-of-Laplacian), configurable.
  3. **Estimate camera poses**: COLMAP through `ns-process-data` (images mode, using the extracted frames) or `ns-process-data video` directly. Choose the cleaner option and document why. Record the % of frames successfully registered.
  4. **Train**: `ns-train splatfacto` with iterations from preset; headless (no viewer blocking); fixed random seed.
  5. **Export**: `ns-export gaussian-splat` → `scene.ply`.
- R1.3 **Presets** in YAML (`fast`, `balanced`, `quality`) controlling frame count, iterations, and any splatfacto parameters. CLI flags can override presets.
- R1.4 **Run manifest** (`manifest.json`) in each run folder: input metadata, preset + resolved config, tool versions (nerfstudio, gsplat, COLMAP, CUDA, torch, GPU model), git commit, per-step start/end/duration, output paths, status.
- R1.5 **Logging**: per-step log files plus a combined log; clear error messages naming the failing step.
- R1.6 **Resumability**: if a run folder already has the output of a step, skip it unless `--force` is passed. This saves hours when tuning later steps.
- R1.7 **Failure handling**: if pose estimation registers < 70% of frames (threshold configurable), fail with a clear message suggesting capture problems (see capture guide).
- R1.8 **Docker image** (`pipeline/Dockerfile`) based on an official CUDA image with Nerfstudio, COLMAP and FFmpeg installed; documented `docker run --gpus all ...` command. Document a non-Docker install path as well.
- R1.9 Pipeline code must be importable as a library (the Celery worker in Step 5 will call it), not only via CLI.
- R1.10 Unit tests for anything that doesn't need a GPU (config resolution, manifest writing, input validation, frame selection). GPU-dependent steps get a smoke test marked `@pytest.mark.gpu` and skipped when no GPU is available.

**Acceptance criteria**
- [ ] On a GPU machine, `cpipe run` on a sample room video produces `scene.ply` and `manifest.json` without manual intervention.
- [ ] Re-running the same command skips completed steps; `--force` re-runs them.
- [ ] A deliberately bad video (e.g. 5 s, or a static shot) fails early with a clear, human-readable error.
- [ ] `docker run --gpus all ...` reproduces the run.
- [ ] `docs/pipeline.md` explains install, usage, presets and outputs.

---

### Step 2: Evaluation and benchmarking

**Goal:** objective numbers so the team can decide "is the reconstruction good enough?".

**Requirements**
- R2.1 `evaluate` step: run Nerfstudio evaluation (`ns-eval` or equivalent) on held-out views and store **PSNR, SSIM, LPIPS** in `metrics.json`.
- R2.2 Also store: number of Gaussians, raw file size, training time, total pipeline time, % frames registered, GPU peak memory if available.
- R2.3 Render a short **orbit/fly-through preview video** (`ns-render` camera path or interpolated path) and a few **side-by-side images** (ground truth vs render) for visual inspection.
- R2.4 `cpipe benchmark --runs data/runs/*` generates/updates `docs/benchmarks.md` with a comparison table across runs (room, preset, metrics, time, size, notes).
- R2.5 Document the **go/no-go criteria** (initial targets, to be tuned by the team):

  | Criterion | Initial target |
  |---|---|
  | Frames registered by COLMAP | ≥ 80% |
  | PSNR on held-out views | ≥ 25 dB (indicative) |
  | Visual check | straight walls, sharp textures, few floaters, no large holes |
  | Total processing time (balanced preset) | ≤ 45 min per room |
  | Web file size after Step 3 | ≤ 50 MB |
  | Mobile performance (Step 4) | ~30 fps on a mid-range phone |

**Acceptance criteria**
- [ ] Every completed run has `metrics.json`, a preview video and comparison images.
- [ ] `docs/benchmarks.md` shows at least the sample runs in one table.

---

### Step 3: Post-processing and compression for mobile

**Goal:** make the splat light and clean enough for a phone.

**Requirements**
- R3.1 **Cleanup**, each configurable and individually toggleable:
  - remove Gaussians with opacity below a threshold,
  - remove statistical outliers / floaters (distance-based),
  - optional crop to a bounding box (auto-estimated from camera positions + margin, or provided manually),
  - optional reduction of spherical-harmonics degree (big size win, small quality cost).
- R3.2 **Scene normalisation**: put the scene upright (gravity-aligned using camera "up" vectors), floor roughly at y=0, sensible scale. **Save the applied transform** in the manifest (needed later to align "after" variants).
- R3.3 **Compression** to the web format chosen in §5 (SPZ preferred). Keep the raw `.ply` as well.
- R3.4 Report size before/after each operation, and re-run evaluation on the cleaned scene to measure the quality cost.
- R3.5 Output: `scene_clean.ply`, `scene.web.<ext>`, updated `manifest.json` and `metrics.json`.

**Acceptance criteria**
- [ ] For the sample room, web file ≤ 50 MB (or the team-agreed target), with the quality drop reported.
- [ ] Cleaned scene visibly has fewer floaters (before/after screenshots in `docs/benchmarks.md`).
- [ ] Applied transform is stored and documented.

---

### Step 4: Web viewer

**Goal:** a standalone, mobile-friendly viewer for one splat file, later embedded in the app.

**Requirements**
- R4.1 Vite + TypeScript + three.js + Spark. Load a splat from a URL query parameter, e.g. `/?src=<url>`.
- R4.2 **Navigation**:
  - Desktop: orbit (mouse) + WASD/arrow keys for walking.
  - Mobile: one-finger drag to look around, two-finger pinch/drag to move; optional on-screen joystick.
  - Start position: inside the room at eye height (~1.6 m), using the normalised scene from Step 3. Allow an optional `start` pose param.
- R4.3 UI: loading progress bar, error message if loading fails, a "reset view" button, an FPS counter in debug mode (`?debug=1`).
- R4.4 Performance: adaptive pixel ratio on mobile; target ~30 fps on a mid-range Android phone.
- R4.5 **Future-proofing**: viewer code structured to support **two variants loaded at once** with a switch (before/after) later. Don't build the UI yet, but don't hard-code "one scene" assumptions deep in the code.
- R4.6 **Embedding API**: when running inside a WebView, communicate via `window.postMessage` (events: `loaded`, `error`, `progress`; commands: `load(src)`, `resetView`). Document the message schema in `docs/viewer.md`.
- R4.7 Serve via Docker Compose service for local dev; static build deployable anywhere.

**Acceptance criteria**
- [ ] Opening the viewer on a phone browser (same network) with the sample scene works and is navigable.
- [ ] FPS measured on at least one iPhone and one Android, recorded in `docs/benchmarks.md`.
- [ ] postMessage events fire and are documented.

---

### Step 5: Backend API and background jobs

**Goal:** the pipeline becomes a service the app can call.

**Requirements**
- R5.1 Django + DRF project in `backend/`, app `scenes` with the models in §7, migrations, admin registration.
- R5.2 Endpoints (prefix `/api/v1/`):

  | Method | Path | Purpose |
  |---|---|---|
  | POST | `/scenes/` | create a scene (name, room_type) |
  | GET | `/scenes/` , `/scenes/{id}/` | list / detail (incl. variants) |
  | POST | `/scenes/{id}/captures/` | request an upload → returns **presigned upload URL** + capture id |
  | POST | `/captures/{id}/complete/` | confirm upload finished → validates file, **creates and enqueues a ReconstructionJob** (body: preset) |
  | GET | `/jobs/{id}/` | status, progress, current step, timings, error |
  | POST | `/jobs/{id}/cancel/` | cancel a queued/running job |
  | GET | `/variants/{id}/` | metadata + **presigned download URL** for the web splat + metrics |
  | GET | `/health/` | health check |

- R5.3 Uploads go **directly from the client to object storage** via presigned URLs (videos can be several hundred MB). Support multipart upload or document size limits.
- R5.4 Celery worker (GPU host) runs the Step 1–3 pipeline as a library, updates job `status`/`progress` per step, uploads outputs to storage, creates the `SceneVariant(kind="before")` on success, and stores logs.
- R5.5 One GPU job at a time per worker (concurrency 1); queue additional jobs.
- R5.6 Job failure: status `failed`, error message stored, partial outputs kept for debugging.
- R5.7 OpenAPI schema (drf-spectacular) served at `/api/schema/` + Swagger UI. `docs/api.md` summarises usage with curl examples.
- R5.8 No auth in Stage 1, but structure code so a simple API-key or token auth can be added in one place. CORS configured for the viewer and dev app.
- R5.9 Tests: model tests, API tests (pytest-django), Celery task tests with the pipeline mocked.

**Acceptance criteria**
- [ ] With Docker Compose + a GPU worker, a curl-only flow works: create scene → upload video → complete → poll job → get variant download URL → open it in the viewer.
- [ ] Job status progresses through the named steps in real time.
- [ ] Tests pass in CI (GPU parts mocked).

---

### Step 6: Mobile app (React Native + Expo)

**Goal:** capture, upload, track and view from a phone.

**Requirements**
- R6.1 Expo (dev build, since vision-camera needs native code) + TypeScript + Expo Router.
- R6.2 Screens:
  1. **Home**: list of scenes with status thumbnails; "New scan" button.
  2. **New scan**: name + room type.
  3. **Capture guide**: short illustrated tips (from `docs/capture-guide.md`): move slowly, walk around, cover 2–3 heights, good lighting, avoid mirrors/windows.
  4. **Capture**: `react-native-vision-camera` video recording; show elapsed time, target duration range (1–3 min), and a gentle "move slower" hint if possible (optional, nice-to-have). Lock exposure/focus if supported.
  5. **Upload**: progress bar, retry on failure, works with large files; allow **importing an existing video from the gallery** as an alternative.
  6. **Processing**: polls job status every few seconds; shows current step and progress; handles failure with the error message.
  7. **Viewer**: `react-native-webview` loading the web viewer with the variant URL; listens to postMessage events; full-screen, landscape allowed.
- R6.3 API base URL configurable (env / settings screen for dev).
- R6.4 Works on iOS and Android; test on at least one physical device of each if available.
- R6.5 Basic error handling everywhere (no network, upload failed, job failed).

**Acceptance criteria**
- [ ] On a real phone: create scene → film or import video → upload → watch processing → open the 3D room in the viewer.
- [ ] App survives backgrounding during upload/processing (resumes polling when reopened).

---

### Step 7: End-to-end validation, docs and hand-over readiness

**Goal:** prove Stage 1 works on real rooms and is understandable by someone new.

**Requirements**
- R7.1 Run the full flow on **at least 3 different rooms** (e.g. living room, bathroom, kitchen) and record results in `docs/benchmarks.md`, including failures and what caused them (mirrors, glossy tiles, low light…).
- R7.2 Compare against a reference app (Scaniverse / Polycam / Luma) on the same room: visual quality, time, file size. Record findings.
- R7.3 Finalise `docs/capture-guide.md` based on what actually worked.
- R7.4 `docs/architecture.md` up to date; `README.md` quick start tested from a fresh clone.
- R7.5 Write a **Stage 1 go/no-go summary** (`docs/stage1-review.md`): criteria from Step 2 vs results, open problems, recommendation for Stage 2/3. This feeds the final feasibility report.

**Acceptance criteria**
- [ ] A teammate who didn't write the code can run the full flow by following the README.
- [ ] `docs/stage1-review.md` exists with a clear recommendation.

---

## 9. Non-functional requirements

- **Reproducibility:** every run records versions, config and seed; Docker image pinned.
- **Performance targets:** see §8 Step 2 table (initial, adjustable).
- **Storage hygiene:** intermediate artefacts (frames, checkpoints) can be purged by a command; final outputs kept.
- **Privacy:** videos of real homes are personal data. Store them only in the project's storage, never in Git, never in third-party services without client approval. Provide a delete command/endpoint for a scene and all its files.
- **Documentation:** every package has a README; code has docstrings for public functions; decisions logged in `docs/decisions.md`.
- **Code quality:** linted, formatted, tested; CI green on `main`.

---

## 10. Future stages (out of scope now, keep in mind)

- **Stage 2, manual "after":** create an "after" variant by manually editing the splat (e.g. SuperSplat) in the **same coordinate frame**; add the before/after switch/slider in the viewer and app.
- **Stage 3, automatic renovation (research):** parse a works list into structured edits (LLM via **OpenRouter**), segment the scene (SAM 2 + 3D grouping), apply 3D-consistent edits (e.g. Instruct-GS2GS, GaussianEditor, DGE, ControlNet-guided diffusion), measure alignment and quality, and document feasibility.
- **Stage 4, polish and demo** on a real home; final feasibility report.

Design choices in Stage 1 (variant model, stored transforms, viewer able to hold two scenes, pipeline as a library) exist so these stages don't require rewrites.

---

## 11. Open questions (to resolve with the team/client)

1. Which GPU do we have? (School cluster vs rented cloud; VRAM; who pays.) This affects presets and timings.
2. Target phones for the demo (iOS, Android, or both; model range)?
3. Will Colibrimo share their existing before/after image pipeline or API?
4. Is a web viewer link acceptable for the client demo, or must it be a store-installable app?
5. Any data-protection constraints from the client on storing home videos?
6. Final file-size and processing-time targets acceptable to the client?

---

## 12. Glossary

- **Gaussian Splatting (3DGS):** represents a scene as millions of small coloured, semi-transparent 3D blobs; renders photorealistic views fast enough for real time.
- **COLMAP:** tool that computes where each frame was taken from (camera poses) and a sparse 3D point cloud.
- **splatfacto:** Nerfstudio's Gaussian Splatting training method.
- **Floaters:** stray blobs floating in empty space; a common reconstruction artefact.
- **SH (spherical harmonics):** how each Gaussian stores view-dependent colour; higher degree = more realism, bigger files.
- **Variant:** one 3D version of a room (`before`, later `after`).
