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

---

## D-005 · Train with gsplat's `simple_trainer.py` instead of Nerfstudio `splatfacto` · 2026-10-08

**Context:** PRD §5 specifies Nerfstudio `splatfacto`. Checks on 2026-10-08:
- Nerfstudio is dormant: last release v1.1.5 (2024-11), last commit 2025-07. It pins `gsplat==1.4.0`.
- On Google Colab (Python 3.13, PyTorch 2.11, CUDA 13.0), `pip install nerfstudio==1.1.5` downgrades `numpy` and `protobuf` (breaking Colab packages), and compiling gsplat 1.4.0 was killed (out of memory).
- gsplat (the library splatfacto is built on) is actively maintained (commits in 2026-09). gsplat 1.5.3 compiled and rendered on Colab (with `MAX_JOBS=2`).

**Decision:** train with gsplat v1.5.3's official `examples/simple_trainer.py` (`default` strategy = original 3DGS densification, same algorithm family as splatfacto). Its requirements are installed except the `numpy<2.0.0` pin (Colab ships numpy 2; downgrading breaks Colab packages) and `fused-bilagrid` (only for an optional feature). *Correction 2026-10-09:* the COLMAP reader it uses (a pycolmap fork) **does** have one numpy-2 incompatibility, missed in the first check; it is patched automatically, see D-009.

**Consequences:**
- No `ns-train` / `ns-export` / `ns-eval`. Export = the PLY gsplat writes; evaluation in Step 2 will use gsplat's eval mode (`--ckpt`) instead of `ns-eval`.
- gsplat normalises the scene (`normalize_world_space`); we recompute that 4x4 transform with gsplat's own parser and store it in the manifest (needed for future "after" variants).
- The training script is not part of the gsplat wheel: the GPU machine needs a gsplat checkout (`GSPLAT_EXAMPLES_DIR`).
- If gsplat changes its example script, pin/adjust the version in one place (notebook + Dockerfile).

---

## D-006 · Call COLMAP directly instead of `ns-process-data` · 2026-10-08

**Context:** PRD R1.2 suggests COLMAP through Nerfstudio's `ns-process-data`. With D-005 Nerfstudio is not installed, and installing it (with PyTorch) on CPU laptops just to run COLMAP would cost several GB.

**Decision:** run `colmap feature_extractor`, `sequential_matcher` (video frames are ordered; exhaustive matching is optional) and `mapper` from the pipeline. Frames are extracted by our own FFmpeg step (blur filtering, downscaling), not by COLMAP.

**Consequences:** COLMAP renamed options between 3.x and 4.x (`SiftExtraction.use_gpu` became `FeatureExtraction.use_gpu`), so the pipeline reads `colmap <command> --help` and uses whichever name exists. Verified with COLMAP 4.2.1 (Windows); Colab/Docker use Ubuntu 24.04's packaged COLMAP. The output layout (`images/`, `sparse/0/`) is what gsplat reads.

---

## D-007 · Google Colab as the GPU until a GPU machine is chosen · 2026-10-08

**Context:** no Linux + NVIDIA machine yet (PRD §11 Q1); the team laptop has an AMD Radeon 760M (no CUDA), so training cannot run locally.

**Decision:** CPU steps (validate, frames, COLMAP) run anywhere, `--until poses` on laptops. Training runs on Google Colab's free T4 GPU through `pipeline/notebooks/colab_run.ipynb`. gsplat and fused-ssim have no prebuilt wheels for Colab's Python 3.13 / PyTorch 2.11, so the notebook compiles them once and caches the wheels on the user's Google Drive.

**Consequences:**
- Videos must be on Google Drive: only the team's own test rooms until Colibrimo approves (privacy, PRD §9).
- Free Colab sessions can disconnect; timings on a T4 will not match the PRD's 45 min target GPU.
- The Dockerfile (R1.8) is written for a future GPU machine but **untested**.

---

## D-008 · Presets ship inside the Python package · 2026-10-08

**Context:** PRD §6 shows `pipeline/presets/`. Presets must be available when the pipeline is installed with `pip install git+...` (Colab, Docker), where only the package is installed.

**Decision:** presets live in `pipeline/colibrimo_pipeline/presets/` (package data). Custom presets can be any YAML file passed with `--preset path/to/file.yaml`.

**Consequences:** none for users; the folder differs from the PRD's tree.

---

## D-009 · Automatic workarounds for gsplat v1.5.3's example trainer · 2026-10-09

**Context:** the first Colab training runs failed before training started:
1. `No module named 'datasets.colmap'`: gsplat's `examples/datasets/` has no `__init__.py`, so the HuggingFace `datasets` package preinstalled on Colab was imported instead.
2. `OverflowError: Python integer -1 out of bounds for uint64`: the pycolmap fork pinned by gsplat v1.5.3 (`rmbrualla/pycolmap@cc7ea4b`) has `INVALID_POINT3D = np.uint64(-1)`, which numpy 2 rejects. (This is why gsplat pins `numpy<2`; gsplat's main branch has since moved to the official `pycolmap` and numpy 2, but there is no release with that yet.)

**Decision:** `colibrimo_pipeline/gsplat_compat.py`, called by the `train` step, (1) adds an empty `examples/datasets/__init__.py` and (2) replaces that one line with `np.uint64(np.iinfo(np.uint64).max)` (same value and type). Both are idempotent and logged. Verified on 2026-10-09 by running gsplat v1.5.3's real COLMAP loader with numpy 2.4 on the room1 COLMAP model (108 images loaded, transform computed); regression tests in `pipeline/tests/test_gsplat_compat.py`.

**Consequences:**
- The pipeline edits files of a third-party checkout/package; acceptable for a pinned version, removable once gsplat releases the main-branch loader.
- Known, not fixed: the same pycolmap fork reads binary files with native `struct` `'L'` (4 bytes on Windows, 8 on Linux), so gsplat v1.5.3 training cannot read COLMAP models on **Windows**. Irrelevant for now (training needs Linux + NVIDIA: Colab, Docker).

---

## D-010 · Several videos per room · 2026-10-13

**Context:** the best room so far (myRoom2) came from three videos filmed at three heights (`bottom.mp4`, `middle.mp4`, `up.mp4`), merged by hand. The pipeline only accepted one video.

**Decision:**
- `cpipe run` takes `--video` several times and/or `--videos <folder>` (every `.mp4/.mov/.m4v/.mkv/.avi/.webm`, sorted by name); `run_pipeline()` takes one path or a list. Each video is named after its file (`bottom`), reduced to letters, digits, `-`, `_`; two videos with the same name are rejected.
- Durations: the 20 s **minimum applies to all videos together**, the 5 min **maximum to each video** (a short extra pass must not fail a run; one very long video is still suspicious).
- `frames.target_count` is shared between the videos **in proportion to their duration** (largest-remainder rounding, so the total is exact), each video keeping at least `frames.min_per_video` (20). The sharpest-frame selection and the static-video check run per video.
- Frames are named after their video, `bottom_00000.jpg`, **also for a single video** (`room1_00000.jpg` instead of `frame_00000.jpg`), so there is one naming rule and every frame knows its source. `frames.json` (run folder) lists each frame's file, video, index and timestamp; the manifest has a per-video summary.
- COLMAP uses one shared camera when all videos have the same resolution/orientation, otherwise one camera per frame (warning in `validate`).

**Consequences:** manifest input is now `input.videos` (list) and `validate` writes `input/videos.json` (was `input/video_info.json`). Run folders from before this change still resume: `extract_frames` reads the old single-video metadata.

---

## D-011 · Matcher chosen automatically (`poses.matcher: auto`) · 2026-10-13

**Context:** sequential matching only compares each frame with its neighbours in time, so frames from different videos are never matched and COLMAP splits the room into one model per video.

**Decision:** new default `poses.matcher: auto`:
- one video: `sequential` (as before);
- several videos: `sequential` with `SequentialMatching.loop_detection=1`, which also matches visually similar frames far apart in time (e.g. the same wall seen in another video).

`exhaustive` (every pair) stays available with `--set poses.matcher=exhaustive`; it also connects the videos but is about twice as slow (measurements below). A first version of this decision chose exhaustive matching up to 400 frames (`poses.exhaustive_max_frames`); the measurements below replaced that with loop detection for every multi-video run, and the setting was removed before being pushed.

Option names checked with COLMAP 4.2.1's `sequential_matcher --help` (`SequentialMatching.loop_detection`, `SequentialMatching.vocab_tree_path`). Loop detection needs a vocabulary tree: per COLMAP's tutorial, "A default tree will be automatically downloaded and cached" when no path is given (internet needed the first time); `poses.vocab_tree_path` can point to a local one. The chosen matcher and the reason are recorded in the manifest (`poses.info.matcher_reason`).

**Measured (2026-10-13, myRoom2, 3 videos, `balanced` = 300 frames, COLMAP 4.2.1 CUDA on an NVIDIA T500 4 GB, `poses.use_gpu=true`):**

| | Exhaustive | Sequential + loop detection (default) |
|---|---|---|
| Feature extraction | 36 s | 56 s |
| Matching | ~35 min | 13 min |
| Mapper | 6.5 min | 6.3 min |
| **Total `poses`** | **~41 min** | **20 min** |
| Frames registered | 300 / 300 | 296 / 300 (99%) |
| Models | 1 | 1 |
| Frames per video in the model (bottom / middle / up) | 121 / 106 / 73 | 117 / 106 / 73 |
| 3D points | 25,106 | 20,233 |

Both connect all three heights in one model. (Exhaustive wall-clock time was longer because the laptop slept for about an hour; the table excludes that pause.)

**Consequences:** multi-video runs need internet the first time (vocabulary tree download).

---

## D-012 · Tone-map HDR videos (iPhone) to SDR when extracting frames · 2026-10-13

**Context:** iPhones film in HDR by default (HEVC 10-bit, HLG transfer `arib-std-b67`, BT.2020, plus a Dolby Vision record). Extracted as is, the frames come out grey and washed out, and the splat inherits those colours. iPhone `.MOV` files also hold 2 audio and 6 data streams besides the video.

**Decision:**
- `validate` records each video's `pix_fmt`, `color_transfer`, `color_primaries`, `color_space` and flags it as HDR when the transfer is `arib-std-b67` (HLG) or `smpte2084` (PQ).
- `extract_frames` always reads only the main video stream (`-map 0:v:0`) and, for HDR, inserts the chain that worked in manual tests between frame sampling and downscaling:
  `zscale=tin=<transfer>:min=<matrix>:pin=<primaries>:t=linear:npl=100, format=gbrpf32le, zscale=p=bt709, tonemap=tonemap=hable:desat=0, zscale=t=bt709:m=bt709:r=tv, format=yuv420p`.
  Transfer, matrix and primaries come from ffprobe (BT.2020 if missing). Settings: `frames.hdr_tonemap` (default `true`), `frames.tonemap` (`hable`), `frames.hdr_peak_nits` (100).
- If the FFmpeg build lacks `zscale`/`tonemap` (they need libzimg), the step fails with a clear message instead of silently producing washed-out frames.

**Verified (2026-10-13, FFmpeg 9.0.2 gyan.dev full build):** `room1.MOV` (iPhone 16, HLG, portrait) is detected as HDR; 12 frames extracted with and without tone-mapping. Tone-mapped frames have normal colours (red door, wood tables, white walls; mean saturation 68 vs 41, darker blacks: 1st percentile 14.5 vs 24.6). Portrait rotation is still applied (720x1280). A synthetic HLG-tagged clip covers this in the tests (skipped if FFmpeg lacks libx264 or zscale).

**Consequences:** the setting is a boolean on purpose: YAML reads `off` as `false`, so an `auto`/`off` choice broke `--set frames.hdr_tonemap=off`. Tone-mapped colours are an approximation of what the phone shows (hable keeps highlights soft; white walls come out very slightly warm); `frames.tonemap=mobius` is an alternative if colours look off. For comparison, COLMAP's own `automatic_reconstructor` (data type video, ~255 frames) took about 3 min on the same videos; it uses its own extraction/matching settings, which were not compared in detail. If `poses` stays the bottleneck, that comparison is the next thing to look at.
