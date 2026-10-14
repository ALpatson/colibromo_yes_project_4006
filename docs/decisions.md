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

**Consequences:** multi-video runs need internet the first time (vocabulary tree download). For comparison, COLMAP's own `automatic_reconstructor` (data type video, ~255 frames) took about 3 min on the same videos; it uses its own extraction/matching settings, which were not compared in detail. If `poses` stays the bottleneck, that comparison is the next thing to look at.

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

**Consequences:** the setting is a boolean on purpose: YAML reads `off` as `false`, so an `auto`/`off` choice broke `--set frames.hdr_tonemap=off`. Tone-mapped colours are an approximation of what the phone shows (hable keeps highlights soft; white walls come out very slightly warm); `frames.tonemap=mobius` is an alternative if colours look off.

---

## D-013 · Brush as a second trainer (`train.trainer: brush`) · 2026-10-13

**Context:** gsplat training needs Linux + an NVIDIA GPU with CUDA (D-005, D-007), so the team trains on Colab. Brush (https://github.com/ArthurBrussee/brush) trains Gaussian Splats through wgpu/Vulkan on NVIDIA, AMD and Intel GPUs, on Windows too; it already produced the team's best room (myRoom2) by hand on a laptop with an NVIDIA T500 4 GB.

**Decision:**
- `train.trainer: gsplat | brush` (default `gsplat`, unchanged). Brush is found through `train.brush_exe`, then `CPIPE_BRUSH`, then `brush_app` on PATH.
- Flags checked with `brush_app.exe --help` (v0.3.0, prints `brush-cli 0.3.0`): the dataset folder (`colmap/`, with `images/` + `sparse/0/`) is a **positional argument** (there is no `train` subcommand); `--total-steps`, `--sh-degree`, `--max-resolution` (default: `frames.max_long_side`, i.e. no further downscaling), `--max-splats` (new `train.max_splats`), `--seed 42`, and `--export-every <total-steps> --export-path train/ --export-name export_{iter}.ply` so only the final splat is written (`{iter}` is replaced by Brush, verified: `export_100.ply`). No `--with-viewer`, so it runs headless.
- `RUST_LOG=info` is set so Brush logs the GPU it uses (`cubecl_wgpu::runtime Using adapter AdapterInfo { name: ..., device_type: ..., backend: ... }`); the pipeline records it in the manifest (`steps.train.info.gpu`) and warns if it is not a discrete GPU.
- Brush trains on all frames (`test_every` only applies to gsplat).
- **Scene transform:** Brush does not normalise the scene: its PLY stays in COLMAP's coordinates (seen in the viewer prototype: COLMAP camera positions sit inside the Brush splat). This is **checked on every run**: COLMAP's sparse points lie on the room's surfaces, where the splat has its Gaussians, so in the same coordinates each COLMAP point has a splat right next to it. The check samples 1,000 COLMAP points and 50,000 splats (fixed seed, < 1 s) and requires the median nearest-splat distance to be below 1.5% of the scene size (median interquartile range of the COLMAP points). If it passes, `train/transform.json` and the manifest record the identity matrix with the check's numbers; if not, the step stops rather than record a wrong transform.
  A first version compared per-axis medians/spreads of both point sets; on myRoom2 one spread ratio came out at 1.50 (limit 1.6) because splats fill walls more evenly than sparse points, too close to the limit to trust, so it was replaced before being pushed.
- Brush is started in a persistent working folder (`~/.cache/colibrimo-pipeline/brush`) because it writes its GPU kernel-tuning cache (`target/autotune`) there; the PLY goes to `train/` through `--export-path`.

**Measured (2026-10-13, myRoom2 exhaustive run: 3 videos, 300 frames; Brush v0.3.0, `max_splats` 1,000,000, `max_resolution` 1280, SH degree 3, 30,000 steps):** Brush used the **NVIDIA T500** (`DiscreteGpu`, Vulkan) after setting `brush_app.exe` to High performance. Training took **58 min 50 s**; growth stopped at step 15,000 with **282,405 splats** (well under the 1,000,000 cap); `scene.ply` is **66.6 MB**. Coordinate check: median nearest-splat distance 0.0072 x scene size (threshold 0.015); the same splat moved by 10% of the scene size gives 0.064, rotated by 10 degrees 0.029.

**Consequences:** the team laptop can run the whole pipeline locally (no Colab needed for Brush). On laptops with two GPUs, Windows may give Brush the integrated one; set `brush_app.exe` to "High performance" in Windows Settings > System > Display > Graphics (the pipeline warns when this happens). gsplat stays the default for Linux GPU machines and Docker.

---

## D-014 · Unused COLMAP models leave `colmap/sparse/` · 2026-10-13

**Context:** COLMAP's mapper can split a capture into several models. The pipeline kept the largest as `sparse/0` and renamed the others `sparse/unused_N`. gsplat reads `sparse/0` explicitly, but **Brush loaded a small leftover model** instead of the real one when others were still inside the folder (manual test).

**Decision:** `poses` keeps only the largest model in `sparse/` (as `0`) and moves the others to `colmap/unused_models/model_N` (kept for debugging; cleared when `poses` re-runs). `train` with Brush also moves anything other than `0` out of `sparse/` before training, so run folders made before this change work too (idempotent, never overwrites).

**Consequences:** `colmap/sparse/` always contains exactly one model; `poses.info.other_model_sizes` still lists the other models' sizes.

---

## D-015 · `cameras.json`: camera poses for the viewer, in scene.ply's coordinates · 2026-10-13

**Context:** the viewer prototype levels the room from the cameras' "up" direction (phones are held upright), takes the eye height from the `middle_*` cameras and limits the walk area to where the cameras were. It read COLMAP's binary files through its own conversion script. The poses must be in the same coordinates as the splat, which differs between trainers (gsplat normalises, Brush does not).

**Decision:** `export` also writes `cameras.json`: for every registered frame its `name`, source `video`, `frame_index` and `time_s` (from `frames.json`; for older run folders the video is taken from the frame name), `position` (camera centre), `rotation_wxyz` (world-from-camera quaternion), `up` and `forward` unit vectors, and `camera_id`; plus `intrinsics` per COLMAP camera, `videos` (frames per video) and the `transform_4x4` used. Poses are converted from COLMAP's camera-from-world (`C = -Rᵀt`, axes = rows of R) and mapped with the run's scene transform `M = [sQ | m]`: `C' = sQC + m`, axes `Q·axis`. A small `images.bin` / `cameras.bin` reader lives in `utils/colmap_model.py` (camera model ids from COLMAP's `models.h`).

**Verified (2026-10-13, myRoom2, Brush/identity):** 300 cameras (bottom 121, middle 106, up 73); positions, up and forward identical (max difference 0.0) to the viewer prototype's independent JavaScript converter on the same `images.bin`. The gsplat case is covered by a test with a known scale + rotation + translation (a point in front of a camera stays in front of it at the scaled distance).

**Consequences:** the viewer only needs `scene.ply` + `cameras.json` from a run folder; no COLMAP files.

---

## D-016 · `cpipe ui`: a local web page for the pipeline (developer tool, not Step 5) · 2026-10-13

**Context:** testing captures means running long `cpipe run` commands with several options (videos, Brush settings, COLMAP on the GPU), watching the logs and resuming after failures. The team wants to do that by dropping videos on a page. The real service for this is PRD **Step 5** (Django + DRF API, Celery worker, MinIO storage), which is not started yet and is much bigger.

**Decision:** a small **local developer tool**, `cpipe ui`, inside the pipeline package:
- Python standard library only (`http.server` + a background thread; one HTML page with plain JavaScript): no new dependency, no database, no login.
- It calls `run_pipeline()` as a library: videos in `data/uploads/<room>/`, runs in `data/runs/<room>/`, one run at a time. Defaults: Brush, `max_splats` 1,000,000, `max_resolution` 1280 (a 4 GB GPU), COLMAP on the GPU when it is a CUDA build; `--set` overrides.
- Progress is read only from the run folder (`manifest.json` + Brush's `Refine iter` lines in `logs/train.log`), so the pipeline itself is unchanged. Resume = calling `run_pipeline()` again on the same folder (its resumability skips finished steps).
- Privacy: it listens on **127.0.0.1 only** (phones and other computers cannot connect), and rejects requests whose `Host`/`Origin` is not this computer (protection against DNS rebinding and cross-site requests from other web pages). Uploaded file and room names are reduced to safe characters; only video extensions are accepted.
- "Open in viewer" copies `scene.ply` + `cameras.json` into the viewer prototype's `public/runs/<room>/` (`--viewer`, `CPIPE_VIEWER_DIR`, or `~/3DViewing`) and opens `http://localhost:5173/?run=runs/<room>`.
- `Manifest.save()` now retries its atomic rename briefly: on Windows it fails while another reader (this page) has `manifest.json` open.

**Update 2026-10-14:** each uploaded video can be labelled bottom / middle / up on the page (default from the file name, e.g. `middle_take2.mp4`; saved in `uploads/<room>/labels.json`, files are not renamed). The label becomes the video's name in the run (`run_pipeline()` accepts named videos), so frames, `cameras.json` and the viewer's eye height (`middle`) work for files named by date. Changing the videos or labels of a room makes the next **Start** a fresh run (`force`), because the old frames carry the old names; **Resume** still reuses finished steps. The page shows each video's resolution/orientation/fps/codec and warns about mixed orientations and a missing "middle".

**Consequences:** **Step 5 replaces this tool** (uploads from the phone, jobs in a queue, storage, several users); nothing in the backend should build on it. No cancel button (stop the server with Ctrl+C; a running Brush process may need ending in Task Manager). Runs stop if the laptop sleeps or the server is closed; they can be resumed.

---

## D-017 · One COLMAP camera per video when the videos differ (never one per frame) · 2026-10-14

**Context:** D-010 used one shared camera when all videos have the same resolution/orientation, and otherwise `ImageReader.single_camera=0`, i.e. **one camera model per frame** (300 unknown focal lengths instead of a few), which is wrong in principle. It showed up in the NewRoom run (Samsung, two portrait passes + one landscape, 300 frames): 68 of 300 frames placed (23%), 5 models split along the walk (not by video), as many camera models as frames, and repeated "Matrix not positive definite" warnings in the mapper. The camera setting was suspected, but the measurements below show it was **not** the cause of that failure.

**Decision:** when the videos differ, `poses` runs COLMAP's `feature_extractor` once per video, with `--image_list_path` (that video's frames, from `frames.json`, written to `colmap/image_lists/<video>.txt`) and `ImageReader.single_camera 1`: one camera per video. Frames stay in the flat `colmap/images/` folder, so gsplat, Brush and `cameras.json` are unaffected (`ImageReader.single_camera_per_folder` would have needed sub-folders). Same-looking videos keep one shared camera. Never one camera per frame. `validate` now names each video's orientation and resolution in its warning and recommends filming all passes the same way (landscape); `cpipe ui` shows it before the run.

**Measured on NewRoom (2026-10-14, COLMAP 4.2.1 CUDA, T500, `balanced`, 300 frames):**

| Run | Cameras | Matcher | Frames placed | Models (sizes) |
|---|---|---|---|---|
| 3 videos, before (one camera per frame) | 300 | sequential + loop detection | 68 (23%) | 5 (68, 26, 25, 25, 20) |
| 3 videos, after (one camera per video) | 3 | sequential + loop detection | 68 (23%) | 5 (68, 26, 25, 20, 3) |
| 2 portrait videos only (one shared camera) | 1 | exhaustive (90 min) | 101 (34%) | 5 (101, 70, 65, 34, 10) |
| Manual method, no pipeline: plain FFmpeg `fps=2` (515 frames, no sharpest-frame selection), `colmap automatic_reconstructor --data_type video --quality medium --dense 0 --single_camera_per_folder 1` (6 min) | 3 | automatic (video: sequential + loop detection) | 136 of 515 (26%) | 8 (136, 52, 46, 43, 35, 19, 16, 14) |

So the camera models were **not** what broke NewRoom: with one shared camera and exhaustive matching, 280 of 300 frames are placed but in 5 pieces. Each piece mixes the videos and covers a stretch of the walk; in both portrait videos the walk breaks at the same moments (about 17 s, 35 s, 49 s and 62-66 s of an 80 s lap, i.e. at the room's corners), where frames are blurrier and overlap less (4-6 verified neighbours per frame vs 11-17 for myRoom2). The manual COLMAP method that worked on myRoom/myRoom2 fails the same way on these videos (largest model 26% of frames, 8 pieces), so the pipeline is not the cause. **Cause: the capture, most likely turning too fast at the corners**; the camera fix did **not** rescue NewRoom. Fix: re-film.

**Consequences:** the per-video camera is still kept: one camera per frame is wrong in principle and slows the mapper down. COLMAP is called once per video for feature extraction (same total work). Videos filmed with different settings still cost reconstruction quality: the capture guide asks for one orientation, 30 fps and H.264, and slow turns at the corners.
