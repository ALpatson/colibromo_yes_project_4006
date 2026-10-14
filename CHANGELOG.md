# Changelog

One entry per PRD step (newest first). Dates are YYYY-MM-DD.

## 2026-10-13 · Step 1: pipeline improvements from manual tests

- Several videos per room (D-010): `--video` repeatable, `--videos <folder>`; per-video validation and metadata; frames shared by duration and named after their video (`bottom_00000.jpg`); `frames.json` records each frame's source video and timestamp.
- `poses.matcher: auto` (D-011): sequential + loop detection for several videos (myRoom2: all 3 videos in one model in 20 min, vs ~41 min with exhaustive, still available); one COLMAP camera per frame when the videos' resolutions differ.
- HDR videos (D-012): `validate` detects HLG/PQ (iPhone); `extract_frames` reads only the main video stream and tone-maps HDR to SDR with zscale + tonemap (`frames.hdr_tonemap`, `frames.tonemap`); clear error if FFmpeg lacks zscale.
- Brush as a second trainer (D-013): `train.trainer: brush` (Windows, any GPU), found via `train.brush_exe` / `CPIPE_BRUSH`; flags from `brush_app --help` v0.3.0 (`train.max_splats`, `train.max_resolution`); logs and records the GPU adapter; checks on every run that the splat is in COLMAP's coordinates and records the identity transform.
- Unused COLMAP models move to `colmap/unused_models/` (D-014), so trainers only ever see `sparse/0`.
- `export` writes `cameras.json` (D-015): every frame's name, source video, timestamp, position and orientation in scene.ply's coordinates (gsplat transform applied), plus intrinsics; `images.bin` / `cameras.bin` readers in `utils/colmap_model.py`.
- Docs: `docs/capture-guide.md` rewritten from the myRoom2 tests (three heights, walk along the walls facing in, slow turns, locked exposure/focus, nobody moving, iPhone HDR); `docs/pipeline.md` test overview; READMEs updated. 132 CPU tests.
- `cpipe ui` (D-016): local web page (127.0.0.1 only, standard library only) to upload a room's videos, run the pipeline with Brush, follow each step and Brush's progress, resume after a failure, and open the result in the viewer prototype. A developer tool; the Step 5 backend will replace it. `Manifest.save()` retries its rename on Windows while the page reads the file.
- One COLMAP camera per video when videos differ in resolution/orientation (D-017), instead of one per frame (correct in principle; it did not rescue the failed NewRoom capture, see D-017). `validate` and `cpipe ui` warn about mixed orientations; `cpipe ui` labels videos bottom / middle / up (names in the run, `run_pipeline()` accepts named videos) and shows each video's resolution/orientation/fps/codec. Capture guide: one orientation, 30 fps, H.264 on Samsung.
- Fix (D-018): leftover COLMAP models now go to `<run>/colmap_unused/`, outside `colmap/`. Brush scans all of `colmap/` and had mixed two leftover models in run "testnew" (2,341 splats, rejected by the coordinate check). Before training, Brush's step moves old leftovers out and checks that `colmap/sparse/0` is the only model.

## 2026-10-08 · Step 1: reconstruction pipeline CLI (in progress)

- `pipeline/`: Python package `colibrimo-pipeline` with the `cpipe` CLI, also usable as a library (`run_pipeline`).
- Steps: `validate` (ffprobe), `extract_frames` (FFmpeg + blur filter + static-video check), `poses` (COLMAP, version-aware options, 70% registration threshold), `train` (gsplat `simple_trainer.py`), `export` (`scene.ply` + scene transform).
- Presets `fast` / `balanced` / `quality` (YAML) + CLI overrides (`--frames`, `--steps`, `--set`).
- `manifest.json` per run (inputs, resolved config, tool versions, git commit, per-step timings/results), combined + per-step logs, resumability (`--force`, `--from-step`, `--until`).
- Google Colab notebook for GPU training (gsplat/fused-ssim compiled once, cached on Drive).
- Dockerfile for a GPU machine (written, not tested).
- 61 CPU tests (incl. real FFmpeg runs on synthetic videos); CI installs FFmpeg and runs them.
- Decisions D-005 to D-008: gsplat instead of Nerfstudio, COLMAP called directly, Colab as GPU, presets inside the package.
- Not yet verified: a full run on a real room video (COLMAP + training on Colab).

## 2026-10-08 · Step 0: repository and tooling setup

- Monorepo layout from PRD §6 (files at the repo root), placeholder READMEs per package.
- Root README: purpose, architecture, prerequisites, quick start, team conventions (branch/PR workflow).
- `.gitignore`, `.gitattributes` (LF line endings).
- Pre-commit: Ruff + Black (Python), ESLint 10 + Prettier 3 (JS/TS, pinned in root `package.json`), general file checks incl. 1 MB max file size.
- `docker-compose.yml`: PostgreSQL 18, Redis 8, MinIO (`pgsty/minio` fork) + one-shot bucket creation.
- `.env.example` with every variable documented.
- `docs/decisions.md` started (PRD §5 decisions + Step 0 deviations: MinIO image, TypeScript pin).
- GitHub Actions CI: pre-commit lint job + one test job per package (skip until code exists).
