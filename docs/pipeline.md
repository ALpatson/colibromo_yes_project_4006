# Reconstruction pipeline (`cpipe`)

Turns a phone video of one room into a 3D Gaussian Splatting scene (`scene.ply`).
Code: [pipeline/](../pipeline/). Design decisions: [decisions.md](decisions.md) (D-005 to D-008).

```
video ──validate──▶ extract_frames ──▶ poses ──▶ train ──▶ export ──▶ scene.ply
        (ffprobe)     (FFmpeg)        (COLMAP)   (gsplat,    (copy +
                                                  NVIDIA GPU)  transform)
```

| Step | What it does | Runs on |
|---|---|---|
| `validate` | Reads the video with ffprobe. Fails if shorter than 20 s or longer than 5 min; warns below 1080p. | any machine |
| `extract_frames` | Extracts 2x the target frame count with FFmpeg, keeps the sharpest frame of each group, downscales. Fails if the video looks static. | any machine |
| `poses` | COLMAP: feature extraction, sequential matching, mapper. Keeps the largest model. Fails if fewer than 70% of frames are registered. | any machine (CPU) |
| `train` | gsplat `simple_trainer.py` (headless, seed 42). | **NVIDIA GPU only** |
| `export` | Copies the trained splat to `scene.ply`, records the scene transform in the manifest. | any machine |

## Where to run it

We have no GPU machine yet (PRD §11 Q1), so the normal workflow is:

1. **Laptop (CPU):** `validate`, `extract_frames`, `poses` with `--until poses`.
   COLMAP is faster on a laptop (6+ cores) than on Colab (2 cores).
2. **Google Colab (free T4 GPU):** copy the run folder to Google Drive, open
   [pipeline/notebooks/colab_run.ipynb](../pipeline/notebooks/colab_run.ipynb), run `train` + `export`.

Or run everything on Colab with the notebook (simplest, slower poses). Completed steps are
skipped automatically, so a run folder can move between machines.

## Install on a laptop (Windows, CPU steps)

1. Python 3.11+ and the repo's virtual environment (see the root README).
2. FFmpeg and COLMAP. On the team laptop they live in `D:\tools`:
   - FFmpeg 9.0.2 essentials build from https://www.gyan.dev/ffmpeg/builds/ (unzip to `D:\tools\ffmpeg`)
   - COLMAP 4.2.1 `colmap-x64-windows-nocuda.zip` from https://github.com/colmap/colmap/releases (unzip to `D:\tools\colmap`)
3. Tell the pipeline where they are (or add their `bin` folders to PATH):
   ```powershell
   $env:CPIPE_FFMPEG  = "D:\tools\ffmpeg\bin\ffmpeg.exe"
   $env:CPIPE_FFPROBE = "D:\tools\ffmpeg\bin\ffprobe.exe"
   $env:CPIPE_COLMAP  = "D:\tools\colmap\bin\colmap.exe"
   ```
   Use `setx NAME value` instead of `$env:NAME = value` to make them permanent (new terminals).
4. Install the pipeline (editable, with test tools):
   ```powershell
   .venv\Scripts\activate
   pip install -e "pipeline[dev]"
   cpipe --version
   ```

On Linux/macOS: install `ffmpeg` and `colmap` with your package manager, then step 4.

## Usage

```bash
cpipe run --video data/videos/room1.mp4 --output data/runs/room1 --preset balanced
```

| Option | Meaning |
|---|---|
| `--preset NAME\|FILE` | `fast`, `balanced` (default), `quality`, or your own YAML file |
| `--frames N` | number of frames kept (`frames.target_count`) |
| `--steps N` | training steps (`train.max_steps`) |
| `--set KEY=VALUE` | any setting, repeatable, e.g. `--set poses.matcher=exhaustive` |
| `--until STEP` | stop after a step, e.g. `--until poses` on a machine without GPU |
| `--from-step STEP` | re-run from this step (and all later ones) |
| `--force` | re-run every step |
| `-v` | print all tool output on the console |

`cpipe config --preset fast --set frames.target_count=200` prints the resolved settings without
running anything. Exit codes: `0` success, `1` a step failed, `2` configuration error.

**Resumability:** a step is skipped if `manifest.json` says it succeeded and its outputs still
exist. When a step re-runs, all later steps re-run too. Changing settings in an existing run
folder only prints a warning: use `--from-step` or `--force` to apply them.

### As a Python library (used by the backend worker in Step 5)

```python
from colibrimo_pipeline import load_config, run_pipeline, StepError

config = load_config("balanced", {"train": {"max_steps": 20000}})
try:
    manifest = run_pipeline("room1.mp4", "data/runs/room1", config)
except StepError as exc:
    print(exc.step, exc.message)
```

## Presets

Defined in [pipeline/colibrimo_pipeline/presets/](../pipeline/colibrimo_pipeline/presets/); all
other settings and their meaning are in
[config.py](../pipeline/colibrimo_pipeline/config.py).

| Preset | Frames | Max frame size | Training steps | Use for |
|---|---|---|---|---|
| `fast` | 150 | 1280 px | 7,000 | checking a capture quickly |
| `balanced` | 300 | 1600 px | 30,000 | normal runs |
| `quality` | 500 | 1920 px | 30,000 | best result (more matching overlap) |

Timings per preset on a Colab T4 are **not measured yet** (Step 2 benchmarks).

## Outputs (run folder)

```
data/runs/room1/
├── manifest.json            inputs, config, versions, git commit, per-step status/timings/results
├── scene.ply                final Gaussian Splat (normalised coordinates, see "transform")
├── input/video_info.json    ffprobe metadata
├── colmap/images/           selected frames (frame_00000.jpg ...)
├── colmap/database.db       COLMAP features and matches
├── colmap/sparse/0/         camera poses + sparse points (largest model; others: unused_N)
├── train/ply/               trained splat from gsplat
├── train/ckpts/             gsplat checkpoint (for evaluation in Step 2)
├── train/transform.json     scene normalisation applied by gsplat
└── logs/                    pipeline.log (everything) + one log per step
```

**`manifest.json`** contains: `input` (video path), `config` (fully resolved), `tool_versions`
(ffmpeg, ffprobe, COLMAP, and on GPU runs Python, torch, CUDA, gsplat, GPU model and memory),
`git_commit`, `pipeline_version`, `system`, `status`, `total_duration_s`, and per step:
`status`, `started_at`, `finished_at`, `duration_s`, `outputs`, `info` (e.g. frames registered by
COLMAP, number of Gaussians) and `error` if it failed.

**`transform`** (in the manifest after `export`): the 4x4 matrix gsplat applied to the COLMAP
coordinates (recentre, rotate, rescale) before training, so `x_scene = M @ [x, y, z, 1]`. Future
"after" variants of the room must use the same frame (PRD §7). Gravity alignment and floor at
y=0 come in Step 3.

## Google Colab

Open [pipeline/notebooks/colab_run.ipynb](../pipeline/notebooks/colab_run.ipynb) in Colab
(*File > Open notebook > GitHub*, paste the repository URL), choose a **T4 GPU** runtime, edit the
settings cell, and *Run all*. The notebook:

- installs COLMAP and FFmpeg (Ubuntu packages);
- compiles **gsplat 1.5.3** and **fused-ssim** once for the current Colab Python/PyTorch and
  caches the builds on your Drive (`MyDrive/colibrimo/wheels/`): about 25 min the first time,
  about 2 min afterwards. Colab upgrades (e.g. Python 3.12 to 3.13 on 2026-09-16) trigger one new
  compile automatically;
- installs gsplat's training script and requirements, and this pipeline from GitHub;
- runs `cpipe` on local disk and copies the run folder to `MyDrive/colibrimo/runs/<RUN_NAME>/`.

To continue a laptop run on Colab: copy the laptop's run folder (e.g. `data/runs/room1`) to
`MyDrive/colibrimo/runs/room1/` (the `frames_candidates` folder is not needed), set
`RUN_NAME = "room1"` and run the notebook. Steps up to `poses` are skipped.

To look at `scene.ply`: drop it on [SuperSplat](https://superspl.at/editor) (in-browser, no
upload). Our viewer comes in Step 4.

## Docker (GPU machine)

[pipeline/Dockerfile](../pipeline/Dockerfile): CUDA 13.0 + PyTorch 2.11 + gsplat 1.5.3 +
COLMAP + FFmpeg + this pipeline.

```bash
docker build -t colibrimo-pipeline --build-arg GIT_COMMIT=$(git rev-parse HEAD) pipeline
docker run --gpus all --rm -v "$PWD/data:/work/data" colibrimo-pipeline \
  run --video data/videos/room1.mp4 --output data/runs/room1 --preset balanced
```

**Not tested yet**: no Linux + NVIDIA machine is available. It mirrors the setup that works on
Colab, but expect to fix small issues the first time it is built.

## Tests

```bash
pytest pipeline            # from the repo root
```

Tests that need FFmpeg (marker `tools`) are skipped if FFmpeg is missing; GPU tests (marker
`gpu`) are skipped without CUDA. The FFmpeg tests generate synthetic videos and check, among
other things, that a 5 s video fails at `validate` and a static shot fails at `extract_frames`.

## Troubleshooting

| Message | Cause / fix |
|---|---|
| `'colmap' was not found` | Install it or set `CPIPE_COLMAP` (same for ffmpeg/ffprobe). |
| `The video is 5.0 s long, but at least 20 s is needed` | Film 1 to 3 minutes (capture guide). |
| `The video looks static` | The camera must move through the room. |
| `COLMAP placed only X of Y frames` | Capture problem (fast motion, blur, mirrors, low light, turning on the spot). Try `--set poses.matcher=exhaustive` (slower) or re-film. |
| `Training needs an NVIDIA GPU with CUDA` | Run `--until poses` here and train on Colab. |
| `gsplat's examples folder is not configured` | Set `GSPLAT_EXAMPLES_DIR` to `<gsplat checkout>/examples`. |
| Colab: compile killed (exit 137) | Out of RAM; keep `MAX_JOBS=2` (set by the notebook). |
