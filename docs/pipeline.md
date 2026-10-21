# Reconstruction pipeline (`cpipe`)

Turns one or more phone videos of one room into a 3D Gaussian Splatting scene (`scene.ply`).
Code: [pipeline/](../pipeline/). Design decisions: [decisions.md](decisions.md) (D-005 to D-019).

```
video(s) ──validate──▶ extract_frames ──▶ poses ──▶ train ──▶ export ──▶ scene.ply
           (ffprobe)     (FFmpeg)        (COLMAP)   (gsplat or   (copy +
                                                     Brush)       transform)
```

| Step | What it does | Runs on |
|---|---|---|
| `validate` | Reads each video with ffprobe. Fails if all videos together are shorter than 20 s or one video is longer than 5 min; warns below 1080p. | any machine |
| `extract_frames` | Shares the target frame count between the videos by duration. Per video: extracts 2x its share with FFmpeg, keeps the sharpest frame of each group, downscales. Fails if a video looks static. | any machine |
| `poses` | COLMAP: feature extraction (one shared camera model, or one per video if the videos differ in resolution/orientation, D-017), matching (sequential for one video, sequential + loop detection for several), mapper. Keeps the largest model (others go to `colmap_unused/`, outside `colmap/`, D-018). Fails if fewer than 70% of frames are registered. | any machine (CPU, or GPU with a CUDA build) |
| `train` | `train.trainer: gsplat` (default): gsplat `simple_trainer.py` (headless, seed 42). `train.trainer: brush`: Brush (headless, seed 42). | gsplat: **Linux + NVIDIA GPU**; Brush: **any GPU, Windows too** |
| `export` | Copies the trained splat to `scene.ply`, writes `cameras.json` (camera poses for the viewer), records the scene transform in the manifest. | any machine |

How to film a room (three heights, slow turns, locked exposure, nobody moving):
[capture-guide.md](capture-guide.md).

## Where to run it

We have no GPU machine yet (PRD §11 Q1), so the normal workflow is:

1. **Laptop:** `validate`, `extract_frames`, `poses` with `--until poses` (in `cpipe ui`:
   *Stop after COLMAP (train on Colab)*). COLMAP is faster on a laptop (6+ cores, CUDA COLMAP)
   than on Colab (2 cores).
2. **Google Colab (free T4 GPU):** `train` (gsplat) + `export`, see
   [Training on Google Colab](#training-on-google-colab).

Or run everything on Colab with the notebook (slower poses). Completed steps are skipped
automatically, so a run folder can move between machines; `cpipe run --output <run>` without
`--video` continues a run folder with the videos its manifest records.

**Or everything on the laptop with Brush** (any GPU; see [Training with Brush](#training-with-brush)):
`cpipe run ... --set train.trainer=brush`.

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

   **Laptop with an NVIDIA GPU** (e.g. NVIDIA T500): use the CUDA build
   `colmap-x64-windows-cuda.zip` instead and run COLMAP on the GPU with
   `--set poses.use_gpu=true` (much faster feature extraction and matching). On the
   test laptop it is unzipped in Downloads:
   ```powershell
   $env:CPIPE_COLMAP = "C:\Users\JOSEPHINE\Downloads\colmap-x64-windows-cuda\bin\colmap.exe"
   ```
4. Install the pipeline (editable, with test tools):
   ```powershell
   .venv\Scripts\activate
   pip install -e "pipeline[dev]"
   cpipe --version
   ```

On Linux/macOS: install `ffmpeg` and `colmap` with your package manager, then step 4.

## Training with Brush

[Brush](https://github.com/ArthurBrussee/brush) trains on NVIDIA, AMD and Intel GPUs, on
Windows too, so the laptop can run the whole pipeline (decision D-013).

1. Download a release (tested: v0.3.0, `brush-app-x86_64-pc-windows-msvc`) and unzip it.
2. Tell the pipeline where `brush_app.exe` is (or add it to PATH, or set `train.brush_exe`):
   ```powershell
   $env:CPIPE_BRUSH = "C:\Users\JOSEPHINE\Downloads\brush-app-x86_64-pc-windows-msvc\brush_app.exe"
   ```
3. **Laptops with two GPUs** (e.g. Intel Iris Xe + NVIDIA T500): Windows may give Brush the
   integrated GPU, which is much slower. In Windows **Settings > System > Display > Graphics**,
   add `brush_app.exe` and set it to **High performance** (the NVIDIA GPU). The pipeline logs
   the GPU Brush uses (`Brush GPU: NVIDIA T500 (DiscreteGpu, Vulkan)`, also in the manifest
   under `steps.train.info.gpu`) and warns if it is not a dedicated GPU.
4. Run, e.g. with the settings that fit a 4 GB GPU:
   ```powershell
   cpipe run --videos C:\path\to\room --output data\runs\room `
     --set train.trainer=brush --set train.max_splats=1000000 --set train.max_resolution=1280
   ```

| Setting | Default | Meaning (Brush) |
|---|---|---|
| `train.max_steps` | 30,000 (`fast`: 7,000) | `--total-steps` |
| `train.sh_degree` | 3 | `--sh-degree` |
| `train.max_splats` | Brush's default (10,000,000) | `--max-splats`; ~1,000,000 for a 4 GB GPU |
| `train.max_resolution` | `frames.max_long_side` | `--max-resolution` (longest image side used for training) |
| `train.extra_args` | `[]` | any other `brush_app` option |

Brush keeps COLMAP's coordinates (no normalisation): the step checks this on every run
(COLMAP's points must lie right next to splats) and records the identity transform (see
"Outputs" below). Brush's GPU tuning cache lives in `~/.cache/colibrimo-pipeline/brush`.

Measured on myRoom2 (3 videos, 300 frames, NVIDIA T500, the settings above, SH degree 3,
30,000 steps): **59 min** of training, 282,405 splats, `scene.ply` 66.6 MB. Long runs: set
Windows sleep to **Never** while plugged in. Brush logs its progress every 200 steps; the
console shows `Brush: step N / 30000, M splats` every 2,000 steps.

## Usage

```bash
cpipe run --video data/videos/room1.mp4 --output data/runs/room1 --preset balanced

# One room filmed as several videos (e.g. at three heights):
cpipe run --video bottom.mp4 --video middle.mp4 --video up.mp4 --output data/runs/room2
cpipe run --videos data/videos/room2 --output data/runs/room2   # every video in the folder
```

| Option | Meaning |
|---|---|
| `--video FILE` | input video; repeat for several videos of the same room |
| `--videos FOLDER` | every video file (`.mp4 .mov .m4v .mkv .avi .webm`) in the folder, sorted by name |
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

### Several videos per room

Film the room in several passes (e.g. three heights, see the
[capture guide](capture-guide.md)) and give all videos to one run:

- Each video is named after its file (`bottom.mp4` → `bottom`); names must be unique.
- The frame budget (`frames.target_count`) is shared in proportion to each video's
  duration, with at least `frames.min_per_video` (20) per video. Example, myRoom2 with
  300 frames: bottom 93.5 s → 121, middle 81.9 s → 106, up 56.9 s → 73.
- Frames are named after their video (`bottom_00000.jpg` …, in time order);
  `frames.json` lists each frame's video and timestamp.
- Matching: plain sequential matching never connects separate videos, so with
  `poses.matcher: auto` (default) several videos use **sequential + loop detection**
  (COLMAP downloads its default vocabulary tree the first time, so internet is needed
  once; or set `poses.vocab_tree_path`). The choice and reason are in the manifest
  (`steps.poses.info.matcher_reason`). `--set poses.matcher=exhaustive` matches every
  pair instead: also connects the videos, but about twice as slow.
- Timing, myRoom2 (3 videos, 300 frames, CUDA COLMAP on an NVIDIA T500), both put all
  three videos in one model (details in [decisions.md](decisions.md) D-011):

  | Matcher | `poses` time | Frames registered |
  |---|---|---|
  | sequential + loop detection (default) | 20 min | 296 / 300 |
  | exhaustive | ~41 min | 300 / 300 |

- Long runs: keep the laptop plugged in and set Windows sleep to **Never** (Settings →
  System → Power), otherwise sleep pauses COLMAP.

### HDR videos (iPhone)

iPhones film in HDR (HLG) by default. `validate` detects HDR (`color_transfer` HLG
`arib-std-b67` or PQ `smpte2084`; see `steps.validate.info.hdr_videos`) and
`extract_frames` tone-maps those frames to normal SDR colours with FFmpeg's `zscale` +
`tonemap` filters (decision D-012). Only the main video stream is read (`-map 0:v:0`).

| Setting | Default | Meaning |
|---|---|---|
| `frames.hdr_tonemap` | `true` | `false` extracts HDR as is (washed-out colours) |
| `frames.tonemap` | `hable` | tone-mapping curve (`hable`, `mobius`, `reinhard`, `clip`) |
| `frames.hdr_peak_nits` | `100` | zscale `npl` when linearising |

This needs an FFmpeg built with libzimg (the gyan.dev **full** build on Windows has it;
check with `ffmpeg -filters | findstr zscale`).

**Resumability:** a step is skipped if `manifest.json` says it succeeded and its outputs still
exist. When a step re-runs, all later steps re-run too. Changing settings in an existing run
folder only prints a warning: use `--from-step` or `--force` to apply them.

### Local web page (`cpipe ui`)

> **Developer tool for the team, not the Stage 1 backend.** The PRD's Step 5 (Django API +
> Celery worker) will replace it; see decision D-016.

```powershell
$env:CPIPE_COLMAP = "C:\Users\JOSEPHINE\Downloads\colmap-x64-windows-cuda\bin\colmap.exe"
$env:CPIPE_BRUSH  = "C:\Users\JOSEPHINE\Downloads\brush-app-x86_64-pc-windows-msvc\brush_app.exe"
cpipe ui            # opens http://127.0.0.1:8765/ in the browser; Ctrl+C to stop
```

On the page: type a **room name**, drop the room's videos (one per height), **label each
one bottom / middle / up** (the label defaults from the file name; the viewer takes the eye
height from "middle"), pick a **preset** and press **Start**. Each video shows its
orientation, resolution, fps and codec, with a warning if portrait and landscape are mixed.
Changing a room's videos or labels makes the next Start a fresh run. Videos go to `data/uploads/<room>/`, the run to
`data/runs/<room>/`. The page shows each step's status (from `manifest.json`) and, during
training, Brush's step and splat count. If a step fails, its error appears with a **Resume**
button (finished steps are skipped). At the end it shows the training time, splats and file
size, and **Open in viewer** copies `scene.ply` + `cameras.json` to the viewer's
`public/runs/<room>/` and opens `http://localhost:5173/?run=runs/<room>` (start the viewer
with `npm run dev` first).

| Option | Default | Meaning |
|---|---|---|
| `--port` | 8765 | page address `http://127.0.0.1:<port>/` |
| `--data` | `./data` | where `uploads/` and `runs/` go |
| `--viewer` | `CPIPE_VIEWER_DIR`, else the repository's `viewer/`, else `~/3DViewing` | viewer project folder (with `public/`) |
| `--set KEY=VALUE` | Brush, `train.max_splats=1000000`, `train.max_resolution=1280` | any run setting; COLMAP runs on the GPU automatically if it is a CUDA build |
| `--no-browser` | | don't open the page |

- **This computer only:** it listens on 127.0.0.1, so a phone or another computer cannot
  open it (room videos are private), and it ignores requests from other web pages.
- **Partial room:** if COLMAP placed too few frames (but at least 20), a **Continue anyway
  (partial room)** button appears. It reuses COLMAP's result and continues with the placed
  frames only; the result is marked "PARTIAL ROOM: N of M frames" (D-019).
- **One run at a time:** Start is disabled while a run is going. There is no cancel button:
  Ctrl+C stops the server (end `brush_app.exe` in Task Manager if it keeps running).
- **Long runs:** keep the laptop plugged in and set Windows sleep to **Never**.

### As a Python library (used by the backend worker in Step 5)

```python
from colibrimo_pipeline import load_config, run_pipeline, StepError

config = load_config("balanced", {"train": {"max_steps": 20000}})
try:
    manifest = run_pipeline("room1.mp4", "data/runs/room1", config)
    # several videos of one room: run_pipeline(["bottom.mp4", "middle.mp4", "up.mp4"], ...)
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
├── scene.ply                final Gaussian Splat (coordinates: see "transform")
├── cameras.json             every frame's camera pose in scene.ply's coordinates (for the viewer)
├── input/videos.json        ffprobe metadata of each video
├── frames.json              every frame: file, source video, index, timestamp
├── poses_attempt.json       last COLMAP result + the settings it used (for "continue anyway")
├── colmap/images/           selected frames, named after their video (bottom_00000.jpg ...)
├── colmap/database.db       COLMAP features and matches
├── colmap/sparse/0/         camera poses + sparse points (the only model in sparse/)
├── colmap_unused/           other, smaller models COLMAP made, if any (model_1 ...); kept
│                            outside colmap/ because Brush reads everything in it (D-018)
├── train/ply/               gsplat: trained splat
├── train/ckpts/             gsplat: checkpoint (for evaluation in Step 2)
├── train/export_30000.ply   Brush: trained splat (number = training steps)
├── train/transform.json     scene transform applied by the trainer
└── logs/                    pipeline.log (everything) + one log per step
```

**`manifest.json`** contains: `input` (`videos`: list of paths), `config` (fully resolved), `tool_versions`
(ffmpeg, ffprobe, COLMAP, and on gsplat runs Python, torch, CUDA, gsplat, GPU model and memory;
on Brush runs the Brush version, with the GPU in `steps.train.info.gpu`),
`git_commit`, `pipeline_version`, `system`, `status`, `total_duration_s`, and per step:
`status`, `started_at`, `finished_at`, `duration_s`, `outputs`, `info` (e.g. frames registered by
COLMAP, number of Gaussians) and `error` if it failed.

**`transform`** (in the manifest after `export`): the 4x4 matrix mapping COLMAP coordinates to
`scene.ply`, `x_scene = M @ [x, y, z, 1]`. gsplat recentres, rotates and rescales the scene
before training (`normalize_world_space`); Brush keeps COLMAP's coordinates, so `M` is the
identity, with `identity_check` showing that the splat really matches COLMAP's points. Future
"after" variants of the room must use the same frame (PRD §7). Gravity alignment and floor at
y=0 come in Step 3.

**`cameras.json`** (written by `export`, decision D-015): where each frame was filmed from, in
the same coordinates as `scene.ply` (the transform above is already applied). The viewer uses it
to level the room, set the eye height (`middle_*` cameras) and limit the walk area.

```json
{
  "schema_version": 1,
  "num_cameras": 300,
  "videos": {"bottom": 121, "middle": 106, "up": 73},
  "intrinsics": {"1": {"model": "SIMPLE_RADIAL", "width": 1600, "height": 900, "params": [785.3, 800, 450, -0.0024]}},
  "cameras": [
    {"name": "bottom_00000.jpg", "video": "bottom", "frame_index": 0, "time_s": 0.36,
     "position": [x, y, z], "rotation_wxyz": [w, x, y, z],
     "up": [x, y, z], "forward": [x, y, z], "camera_id": 1}
  ]
}
```

Camera axes follow COLMAP/OpenCV (x right, y down, z forward): `rotation_wxyz` is the
world-from-camera rotation, `up` = −y and `forward` = +z as unit vectors.

## Training on Google Colab

The laptop does validate → extract_frames → poses; Colab does train + export with gsplat on a
T4 (decision D-020).

**Privacy:** the zip holds the room's frames. Only put our own test rooms on Google Drive until
Colibrimo approves storing client data there.

1. **Run up to COLMAP.** In `cpipe ui`, tick *Stop after COLMAP (train on Colab)* and click
   *Start* (CLI: `cpipe run ... --until poses`). The run stops with status "ready for Colab".
2. **Zip.** Click *Export for Colab*. The page shows the zip's path and size:
   `data/colab/<room>_for_colab.zip` (myRoom2: 108 MB, 310 files). It holds only the manifest,
   `frames.json`, `input/videos.json`, the frames, COLMAP's model and database: no videos, no
   logs, no candidate frames, no unused COLMAP models.
3. **Upload** the zip to Google Drive: `MyDrive/colibrimo/runs/<room>/`.
4. **Open the notebook** in Colab: *File > Open notebook > GitHub*, repository
   `ALpatson/colibromo_yes_project_4006`, branch `step-1-pipeline`,
   `pipeline/notebooks/colab_run.ipynb`.
5. **T4 runtime:** *Runtime > Change runtime type > T4 GPU*.
6. **Settings cell:** `RUN_NAME = "<room>"`; `PRESET` as on the laptop (`balanced` by default);
   leave `VIDEOS = []`.
7. **Run all.** Allow Drive access when asked. The first session compiles **gsplat 1.5.3** and
   **fused-ssim** for the current Colab Python/PyTorch (**about 25 min**, cached in
   `MyDrive/colibrimo/wheels/`); later sessions install in about 2 min. Colab upgrades (e.g.
   Python 3.12 to 3.13 on 2026-09-16) trigger one new compile automatically. The notebook then
   unzips the run, runs `cpipe run --output <run> --set train.trainer=gsplat` (validate,
   extract_frames and poses are reused), and prints `cpipe check`: the coordinate check and the
   camera count. All lines should say OK.
8. **Download the result:** `MyDrive/colibrimo/runs/<room>/<room>_colab_result.zip` (scene.ply,
   cameras.json, manifest.json, gsplat's transform, train/export logs). The notebook's last cell
   can also download it directly. Colab's logs are copied to `logs_colab/` in the same Drive folder.
9. **Import.** In `cpipe ui`, open the room and click *Import Colab result*. Choose the zip (or
   `scene.ply`, `cameras.json` and `manifest.json` together). The import is refused, and nothing
   changes, unless the result was trained from this room's current COLMAP result and passes the
   checks. Then *Open in viewer* works as usual; the result card says "Trained on Google Colab".

*Train on this laptop instead* (on the Colab card) continues the run here with Brush.

**What `cpipe check` checks** (also run on import, and by the `train` step itself after gsplat):

- *Coordinate check:* COLMAP's 3D points, moved by the scene transform recorded in the manifest
  (gsplat's normalisation; identity for Brush), lie right next to splats of `scene.ply`: median
  distance below 0.015 × scene size (D-013).
- *Cameras:* `cameras.json` has one camera per frame COLMAP placed, built with the same transform.

**Full run on Colab** (no laptop): put the videos on Drive, set
`VIDEOS = ["/content/drive/MyDrive/colibrimo/videos/room1.mp4", ...]` and run the notebook; it
then also installs COLMAP and FFmpeg and keeps the whole run folder on Drive.

To look at `scene.ply` without our viewer: drop it on [SuperSplat](https://superspl.at/editor)
(in-browser, no upload).

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

| Test file | Covers |
|---|---|
| `test_multi_video.py`, `test_end_to_end_cpu.py` | several videos: names, frame split by duration, per-video naming and `frames.json`, duration rules, matcher choice, `--videos` folder |
| `test_hdr.py` | HDR detection (HLG/PQ/SDR), the tone-mapping filter chain, a real FFmpeg run on a synthetic HLG clip (skipped without libx264/zscale), missing zscale |
| `test_brush.py` | Brush command line (v0.3.0 flags), GPU adapter/progress parsing, moving unused COLMAP models (incl. old layout), the coordinate check, PLY/points readers |
| `test_cameras.py` | `cameras.json`: COLMAP readers, quaternions, identity and gsplat transforms |
| `test_colab.py`, `test_ui_colab.py` | continuing a run without its videos, gsplat's coordinate check (fake trainer), `cpipe check`, the Colab zip (contents), result zip round trip, refused imports, `cpipe ui` Colab buttons |

Not covered by automated tests (needs the real tools; checked by hand on myRoom2, see
decisions D-011 to D-015): a full COLMAP run, Brush training. gsplat training on Colab with the
current code is not verified yet (D-020).

## Troubleshooting

| Message | Cause / fix |
|---|---|
| `'colmap' was not found` | Install it or set `CPIPE_COLMAP` (same for ffmpeg/ffprobe). |
| `The video is 5.0 s long, but at least 20 s is needed` | Film 1 to 3 minutes (capture guide). |
| `The video looks static` | The camera must move through the room. |
| `COLMAP placed only X of Y frames` | Capture problem (fast motion, fast turns at corners, blur, mirrors, low light, turning on the spot): re-film. For a preview from the placed frames only: `--set poses.allow_partial=true` (or "Continue anyway" in `cpipe ui`); the room is marked partial and has holes (D-019). |
| `HDR video ... needs FFmpeg's zscale ... filter(s)` | Install an FFmpeg build with libzimg (gyan.dev "full" on Windows), or `--set frames.hdr_tonemap=false` (washed-out colours). |
| `Training needs an NVIDIA GPU with CUDA` | Run `--until poses` here and train on Colab, or train with Brush (`--set train.trainer=brush`). |
| `'brush_app' was not found` | Set `CPIPE_BRUSH` (or `train.brush_exe`) to `brush_app.exe`. |
| `Warning: Brush is not using a dedicated GPU` | Windows Settings > System > Display > Graphics: set `brush_app.exe` to High performance. |
| `The Brush splat does not seem to be in COLMAP's coordinates` | A Brush version that normalises the scene; the transform must be worked out before using it (D-013). |
| `gsplat's examples folder is not configured` | Set `GSPLAT_EXAMPLES_DIR` to `<gsplat checkout>/examples`. |
| `No module named 'datasets.colmap'` or `OverflowError: ... out of bounds for uint64` | Fixed automatically by the `train` step (decisions D-009): gsplat v1.5.3's examples clash with Colab's `datasets` package and its pycolmap fork predates numpy 2. If you see them, the pipeline version is too old: reinstall it. |
| Colab: compile killed (exit 137) | Out of RAM; keep `MAX_JOBS=2` (set by the notebook). |
| Import: `not trained from this room's current COLMAP result` | COLMAP ran again on the laptop after the export. Export for Colab again and retrain. |
| `The gsplat splat is not where the recorded scene transform puts COLMAP's points` | gsplat trained in other coordinates than the transform we recorded: `cameras.json` would not match. Keep the logs (`logs_colab/`) and report it (D-020); don't import that result. |
