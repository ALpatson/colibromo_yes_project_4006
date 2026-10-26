# Benchmarks

Go/no-go criteria (initial targets, PRD Step 2):

| Criterion | Initial target |
|---|---|
| Frames registered by COLMAP | ≥ 80% |
| PSNR on held-out views | ≥ 25 dB (indicative) |
| Visual check | straight walls, sharp textures, few floaters, no large holes |
| Total processing time (balanced preset) | ≤ 45 min per room |
| Web file size after Step 3 | ≤ 50 MB |
| Mobile performance (Step 4) | ~30 fps on a mid-range phone |

## Training benchmark: visual quality, measured (myRoom2)

Goal: improve the splat's visual quality with numbers, not impressions. One room
(myRoom2: 3 videos, 300 frames, COLMAP of 2026-10-15 with all 300 frames placed),
the **same frames and COLMAP result for every run**, so only training changes,
**one change at a time**. Decision log: D-021.

### How a run is scored

- **Held-out frames:** every 8th frame, sorted by name (`train.test_every: 8`), is kept out
  of training: 38 of myRoom2's 300 frames, spread over bottom/middle/up. gsplat v1.5.3 does
  exactly this split itself (`examples/datasets/colmap.py`, frames 0, 8, 16, ...); Brush gets a
  copy of the dataset without them.
- **Scores** of the exported `scene.ply` (what the viewer gets), rendered at those 38 frames
  with gsplat's rasterizer and compared with the real frames as gsplat's loader prepares them
  (undistorted): **PSNR** (dB, higher is better), **SSIM** (0-1, higher is better), **LPIPS**
  (AlexNet, lower is better), the same metrics and settings as gsplat's own evaluation. All
  trainers are scored by this one scorer.
- **Side-by-side images** (real | rendered) for 5 fixed held-out frames, the same 5 for every
  run, plus a render with the plain COLMAP camera for the viewer check.
- Also recorded: splats, file size, training time, GPU.
- For a room's **final** result, once the settings are chosen: `train.test_every: 0` trains on
  all frames (no score).

### Before the benchmark: what the 15 Oct Colab run used

| Setting | Colab gsplat run (myRoom2, 15 Oct) |
|---|---|
| Strategy | `default` (original 3DGS growth/pruning) |
| Frames | 1600x900 as extracted (`data_factor 1`); the videos are 1920x1080 |
| Held-out frames | `test_every 8`: **38 of 300 frames were never trained on**, and nothing was scored (`--eval_steps -1`) |
| COLMAP | the 13 Oct result (myRoom2's COLMAP was redone on 15 Oct) |
| Result | 732,121 splats, 172.8 MB, 30.4 min on a T4 |

Brush on the laptop used `max_resolution 1280` (frames downscaled to 1280x720), max 1M splats,
all 300 frames.

### Variants

| Variant | Change | Compared with |
|---|---|---|
| baseline | gsplat `default` strategy, 1600x900 (as the 15 Oct run, on the current COLMAP) | – |
| mcmc-1M | MCMC strategy (its default budget, 1M splats) | baseline |
| fullres | 1920x1080 frames: the same frames re-extracted from the videos, COLMAP's cameras scaled by 1.2, poses identical (worst frame vs original: 44.4 dB, i.e. the same frames) | baseline |
| bilagrid | per-frame colour/exposure correction (`use_bilateral_grid`, gsplat v1.5.3) | baseline |
| antialiased | anti-aliased rasterization (`antialiased`, gsplat v1.5.3) | baseline |
| mcmc-2M, mcmc-3M | MCMC with a 2M / 3M splat budget | mcmc-1M |
| brush-ref | Brush on the laptop (NVIDIA T500), 1280 px, max 1M splats, same 262 training frames | reference |

Not tested: gsplat's `app_opt` (appearance embedding). Its exported PLY bakes the colours
without view-dependent colour (SH degree 0), so the viewer would lose that.

### How to run it

1. **Laptop** (done for myRoom2), files in `data/bench/upload/`:
   - `myRoom2_for_colab.zip`: frames + COLMAP (`colab.export_for_colab`);
   - `myRoom2-fullres_for_colab.zip`: `bench.prepare_fullres`, then `export_for_colab`;
   - `myRoom2-brush_for_scoring.zip`: Brush trained with `train.test_every=8`, then
     `bench.pack_for_scoring`.
2. Upload the three zips to `MyDrive/colibrimo/bench/myRoom2/`.
3. Open `pipeline/notebooks/colab_benchmark.ipynb` (branch `step-1-pipeline`) in Colab, T4 GPU,
   *Run all*. Each variant takes about 30-60 min and is saved to Drive when done; run the
   notebook again in a new session to continue.
4. Download `myRoom2_bench_results.zip` (scores, manifests, side-by-side images; no splats).

### Results (myRoom2, Colab T4, 30,000 steps, 2026-10-21)

Table from `bench.results_markdown`; per-variant files in `data/bench/myRoom2_results/` (team
laptop, outside git). 38 held-out frames each.

| Variant | Change | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Splats | Size (MB) | Training (min) | GPU mem (GB) |
|---|---|---|---|---|---|---|---|---|
| baseline | gsplat default strategy, 1600x900 frames (the Colab run of 15 Oct) | 29.73 | 0.9530 | 0.1275 | 714,621 | 168.7 | 29.1 | – |
| mcmc-1M | MCMC strategy (its default budget: 1M splats) | 29.39 | 0.9512 | 0.1420 | 1,000,000 | 236.0 | 57.9 | – |
| fullres | 1920x1080 frames (video resolution), same poses | 29.38 | 0.9514 | 0.1599 | 679,700 | 160.4 | 34.6 | – |
| bilagrid | per-frame colour/exposure correction (use_bilateral_grid) | 25.49 | 0.9461 | 0.1404 | 691,865 | 163.3 | 54.9 | – |
| antialiased | anti-aliased rasterization | 29.54 | 0.9521 | 0.1284 | 726,446 | 171.4 | 28.8 | – |

Not run: `mcmc-2M` (CUDA out of memory on the T4's 15 GB), `mcmc-3M`, `brush-ref`.
GPU memory was not recorded yet for these runs (added in D-023).

**Reading:**
- **The baseline (default settings) is the best and among the fastest.** No variant is worth its
  extra time on this room; PSNR 29.7 dB is well above the 25 dB target.
- `fullres` is scored on 1920x1080 frames, the others on 1600x900, so its numbers are not
  directly comparable (larger images are harder to match pixel for pixel).
- `bilagrid` learns a colour correction per *training* frame; held-out frames are rendered without
  one, which lowers PSNR by construction. It may still look better; judge it on the images.

### Next: quality against training time (Room4, D-023)

Variant `steps-curve`: baseline settings, scored at 7k, 15k and 30k steps in one run, plus
`brush-ref` (Brush on the team laptop's AMD Radeon 760M, 1280 px, max 1M splats, 38 frames held
out). *Pending: the Colab run.*

Side-by-side images show a private interior, so they stay **outside the repository**, in
`C:\Users\JOSEPHINE\3DTests\benchmarks\myRoom2\<variant>\` (team laptop).

### Viewer check

Question: does the web viewer (`viewer/`, three.js + Spark 2.3.1) show less than the splat
contains? Method: the viewer's snapshot mode renders a held-out frame's exact view (pose and
field of view from `cameras.json`, 1600x900, one canvas pixel per image pixel) and downloads it
with an info file (GPU, drawing size, SH degree, Spark settings). `bench.compare_viewer`
compares it with gsplat's render of the same splat and view (`*_render_pinhole.png` from the
scorer) and with the real frame.

```
http://localhost:5173/?run=runs/myRoom2&snap=bottom_00000.jpg              Spark defaults
http://localhost:5173/?run=runs/myRoom2&snap=bottom_00000.jpg&blur=classic  3DGS-style blur
...&sh=0  (no view-dependent colour)     ...&lod=0  (no level of detail)
```

What to look at, from the viewer's code and Spark's defaults:

| Setting | Viewer now | Why it may matter |
|---|---|---|
| GPU | browser's choice (no `powerPreference`) | a laptop browser may render on the Intel GPU (slower, not lower quality per se) |
| Resolution | window size x pixel ratio, ratio capped at 2 (desktop) / 1.5 (phone) | fewer pixels than the screen on high-DPI phones |
| SH degree | 3 (all), K cycles it | none if 3 |
| Splat blur | Spark default `blurAmount 0.3`, the anti-aliased-training mode with an opacity correction | our splats are trained without anti-aliasing (gsplat `classic`); the matching Spark setting is `preBlurAmount 0.3`, `blurAmount 0`. A mismatch can make splats look thinner/more transparent |
| Level of detail | Spark default `enableLod: true` | could drop splats on weaker devices |

*Pending: the snapshot comparisons (needs the scorer's renders from Colab).*
