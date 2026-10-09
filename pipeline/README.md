# pipeline/: reconstruction pipeline (`cpipe`)

Phone video of a room → frames (FFmpeg) → camera poses (COLMAP) → Gaussian Splatting training
(gsplat, NVIDIA GPU) → `scene.ply`, with a `manifest.json` recording everything about the run.

```bash
pip install -e "pipeline[dev]"
cpipe run --video data/videos/room1.mp4 --output data/runs/room1 --preset balanced
cpipe run --video data/videos/room1.mp4 --output data/runs/room1 --until poses   # no GPU
pytest pipeline
```

Full guide (install, laptop + Colab workflow, presets, outputs, Docker, troubleshooting):
**[docs/pipeline.md](../docs/pipeline.md)**.

| Path | Content |
|---|---|
| `colibrimo_pipeline/cli.py` | `cpipe` command line |
| `colibrimo_pipeline/pipeline.py` | `run_pipeline()`: step order, resumability, manifest, logs |
| `colibrimo_pipeline/config.py` | typed settings (pydantic) |
| `colibrimo_pipeline/presets/` | `fast`, `balanced`, `quality` |
| `colibrimo_pipeline/steps/` | `validate`, `extract_frames`, `poses`, `train`, `export` |
| `colibrimo_pipeline/manifest.py` | `manifest.json` |
| `notebooks/colab_run.ipynb` | run on a free Google Colab GPU |
| `Dockerfile` | GPU image (not tested yet) |
| `tests/` | pytest (CPU; FFmpeg tests skip if FFmpeg is missing) |
