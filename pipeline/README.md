# pipeline/: GPU reconstruction pipeline

> Placeholder (Step 0). Implemented in **Step 1** (CLI), **Step 2** (evaluation), **Step 3** (post-processing + compression).

Turns a phone video of a room into a Gaussian Splatting scene: FFmpeg frames → COLMAP poses (`ns-process-data`) → Nerfstudio `splatfacto` training → export → cleanup → compression → metrics.

Planned usage (PRD R1.1):

```bash
cpipe run --video data/videos/room1.mp4 --output data/runs/room1 --preset balanced
```

Planned layout (PRD §6):

```
pipeline/
├── Dockerfile               # CUDA base image + Nerfstudio + COLMAP + FFmpeg
├── pyproject.toml
├── colibrimo_pipeline/
│   ├── cli.py               # `cpipe` entry point
│   ├── config.py            # pydantic config + YAML presets
│   ├── steps/               # extract_frames, poses, train, export, postprocess, compress, evaluate
│   ├── manifest.py          # run manifest
│   └── utils/
├── presets/                 # fast.yaml, balanced.yaml, quality.yaml
└── tests/
```

Requires a Linux machine with an NVIDIA GPU for training (not the Windows dev laptops).
