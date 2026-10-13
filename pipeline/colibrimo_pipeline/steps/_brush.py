"""Brush trainer (https://github.com/ArthurBrussee/brush), used by the ``train`` step.

Brush trains Gaussian Splats on any GPU (NVIDIA, AMD, Intel) through
wgpu/Vulkan, on Windows too. Command-line options checked with
``brush_app.exe --help`` (v0.3.0): the dataset folder is a positional argument
(no ``train`` subcommand); without ``--with-viewer`` it runs headless and
exports ``--export-name`` into ``--export-path`` every ``--export-every`` steps.
With ``RUST_LOG=info`` it logs the GPU adapter it uses.

Brush keeps COLMAP's coordinates (no normalisation): ``identity_check``
verifies that on every run (COLMAP's points must lie right next to splats).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from colibrimo_pipeline.config import PipelineConfig
from colibrimo_pipeline.errors import StepError
from colibrimo_pipeline.utils.proc import find_tool, run_capture

NAME = "train"
EXPORT_NAME = "export_{iter}.ply"  # Brush replaces {iter} with the step number

_ADAPTER = re.compile(
    r'Using adapter AdapterInfo \{ name: "(?P<name>[^"]*)".*?'
    r"device_type: (?P<device_type>\w+).*?backend: (?P<backend>\w+)"
)
# Logged every --refine-every (200) steps, e.g. "Refine iter 6001, 110507 splats."
_REFINE = re.compile(r"Refine iter (\d+), (\d+) splats")


def parse_progress(line: str) -> tuple[int, int] | None:
    """(step, number of splats) from Brush's 'Refine iter' log line."""
    match = _REFINE.search(line)
    return (int(match.group(1)), int(match.group(2))) if match else None


def brush_executable(cfg: PipelineConfig) -> str:
    """train.brush_exe, else CPIPE_BRUSH, else brush_app on PATH."""
    configured = cfg.train.brush_exe
    if configured:
        if not Path(configured).is_file():
            raise StepError(
                NAME, f"train.brush_exe is set to '{configured}', which does not exist."
            )
        return configured
    return find_tool("brush_app")


def brush_version(exe: str) -> str | None:
    result = run_capture([exe, "--version"], timeout=60)
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[0] if text else None


def build_brush_command(
    exe: str, data_dir: Path, out_dir: Path, cfg: PipelineConfig
) -> list[str]:
    """Headless Brush training on ``data_dir`` (images/ + sparse/0), final PLY in ``out_dir``."""
    t = cfg.train
    steps = str(t.max_steps)
    command = [
        exe,
        str(data_dir),
        "--total-steps",
        steps,
        "--sh-degree",
        str(t.sh_degree),
        "--max-resolution",
        str(t.max_resolution or cfg.frames.max_long_side),
        "--seed",
        "42",
        "--export-every",
        steps,  # only the final splat
        "--export-path",
        str(out_dir),
        "--export-name",
        EXPORT_NAME,
    ]
    if t.max_splats is not None:
        command += ["--max-splats", str(t.max_splats)]
    return command + list(t.extra_args)


def expected_brush_ply(out_dir: Path, max_steps: int) -> Path:
    return out_dir / EXPORT_NAME.replace("{iter}", str(max_steps))


def parse_adapter(line: str) -> dict[str, str] | None:
    """GPU adapter from Brush's 'Using adapter AdapterInfo { ... }' log line."""
    match = _ADAPTER.search(line)
    return match.groupdict() if match else None


# Median distance from a COLMAP point to the nearest splat, as a fraction of the
# scene size, below which the splat counts as being in COLMAP's coordinates.
# myRoom2 (Brush v0.3.0, 30k steps): 0.0072 as is; 0.029 after a 10 degree
# rotation, 0.064 after a 10% shift (docs/decisions.md, D-013).
IDENTITY_THRESHOLD = 0.015


def _nearest_distances(points: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Distance from each point to its nearest neighbour in ``others`` (brute force)."""
    others_sq = (others**2).sum(axis=1)
    result = []
    for start in range(0, len(points), 250):
        chunk = points[start : start + 250]
        d2 = (chunk**2).sum(axis=1)[:, None] + others_sq[None, :] - 2 * chunk @ others.T
        result.append(np.sqrt(np.maximum(d2.min(axis=1), 0)))
    return np.concatenate(result)


def identity_check(
    colmap_xyz: np.ndarray,
    splat_xyz: np.ndarray,
    max_points: int = 1000,
    max_splats: int = 50_000,
) -> dict[str, Any]:
    """Do the splat and COLMAP's 3D points share the same coordinates?

    COLMAP's sparse points lie on the room's surfaces, where the trained splat
    has its Gaussians. In the same coordinates, every COLMAP point has a splat
    right next to it; after any recentring, rescaling or rotation (what
    gsplat's normalisation does) most of them don't. Distances are measured
    relative to the scene size (median interquartile range of the COLMAP
    points), on random samples (fixed seed) to keep it under a second.
    """
    rng = np.random.default_rng(0)

    def sample(xyz: np.ndarray, n: int) -> np.ndarray:
        if len(xyz) > n:
            xyz = xyz[rng.choice(len(xyz), n, replace=False)]
        return xyz.astype(np.float32)

    q = np.percentile(colmap_xyz, [25, 75], axis=0)
    scene_size = float(np.median(q[1] - q[0])) or 1.0
    rel = (
        _nearest_distances(
            sample(colmap_xyz, max_points), sample(splat_xyz, max_splats)
        )
        / scene_size
    )
    median = float(np.median(rel))
    return {
        "same_coordinates": median < IDENTITY_THRESHOLD,
        "median_nearest_splat_rel": round(median, 5),
        "p90_nearest_splat_rel": round(float(np.percentile(rel, 90)), 5),
        "threshold_rel": IDENTITY_THRESHOLD,
        "scene_size": round(scene_size, 4),
        "num_colmap_points": int(len(colmap_xyz)),
        "num_splats": int(len(splat_xyz)),
    }
