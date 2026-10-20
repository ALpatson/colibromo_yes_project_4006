"""Training benchmarks on one room (D-021): variants, full-resolution frames, table.

Every variant trains from the *same* frames and COLMAP result, so only training
changes. Variants are settings overrides (``VARIANTS``) applied with
``cpipe run --output <copy> --from-step train --set ...`` on a copy of the run.

``prepare_fullres`` makes the one variant that needs other frames: the same
frames (same video and time) re-extracted at the video's own resolution, with
COLMAP's model rescaled to match, so the camera poses stay identical.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from colibrimo_pipeline.config import PipelineConfig
from colibrimo_pipeline.errors import PipelineError
from colibrimo_pipeline.manifest import MANIFEST_NAME, Manifest
from colibrimo_pipeline.steps.extract_frames import (
    FRAMES_FILE,
    split_frame_counts,
    video_filter,
)
from colibrimo_pipeline.utils.colmap_model import read_cameras, write_scaled_model
from colibrimo_pipeline.utils.proc import find_tool, run_capture

# name -> (what changes, settings). One change at a time against `compared_to`.
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {
        "about": "gsplat default strategy, 1600x900 frames (the Colab run of 15 Oct)",
        "compared_to": None,
        "set": {},
    },
    "mcmc-1M": {
        "about": "MCMC strategy (its default budget: 1M splats)",
        "compared_to": "baseline",
        "set": {"strategy": "mcmc"},
    },
    "mcmc-2M": {
        "about": "MCMC, 2M splats",
        "compared_to": "mcmc-1M",
        "set": {"strategy": "mcmc", "max_splats": 2_000_000},
    },
    "mcmc-3M": {
        "about": "MCMC, 3M splats",
        "compared_to": "mcmc-1M",
        "set": {"strategy": "mcmc", "max_splats": 3_000_000},
    },
    "fullres": {
        "about": "1920x1080 frames (video resolution), same poses",
        "compared_to": "baseline",
        "set": {},
        "frames": "fullres",
    },
    "bilagrid": {
        "about": "per-frame colour/exposure correction (use_bilateral_grid)",
        "compared_to": "baseline",
        "set": {"bilateral_grid": True},
    },
    "antialiased": {
        "about": "anti-aliased rasterization",
        "compared_to": "baseline",
        "set": {"antialiased": True},
    },
    "brush-ref": {
        "about": "Brush on the laptop (NVIDIA T500), 1280 px, max 1M splats (reference)",
        "compared_to": None,
        "set": {},
        "trained_elsewhere": True,
    },
}


def set_args(variant: str) -> list[str]:
    """``--set train.x=y`` arguments for a variant (always gsplat)."""
    settings = {"trainer": "gsplat", **VARIANTS[variant]["set"]}
    return [
        arg
        for key, value in settings.items()
        for arg in (
            "--set",
            f"train.{key}={str(value).lower() if isinstance(value, bool) else value}",
        )
    ]


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)


def prepare_fullres(run_dir: Path, out_dir: Path, min_psnr: float = 30.0) -> dict:
    """Copy a run stopped after (or past) COLMAP with its frames re-extracted at
    the videos' own resolution and COLMAP's model rescaled; returns a summary.

    Each new frame is checked against the old one (downscaled, PSNR): a low
    value means another video frame was picked, and nothing is written.
    """
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    manifest = Manifest.load_or_create(run_dir)
    if not (run_dir / MANIFEST_NAME).is_file() or not manifest.is_step_done("poses"):
        raise PipelineError(f"{run_dir}: run it up to COLMAP (poses) first.")
    if out_dir.exists():
        raise PipelineError(f"{out_dir} already exists.")
    config = PipelineConfig.model_validate(manifest.data["config"])
    cfg = config.frames
    videos = manifest.step_info("validate")["videos"]
    frames = json.loads((run_dir / FRAMES_FILE).read_text(encoding="utf-8"))["frames"]
    images = run_dir / "colmap" / "images"
    native = cfg.model_copy(update={"max_long_side": 100_000})  # no downscaling
    ffmpeg = find_tool("ffmpeg")

    counts = split_frame_counts(
        [float(v["duration_s"]) for v in videos], cfg.target_count, cfg.min_per_video
    )
    staging = Path(tempfile.mkdtemp(prefix="fullres_"))
    try:
        new_images = staging / "images"
        new_images.mkdir()
        worst = math.inf
        for video, count in zip(videos, counts, strict=True):
            n_wanted = math.ceil(count * cfg.oversample) if cfg.blur_filter else count
            fps = n_wanted / float(video["duration_s"])
            cand = staging / video["name"]
            cand.mkdir()
            result = run_capture(
                [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-i",
                 video["path"], "-map", "0:v:0", "-vf", video_filter(video, fps, native),
                 "-q:v", str(cfg.jpeg_quality), "-start_number", "0",
                 str(cand / "%05d.jpg")],
                timeout=3600,
            )  # fmt: skip
            if result.returncode != 0:
                raise PipelineError(
                    f"FFmpeg failed on {video['name']}: {result.stderr[-500:]}"
                )
            for frame in (f for f in frames if f["video"] == video["name"]):
                candidate = cand / f"{round(frame['time_s'] * fps):05d}.jpg"
                if not candidate.is_file():
                    raise PipelineError(f"No candidate frame for {frame['file']}.")
                old = Image.open(images / frame["file"]).convert("RGB")
                new = Image.open(candidate).convert("RGB")
                small = new.resize(old.size, Image.Resampling.BICUBIC)
                worst = min(worst, _psnr(np.asarray(old), np.asarray(small)))
                shutil.copy2(candidate, new_images / frame["file"])
        if worst < min_psnr:
            raise PipelineError(
                f"A re-extracted frame differs from the original (PSNR {worst:.1f} dB "
                f"< {min_psnr}): not the same video frame. Nothing written."
            )

        sparse = manifest.step_output("poses", "sparse_model")
        old_cam = next(iter(read_cameras(sparse / "cameras.bin").values()))
        new_w = Image.open(next(new_images.iterdir())).size[0]
        scale = new_w / old_cam["width"]

        out_dir.mkdir(parents=True)
        for name in (MANIFEST_NAME, FRAMES_FILE, "input"):
            src = run_dir / name
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, out_dir / name)
        (out_dir / "colmap").mkdir()
        shutil.copytree(new_images, out_dir / "colmap" / "images")
        shutil.copy2(
            run_dir / "colmap" / "database.db", out_dir / "colmap" / "database.db"
        )
        write_scaled_model(sparse, out_dir / "colmap" / "sparse" / "0", scale)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    copy = Manifest.load_or_create(out_dir)
    for step in ("train", "export"):
        copy.data.get("steps", {}).pop(step, None)
    copy.data["config"]["frames"]["max_long_side"] = max(
        Image.open(next((out_dir / "colmap" / "images").iterdir())).size
    )
    copy.data["fullres_from"] = {
        "run": str(run_dir),
        "scale": scale,
        "min_psnr": round(worst, 2),
    }
    copy.save()
    return {
        "scale": round(scale, 4),
        "frames": len(frames),
        "min_psnr_vs_original": round(worst, 2),
    }


def compare_viewer(
    viewer_png: Path, reference_png: Path, real_frame: Path | None = None
) -> dict[str, Any]:
    """Viewer check (D-021): the web viewer's snapshot of a frame's exact view vs
    gsplat's render of the same splat and view (``*_render_pinhole.png``).

    PSNR between the two renderers measures what the viewer changes; against the
    real frame for context. ``black_fraction`` counts near-black pixels (holes
    where no splat covers the image: the viewer's background is black).
    """

    def load(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"))

    viewer, reference = load(viewer_png), load(reference_png)
    if viewer.shape != reference.shape:
        raise PipelineError(
            f"Different sizes: viewer {viewer.shape[1]}x{viewer.shape[0]}, "
            f"reference {reference.shape[1]}x{reference.shape[0]}."
        )

    def black(img: np.ndarray) -> float:
        return round(float((img.max(axis=2) < 8).mean()), 4)

    result = {
        "psnr_viewer_vs_gsplat": round(_psnr(viewer, reference), 2),
        "mean_abs_diff": round(float(np.abs(viewer.astype(int) - reference).mean()), 2),
        "brightness_viewer": round(float(viewer.mean()), 1),
        "brightness_gsplat": round(float(reference.mean()), 1),
        "black_fraction_viewer": black(viewer),
        "black_fraction_gsplat": black(reference),
    }
    if real_frame is not None:
        real = load(real_frame)
        if real.shape == viewer.shape:
            result["psnr_viewer_vs_real"] = round(_psnr(viewer, real), 2)
            result["psnr_gsplat_vs_real"] = round(_psnr(reference, real), 2)
    return result


def pack_for_scoring(run_dir: Path, zip_path: Path) -> dict[str, Any]:
    """Zip a run trained elsewhere (e.g. Brush on the laptop) so Colab can score it
    with ``cpipe evaluate``: what training needed, plus scene.ply and cameras.json."""
    from colibrimo_pipeline.colab import _write_zip, files_for_colab

    run_dir = Path(run_dir)
    if not (run_dir / "scene.ply").is_file():
        raise PipelineError(f"{run_dir} has no scene.ply: finish the run first.")
    extra = [
        (run_dir / n, n)
        for n in ("scene.ply", "cameras.json")
        if (run_dir / n).is_file()
    ]
    return _write_zip(zip_path, files_for_colab(run_dir) + extra)


def collect_results(runs: dict[str, Path]) -> list[dict[str, Any]]:
    """One row per variant run folder: scores, splats, size, training time."""
    rows = []
    for name, run_dir in runs.items():
        path = Path(run_dir) / MANIFEST_NAME
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        info = data.get("steps", {}).get("train", {}).get("info", {})
        ev = info.get("evaluation") or {}
        rows.append(
            {
                "variant": name,
                "about": VARIANTS.get(name, {}).get("about", ""),
                "psnr": ev.get("psnr"),
                "ssim": ev.get("ssim"),
                "lpips": ev.get("lpips"),
                "splats": info.get("num_gaussians"),
                "size_mb": (
                    round(info["ply_size_bytes"] / 1e6, 1)
                    if info.get("ply_size_bytes")
                    else None
                ),
                "train_min": (
                    round(info["train_seconds"] / 60, 1)
                    if info.get("train_seconds")
                    else None
                ),
                "trainer": info.get("trainer"),
                "gpu": (info.get("gpu") or {}).get("name"),
                "test_frames": ev.get("num_test_frames"),
            }
        )
    return rows


def results_markdown(rows: list[dict[str, Any]]) -> str:
    """The comparison table (higher PSNR/SSIM and lower LPIPS are better)."""

    def fmt(value, spec):
        return "–" if value is None else format(value, spec)

    lines = [
        "| Variant | Change | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Splats | Size (MB) | Training (min) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['about']} | {fmt(r['psnr'], '.2f')} | {fmt(r['ssim'], '.4f')} "
            f"| {fmt(r['lpips'], '.4f')} | {fmt(r['splats'], ',')} | {fmt(r['size_mb'], '.1f')} "
            f"| {fmt(r['train_min'], '.1f')} |"
        )
    return "\n".join(lines)
