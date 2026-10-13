"""Typed pipeline configuration.

A configuration is built from a YAML preset (``fast``, ``balanced``, ``quality``
or a path to your own YAML file) plus optional overrides, e.g. from the CLI::

    load_config("balanced", {"frames": {"target_count": 200}})

Every field has a default, so a preset only lists what it changes.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from colibrimo_pipeline.errors import ConfigError

PRESETS_DIR = Path(__file__).parent / "presets"


class _Section(BaseModel):
    # Reject unknown keys so typos in presets/overrides fail loudly.
    model_config = ConfigDict(extra="forbid")


class ValidationConfig(_Section):
    """Input video checks (step ``validate``)."""

    min_duration_s: float = Field(
        20.0,
        gt=0,
        description="Fail if all videos together are shorter than this.",
    )
    max_duration_s: float = Field(
        300.0, gt=0, description="Fail if any single video is longer than this."
    )
    min_resolution: int = Field(
        1080, gt=0, description="Warn (don't fail) if the short side is below this."
    )


class FramesConfig(_Section):
    """Frame extraction (step ``extract_frames``)."""

    target_count: int = Field(
        300,
        ge=10,
        description="Frames kept for reconstruction, in total. With several videos "
        "they are shared in proportion to each video's duration.",
    )
    min_per_video: int = Field(
        20,
        ge=1,
        description="With several videos, each one keeps at least this many frames "
        "(if target_count allows).",
    )
    max_long_side: int = Field(
        1600, ge=320, description="Frames are downscaled to this."
    )
    blur_filter: bool = Field(True, description="Drop the blurriest frames.")
    oversample: float = Field(
        2.0,
        ge=1.0,
        description="With blur_filter, extract oversample x target_count candidates "
        "and keep the sharpest one in each group.",
    )
    jpeg_quality: int = Field(
        2, ge=1, le=31, description="FFmpeg -q:v (2 = high quality)."
    )
    min_motion: float = Field(
        5.0,
        ge=0,
        description="Fail if the video looks static: largest mean pixel difference "
        "(0-255) between any frame and the first frame must exceed this.",
    )
    keep_candidates: bool = Field(
        False, description="Keep all candidate frames on disk."
    )
    hdr_tonemap: bool = Field(
        True,
        description="Tone-map HDR videos (HLG/PQ, e.g. iPhone) to SDR frames with "
        "FFmpeg zscale + tonemap. false: extract as is (washed-out colours).",
    )
    tonemap: Literal["hable", "mobius", "reinhard", "clip"] = Field(
        "hable", description="FFmpeg tonemap operator for HDR videos."
    )
    hdr_peak_nits: float = Field(
        100.0,
        gt=0,
        description="zscale 'npl' (nominal peak luminance) when converting HDR to linear.",
    )


class PosesConfig(_Section):
    """Camera pose estimation with COLMAP (step ``poses``)."""

    matcher: Literal["auto", "sequential", "exhaustive"] = Field(
        "auto",
        description="auto: 'sequential' for one video; 'sequential' with loop "
        "detection for several videos (plain sequential matching does not connect "
        "them). 'exhaustive' matches every pair: also connects videos, but slower.",
    )
    sequential_overlap: int = Field(
        10, ge=1, description="Neighbours matched per frame."
    )
    loop_detection: bool = Field(
        False,
        description="Sequential matching also matches visually similar frames far "
        "apart in time (vocabulary tree). Switched on automatically by matcher=auto "
        "for several videos.",
    )
    vocab_tree_path: str | None = Field(
        None,
        description="Vocabulary tree for loop detection. Empty = COLMAP downloads "
        "and caches its default tree (needs internet the first time).",
    )
    camera_model: Literal[
        "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"
    ] = "SIMPLE_RADIAL"
    max_num_features: int = Field(8192, ge=500)
    min_registered_ratio: float = Field(
        0.7, ge=0, le=1, description="Fail if COLMAP registers fewer frames than this."
    )
    use_gpu: bool = Field(False, description="GPU SIFT (needs a CUDA build of COLMAP).")
    num_threads: int = Field(-1, description="-1 = all CPU cores.")
    random_seed: int = 42


class TrainConfig(_Section):
    """Gaussian Splatting training (step ``train``) with gsplat or Brush."""

    trainer: Literal["gsplat", "brush"] = Field(
        "gsplat",
        description="gsplat: NVIDIA GPU + Linux (Colab, Docker). brush: Brush app, "
        "runs on Windows/Linux/macOS with any GPU (NVIDIA, AMD, Intel).",
    )
    max_steps: int = Field(30_000, ge=100)
    strategy: Literal["default", "mcmc"] = Field(
        "default", description="gsplat densification strategy (gsplat only)."
    )
    sh_degree: int = Field(3, ge=0, le=3, description="Spherical harmonics degree.")
    test_every: int = Field(
        8,
        ge=1,
        description="gsplat only: every Nth frame is held out for evaluation. "
        "Brush trains on all frames.",
    )
    brush_exe: str | None = Field(
        None,
        description="Brush executable (brush_app). Defaults to the CPIPE_BRUSH "
        "environment variable, then brush_app on PATH.",
    )
    max_splats: int | None = Field(
        None,
        ge=1000,
        description="Brush only: upper bound on the number of splats (Brush's "
        "default is 10,000,000; ~1,000,000 fits a 4 GB GPU).",
    )
    max_resolution: int | None = Field(
        None,
        ge=64,
        description="Brush only: longest image side used for training. Default: "
        "frames.max_long_side (no further downscaling).",
    )
    gsplat_examples_dir: str | None = Field(
        None,
        description="Folder containing gsplat's examples/simple_trainer.py. "
        "Defaults to the GSPLAT_EXAMPLES_DIR environment variable.",
    )
    python: str | None = Field(
        None, description="Python used to run the trainer (default: the current one)."
    )
    extra_args: list[str] = Field(
        default_factory=list,
        description="Extra arguments for the trainer (simple_trainer.py or brush_app).",
    )


class PipelineConfig(_Section):
    """Full, resolved configuration for one pipeline run."""

    preset: str = "balanced"
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    frames: FramesConfig = Field(default_factory=FramesConfig)
    poses: PosesConfig = Field(default_factory=PosesConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)


def available_presets() -> list[str]:
    """Names of the built-in presets."""
    return sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``base`` with ``overrides`` merged in (nested dicts merged)."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_set_overrides(items: list[str]) -> dict[str, Any]:
    """Turn ``["frames.target_count=200", "poses.matcher=exhaustive"]`` into nested dicts.

    Values are parsed as YAML, so ``200`` becomes an int and ``true`` a bool.
    """
    result: dict[str, Any] = {}
    for item in items:
        key, sep, raw_value = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"Invalid override '{item}': expected KEY=VALUE.")
        node = result
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw_value) if raw_value.strip() else ""
    return result


def load_config(
    preset: str | Path = "balanced", overrides: dict[str, Any] | None = None
) -> PipelineConfig:
    """Load a preset (built-in name or YAML path) and apply overrides."""
    preset_path = PRESETS_DIR / f"{preset}.yaml"
    if not preset_path.is_file():
        preset_path = Path(preset)
    if not preset_path.is_file():
        raise ConfigError(
            f"Unknown preset '{preset}'. Use one of {available_presets()} "
            "or the path to a YAML file."
        )

    data = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Preset {preset_path} must contain a YAML mapping.")
    data.setdefault("preset", preset_path.stem)
    data = deep_merge(data, overrides or {})

    try:
        return PipelineConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration:\n{exc}") from exc
