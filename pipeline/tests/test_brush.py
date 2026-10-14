"""Brush trainer: command line, unused COLMAP models, coordinates check, file readers."""

import struct
from pathlib import Path

import numpy as np
import pytest

from colibrimo_pipeline.config import load_config
from colibrimo_pipeline.errors import ConfigError, StepError
from colibrimo_pipeline.steps._brush import (
    brush_executable,
    build_brush_command,
    expected_brush_ply,
    identity_check,
    parse_adapter,
    parse_progress,
)
from colibrimo_pipeline.utils.colmap_model import (
    find_model_dirs,
    move_unused_models,
    promote_best_model,
    read_points3d_xyz,
)
from colibrimo_pipeline.utils.ply import read_vertex_count, read_vertex_xyz

# ---- command line ----------------------------------------------------------------


def brush_config(**train):
    return load_config("balanced", {"train": {"trainer": "brush", **train}})


def test_brush_command_uses_real_v030_options(tmp_path):
    cfg = brush_config(max_splats=1_000_000, sh_degree=2, max_resolution=1280)
    cmd = build_brush_command(
        "brush_app.exe", tmp_path / "colmap", tmp_path / "train", cfg
    )
    assert cmd[:2] == [
        "brush_app.exe",
        str(tmp_path / "colmap"),
    ]  # positional, no subcommand
    opts = dict(zip(cmd[2::2], cmd[3::2], strict=True))
    assert opts == {
        "--total-steps": "30000",
        "--sh-degree": "2",
        "--max-resolution": "1280",
        "--seed": "42",
        "--export-every": "30000",
        "--export-path": str(tmp_path / "train"),
        "--export-name": "export_{iter}.ply",
        "--max-splats": "1000000",
    }
    assert "--with-viewer" not in cmd  # headless


def test_brush_defaults_to_frame_size_and_brush_splat_limit(tmp_path):
    cmd = build_brush_command("b", tmp_path, tmp_path, brush_config())
    assert cmd[cmd.index("--max-resolution") + 1] == "1600"  # balanced frames
    assert "--max-splats" not in cmd


def test_expected_ply_name():
    assert expected_brush_ply(Path("train"), 30000) == Path("train/export_30000.ply")


def test_trainer_must_be_known():
    with pytest.raises(ConfigError):
        load_config("fast", {"train": {"trainer": "nerfstudio"}})


def test_brush_exe_from_config_must_exist(tmp_path):
    with pytest.raises(StepError, match="does not exist"):
        brush_executable(brush_config(brush_exe=str(tmp_path / "nope.exe")))


def test_brush_exe_from_environment(monkeypatch, tmp_path):
    exe = tmp_path / "brush_app.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("CPIPE_BRUSH", str(exe))
    assert brush_executable(brush_config()) == str(exe)


# ---- GPU adapter ----------------------------------------------------------------

# Real line from brush_app.exe v0.3.0 with RUST_LOG=info (shortened in the middle).
ADAPTER_LINE = (
    "[2026-10-13T12:18:33Z INFO  cubecl_wgpu::runtime] Using adapter AdapterInfo "
    '{ name: "NVIDIA T500", vendor: 4318, device: 8123, device_type: DiscreteGpu, '
    'driver: "NVIDIA", driver_info: "595.95", backend: Vulkan }'
)


def test_parse_adapter():
    assert parse_adapter(ADAPTER_LINE) == {
        "name": "NVIDIA T500",
        "device_type": "DiscreteGpu",
        "backend": "Vulkan",
    }


def test_parse_adapter_integrated_gpu():
    line = ADAPTER_LINE.replace("NVIDIA T500", "Intel(R) Iris(R) Xe Graphics").replace(
        "DiscreteGpu", "IntegratedGpu"
    )
    assert parse_adapter(line)["device_type"] == "IntegratedGpu"


def test_parse_adapter_other_lines():
    assert parse_adapter("[INFO brush_cli] Starting up") is None


def test_parse_progress():
    line = "[2026-10-13T12:37:13Z INFO  brush_cli] Refine iter 6001, 110507 splats."
    assert parse_progress(line) == (6001, 110507)
    assert parse_progress(ADAPTER_LINE) is None


# ---- unused COLMAP models ------------------------------------------------------------


def write_fake_model(folder, num_images):
    folder.mkdir(parents=True)
    (folder / "images.bin").write_bytes(struct.pack("<Q", num_images))


def test_promote_moves_unused_models_out_of_colmap(tmp_path):
    sparse = tmp_path / "colmap" / "sparse"
    write_fake_model(sparse / "0", 290)
    write_fake_model(sparse / "1", 4)  # the tiny model Brush loaded by mistake
    promote_best_model(sparse)
    # Nothing but sparse/0 is left anywhere under colmap/ (Brush scans all of it).
    assert find_model_dirs(tmp_path / "colmap") == [sparse / "0"]
    assert (tmp_path / "colmap_unused" / "model_1" / "images.bin").is_file()


def test_move_unused_models_handles_both_old_layouts_and_is_idempotent(tmp_path):
    colmap = tmp_path / "colmap"
    write_fake_model(colmap / "sparse" / "0", 290)
    write_fake_model(colmap / "sparse" / "unused_1", 4)  # layout before D-014
    write_fake_model(colmap / "unused_models" / "model_1", 18)  # D-014 (testnew)
    write_fake_model(colmap / "unused_models" / "model_2", 2)

    assert move_unused_models(colmap / "sparse") == [
        "sparse/unused_1",
        "unused_models/model_1",
        "unused_models/model_2",
    ]
    assert find_model_dirs(colmap) == [colmap / "sparse" / "0"]
    assert not (colmap / "unused_models").exists()
    unused = tmp_path / "colmap_unused"
    assert sorted(p.name for p in unused.iterdir()) == [
        "model_1",
        "model_2",
        "unused_1",
    ]
    assert move_unused_models(colmap / "sparse") == []  # second call: nothing to do


def test_move_unused_models_never_overwrites(tmp_path):
    sparse = tmp_path / "colmap" / "sparse"
    write_fake_model(sparse / "0", 10)
    write_fake_model(sparse / "unused_1", 3)
    write_fake_model(tmp_path / "colmap_unused" / "unused_1", 7)
    move_unused_models(sparse)
    unused = tmp_path / "colmap_unused"
    assert sorted(p.name for p in unused.iterdir()) == ["unused_1", "unused_1_1"]


def brush_run(tmp_path, monkeypatch):
    """A run folder after poses, with Brush itself replaced by a stub that records it ran."""
    from colibrimo_pipeline.manifest import Manifest
    from colibrimo_pipeline.steps import train
    from colibrimo_pipeline.steps.base import StepContext
    from colibrimo_pipeline.utils.logs import setup_logging

    sparse0 = tmp_path / "colmap" / "sparse" / "0"
    write_fake_model(sparse0, 260)
    manifest = Manifest.load_or_create(tmp_path)
    manifest.mark_succeeded("poses", {"sparse_model": sparse0}, {}, 0)
    started = []

    def fake_run_tool(cmd, **kwargs):
        started.append(cmd)
        raise RuntimeError("stop after starting Brush")  # enough for this test

    monkeypatch.setattr(train, "brush_executable", lambda cfg: "brush_app.exe")
    monkeypatch.setattr(train, "run_tool", fake_run_tool)
    ctx = StepContext(
        tmp_path, [], brush_config(), manifest, setup_logging(tmp_path / "logs")
    )
    return ctx, train, started


def test_brush_cleans_up_an_old_run_folder_before_training(tmp_path, monkeypatch):
    ctx, train, started = brush_run(tmp_path, monkeypatch)
    write_fake_model(
        tmp_path / "colmap" / "unused_models" / "model_1", 18
    )  # like testnew
    with pytest.raises(RuntimeError, match="stop after starting Brush"):
        train.run_brush(ctx)
    assert len(started) == 1  # Brush started, after the leftover was moved out
    assert find_model_dirs(tmp_path / "colmap") == [
        tmp_path / "colmap" / "sparse" / "0"
    ]


def test_brush_refuses_any_other_model_inside_colmap(tmp_path, monkeypatch):
    ctx, train, started = brush_run(tmp_path, monkeypatch)
    write_fake_model(tmp_path / "colmap" / "copy_of_old_model", 18)  # unexpected place
    with pytest.raises(StepError, match="exactly one COLMAP model"):
        train.run_brush(ctx)
    assert started == []  # Brush never ran


# ---- readers ------------------------------------------------------------------------


def write_points3d(path: Path, xyz: np.ndarray, track_length: int = 2) -> None:
    """COLMAP points3D.bin with the given positions (format: colmap.github.io/format.html)."""
    out = struct.pack("<Q", len(xyz))
    for i, (x, y, z) in enumerate(xyz):
        out += struct.pack("<Q3d3Bd", i + 1, x, y, z, 10, 20, 30, 0.5)
        out += struct.pack("<Q", track_length)
        out += struct.pack(f"<{2 * track_length}i", *range(2 * track_length))
    path.write_bytes(out)


def write_splat_ply(path: Path, xyz: np.ndarray) -> None:
    """Binary PLY with the property layout Brush writes (x/y/z after other floats)."""
    names = ["f_dc_0", "f_dc_1", "f_dc_2", "opacity", "x", "y", "z", "scale_0"]
    header = "ply\nformat binary_little_endian 1.0\ncomment Exported from Brush\n"
    header += f"element vertex {len(xyz)}\n"
    header += "".join(f"property float {n}\n" for n in names) + "end_header\n"
    data = np.zeros((len(xyz), len(names)), dtype="<f4")
    data[:, 4:7] = xyz
    path.write_bytes(header.encode() + data.tobytes())


def test_read_points3d_xyz(tmp_path):
    xyz = np.array([[1.0, 2.0, 3.0], [-4.5, 0.25, 9.0], [0, 0, 0]])
    write_points3d(tmp_path / "points3D.bin", xyz, track_length=3)
    np.testing.assert_allclose(read_points3d_xyz(tmp_path / "points3D.bin"), xyz)


def test_read_vertex_xyz(tmp_path):
    xyz = np.random.default_rng(0).normal(size=(50, 3)).astype("<f4")
    write_splat_ply(tmp_path / "s.ply", xyz)
    np.testing.assert_allclose(read_vertex_xyz(tmp_path / "s.ply"), xyz)
    assert read_vertex_count(tmp_path / "s.ply") == 50


# ---- coordinates check ---------------------------------------------------------------


def room_points(rng, n):
    """Points on the walls/floor of a 6 x 3 x 4 box room (COLMAP-like units)."""
    p = rng.uniform([-3, -1.5, -2], [3, 1.5, 2], size=(n, 3))
    axis = rng.integers(0, 3, n)
    side = rng.choice([-1, 1], n)
    half = np.array([3, 1.5, 2])
    p[np.arange(n), axis] = side * half[axis]
    return p


def trained_splat(rng, colmap):
    """Like a trained splat: Gaussians grown around COLMAP's points (training starts
    from them), more on the surfaces, and some floaters in the middle of the room."""
    around_points = np.repeat(colmap, 10, axis=0) + rng.normal(
        0, 0.01, (len(colmap) * 10, 3)
    )
    surfaces = room_points(rng, 30_000)
    floaters = rng.uniform(-1, 1, (2000, 3))
    return np.vstack([around_points, surfaces, floaters])


def test_identity_check_accepts_same_coordinates():
    rng = np.random.default_rng(1)
    colmap = room_points(rng, 2000)
    result = identity_check(colmap, trained_splat(rng, colmap))
    assert result["same_coordinates"] is True
    assert result["median_nearest_splat_rel"] < result["threshold_rel"]


def rotate_y(degrees):
    a = np.radians(degrees)
    m = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return lambda p: p @ m.T


@pytest.mark.parametrize(
    "transform",
    [
        lambda p: p * 0.2,  # rescaled (like gsplat's normalisation)
        lambda p: p - np.array([0.5, 0.0, 0.0]),  # recentred by a small amount
        rotate_y(90),
        rotate_y(10),  # even a small rotation is caught
    ],
)
def test_identity_check_rejects_normalised_scenes(transform):
    rng = np.random.default_rng(2)
    colmap = room_points(rng, 2000)
    splats = transform(trained_splat(rng, colmap))
    assert identity_check(colmap, splats)["same_coordinates"] is False
