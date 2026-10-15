"""Training on Colab (D-020): continue a run folder without its videos, gsplat's
coordinate check, and ``cpipe check`` on the result."""

import json
import shutil

import numpy as np
import pytest
from test_brush import room_points, trained_splat, write_points3d, write_splat_ply
from test_cameras import POSES, rotation, write_model
from test_pipeline import Recorder, fake_step

from colibrimo_pipeline import cli
from colibrimo_pipeline.check import check_run, describe, transform_points
from colibrimo_pipeline.config import load_config
from colibrimo_pipeline.errors import ConfigError, StepError
from colibrimo_pipeline.manifest import Manifest
from colibrimo_pipeline.pipeline import run_pipeline
from colibrimo_pipeline.steps import export, train
from colibrimo_pipeline.steps.base import StepContext
from colibrimo_pipeline.utils.cameras import build_cameras
from colibrimo_pipeline.utils.logs import setup_logging

# ---- continue a run folder without its videos --------------------------------------


def test_continue_on_another_computer_without_the_videos(tmp_path):
    video = tmp_path / "middle.mp4"
    video.write_text("fake")
    recorder = Recorder()
    steps = [fake_step(n, recorder) for n in ("a", "b", "c")]
    laptop = tmp_path / "laptop" / "room"
    run_pipeline(video, laptop, load_config("fast"), steps=steps, until="a")
    assert json.loads((laptop / "manifest.json").read_text())["until"] == "a"

    # On Colab: the run folder (with a Windows path to a video that isn't there).
    colab = tmp_path / "colab" / "room"
    shutil.copytree(laptop, colab)
    data = json.loads((colab / "manifest.json").read_text())
    data["input"] = {"videos": [r"C:\Users\me\room\middle.mp4"], "names": ["middle"]}
    data["steps"]["a"]["outputs"]["out"] = "a.out"
    (colab / "manifest.json").write_text(json.dumps(data))
    recorder.calls.clear()

    manifest = run_pipeline(None, colab, load_config("fast"), steps=steps)
    assert recorder.calls == ["b", "c"]  # "a" (needing the video) was reused
    assert manifest.data["status"] == "succeeded"
    assert manifest.data["input"] == data["input"]  # kept as recorded
    assert manifest.data["until"] is None


def test_continue_needs_a_run_folder(tmp_path):
    with pytest.raises(ConfigError, match="no manifest.json to continue"):
        run_pipeline(None, tmp_path / "nothing", load_config("fast"))
    assert not (tmp_path / "nothing").exists()


def test_cli_without_videos_continues_the_output_folder(tmp_path, capsys):
    assert cli.main(["run", "--output", str(tmp_path / "nothing")]) == 2
    assert "give --video" in capsys.readouterr().err


# ---- gsplat on Colab: coordinate check, export, cpipe check -----------------------


def similarity(scale, rot, shift):
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rot
    matrix[:3, 3] = shift
    return matrix


# A gsplat-like normalisation: rescale, rotate, recentre.
GSPLAT_MATRIX = similarity(
    0.31, rotation("x", 180) @ rotation("z", 25), np.array([0.4, -1.2, 0.3])
)


def gsplat_run(tmp_path, monkeypatch, recorded_matrix):
    """A run folder after poses; gsplat replaced by a fake that trains in
    GSPLAT_MATRIX's coordinates, and reports ``recorded_matrix`` as its transform."""
    rng = np.random.default_rng(3)
    sparse = tmp_path / "colmap" / "sparse" / "0"
    write_model(sparse, POSES)
    colmap_xyz = room_points(rng, 2000)
    write_points3d(sparse / "points3D.bin", colmap_xyz)
    manifest = Manifest.load_or_create(tmp_path)
    manifest.mark_succeeded("poses", {"sparse_model": sparse}, {"num_registered": 3}, 0)
    manifest.save()

    def fake_trainer(cmd, **kwargs):
        ply = train.expected_ply(tmp_path / "train", 7000)
        ply.parent.mkdir(parents=True)
        splat = transform_points(GSPLAT_MATRIX, trained_splat(rng, colmap_xyz))
        write_splat_ply(ply, splat.astype("<f4"))

    monkeypatch.setattr(train, "gsplat_examples_dir", lambda cfg: tmp_path)
    monkeypatch.setattr(train, "apply_gsplat_fixes", lambda *args: [])
    monkeypatch.setattr(
        train,
        "gpu_env_versions",
        lambda python: {"cuda_available": True, "gpu": "Tesla T4"},
    )
    monkeypatch.setattr(train, "run_tool", fake_trainer)
    monkeypatch.setattr(
        train,
        "compute_transform",
        lambda *args: {"matrix_4x4": recorded_matrix.tolist(), "scene_scale": 1.0},
    )
    config = load_config("fast")
    assert config.train.trainer == "gsplat"
    return StepContext(tmp_path, [], config, manifest, setup_logging(tmp_path / "logs"))


def test_gsplat_run_passes_and_cpipe_check_confirms(tmp_path, monkeypatch, capsys):
    ctx = gsplat_run(tmp_path, monkeypatch, GSPLAT_MATRIX)
    result = train.run(ctx)
    assert result.info["coordinate_check"]["same_coordinates"] is True
    assert result.info["gpu"] == {"name": "Tesla T4"}
    ctx.manifest.mark_succeeded("train", result.outputs, result.info, 1)
    done = export.run(ctx)
    ctx.manifest.mark_succeeded("export", done.outputs, done.info, 1)
    ctx.manifest.save()

    checked = check_run(tmp_path)
    assert checked["ok"] is True
    assert checked["num_cameras"] == 3
    assert checked["cameras_use_manifest_transform"] is True
    assert describe(checked)[-1] == "Overall: PASS (trainer: gsplat simple_trainer.py)"

    assert cli.main(["check", "--output", str(tmp_path)]) == 0
    assert "Coordinate check" in capsys.readouterr().out

    # cameras.json built without gsplat's transform: caught.
    wrong = build_cameras(tmp_path / "colmap" / "sparse" / "0", np.eye(4))
    (tmp_path / "cameras.json").write_text(json.dumps(wrong))
    checked = check_run(tmp_path)
    assert checked["ok"] is False
    assert checked["cameras_use_manifest_transform"] is False
    assert cli.main(["check", "--output", str(tmp_path)]) == 1


def test_gsplat_run_with_the_wrong_transform_stops(tmp_path, monkeypatch):
    ctx = gsplat_run(tmp_path, monkeypatch, np.eye(4))  # recorded != trained-in
    with pytest.raises(StepError, match="not where the recorded scene transform"):
        train.run(ctx)
