"""Training on Colab (D-020): continue a run folder without its videos, gsplat's
coordinate check, and ``cpipe check`` on the result."""

import json
import shutil
import zipfile

import numpy as np
import pytest
from test_brush import room_points, trained_splat, write_points3d, write_splat_ply
from test_cameras import POSES, rotation, write_model
from test_pipeline import Recorder, fake_step

from colibrimo_pipeline import cli, colab
from colibrimo_pipeline.check import check_run, describe, transform_points
from colibrimo_pipeline.config import load_config
from colibrimo_pipeline.errors import ConfigError, PipelineError, StepError
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
    monkeypatch.setattr(train, "run_evaluation", lambda *a, **k: {"psnr": 0.0})
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


# ---- zips: laptop -> Colab -> laptop -----------------------------------------------


def laptop_run_after_poses(run_dir):
    """A run folder stopped after poses, with the clutter a real one has."""
    files = {
        "input/videos.json": "{}",
        "frames.json": '{"frames": []}',
        "colmap/images/middle_00000.jpg": "jpg",
        "colmap/images/middle_00001.jpg": "jpg",
        "colmap/sparse/0/images.bin": "bin",
        "colmap/database.db": "db",
        "colmap/image_lists/middle.txt": "list",
        "colmap_unused/model_1/images.bin": "old",
        "frames_candidates/middle_c0001.jpg": "candidate",
        "logs/pipeline.log": "log" * 1000,
    }
    for rel, text in files.items():
        (run_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / rel).write_text(text)
    manifest = Manifest.load_or_create(run_dir)
    manifest.mark_succeeded(
        "validate", {"videos_info": run_dir / "input/videos.json"}, {}, 1
    )
    manifest.mark_succeeded(
        "extract_frames",
        {"images_dir": run_dir / "colmap/images", "frames": run_dir / "frames.json"},
        {},
        1,
    )
    manifest.mark_succeeded(
        "poses",
        {
            "sparse_model": run_dir / "colmap/sparse/0",
            "database": run_dir / "colmap/database.db",
        },
        {},
        1,
    )
    manifest.save()
    return manifest


def test_export_for_colab_zips_only_what_training_needs(tmp_path):
    laptop_run_after_poses(tmp_path / "run")
    info = colab.export_for_colab(tmp_path / "run", tmp_path / "room_for_colab.zip")
    assert info["num_files"] == 7
    with zipfile.ZipFile(info["path"]) as z:
        assert sorted(z.namelist()) == [
            "colmap/database.db",
            "colmap/images/middle_00000.jpg",
            "colmap/images/middle_00001.jpg",
            "colmap/sparse/0/images.bin",
            "frames.json",
            "input/videos.json",
            "manifest.json",
        ]
    assert info["size_bytes"] == (tmp_path / "room_for_colab.zip").stat().st_size


def test_export_for_colab_needs_poses(tmp_path):
    manifest = laptop_run_after_poses(tmp_path)
    manifest.data["steps"]["poses"]["status"] = "failed"
    manifest.save()
    with pytest.raises(PipelineError, match="'poses' is not finished"):
        colab.export_for_colab(tmp_path, tmp_path / "x.zip")


@pytest.fixture
def trained_on_colab(tmp_path, monkeypatch):
    """(laptop run folder stopped after poses, the same run finished on 'Colab',
    the result zip)."""
    colab_run = tmp_path / "colab"
    ctx = gsplat_run(colab_run, monkeypatch, GSPLAT_MATRIX)
    laptop = tmp_path / "laptop"
    shutil.copytree(colab_run, laptop, ignore=shutil.ignore_patterns("logs"))
    for step in (train, export):
        result = step.run(ctx)
        ctx.manifest.mark_succeeded(step.NAME, result.outputs, result.info, 1)
    ctx.manifest.set_status("succeeded")
    ctx.manifest.save()
    (colab_run / "logs" / "train.log").write_text("trained")
    packed = colab.pack_result(colab_run, tmp_path / "room_colab_result.zip")
    return laptop, colab_run, packed


def test_result_zip_round_trip_and_import(trained_on_colab, tmp_path):
    laptop, colab_run, packed = trained_on_colab
    with zipfile.ZipFile(packed["path"]) as z:
        assert sorted(z.namelist()) == [
            "cameras.json",
            "logs/train.log",
            "manifest.json",
            "scene.ply",
            "train/transform.json",
        ]
    staging = tmp_path / "staging"
    found = colab.extract_result_zip(packed["path"], staging)
    assert sorted(found) == sorted(
        ["cameras.json", "manifest.json", "scene.ply", "transform.json", "train.log"]
    )

    result = colab.import_result(laptop, staging)
    assert result["ok"] is True
    assert (laptop / "scene.ply").read_bytes() == (colab_run / "scene.ply").read_bytes()
    assert (laptop / "train" / "transform.json").is_file()
    assert (laptop / "logs" / "train.log").read_text() == "trained"
    data = json.loads((laptop / "manifest.json").read_text())
    assert data["status"] == "succeeded"
    assert data["imported_result"]["check"]["ok"] is True
    assert check_run(laptop)["ok"] is True
    assert not staging.exists()


def test_import_refuses_a_result_from_other_poses(trained_on_colab, tmp_path):
    laptop, _, packed = trained_on_colab
    data = json.loads((laptop / "manifest.json").read_text())
    data["steps"]["poses"]["finished_at"] = "2026-01-01T00:00:00+00:00"  # COLMAP re-run
    (laptop / "manifest.json").write_text(json.dumps(data))
    staging = tmp_path / "staging"
    colab.extract_result_zip(packed["path"], staging)
    with pytest.raises(PipelineError, match="not trained from this room's current"):
        colab.import_result(laptop, staging)
    assert not (laptop / "scene.ply").exists()  # nothing installed


def test_import_refuses_a_result_that_fails_the_check(trained_on_colab, tmp_path):
    laptop, _, packed = trained_on_colab
    staging = tmp_path / "staging"
    colab.extract_result_zip(packed["path"], staging)
    cameras = json.loads((staging / "cameras.json").read_text())
    cameras["num_cameras"] = 2
    (staging / "cameras.json").write_text(json.dumps(cameras))
    with pytest.raises(PipelineError, match="failed the checks"):
        colab.import_result(laptop, staging)
    assert not (laptop / "scene.ply").exists()


def test_extract_result_zip_takes_only_known_names(tmp_path):
    path = tmp_path / "r.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("../../evil.txt", "x")
        z.writestr("myRoom2/../../manifest.json", "{}")
        z.writestr("notes/readme.md", "x")
    dest = tmp_path / "dest"
    assert colab.extract_result_zip(path, dest) == ["manifest.json"]
    assert sorted(p.name for p in dest.iterdir()) == ["manifest.json"]
    assert not (tmp_path / "evil.txt").exists()
    (tmp_path / "bad.zip").write_text("not a zip")
    with pytest.raises(PipelineError, match="Not a valid zip"):
        colab.extract_result_zip(tmp_path / "bad.zip", dest)
