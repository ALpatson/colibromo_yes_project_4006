"""Scoring on held-out frames (D-021): frame split, PLY reading, camera moves,
COLMAP model copies, trainer options, and the train step's bookkeeping."""

import json
import struct
import textwrap

import numpy as np
import pytest
from test_brush import room_points, write_points3d
from test_cameras import POSES, rotation, write_model
from test_colab import GSPLAT_MATRIX, gsplat_run

from colibrimo_pipeline import bench, gsplat_compat
from colibrimo_pipeline.config import TrainConfig, load_config
from colibrimo_pipeline.errors import PipelineError, StepError
from colibrimo_pipeline.evaluate import (
    evaluate_existing_run,
    frames_to_save,
)
from colibrimo_pipeline.evaluate import (  # renamed: pytest collects test_* names
    test_frame_names as held_out_frames,
)
from colibrimo_pipeline.manifest import Manifest
from colibrimo_pipeline.steps import train
from colibrimo_pipeline.steps.base import StepContext
from colibrimo_pipeline.utils.cameras import scene_camtoworld
from colibrimo_pipeline.utils.colmap_model import (
    read_cameras,
    read_images,
    write_images_subset,
    write_scaled_model,
)
from colibrimo_pipeline.utils.logs import setup_logging
from colibrimo_pipeline.utils.ply import read_splat_ply

LOOK = POSES[0][1]


def many_poses(n):
    """n frames over three videos, given out of name order (COLMAP doesn't sort)."""
    names = [
        f"{v}_{i:05d}.jpg" for v in ("up", "bottom", "middle") for i in range(n // 3)
    ]
    return [(name, LOOK, np.array([i * 0.1, 0.0, 0.0])) for i, name in enumerate(names)]


# ---- held-out frames ------------------------------------------------------------


def test_held_out_frames_match_gsplats_split(tmp_path):
    poses = many_poses(30)
    write_model(tmp_path / "0", poses)
    names = sorted(p[0] for p in poses)
    held_out = held_out_frames(tmp_path / "0", 8)
    # gsplat v1.5.3: sorted names, indices 0, 8, 16, ... are the test split.
    assert held_out == [n for i, n in enumerate(names) if i % 8 == 0]
    assert held_out[0] == "bottom_00000.jpg"
    assert held_out_frames(tmp_path / "0", 0) == []


def test_saved_frames_are_fixed_and_spread():
    frames = [f"f{i:02d}" for i in range(38)]
    picked = frames_to_save(frames, 5)
    assert picked == ["f00", "f09", "f18", "f28", "f37"]
    assert frames_to_save(frames, 5) == picked  # same every run
    assert frames_to_save(frames[:3], 5) == frames[:3]
    assert frames_to_save(frames, 0) == []


# ---- PLY and cameras -------------------------------------------------------------


def write_full_ply(path, xyz, sh_rest_per_channel):
    """3DGS PLY: x y z, f_dc_0..2, f_rest_0..(3*R-1) per channel, opacity, scale, rot."""
    n, rest = len(xyz), sh_rest_per_channel
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * rest)]
    names += [
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    data = np.zeros((n, len(names)), dtype="<f4")
    data[:, :3] = xyz
    col = {name: i for i, name in enumerate(names)}
    for c in range(3):
        data[:, col[f"f_dc_{c}"]] = c + 1
        for k in range(rest):  # channel-major: f_rest_{c * rest + k}
            data[:, col[f"f_rest_{c * rest + k}"]] = 10 * (c + 1) + k
    data[:, col["opacity"]] = 0.5
    data[:, col["rot_0"]] = 1.0
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    path.write_bytes(header.encode() + data.tobytes())


def test_read_splat_ply_reorders_sh_like_gsplat(tmp_path):
    xyz = np.random.default_rng(0).normal(size=(4, 3)).astype("<f4")
    write_full_ply(tmp_path / "s.ply", xyz, sh_rest_per_channel=15)
    splat = read_splat_ply(tmp_path / "s.ply")
    assert splat["sh_degree"] == 3 and splat["count"] == 4
    assert splat["sh"].shape == (4, 16, 3)
    np.testing.assert_allclose(splat["sh"][0, 0], [1, 2, 3])  # f_dc per channel
    np.testing.assert_allclose(
        splat["sh"][0, 1], [10, 20, 30]
    )  # first f_rest of each channel
    np.testing.assert_allclose(splat["sh"][0, 15], [24, 34, 44])
    np.testing.assert_allclose(splat["means"], xyz)

    write_full_ply(tmp_path / "dc.ply", xyz, sh_rest_per_channel=0)  # app_opt export
    assert read_splat_ply(tmp_path / "dc.ply")["sh_degree"] == 0


def test_moved_camera_sees_the_moved_scene_identically():
    rng = np.random.default_rng(1)
    c2w = np.eye(4)
    c2w[:3, :3] = rotation("y", 25) @ rotation("x", -10)
    c2w[:3, 3] = [0.5, -1.0, 2.0]
    points = rng.normal(size=(20, 3)) + [0, 0, 6]
    moved_points = points @ GSPLAT_MATRIX[:3, :3].T + GSPLAT_MATRIX[:3, 3]

    def project(cam, pts):
        w2c = np.linalg.inv(cam)
        p = pts @ w2c[:3, :3].T + w2c[:3, 3]
        return p[:, :2] / p[:, 2:]

    np.testing.assert_allclose(
        project(c2w, points),
        project(scene_camtoworld(c2w, GSPLAT_MATRIX.tolist()), moved_points),
        atol=1e-9,
    )


# ---- COLMAP model copies ---------------------------------------------------------


def test_images_subset_keeps_records_byte_for_byte(tmp_path):
    write_model(tmp_path / "0", POSES)
    keep = {"middle_00000.jpg", "up_00001.jpg"}
    assert (
        write_images_subset(tmp_path / "0" / "images.bin", tmp_path / "sub.bin", keep)
        == 2
    )
    original = {i["name"]: i for i in read_images(tmp_path / "0" / "images.bin")}
    for image in read_images(tmp_path / "sub.bin"):
        assert image == original[image["name"]]
    assert {i["name"] for i in read_images(tmp_path / "sub.bin")} == keep


def test_scaled_model_keeps_poses_and_scales_pixels(tmp_path):
    write_model(tmp_path / "0", POSES)
    write_points3d(tmp_path / "0" / "points3D.bin", np.zeros((3, 3)))
    write_scaled_model(tmp_path / "0", tmp_path / "big", 1.2)
    cam = read_cameras(tmp_path / "big" / "cameras.bin")[1]
    assert (cam["width"], cam["height"]) == (1920, 1080)
    np.testing.assert_allclose(cam["params"], [785.3 * 1.2, 960.0, 540.0, -0.0024])
    assert read_images(tmp_path / "big" / "images.bin") == read_images(
        tmp_path / "0" / "images.bin"
    )
    data = (tmp_path / "big" / "images.bin").read_bytes()
    first_point = data.index(b"middle_00000.jpg\0") + len("middle_00000.jpg\0") + 8
    assert struct.unpack_from("<dd", data, first_point) == pytest.approx((1.2, 2.4))
    assert (tmp_path / "big" / "points3D.bin").read_bytes() == (
        tmp_path / "0" / "points3D.bin"
    ).read_bytes()


# ---- gsplat: training on all frames, options --------------------------------------

V153_SPLIT = """        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]
"""


def test_all_frames_patch(tmp_path):
    (tmp_path / "datasets").mkdir()
    colmap_py = tmp_path / "datasets" / "colmap.py"
    colmap_py.write_text(V153_SPLIT)
    assert gsplat_compat.allow_training_on_all_frames(tmp_path) is True
    assert gsplat_compat.allow_training_on_all_frames(tmp_path) is False  # idempotent
    code = textwrap.dedent(colmap_py.read_text())

    class Split:  # runs the patched lines as gsplat's Dataset.__init__ would
        def __init__(self, split, test_every):
            self.parser = type(
                "P", (), {"image_names": list("abcdefghij"), "test_every": test_every}
            )
            exec(code, {"np": np}, {"self": self, "split": split})

    assert list(Split("train", 0).indices) == list(range(10))
    assert list(Split("val", 0).indices) == []
    assert list(Split("train", 8).indices) == [1, 2, 3, 4, 5, 6, 7, 9]
    colmap_py.write_text("something else")
    with pytest.raises(RuntimeError, match="not gsplat v1.5.3"):
        gsplat_compat.allow_training_on_all_frames(tmp_path)


def test_trainer_options_on_the_command_line(tmp_path):
    def command(**options):
        return train.build_train_command(
            "py",
            tmp_path,
            tmp_path / "colmap",
            tmp_path / "out",
            TrainConfig(**options),
        )

    plain = command()
    assert "--antialiased" not in plain and "--use_bilateral_grid" not in plain
    mcmc = command(
        strategy="mcmc", max_splats=2_000_000, antialiased=True, bilateral_grid=True
    )
    assert mcmc[2] == "mcmc"
    joined = " ".join(mcmc)
    assert "--strategy.cap_max 2000000" in joined
    assert "--antialiased" in mcmc and "--use_bilateral_grid" in mcmc
    assert "--test_every 0" in " ".join(command(test_every=0))
    assert "cap_max" not in " ".join(command(max_splats=2_000_000))  # default strategy


def test_flag_speller_reads_the_trainers_help(tmp_path, monkeypatch):
    help_text = (
        "usage: ...\n  --strategy.cap-max INT\n  --antialiased, --no-antialiased\n"
    )
    monkeypatch.setattr(
        train,
        "run_capture",
        lambda *a, **k: type("R", (), {"stdout": help_text, "stderr": ""})(),
    )
    flag = train.trainer_flag_speller("py", tmp_path, "mcmc")
    assert flag("strategy.cap_max") == "--strategy.cap-max"
    assert flag("antialiased") == "--antialiased"
    with pytest.raises(StepError, match="no option --use_bilateral_grid"):
        flag("use_bilateral_grid")


# ---- train step bookkeeping ---------------------------------------------------------


def test_gsplat_records_held_out_frames_and_score(tmp_path, monkeypatch):
    ctx = gsplat_run(tmp_path, monkeypatch, GSPLAT_MATRIX)
    ctx.config.train.test_every = 2
    ctx.manifest.data["imported_result"] = {"imported_at": "earlier"}
    calls = []

    def fake_eval(run_dir, ply, sparse, matrix, frames, **kw):
        calls.append((frames, matrix, kw["antialiased"], kw["eval_images"]))
        return {"psnr": 25.0, "ssim": 0.8, "lpips": 0.2}

    monkeypatch.setattr(train, "run_evaluation", fake_eval)
    result = train.run(ctx)
    held_out = ["bottom_00000.jpg", "up_00001.jpg"]  # sorted names, every 2nd
    assert result.info["test_frames"] == held_out
    assert result.info["evaluation"]["psnr"] == 25.0
    assert calls == [(held_out, GSPLAT_MATRIX.tolist(), False, 5)]
    saved = json.loads((tmp_path / "train" / "test_frames.json").read_text())
    assert saved == {"test_every": 2, "frames": held_out}
    assert "imported_result" not in ctx.manifest.data  # retrained: no longer imported


def test_failed_scoring_keeps_the_trained_splat(tmp_path, monkeypatch):
    ctx = gsplat_run(tmp_path, monkeypatch, GSPLAT_MATRIX)

    def broken(*args, **kwargs):
        raise PipelineError("Evaluation failed: no lpips weights")

    monkeypatch.setattr(train, "run_evaluation", broken)
    result = train.run(ctx)
    assert result.info["evaluation"] == {"error": "Evaluation failed: no lpips weights"}
    assert result.outputs["ply"].is_file()


def test_brush_trains_without_the_held_out_frames(tmp_path, monkeypatch):
    sparse = tmp_path / "colmap" / "sparse" / "0"
    write_model(sparse, many_poses(24))
    write_points3d(sparse / "points3D.bin", room_points(np.random.default_rng(0), 50))
    (tmp_path / "colmap" / "images").mkdir()
    for image in read_images(sparse / "images.bin"):
        (tmp_path / "colmap" / "images" / image["name"]).write_bytes(b"jpg")
    manifest = Manifest.load_or_create(tmp_path)
    manifest.mark_succeeded("poses", {"sparse_model": sparse}, {}, 0)
    seen = {}

    def fake_brush(cmd, **kwargs):
        data = tmp_path / "train" / "brush_data"
        seen["data_arg"] = cmd[1]
        seen["images"] = sorted(p.name for p in (data / "images").iterdir())
        seen["model"] = sorted(
            i["name"] for i in read_images(data / "sparse" / "0" / "images.bin")
        )
        raise RuntimeError("stop: enough for this test")

    monkeypatch.setattr(train, "brush_executable", lambda cfg: "brush_app.exe")
    monkeypatch.setattr(train, "run_tool", fake_brush)
    config = load_config("balanced", {"train": {"trainer": "brush", "test_every": 8}})
    ctx = StepContext(tmp_path, [], config, manifest, setup_logging(tmp_path / "logs"))
    with pytest.raises(RuntimeError, match="stop"):
        train.run_brush(ctx)
    held_out = held_out_frames(sparse, 8)
    assert len(held_out) == 3
    assert seen["data_arg"] == str(tmp_path / "train" / "brush_data")
    assert seen["images"] == seen["model"]
    assert not set(held_out) & set(seen["model"])
    assert len(seen["model"]) == 21


def test_evaluate_needs_held_out_frames(tmp_path):
    Manifest.load_or_create(tmp_path).save()
    with pytest.raises(PipelineError, match="kept no frames out of training"):
        evaluate_existing_run(tmp_path)


# ---- benchmark variants -----------------------------------------------------------


def test_variants_change_one_thing_and_are_valid_settings():
    for name, variant in bench.VARIANTS.items():
        args = bench.set_args(name)
        overrides = dict(a.split("=", 1) for a in args[1::2])
        config = load_config(
            "balanced",
            {"train": {k.split(".", 1)[1]: v for k, v in overrides.items()}},
        )
        assert config.train.trainer == "gsplat"
        reference = variant["compared_to"]
        if reference and "frames" not in variant:
            changed = set(variant["set"]) ^ set(bench.VARIANTS[reference]["set"])
            changed |= {
                k
                for k in set(variant["set"]) & set(bench.VARIANTS[reference]["set"])
                if variant["set"][k] != bench.VARIANTS[reference]["set"][k]
            }
            assert len(changed) == 1, (name, changed)
    assert bench.set_args("antialiased") == [
        "--set", "train.trainer=gsplat", "--set", "train.antialiased=true",
    ]  # fmt: skip


def test_results_table(tmp_path):
    run = tmp_path / "mcmc-1M"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "steps": {
                    "train": {
                        "info": {
                            "num_gaussians": 1_000_000,
                            "ply_size_bytes": 236_000_000,
                            "train_seconds": 2400,
                            "evaluation": {
                                "psnr": 27.123,
                                "ssim": 0.9,
                                "lpips": 0.15,
                                "num_test_frames": 38,
                            },
                        }
                    }
                }
            }
        )
    )
    rows = bench.collect_results({"mcmc-1M": run, "missing": tmp_path / "nope"})
    table = bench.results_markdown(rows)
    assert (
        "| mcmc-1M | MCMC strategy (its default budget: 1M splats) | 27.12 | 0.9000 "
        "| 0.1500 | 1,000,000 | 236.0 | 40.0 |" in table
    )
    assert "missing" not in table
