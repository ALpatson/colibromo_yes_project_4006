"""Quality against training time: gsplat checkpoints scored in one run (D-023)."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from colibrimo_pipeline.config import TrainConfig, load_config, parse_set_overrides
from colibrimo_pipeline.errors import ConfigError
from colibrimo_pipeline.steps import train


def test_checkpoints_are_saved_with_the_final_step():
    cfg = TrainConfig(max_steps=30000, checkpoint_steps=[15000, 7000])
    cmd = train.build_train_command(
        "python", Path("ex"), Path("run/colmap"), Path("run/train"), cfg
    )
    joined = " ".join(cmd)
    assert "--save_steps 7000 15000 30000" in joined
    assert "--ply_steps 7000 15000 30000" in joined
    assert "--max_steps 30000" in joined


def test_without_checkpoints_only_the_last_step_is_saved():
    cmd = " ".join(
        train.build_train_command(
            "python", Path("ex"), Path("d"), Path("r"), TrainConfig(max_steps=7000)
        )
    )
    assert "--save_steps 7000 --ply_steps 7000 " in cmd


def test_checkpoint_steps_are_sorted_and_deduplicated():
    cfg = TrainConfig(max_steps=30000, checkpoint_steps=[15000, 7000, 7000])
    assert cfg.checkpoint_steps == [7000, 15000]


@pytest.mark.parametrize("steps", [[30000], [40000], [0]])
def test_checkpoint_steps_must_be_before_the_last_step(steps):
    with pytest.raises(ConfigError, match="checkpoint_steps"):
        load_config("balanced", {"train": {"checkpoint_steps": steps}})


@pytest.mark.parametrize("text", ["[7000,15000]", "7000, 15000", "[15000, 7000]"])
def test_checkpoint_steps_as_text(text):
    assert TrainConfig(checkpoint_steps=text).checkpoint_steps == [7000, 15000]


def test_checkpoint_steps_from_the_command_line():
    config = load_config(
        "balanced", parse_set_overrides(["train.checkpoint_steps=[7000,15000]"])
    )
    assert config.train.checkpoint_steps == [7000, 15000]


def write_stats(result_dir: Path, step: int, seconds: float, mem: float, splats: int):
    stats = result_dir / "stats"
    stats.mkdir(parents=True, exist_ok=True)
    (stats / f"train_step{step - 1:04d}_rank0.json").write_text(
        json.dumps({"mem": mem, "ellipse_time": seconds, "num_GS": splats})
    )


def test_read_train_stats(tmp_path):
    write_stats(tmp_path, 7000, 412.345, 3.217, 512345)
    assert train.read_train_stats(tmp_path, 7000) == {
        "train_seconds": 412.3,
        "gpu_peak_mem_gb": 3.22,
        "num_gaussians": 512345,
    }
    assert train.read_train_stats(tmp_path, 15000) == {}


def write_ply(path: Path, vertices: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\n"
        + f"element vertex {vertices}\n".encode()
        + b"end_header\n"
    )


def test_score_checkpoints(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    result_dir = run_dir / "train"
    write_ply(train.expected_ply(result_dir, 7000), 500000)  # 15000 is missing
    write_stats(result_dir, 7000, 400.0, 3.0, 500000)
    calls = []

    def fake_evaluation(run, ply, *args, out_subdir, **kwargs):
        calls.append((Path(ply).name, out_subdir))
        return {"psnr": 28.5}

    monkeypatch.setattr(train, "run_evaluation", fake_evaluation)
    ctx = SimpleNamespace(
        config=load_config("balanced", {"train": {"checkpoint_steps": [7000, 15000]}}),
        run_dir=run_dir,
        logger=logging.getLogger("test"),
    )
    entries = train._score_checkpoints(
        ctx,
        result_dir,
        Path("sparse/0"),
        [[1.0]],
        ["a.jpg"],
        python="python",
        examples_dir=Path("ex"),
    )

    assert calls == [("point_cloud_6999.ply", "eval/step_7000")]
    assert entries[0]["step"] == 7000
    assert entries[0]["train_seconds"] == 400.0
    assert entries[0]["gpu_peak_mem_gb"] == 3.0
    assert entries[0]["num_gaussians"] == 500000
    assert entries[0]["evaluation"] == {"psnr": 28.5}
    assert entries[1] == {
        "step": 15000,
        "error": "point_cloud_14999.ply was not written",
    }


def test_results_table_has_one_row_per_checkpoint(tmp_path):
    from colibrimo_pipeline import bench

    run = tmp_path / "steps-curve"
    run.mkdir()
    info = {
        "max_steps": 30000,
        "num_gaussians": 700_000,
        "ply_size_bytes": 170_000_000,
        "train_seconds": 1800,
        "gpu_peak_mem_gb": 9.5,
        "evaluation": {"psnr": 29.7, "ssim": 0.953, "lpips": 0.128},
        "checkpoints": [
            {
                "step": 7000,
                "train_seconds": 300,
                "num_gaussians": 400_000,
                "ply_size_bytes": 95_000_000,
                "gpu_peak_mem_gb": 4.0,
                "evaluation": {"psnr": 28.0, "ssim": 0.93, "lpips": 0.18},
            }
        ],
    }
    (run / "manifest.json").write_text(json.dumps({"steps": {"train": {"info": info}}}))

    table = bench.results_markdown(bench.collect_results({"steps-curve": run}))

    assert "| steps-curve @ 7k | same run, after 7,000 steps | 28.00 |" in table
    assert "| 400,000 | 95.0 | 5.0 | 4.0 |" in table
    assert "| steps-curve @ 30k | baseline settings" in table
    assert table.index("@ 7k") < table.index("@ 30k")


def test_no_scores_when_training_on_all_frames(tmp_path, monkeypatch):
    result_dir = tmp_path / "train"
    write_ply(train.expected_ply(result_dir, 7000), 10)
    monkeypatch.setattr(train, "run_evaluation", lambda *a, **k: pytest.fail("scored"))
    ctx = SimpleNamespace(
        config=load_config("balanced", {"train": {"checkpoint_steps": [7000]}}),
        run_dir=tmp_path,
        logger=logging.getLogger("test"),
    )
    entries = train._score_checkpoints(
        ctx, result_dir, Path("s"), [[1.0]], [], python="p", examples_dir=Path("e")
    )
    size = train.expected_ply(result_dir, 7000).stat().st_size
    assert entries == [{"step": 7000, "num_gaussians": 10, "ply_size_bytes": size}]
