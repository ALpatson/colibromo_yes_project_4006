import os
import subprocess
import sys
from pathlib import Path

import pytest

from colibrimo_pipeline.config import TrainConfig
from colibrimo_pipeline.errors import StepError
from colibrimo_pipeline.steps import train


def test_build_train_command_uses_verified_flags():
    cfg = TrainConfig(
        max_steps=7000, strategy="mcmc", sh_degree=2, extra_args=["--packed"]
    )
    cmd = train.build_train_command(
        "python", Path("ex"), Path("run/colmap"), Path("run/train"), cfg
    )
    assert cmd[:3] == ["python", str(Path("ex") / "simple_trainer.py"), "mcmc"]
    joined = " ".join(cmd)
    for expected in (
        "--data_dir " + str(Path("run/colmap")),
        "--data_factor 1",
        "--max_steps 7000",
        "--sh_degree 2",
        "--eval_steps -1",
        "--ply_steps 7000",
        "--save_ply",
        "--disable_viewer",
        "--disable_video",
    ):
        assert expected in joined
    assert cmd[-1] == "--packed"


def test_expected_ply_name_matches_gsplat_zero_based_steps():
    assert (
        train.expected_ply(Path("r"), 30000)
        == Path("r") / "ply" / "point_cloud_29999.ply"
    )


def test_missing_examples_dir_gives_clear_error(monkeypatch):
    monkeypatch.delenv("GSPLAT_EXAMPLES_DIR", raising=False)
    with pytest.raises(StepError, match="GSPLAT_EXAMPLES_DIR"):
        train.gsplat_examples_dir(TrainConfig())


def test_wrong_examples_dir_gives_clear_error(tmp_path):
    with pytest.raises(StepError, match="No simple_trainer.py"):
        train.gsplat_examples_dir(TrainConfig(gsplat_examples_dir=str(tmp_path)))


def test_examples_dir_from_environment(tmp_path, monkeypatch):
    (tmp_path / "simple_trainer.py").write_text("")
    monkeypatch.setenv("GSPLAT_EXAMPLES_DIR", str(tmp_path))
    assert train.gsplat_examples_dir(TrainConfig()) == tmp_path


def test_gsplat_datasets_folder_wins_over_installed_datasets_package(tmp_path):
    """Regression (Colab 2026-10-09): HuggingFace `datasets` shadowed gsplat's folder."""
    examples = tmp_path / "examples"
    (examples / "datasets").mkdir(parents=True)
    (examples / "datasets" / "colmap.py").write_text("PARSER = 'gsplat'\n")
    (examples / "trainer.py").write_text(
        "from datasets.colmap import PARSER\nprint(PARSER)\n"
    )
    # An unrelated, installed "datasets" package (regular package) elsewhere on the path.
    site = tmp_path / "site"
    (site / "datasets").mkdir(parents=True)
    (site / "datasets" / "__init__.py").write_text("")
    env = {**os.environ, "PYTHONPATH": str(site)}

    def run_trainer():
        return subprocess.run(
            [sys.executable, str(examples / "trainer.py")],
            capture_output=True,
            text=True,
            env=env,
        )

    assert run_trainer().returncode != 0  # reproduces the Colab failure
    assert train.make_datasets_importable(examples) is True
    fixed = run_trainer()
    assert fixed.returncode == 0, fixed.stderr
    assert fixed.stdout.strip() == "gsplat"
    assert train.make_datasets_importable(examples) is False  # idempotent


@pytest.mark.gpu
def test_gpu_environment_is_detected():
    """Smoke test on a GPU machine: torch sees CUDA and gsplat imports."""
    from colibrimo_pipeline.utils.versions import gpu_env_versions

    info = gpu_env_versions()
    assert info["cuda_available"] is True
    assert "gsplat" in info
