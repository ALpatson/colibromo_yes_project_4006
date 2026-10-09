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


@pytest.mark.gpu
def test_gpu_environment_is_detected():
    """Smoke test on a GPU machine: torch sees CUDA and gsplat imports."""
    from colibrimo_pipeline.utils.versions import gpu_env_versions

    info = gpu_env_versions()
    assert info["cuda_available"] is True
    assert "gsplat" in info
