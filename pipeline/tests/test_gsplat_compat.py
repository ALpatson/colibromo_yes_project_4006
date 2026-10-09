"""Regression tests for the gsplat v1.5.3 workarounds (found on Colab, 2026-10-09)."""

import os
import subprocess
import sys

from colibrimo_pipeline import gsplat_compat


def run_script(script, pythonpath):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(p) for p in pythonpath)}
    return subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env
    )


def test_gsplat_datasets_folder_wins_over_installed_datasets_package(tmp_path):
    """HuggingFace `datasets` (installed on Colab) shadowed gsplat's examples/datasets."""
    examples = tmp_path / "examples"
    (examples / "datasets").mkdir(parents=True)
    (examples / "datasets" / "colmap.py").write_text("PARSER = 'gsplat'\n")
    trainer = examples / "trainer.py"
    trainer.write_text("from datasets.colmap import PARSER\nprint(PARSER)\n")
    # An unrelated, installed "datasets" package (regular package) elsewhere on the path.
    site = tmp_path / "site"
    (site / "datasets").mkdir(parents=True)
    (site / "datasets" / "__init__.py").write_text("")

    assert run_script(trainer, [site]).returncode != 0  # reproduces the Colab failure
    assert gsplat_compat.make_datasets_importable(examples) is True
    fixed = run_script(trainer, [site])
    assert fixed.returncode == 0, fixed.stderr
    assert fixed.stdout.strip() == "gsplat"
    assert gsplat_compat.make_datasets_importable(examples) is False  # idempotent


def make_fake_pycolmap(site):
    """Minimal package with the exact line that breaks on numpy 2."""
    package = site / "pycolmap"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from .scene_manager import SceneManager\n")
    (package / "scene_manager.py").write_text(
        "import numpy as np\n\n\nclass SceneManager:\n"
        f"    {gsplat_compat.PYCOLMAP_OLD_LINE}\n"
    )
    return package


def test_pycolmap_patch_gives_same_value_and_imports(tmp_path, monkeypatch):
    import numpy as np

    site = tmp_path / "site"
    package = make_fake_pycolmap(site)
    check = tmp_path / "check.py"
    check.write_text(
        "import numpy as np\nfrom pycolmap import SceneManager\n"
        "assert SceneManager.INVALID_POINT3D == 2**64 - 1\n"
        "assert SceneManager.INVALID_POINT3D.dtype == np.uint64\nprint('ok')\n"
    )

    if int(np.__version__.split(".")[0]) >= 2:
        assert run_script(check, [site]).returncode != 0  # the Colab failure

    monkeypatch.setenv("PYTHONPATH", str(site))  # inherited by the lookup subprocess
    assert gsplat_compat.find_pycolmap_dir(sys.executable) == package
    assert gsplat_compat.patch_pycolmap_for_numpy2(package) is True
    result = run_script(check, [site])
    assert result.returncode == 0, result.stderr
    assert gsplat_compat.patch_pycolmap_for_numpy2(package) is False  # idempotent


def test_patch_ignores_other_pycolmap_packages(tmp_path):
    """The official pycolmap (no scene_manager.py / no such line) is left alone."""
    official = tmp_path / "pycolmap"
    official.mkdir()
    (official / "__init__.py").write_text("")
    assert gsplat_compat.patch_pycolmap_for_numpy2(official) is False
