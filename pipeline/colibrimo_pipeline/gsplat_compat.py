"""Workarounds so gsplat v1.5.3's example trainer runs on current Python stacks.

gsplat v1.5.3 (latest release) pins old dependencies in ``examples/``. On
current environments (e.g. Google Colab with Python 3.13 + numpy 2) two things
break. Both fixes are small, idempotent and logged; remove them when moving to
a gsplat release that no longer needs them (gsplat's main branch already uses
the official ``pycolmap`` and numpy 2). See docs/decisions.md D-009.

A third change lets ``test_every 0`` train on every frame (D-021): v1.5.3 always
holds out frames 0, N, 2N, ... (``indices % test_every``), even frame 0 for a
huge N.
"""

from __future__ import annotations

from pathlib import Path

from colibrimo_pipeline.utils.proc import run_capture

# Exact line in rmbrualla/pycolmap@cc7ea4b (the COLMAP reader gsplat v1.5.3 uses).
PYCOLMAP_OLD_LINE = "INVALID_POINT3D = np.uint64(-1)"
PYCOLMAP_NEW_LINE = (
    "INVALID_POINT3D = np.uint64(np.iinfo(np.uint64).max)"
    "  # numpy 2 fix by colibrimo_pipeline"
)

_FIND_PYCOLMAP = (
    "import importlib.util\n"
    "spec = importlib.util.find_spec('pycolmap')\n"
    "print(spec.submodule_search_locations[0] if spec and spec.submodule_search_locations"
    " else '')\n"
)


def make_datasets_importable(examples_dir: Path) -> bool:
    """Make gsplat's ``examples/datasets`` folder win over an installed ``datasets`` package.

    gsplat v1.5.3 ships ``examples/datasets/`` without ``__init__.py``, so Python
    treats it as a namespace package. If the unrelated HuggingFace ``datasets``
    package is installed (it is on Google Colab), Python imports that instead and
    ``from datasets.colmap import ...`` fails. An empty ``__init__.py`` turns the
    folder into a regular package, found first because the script's folder is at
    the front of ``sys.path``. Returns True if the file was created.
    """
    init_file = examples_dir / "datasets" / "__init__.py"
    if init_file.parent.is_dir() and not init_file.exists():
        init_file.write_text(
            "# Added by colibrimo_pipeline (gsplat_compat.make_datasets_importable)\n",
            encoding="utf-8",
        )
        return True
    return False


def find_pycolmap_dir(python: str) -> Path | None:
    """Folder of the ``pycolmap`` package seen by ``python`` (without importing it)."""
    result = run_capture([python, "-c", _FIND_PYCOLMAP])
    location = result.stdout.strip()
    return Path(location) if result.returncode == 0 and location else None


def patch_pycolmap_for_numpy2(package_dir: Path) -> bool:
    """Fix ``np.uint64(-1)`` (an OverflowError on numpy 2) in the pycolmap fork.

    ``np.uint64(np.iinfo(np.uint64).max)`` is the same value (2**64 - 1) and
    type as the original on numpy 1. Returns True if
    the file was changed, False if already patched or not this fork.
    """
    scene_manager = package_dir / "scene_manager.py"
    if not scene_manager.is_file():
        return False
    text = scene_manager.read_text(encoding="utf-8")
    if PYCOLMAP_OLD_LINE not in text:
        return False
    scene_manager.write_text(
        text.replace(PYCOLMAP_OLD_LINE, PYCOLMAP_NEW_LINE), encoding="utf-8"
    )
    return True


# Exact lines in gsplat v1.5.3's examples/datasets/colmap.py (Dataset.__init__).
SPLIT_OLD_LINES = (
    "self.indices = indices[indices % self.parser.test_every != 0]",
    "self.indices = indices[indices % self.parser.test_every == 0]",
)
SPLIT_NEW_LINES = (
    "self.indices = indices[indices % self.parser.test_every != 0]"
    " if self.parser.test_every > 0 else indices  # colibrimo_pipeline: 0 = all frames",
    "self.indices = indices[indices % self.parser.test_every == 0]"
    " if self.parser.test_every > 0 else indices[:0]  # colibrimo_pipeline",
)


def allow_training_on_all_frames(examples_dir: Path) -> bool:
    """Make ``test_every 0`` mean "no held-out frames" in gsplat's Dataset.

    Returns True if the file was changed, False if already patched; raises if
    the expected lines are missing (another gsplat version)."""
    colmap_py = examples_dir / "datasets" / "colmap.py"
    text = colmap_py.read_text(encoding="utf-8")
    if SPLIT_NEW_LINES[0] in text:
        return False
    if not all(line in text for line in SPLIT_OLD_LINES):
        raise RuntimeError(
            f"{colmap_py} is not gsplat v1.5.3's: cannot set up training on all frames."
        )
    for old, new in zip(SPLIT_OLD_LINES, SPLIT_NEW_LINES, strict=True):
        text = text.replace(old, new)
    colmap_py.write_text(text, encoding="utf-8")
    return True


def apply_gsplat_fixes(examples_dir: Path, python: str) -> list[str]:
    """Apply all workarounds; return a description of each change made."""
    changes = []
    if make_datasets_importable(examples_dir):
        changes.append(f"added {examples_dir / 'datasets' / '__init__.py'}")
    if (examples_dir / "datasets" / "colmap.py").is_file() and (
        allow_training_on_all_frames(examples_dir)
    ):
        changes.append(
            f"patched {examples_dir / 'datasets' / 'colmap.py'}: test_every 0"
        )
    pycolmap_dir = find_pycolmap_dir(python)
    if pycolmap_dir and patch_pycolmap_for_numpy2(pycolmap_dir):
        changes.append(f"patched {pycolmap_dir / 'scene_manager.py'} for numpy 2")
    return changes
