import numpy as np
from PIL import Image, ImageFilter

from colibrimo_pipeline.steps.extract_frames import (
    laplacian_variance,
    motion_score,
    score_frame,
    select_sharpest,
)


def checkerboard(size=128, square=8):
    y, x = np.indices((size, size))
    return (((x // square) + (y // square)) % 2 * 255).astype(np.uint8)


def test_sharp_image_scores_higher_than_blurred(tmp_path):
    sharp = Image.fromarray(checkerboard())
    blurred = sharp.filter(ImageFilter.GaussianBlur(3))
    assert laplacian_variance(np.asarray(sharp)) > 5 * laplacian_variance(
        np.asarray(blurred)
    )

    sharp.save(tmp_path / "sharp.jpg")
    blurred.save(tmp_path / "blurred.jpg")
    assert (
        score_frame(tmp_path / "sharp.jpg")[0]
        > score_frame(tmp_path / "blurred.jpg")[0]
    )


def test_flat_image_has_zero_sharpness():
    assert laplacian_variance(np.full((50, 50), 128, np.uint8)) == 0.0


def test_select_sharpest_keeps_one_per_group_in_order():
    #        group 0     group 1     group 2
    scores = [1, 9, 2, 5, 3, 4, 0, 0, 7]
    assert select_sharpest(scores, 3) == [1, 3, 8]


def test_select_sharpest_returns_all_when_too_few():
    assert select_sharpest([3, 1], 5) == [0, 1]


def test_select_sharpest_spreads_over_whole_video():
    selected = select_sharpest(list(range(600)), 300)
    assert len(selected) == 300
    assert selected == sorted(selected)
    assert selected[0] < 2 and selected[-1] >= 598


def test_motion_score_static_vs_moving():
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 255, (64, 64)).astype(np.float32)
    static = [base + rng.normal(0, 1, base.shape) for _ in range(10)]
    moving = [np.roll(base, shift * 5, axis=1) for shift in range(10)]
    assert motion_score(static) < 2
    assert motion_score(moving) > 30
    assert motion_score([base]) == 0.0
