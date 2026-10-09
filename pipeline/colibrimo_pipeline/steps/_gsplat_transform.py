"""Helper run in the GPU environment: print gsplat's scene normalisation as JSON.

gsplat's trainer recentres, rotates and rescales the COLMAP scene before
training (``normalize_world_space``), so the exported PLY is in those
normalised coordinates. We rebuild gsplat's own ``Parser`` with the same
arguments the trainer uses to read back that 4x4 transform.

Usage: python _gsplat_transform.py <gsplat_examples_dir> <data_dir> <test_every>
"""

import json
import sys


def main() -> None:
    examples_dir, data_dir, test_every = sys.argv[1], sys.argv[2], int(sys.argv[3])
    sys.path.insert(0, examples_dir)
    from datasets.colmap import Parser  # gsplat examples package

    parser = Parser(data_dir=data_dir, factor=1, normalize=True, test_every=test_every)
    print(
        json.dumps(
            {
                "matrix_4x4": parser.transform.tolist(),
                "scene_scale": float(parser.scene_scale),
            }
        )
    )


if __name__ == "__main__":
    main()
