"""Minimal PLY header reader (enough to count Gaussians without loading the file)."""

from __future__ import annotations

from pathlib import Path


def read_vertex_count(path: Path) -> int:
    """Number of vertices (= Gaussians for a splat file) declared in a PLY header."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file.")
        for raw in f:
            line = raw.strip()
            if line.startswith(b"element vertex "):
                return int(line.split()[2])
            if line == b"end_header":
                break
    raise ValueError(f"{path} has no 'element vertex' line in its header.")
