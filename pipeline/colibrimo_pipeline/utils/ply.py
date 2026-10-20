"""Minimal PLY reading: count Gaussians, and read their positions.

Splat PLYs (gsplat, Brush) are ``binary_little_endian`` with one ``vertex``
element whose properties are all ``float`` (x, y, z, f_dc_*, opacity, ...).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_PLY_TYPES = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int8": "i1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
}


def read_vertex_xyz(path: Path) -> np.ndarray:
    """Positions (N x 3, float64) of all vertices of a binary little-endian PLY.

    The vertex data is memory-mapped, so only x/y/z are actually read (a
    1M-splat file has ~60 floats per splat).
    """
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file.")
        count, fields, in_vertex = 0, [], False
        while raw := f.readline():
            parts = raw.decode("ascii", errors="replace").split()
            if not parts:
                continue
            if parts[0] == "format" and parts[1] != "binary_little_endian":
                raise ValueError(f"{path}: only binary_little_endian PLY is supported.")
            if parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError(f"{path}: list properties are not supported.")
                fields.append((parts[2], _PLY_TYPES[parts[1]]))
            elif parts[0] == "end_header":
                break
        header_size = f.tell()
    data = np.memmap(
        path, dtype=np.dtype(fields), mode="r", offset=header_size, shape=(count,)
    )
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    del data  # release the file (Windows keeps memory-mapped files locked)
    return xyz


def read_splat_ply(path: Path) -> dict:
    """All splat parameters of a 3DGS PLY (INRIA layout, as gsplat and Brush write).

    Returns raw values as stored: ``means`` [N,3], ``quats`` (w,x,y,z) [N,4],
    ``log_scales`` [N,3], ``opacity_logits`` [N], ``sh`` [N,K,3] (K = (degree+1)^2;
    ``f_rest_*`` is stored per colour channel, so it is reshaped [N,3,K-1] and
    transposed), ``sh_degree`` and ``count``.
    """
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file.")
        fields, count = [], 0
        while (line := f.readline().decode("ascii", errors="replace").split()) != [
            "end_header"
        ]:
            if not line:
                continue
            if line[0] == "format" and line[1] != "binary_little_endian":
                raise ValueError(f"{path}: only binary_little_endian PLY is supported.")
            if line[:2] == ["element", "vertex"]:
                count = int(line[2])
            elif line[0] == "property":
                fields.append((line[2], _PLY_TYPES[line[1]]))
        data = np.fromfile(f, dtype=np.dtype(fields), count=count)
    names = data.dtype.names

    def columns(prefix: str) -> np.ndarray:
        keys = sorted(
            (n for n in names if n.startswith(prefix)),
            key=lambda n: int(n.rsplit("_", 1)[1]),
        )
        if not keys:
            return np.zeros((count, 0), dtype=np.float32)
        return np.stack([data[k] for k in keys], axis=1).astype(np.float32)

    sh0 = columns("f_dc_")[:, None, :]
    rest = columns("f_rest_")
    sh = (
        np.concatenate([sh0, rest.reshape(count, 3, -1).transpose(0, 2, 1)], axis=1)
        if rest.shape[1]
        else sh0
    )
    return {
        "means": np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32),
        "quats": columns("rot_"),
        "log_scales": columns("scale_"),
        "opacity_logits": data["opacity"].astype(np.float32),
        "sh": np.ascontiguousarray(sh),
        "sh_degree": int(round(np.sqrt(sh.shape[1]))) - 1,
        "count": count,
    }


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
