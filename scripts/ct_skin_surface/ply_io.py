#!/usr/bin/env python3
"""
PLY 点云读写（ASCII），与 stereo_depth.save_ply 风格一致。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def save_ply(
    points: np.ndarray,
    path: Path,
    *,
    colors: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
) -> Path:
    """
    写出 ASCII PLY。

    points: (N,3)
    colors: 可选 (N,3) uint8 RGB
    faces: 可选 (M,3) 三角面索引；有面时写成 mesh，便于 MeshLab 查看
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points 须为 (N,3)，得到 {points.shape}")

    n = points.shape[0]
    has_color = colors is not None and colors.shape[0] == n
    has_faces = faces is not None and len(faces) > 0
    m = int(faces.shape[0]) if has_faces else 0

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if has_faces:
            f.write(f"element face {m}\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            if has_color:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        if has_faces:
            for tri in faces:
                f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")

    logger.info(
        "已写入 PLY: %s（顶点=%d，面=%d）",
        path,
        n,
        m,
    )
    return path.resolve()
