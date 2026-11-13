#!/usr/bin/env python3
"""
标记球位姿与半径数据（毫米制）。

表中坐标与半径单位均为 mm；HU 为 CT 灰度参考值，写入物体自定义属性。
本模块不含 bpy，可在 Blender 外被导入或单独检查。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class MarkerSphere:
    """单个标记球：编号、中心坐标 (mm)、HU、半径 (mm)。"""

    index: int
    x_mm: float
    y_mm: float
    z_mm: float
    hu: float
    radius_mm: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 用户给定的 6 个圆球（单位：mm）
MARKERS: List[MarkerSphere] = [
    MarkerSphere(1, 93.0, -76.2, 167.1, 3988, 4.6),
    MarkerSphere(2, 126.2, -68.4, 147.5, 3961, 4.7),
    MarkerSphere(3, 74.2, -77.1, 123.6, 3961, 4.7),
    MarkerSphere(4, 142.2, -58.5, 114.0, 3975, 4.7),
    MarkerSphere(5, 55.9, -71.5, 84.2, 3958, 4.7),
    MarkerSphere(6, 100.4, -70.3, 82.7, 4006, 4.6),
    MarkerSphere(7, 108.1, -8.2, 112.9, 3980, 4.7),
]


def markers_as_dicts() -> List[Dict[str, Any]]:
    """返回可序列化的标记球列表。"""
    return [m.to_dict() for m in MARKERS]
