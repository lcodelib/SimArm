#!/usr/bin/env python3
"""
从 2D 平面图（PNG/JPG）提取墙体中心线线段。

流程：读图 → 灰度/二值 → 形态学 →（可选 Canny）→ 概率霍夫 →
像素坐标转米 → 共线合并 → 过滤短段 → 写出 walls.json 与调试图。
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessParams:
    """图像预处理参数。"""

    blur_ksize: int = 3
    threshold_mode: str = "otsu"  # otsu | adaptive | fixed
    fixed_threshold: int = 128
    walls_are_dark: bool = True
    morph_close_ksize: int = 3
    morph_open_ksize: int = 0
    # 对二值墙体做形态学骨架化，减轻厚线双边检测
    skeletonize: bool = True


@dataclass
class HoughParams:
    """概率霍夫直线参数（像素单位）。"""

    threshold: int = 40
    min_line_length_px: int = 30
    max_line_gap_px: int = 8
    # 骨架化场景默认不做 Canny
    canny_low: Optional[int] = None
    canny_high: Optional[int] = None


@dataclass
class MergeParams:
    """共线线段合并参数（米制角度容差）。"""

    angle_tol_deg: float = 8.0
    dist_tol_m: float = 0.15
    gap_tol_m: float = 0.25


@dataclass
class ExtractParams:
    """墙体提取总参数。"""

    meters_per_pixel: float = 0.02
    wall_thickness_m: float = 0.2
    wall_height_m: float = 2.8
    min_wall_length_m: float = 0.3
    preprocess: PreprocessParams = field(default_factory=PreprocessParams)
    hough: HoughParams = field(default_factory=HoughParams)
    merge: MergeParams = field(default_factory=MergeParams)


@dataclass
class WallSegment:
    """一条墙体中心线（平面 XY，单位米；Z 向上由 Blender 挤出）。"""

    x1: float
    y1: float
    x2: float
    y2: float
    thickness_m: float
    height_m: float

    @property
    def length_m(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    def to_dict(self) -> Dict[str, float]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "thickness_m": self.thickness_m,
            "height_m": self.height_m,
            "length_m": self.length_m,
        }


@dataclass
class ExtractResult:
    """提取结果：墙段列表与调试图像路径。"""

    walls: List[WallSegment]
    image_size_px: Tuple[int, int]  # (width, height)
    meters_per_pixel: float
    debug_paths: Dict[str, str] = field(default_factory=dict)


def setup_logging(level: str = "INFO") -> None:
    """配置根日志：时间 + 级别 + 模块 + 消息。"""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def params_from_config(cfg: Dict[str, Any]) -> ExtractParams:
    """从 YAML/字典构造 ExtractParams。"""
    pre = dict(cfg.get("preprocess") or {})
    hog = dict(cfg.get("hough") or {})
    mer = dict(cfg.get("merge") or {})

    preprocess = PreprocessParams(
        blur_ksize=int(pre.get("blur_ksize", 3)),
        threshold_mode=str(pre.get("threshold_mode", "otsu")),
        fixed_threshold=int(pre.get("fixed_threshold", 128)),
        walls_are_dark=bool(pre.get("walls_are_dark", True)),
        morph_close_ksize=int(pre.get("morph_close_ksize", 3)),
        morph_open_ksize=int(pre.get("morph_open_ksize", 0)),
        skeletonize=bool(pre.get("skeletonize", True)),
    )
    hough = HoughParams(
        threshold=int(hog.get("threshold", 40)),
        min_line_length_px=int(hog.get("min_line_length_px", 30)),
        max_line_gap_px=int(hog.get("max_line_gap_px", 8)),
        canny_low=hog.get("canny_low", None),
        canny_high=hog.get("canny_high", None),
    )
    merge = MergeParams(
        angle_tol_deg=float(mer.get("angle_tol_deg", 8.0)),
        dist_tol_m=float(mer.get("dist_tol_m", 0.15)),
        gap_tol_m=float(mer.get("gap_tol_m", 0.25)),
    )
    return ExtractParams(
        meters_per_pixel=float(cfg.get("meters_per_pixel", 0.02)),
        wall_thickness_m=float(cfg.get("wall_thickness_m", 0.2)),
        wall_height_m=float(cfg.get("wall_height_m", 2.8)),
        min_wall_length_m=float(cfg.get("min_wall_length_m", 0.3)),
        preprocess=preprocess,
        hough=hough,
        merge=merge,
    )


def _odd_ksize(k: int) -> int:
    """保证形态学核为正奇数；<=0 返回 0 表示跳过。"""
    if k <= 0:
        return 0
    return k if k % 2 == 1 else k + 1


def morphological_skeleton(binary: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """
    形态学骨架化：将粗墙线收成约 1px 宽的中心线。

    不依赖 opencv-contrib / skimage，仅用 erode / open。
    """
    img = binary.copy()
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max_iter):
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        residual = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, residual)
        img = cv2.erode(img, element)
        if cv2.countNonZero(img) == 0:
            break
    return skel


def binarize_walls(image_bgr: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """
    将平面图转为墙体二值图（墙=255，背景=0）。

    支持 otsu / adaptive / fixed；walls_are_dark 控制是否反色；
    可选骨架化以得到中心线。
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    k = _odd_ksize(params.blur_ksize)
    if k >= 3:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    mode = params.threshold_mode.lower().strip()
    if mode == "otsu":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "adaptive":
        bw = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
    elif mode == "fixed":
        _, bw = cv2.threshold(gray, params.fixed_threshold, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"未知 threshold_mode: {params.threshold_mode}")

    # 统一为「墙=白」：深色墙线时需要反色
    if params.walls_are_dark:
        bw = cv2.bitwise_not(bw)

    close_k = _odd_ksize(params.morph_close_ksize)
    if close_k >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    open_k = _odd_ksize(params.morph_open_ksize)
    if open_k >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    if params.skeletonize:
        bw = morphological_skeleton(bw)
        logger.info("已对墙体二值图做形态学骨架化")

    return bw


def detect_line_segments_px(
    binary_walls: np.ndarray,
    hough: HoughParams,
) -> List[Tuple[float, float, float, float]]:
    """
    从墙体二值图检测线段（像素坐标）。

    若配置了 Canny，则对二值图做边缘再霍夫；否则直接对二值图霍夫。
    返回 [(x1,y1,x2,y2), ...]，图像坐标系：原点左上，y 向下。
    """
    if hough.canny_low is not None and hough.canny_high is not None:
        edges = cv2.Canny(
            binary_walls,
            int(hough.canny_low),
            int(hough.canny_high),
            apertureSize=3,
        )
        src = edges
    else:
        src = binary_walls

    raw = cv2.HoughLinesP(
        src,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(hough.threshold),
        minLineLength=int(hough.min_line_length_px),
        maxLineGap=int(hough.max_line_gap_px),
    )
    if raw is None:
        logger.warning("霍夫未检测到任何线段，请调大对比或降低 hough.threshold")
        return []

    segments: List[Tuple[float, float, float, float]] = []
    for item in raw:
        x1, y1, x2, y2 = item[0]
        segments.append((float(x1), float(y1), float(x2), float(y2)))
    logger.info("霍夫检测到 %d 条原始线段", len(segments))
    return segments


def pixels_to_meters(
    segments_px: Sequence[Tuple[float, float, float, float]],
    meters_per_pixel: float,
    image_height_px: int,
) -> List[Tuple[float, float, float, float]]:
    """
    像素线段 → 米制平面坐标。

    图像 y 向下 → Blender/建筑平面 y 向前（向上翻转为数学坐标系）：
    X_m = x_px * mpp
    Y_m = (H - 1 - y_px) * mpp
    """
    if meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel 必须 > 0")
    mpp = meters_per_pixel
    h = float(image_height_px)
    out: List[Tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in segments_px:
        out.append(
            (
                x1 * mpp,
                (h - 1.0 - y1) * mpp,
                x2 * mpp,
                (h - 1.0 - y2) * mpp,
            )
        )
    return out


def _segment_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """线段方位角，归一化到 (-pi/2, pi/2]。"""
    ang = math.atan2(y2 - y1, x2 - x1)
    # 无向线段：把方向折叠到半圆
    if ang <= -math.pi / 2:
        ang += math.pi
    elif ang > math.pi / 2:
        ang -= math.pi
    return ang


def _point_line_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """点到无限直线距离。"""
    dx = x2 - x1
    dy = y2 - y1
    denom = math.hypot(dx, dy)
    if denom < 1e-12:
        return math.hypot(px - x1, py - y1)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / denom


def _project_t(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """点在线段方向上的投影参数 t（端点为 0/1，可超出）。"""
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom < 1e-18:
        return 0.0
    return ((px - x1) * dx + (py - y1) * dy) / denom


def merge_colinear_segments(
    segments_m: Sequence[Tuple[float, float, float, float]],
    params: MergeParams,
) -> List[Tuple[float, float, float, float]]:
    """
    迭代合并近似共线且端点间隙不大的线段。

    合并后取投影区间的两端点作为新线段。
    """
    if not segments_m:
        return []

    segs = [list(s) for s in segments_m]
    angle_tol = math.radians(params.angle_tol_deg)
    changed = True
    while changed:
        changed = False
        used = [False] * len(segs)
        merged: List[List[float]] = []
        for i in range(len(segs)):
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = segs[i]
            a_ang = _segment_angle(ax1, ay1, ax2, ay2)
            group = [(ax1, ay1), (ax2, ay2)]
            used[i] = True
            for j in range(i + 1, len(segs)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = segs[j]
                b_ang = _segment_angle(bx1, by1, bx2, by2)
                dang = abs(a_ang - b_ang)
                dang = min(dang, math.pi - dang)
                if dang > angle_tol:
                    continue
                # 中点到 a 直线距离
                mx = 0.5 * (bx1 + bx2)
                my = 0.5 * (by1 + by2)
                if _point_line_distance(mx, my, ax1, ay1, ax2, ay2) > params.dist_tol_m:
                    continue
                # 端点间隙：任一端点落在 a 延长带附近即可
                pts_b = [(bx1, by1), (bx2, by2)]
                close_enough = False
                length_a = math.hypot(ax2 - ax1, ay2 - ay1)
                gap_t = params.gap_tol_m / max(length_a, 1e-9)
                for px, py in pts_b:
                    t = _project_t(px, py, ax1, ay1, ax2, ay2)
                    if -gap_t <= t <= 1.0 + gap_t:
                        if _point_line_distance(px, py, ax1, ay1, ax2, ay2) <= params.dist_tol_m:
                            close_enough = True
                            break
                # 也允许两线段端点欧氏距离足够近
                if not close_enough:
                    ends_a = [(ax1, ay1), (ax2, ay2)]
                    for pa in ends_a:
                        for pb in pts_b:
                            if math.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= params.gap_tol_m:
                                close_enough = True
                                break
                        if close_enough:
                            break
                if not close_enough:
                    continue
                group.extend(pts_b)
                used[j] = True
                changed = True

            # 用主方向投影取极值端点
            # 取组内点协方差主方向
            pts = np.array(group, dtype=np.float64)
            mean = pts.mean(axis=0)
            centered = pts - mean
            if len(pts) >= 2:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                direction = vh[0]
            else:
                direction = np.array([1.0, 0.0])
            projs = centered @ direction
            i_min = int(np.argmin(projs))
            i_max = int(np.argmax(projs))
            p0 = pts[i_min]
            p1 = pts[i_max]
            merged.append([float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])])
        segs = merged

    return [(s[0], s[1], s[2], s[3]) for s in segs]


def build_wall_segments(
    segments_m: Sequence[Tuple[float, float, float, float]],
    thickness_m: float,
    height_m: float,
    min_length_m: float,
) -> List[WallSegment]:
    """过滤短段并封装为 WallSegment。"""
    walls: List[WallSegment] = []
    for x1, y1, x2, y2 in segments_m:
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_length_m:
            continue
        walls.append(
            WallSegment(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                thickness_m=thickness_m,
                height_m=height_m,
            )
        )
    logger.info("过滤后保留 %d 条墙段（min_length=%.3f m）", len(walls), min_length_m)
    return walls


def draw_debug_overlay(
    image_bgr: np.ndarray,
    segments_px: Sequence[Tuple[float, float, float, float]],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """在原图上绘制像素线段，便于调参。"""
    canvas = image_bgr.copy()
    for x1, y1, x2, y2 in segments_px:
        cv2.line(
            canvas,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
    return canvas


def meters_to_pixels(
    segments_m: Sequence[Tuple[float, float, float, float]],
    meters_per_pixel: float,
    image_height_px: int,
) -> List[Tuple[float, float, float, float]]:
    """米制坐标还原为像素，用于调试叠加。"""
    mpp = meters_per_pixel
    h = float(image_height_px)
    out: List[Tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in segments_m:
        out.append(
            (
                x1 / mpp,
                h - 1.0 - y1 / mpp,
                x2 / mpp,
                h - 1.0 - y2 / mpp,
            )
        )
    return out


def extract_walls_from_image(
    image_bgr: np.ndarray,
    params: ExtractParams,
) -> Tuple[List[WallSegment], np.ndarray, List[Tuple[float, float, float, float]]]:
    """
    对已加载的 BGR 图执行完整抽线。

    返回：(墙段列表, 二值图, 合并后像素线段用于调试)。
    """
    binary = binarize_walls(image_bgr, params.preprocess)
    segs_px = detect_line_segments_px(binary, params.hough)
    h_px = image_bgr.shape[0]
    segs_m = pixels_to_meters(segs_px, params.meters_per_pixel, h_px)
    merged_m = merge_colinear_segments(segs_m, params.merge)
    walls = build_wall_segments(
        merged_m,
        thickness_m=params.wall_thickness_m,
        height_m=params.wall_height_m,
        min_length_m=params.min_wall_length_m,
    )
    merged_px = meters_to_pixels(
        [(w.x1, w.y1, w.x2, w.y2) for w in walls],
        params.meters_per_pixel,
        h_px,
    )
    return walls, binary, merged_px


def save_walls_json(
    walls: Sequence[WallSegment],
    output_path: Path,
    *,
    meters_per_pixel: float,
    image_path: Optional[str] = None,
    image_size_px: Optional[Tuple[int, int]] = None,
) -> None:
    """写出供 Blender 读取的 walls.json。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "unit": "meter",
        "coordinate_system": "xy_floor_z_up",
        "meters_per_pixel": meters_per_pixel,
        "image_path": image_path,
        "image_size_px": (
            {"width": image_size_px[0], "height": image_size_px[1]}
            if image_size_px
            else None
        ),
        "wall_count": len(walls),
        "walls": [w.to_dict() for w in walls],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("已写入墙体 JSON: %s（%d 段）", output_path, len(walls))


def run_extract(
    image_path: Path,
    output_dir: Path,
    params: ExtractParams,
) -> ExtractResult:
    """端到端：读图 → 抽线 → 写 JSON/调试图。"""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    h, w = image_bgr.shape[:2]
    logger.info(
        "读取平面图 %s (%dx%d)，mpp=%.6f m/px",
        image_path,
        w,
        h,
        params.meters_per_pixel,
    )

    walls, binary, merged_px = extract_walls_from_image(image_bgr, params)

    binary_path = output_dir / "debug_binary.png"
    overlay_path = output_dir / "debug_walls_overlay.png"
    json_path = output_dir / "walls.json"

    cv2.imwrite(str(binary_path), binary)
    overlay = draw_debug_overlay(image_bgr, merged_px)
    cv2.imwrite(str(overlay_path), overlay)
    save_walls_json(
        walls,
        json_path,
        meters_per_pixel=params.meters_per_pixel,
        image_path=str(image_path.resolve()),
        image_size_px=(w, h),
    )

    # 同步写出参数快照，便于复现
    meta_path = output_dir / "extract_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "image": str(image_path.resolve()),
                "params": {
                    "meters_per_pixel": params.meters_per_pixel,
                    "wall_thickness_m": params.wall_thickness_m,
                    "wall_height_m": params.wall_height_m,
                    "min_wall_length_m": params.min_wall_length_m,
                    "preprocess": asdict(params.preprocess),
                    "hough": asdict(params.hough),
                    "merge": asdict(params.merge),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return ExtractResult(
        walls=walls,
        image_size_px=(w, h),
        meters_per_pixel=params.meters_per_pixel,
        debug_paths={
            "walls_json": str(json_path),
            "binary": str(binary_path),
            "overlay": str(overlay_path),
            "meta": str(meta_path),
        },
    )
