"""
双目立体视觉：立体校正 → 视差 → 深度图 → 点云。

约定（与 OpenCV stereoRectify 一致）：
- 左相机为 camera1，右相机为 camera2。
- 外参 R、T：p_right = R @ p_left + T（T 为 OpenCV 立体标定语义，见 extrinsic_translation_simarm）。
- 畸变 D 为 (k1, k2, p1, p2[, k3])，与 hand_eye 配置相同。

输出 3D（depth.npy / pointcloud.ply）坐标系：
- **左目极线校正后的 OpenCV 相机系**（与 reprojectImageTo3D(Q) 一致）。
- +X：图像 u 向右；+Y：图像 v 向下；+Z：光轴前方（场景深度为正）。
- SimArm 支架上右目在左目 +X 侧、基线 5 cm 时，应使用 T_opencv = [-baseline, 0, 0]，
  勿用支架向量 [+baseline, 0, 0] 直接填入（否则 Z 会为负且与 depth 回退公式不一致）。
- 转 Three.js 左目局部系（Y 上、看向 −Z）：p_three = [X, -Y, -Z]。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import yaml

ArrayLike = Union[Sequence[float], np.ndarray, List[List[float]]]


@dataclass
class StereoCameraParams:
    K: np.ndarray  # 3x3
    D: np.ndarray  # (n,) n>=4


@dataclass
class StereoExtrinsic:
    R: np.ndarray  # 3x3
    T: np.ndarray  # (3,1) 米


@dataclass
class StereoDepthResult:
    disparity: np.ndarray
    depth: np.ndarray
    Q: np.ndarray
    map1_left: Tuple[np.ndarray, np.ndarray]
    map1_right: Tuple[np.ndarray, np.ndarray]
    P1: np.ndarray
    P2: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    roi_left: Tuple[int, int, int, int]
    roi_right: Tuple[int, int, int, int]
    rect_left: np.ndarray
    rect_right: np.ndarray
    points_xyz: np.ndarray  # HxWx3
    valid_mask: np.ndarray


def _as_float64(a: ArrayLike) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def parse_camera_matrix(K: ArrayLike) -> np.ndarray:
    m = _as_float64(K)
    if m.shape == (3, 3):
        return m
    if m.size == 9:
        return m.reshape(3, 3)
    if m.size == 4:
        fx, fy, cx, cy = m.ravel()
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"无法解析内参 K，shape={m.shape}")


def parse_distortion(D: ArrayLike) -> np.ndarray:
    d = _as_float64(D).ravel()
    if d.size < 4:
        raise ValueError("畸变系数至少 4 个 (k1,k2,p1,p2)")
    return d


def parse_rotation(R: ArrayLike) -> np.ndarray:
    r = _as_float64(R)
    if r.shape == (3, 3):
        return r
    if r.size == 9:
        return r.reshape(3, 3)
    if r.size == 3:
        return cv2.Rodrigues(r.reshape(3, 1))[0]
    raise ValueError(f"无法解析旋转 R，shape={r.shape}")


def parse_translation(T: ArrayLike) -> np.ndarray:
    t = _as_float64(T).ravel()
    if t.size != 3:
        raise ValueError("平移 T 须为 3 维向量 (tx, ty, tz)，单位：米")
    return t.reshape(3, 1)


def load_stereo_params_from_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("配置文件须为 YAML 字典")
    return cfg


def camera_from_yaml_block(block: Dict[str, Any]) -> StereoCameraParams:
    """从内联 left/right 块或其中的 camera 子块解析 K、D。"""
    if "camera" in block and isinstance(block["camera"], dict):
        block = block["camera"]
    if "camera_matrix" in block:
        K = parse_camera_matrix(block["camera_matrix"])
    else:
        K = parse_camera_matrix(
            [block["fx"], block["fy"], block["cx"], block["cy"]]
        )
    D = parse_distortion(block.get("dist_coeffs", block.get("D", [0, 0, 0, 0])))
    return StereoCameraParams(K=K, D=D)


def load_camera_from_hand_eye_config(path: Union[str, Path]) -> StereoCameraParams:
    """读取 SimArm 手眼 config.yaml（含顶层 camera: fx/fy/cx/cy/dist_coeffs）。"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"手眼配置须为 YAML 字典: {path}")
    if "camera" not in cfg:
        raise ValueError(f"手眼配置缺少 camera 段: {path}")
    return camera_from_yaml_block(cfg)


def resolve_stereo_cameras(cfg: Dict[str, Any]) -> Tuple[StereoCameraParams, StereoCameraParams]:
    """支持 left/right 内联，或 left_config/right_config 指向手眼 YAML。"""
    if "left_config" in cfg or "right_config" in cfg:
        if "left_config" not in cfg or "right_config" not in cfg:
            raise ValueError("使用 left_config 时须同时提供 right_config")
        left = load_camera_from_hand_eye_config(cfg["left_config"])
        right = load_camera_from_hand_eye_config(cfg["right_config"])
        return left, right
    if "left" not in cfg or "right" not in cfg:
        raise ValueError("配置须包含 left/right 或 left_config/right_config")
    return camera_from_yaml_block(cfg["left"]), camera_from_yaml_block(cfg["right"])


def extrinsic_translation_simarm(
    baseline_m: float,
    *,
    axis: str = "x",
    simarm_positive_x: bool = True,
) -> np.ndarray:
    """
    SimArm 支架基线（右目在左目 +X）→ OpenCV stereoRectify 用的 T。

    右目在左目 +X 时 OpenCV 的 Tx 为负，才能使 reprojectImageTo3D 的 Z>0 表示前方深度。
    """
    b = float(baseline_m)
    if axis.lower() != "x":
        raise ValueError(f"暂仅支持 axis=x，收到 {axis!r}")
    sign = -1.0 if simarm_positive_x else 1.0
    return np.array([[sign * b], [0.0], [0.0]], dtype=np.float64)


def extrinsic_from_yaml(cfg: Dict[str, Any]) -> StereoExtrinsic:
    ext = cfg.get("extrinsic", cfg)
    R = parse_rotation(ext["R"])
    if "T" in ext:
        T = parse_translation(ext["T"])
    elif cfg.get("baseline_m") is not None:
        axis = str(cfg.get("baseline_axis", "x"))
        simarm = bool(cfg.get("simarm_baseline_positive_x", True))
        T = extrinsic_translation_simarm(float(cfg["baseline_m"]), axis=axis, simarm_positive_x=simarm)
    else:
        raise ValueError("extrinsic 须含 T，或配置 baseline_m")
    return StereoExtrinsic(R=R, T=T)


def stereo_rectify_maps(
    *,
    image_size: Tuple[int, int],
    left: StereoCameraParams,
    right: StereoCameraParams,
    extrinsic: StereoExtrinsic,
    alpha: float = 0.0,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Tuple[int, int, int, int],
    Tuple[int, int, int, int],
]:
    """计算校正映射、重投影矩阵 Q，以及左右校正旋转 R1/R2。"""
    w, h = image_size
    flags = cv2.CALIB_ZERO_DISPARITY
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left.K,
        left.D,
        right.K,
        right.D,
        (w, h),
        extrinsic.R,
        extrinsic.T,
        flags=flags,
        alpha=alpha,
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(
        left.K, left.D, R1, P1, (w, h), cv2.CV_32FC1
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        right.K, right.D, R2, P2, (w, h), cv2.CV_32FC1
    )
    return (map1_l, map2_l), (map1_r, map2_r), Q, P1, P2, R1, R2, roi1, roi2


def rectify_pair(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    map_left: Tuple[np.ndarray, np.ndarray],
    map_right: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    rect_l = cv2.remap(
        left_bgr, map_left[0], map_left[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    rect_r = cv2.remap(
        right_bgr, map_right[0], map_right[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return rect_l, rect_r


def create_stereo_matcher(
    *,
    algorithm: str = "sgbm",
    num_disparities: int = 128,
    block_size: int = 5,
    min_disparity: int = 0,
) -> cv2.StereoMatcher:
    num_disparities = int(np.ceil(num_disparities / 16) * 16)
    if num_disparities < 16:
        num_disparities = 16
    block_size = max(3, block_size | 1)

    if algorithm.lower() in ("sgbm", "semi-global"):
        matcher = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
    elif algorithm.lower() in ("bm", "block"):
        matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
        matcher.setMinDisparity(min_disparity)
    else:
        raise ValueError(f"未知立体匹配算法: {algorithm}")
    return matcher


def compute_disparity(
    rect_left: np.ndarray,
    rect_right: np.ndarray,
    matcher: cv2.StereoMatcher,
) -> np.ndarray:
    gray_l = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY) if rect_left.ndim == 3 else rect_left
    gray_r = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY) if rect_right.ndim == 3 else rect_right
    disp_raw = matcher.compute(gray_l, gray_r)
    disp = disp_raw.astype(np.float32) / 16.0
    disp[disp <= min_disparity_effective(matcher)] = 0.0
    return disp


def min_disparity_effective(matcher: cv2.StereoMatcher) -> float:
    if hasattr(matcher, "getMinDisparity"):
        return float(matcher.getMinDisparity())
    return 0.0


def reproject_disparity_to_points(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """视差 → 左目校正 OpenCV 坐标系 3D 点（Z>0 为前方）。"""
    points = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    # OpenCV Q 在 Tx>0（误配 SimArm +X 基线）时 Z 为负；统一为「前方为正深度」
    z = points[:, :, 2]
    if np.nanmedian(z[np.isfinite(z) & (disparity > 0)]) < 0:
        points = points.copy()
        points[:, :, 0] *= -1.0
        points[:, :, 1] *= -1.0
        points[:, :, 2] *= -1.0
    return points


def disparity_to_depth(disparity: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """由 Q 重投影得到 HxWx3 与深度 Z（米，光轴前方为正）。"""
    points = reproject_disparity_to_points(disparity, Q)
    depth = points[:, :, 2].astype(np.float32)
    valid = np.isfinite(depth) & (disparity > 0) & (depth > 0)
    depth[~valid] = 0.0
    return depth, valid


def depth_baseline_fallback(
    disparity: np.ndarray,
    fx: float,
    baseline_m: float,
) -> np.ndarray:
    """Z = fx * B / d；d 为像素视差。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (fx * baseline_m) / disparity
    depth[disparity <= 0] = 0.0
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


def extract_point_cloud(
    points_xyz: np.ndarray,
    valid_mask: np.ndarray,
    colors_bgr: Optional[np.ndarray] = None,
    max_points: int = 500_000,
    stride: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    stride = max(1, stride)
    pts = points_xyz[::stride, ::stride][valid_mask[::stride, ::stride]]
    cols = None
    if colors_bgr is not None:
        cols = colors_bgr[::stride, ::stride][valid_mask[::stride, ::stride]]
        cols = cols[:, ::-1]  # BGR -> RGB for PLY
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
        if cols is not None:
            cols = cols[idx]
    return pts, cols


def save_depth_image(depth_m: np.ndarray, path: Path, max_depth_m: float = 10.0) -> None:
    """16 位 PNG：0 表示无效，否则 depth_mm = depth_m * 1000。"""
    depth_mm = np.clip(depth_m * 1000.0, 0, 65534).astype(np.uint16)
    depth_mm[depth_m <= 0] = 0
    cv2.imwrite(str(path), depth_mm)


def save_disparity_image(disparity: np.ndarray, path: Path) -> None:
    disp_vis = np.clip(disparity, 0, None)
    if disp_vis.max() > 0:
        disp_vis = (disp_vis / disp_vis.max() * 255).astype(np.uint8)
    else:
        disp_vis = np.zeros_like(disparity, dtype=np.uint8)
    cv2.imwrite(str(path), disp_vis)


def save_ply(
    points: np.ndarray,
    path: Path,
    colors: Optional[np.ndarray] = None,
) -> None:
    path = Path(path)
    n = points.shape[0]
    has_color = colors is not None and colors.shape[0] == n
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            if has_color:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def run_stereo_depth(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    left: StereoCameraParams,
    right: StereoCameraParams,
    extrinsic: StereoExtrinsic,
    *,
    alpha: float = 0.0,
    algorithm: str = "sgbm",
    num_disparities: int = 128,
    block_size: int = 5,
    min_disparity: int = 0,
    baseline_m: Optional[float] = None,
) -> StereoDepthResult:
    if left_bgr.shape[:2] != right_bgr.shape[:2]:
        raise ValueError(
            f"左右图尺寸须一致，得到 left={left_bgr.shape[:2]} right={right_bgr.shape[:2]}"
        )
    h, w = left_bgr.shape[:2]
    map_l, map_r, Q, P1, P2, R1, R2, roi_l, roi_r = stereo_rectify_maps(
        image_size=(w, h),
        left=left,
        right=right,
        extrinsic=extrinsic,
        alpha=alpha,
    )
    rect_l, rect_r = rectify_pair(left_bgr, right_bgr, map_l, map_r)
    matcher = create_stereo_matcher(
        algorithm=algorithm,
        num_disparities=num_disparities,
        block_size=block_size,
        min_disparity=min_disparity,
    )
    disp = compute_disparity(rect_l, rect_r, matcher)
    points = reproject_disparity_to_points(disp, Q)
    depth = points[:, :, 2].astype(np.float32)
    valid = np.isfinite(depth) & (disp > 0) & (depth > 0)
    depth[~valid] = 0.0

    if baseline_m is None:
        baseline_m = abs(float(extrinsic.T[0, 0])) or float(np.linalg.norm(extrinsic.T))
    if baseline_m > 0 and np.count_nonzero(valid) < 0.01 * valid.size:
        depth = depth_baseline_fallback(disp, float(P1[0, 0]), baseline_m)
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        fx_f, fy_f = float(P1[0, 0]), float(P1[1, 1])
        cx_f, cy_f = float(P1[0, 2]), float(P1[1, 2])
        points = np.stack(
            [
                (uu - cx_f) * depth / fx_f,
                (vv - cy_f) * depth / fy_f,
                depth,
            ],
            axis=-1,
        )
        valid = (disp > 0) & (depth > 0)
        depth[~valid] = 0.0

    return StereoDepthResult(
        disparity=disp,
        depth=depth,
        Q=Q,
        map1_left=map_l,
        map1_right=map_r,
        P1=P1,
        P2=P2,
        R1=R1,
        R2=R2,
        roi_left=roi_l,
        roi_right=roi_r,
        rect_left=rect_l,
        rect_right=rect_r,
        points_xyz=points,
        valid_mask=valid,
    )


def coordinate_frame_metadata(
    result: StereoDepthResult,
    *,
    baseline_m: float,
    extrinsic_T: np.ndarray,
) -> Dict[str, Any]:
    """输出坐标系说明，便于与 SimArm / 手眼对齐。"""
    z_med = float(np.median(result.depth[result.valid_mask])) if np.any(result.valid_mask) else 0.0
    return {
        "frame": "opencv_rectified_left",
        "axes": {"x": "image_u_right", "y": "image_v_down", "z": "optical_forward_positive"},
        "units": "m",
        "baseline_m": baseline_m,
        "extrinsic_T_opencv_m": [float(x) for x in extrinsic_T.ravel()[:3]],
        "median_depth_m": z_med,
        "simarm_note": "支架右目在左目 +X；OpenCV T 的 Tx 应为负。转 Three 左目: [X,-Y,-Z]。",
        "hand_eye": "p_tcp = T_cam2tcp @ p_cam（eye_in_hand，见 hand_eye_result.yaml）",
    }


def save_stereo_outputs(
    result: StereoDepthResult,
    output_dir: Union[str, Path],
    *,
    left_bgr_for_color: Optional[np.ndarray] = None,
    max_points: int = 500_000,
    stride: int = 2,
    max_depth_m: float = 10.0,
    frame_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    p_rect_l = out / "rectified_left.png"
    p_rect_r = out / "rectified_right.png"
    cv2.imwrite(str(p_rect_l), result.rect_left)
    cv2.imwrite(str(p_rect_r), result.rect_right)
    paths["rectified_left"] = str(p_rect_l)
    paths["rectified_right"] = str(p_rect_r)

    p_disp = out / "disparity.png"
    save_disparity_image(result.disparity, p_disp)
    np.save(out / "disparity.npy", result.disparity)
    paths["disparity"] = str(p_disp)

    p_depth = out / "depth.png"
    save_depth_image(result.depth, p_depth, max_depth_m=max_depth_m)
    np.save(out / "depth.npy", result.depth)
    paths["depth"] = str(p_depth)

    color_src = left_bgr_for_color if left_bgr_for_color is not None else result.rect_left
    pts, cols = extract_point_cloud(
        result.points_xyz,
        result.valid_mask,
        color_src,
        max_points=max_points,
        stride=stride,
    )
    p_ply = out / "pointcloud.ply"
    save_ply(pts, p_ply, cols)
    paths["pointcloud"] = str(p_ply)

    np.savez(
        out / "stereo_rectify.npz",
        Q=result.Q,
        P1=result.P1,
        P2=result.P2,
        R1=result.R1,
        R2=result.R2,
        roi_left=np.array(result.roi_left),
        roi_right=np.array(result.roi_right),
    )
    paths["rectify_npz"] = str(out / "stereo_rectify.npz")
    if frame_meta is not None:
        meta_path = out / "coordinate_frame.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(frame_meta, f, ensure_ascii=False, indent=2)
        paths["coordinate_frame"] = str(meta_path)
    return paths


def run_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    left_path = Path(cfg["left_image"])
    right_path = Path(cfg["right_image"])
    left_bgr = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right_bgr = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left_bgr is None:
        raise FileNotFoundError(f"无法读取左图: {left_path}")
    if right_bgr is None:
        raise FileNotFoundError(f"无法读取右图: {right_path}")

    left_cam, right_cam = resolve_stereo_cameras(cfg)
    ext = extrinsic_from_yaml(cfg)

    stereo_cfg = cfg.get("stereo", {})
    result = run_stereo_depth(
        left_bgr,
        right_bgr,
        left_cam,
        right_cam,
        ext,
        alpha=float(stereo_cfg.get("alpha", 0.0)),
        algorithm=str(stereo_cfg.get("algorithm", "sgbm")),
        num_disparities=int(stereo_cfg.get("num_disparities", 128)),
        block_size=int(stereo_cfg.get("block_size", 5)),
        min_disparity=int(stereo_cfg.get("min_disparity", 0)),
        baseline_m=cfg.get("baseline_m"),
    )

    output_dir = cfg.get("output_dir", "stereo_output")
    baseline_m = float(cfg.get("baseline_m", abs(float(ext.T[0, 0])) or np.linalg.norm(ext.T)))
    frame_meta = coordinate_frame_metadata(result, baseline_m=baseline_m, extrinsic_T=ext.T)
    out_paths = save_stereo_outputs(
        result,
        output_dir,
        max_points=int(cfg.get("max_points", 500_000)),
        stride=int(cfg.get("point_stride", 2)),
        max_depth_m=float(cfg.get("max_depth_m", 10.0)),
        frame_meta=frame_meta,
    )
    return {
        "success": True,
        "output_dir": str(Path(output_dir).resolve()),
        "files": out_paths,
        "baseline_m": baseline_m,
        "coordinate_frame": frame_meta,
        "image_size": [int(left_bgr.shape[1]), int(left_bgr.shape[0])],
    }


def _backproject_uvz(
    u: float,
    v: float,
    z: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    return np.array(
        [(u - cx) * z / fx, (v - cy) * z / fy, z],
        dtype=np.float64,
    )


def _orthonormal_frame_from_normal(n: np.ndarray) -> np.ndarray:
    """由法向构造右手系旋转（列向量为局部 X/Y/Z 在相机系下的方向），Z=法向。"""
    z_axis = n / (np.linalg.norm(n) + 1e-15)
    x_ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_ref - float(np.dot(x_ref, z_axis)) * z_axis
    if float(np.linalg.norm(x_axis)) < 1e-8:
        x_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = x_ref - float(np.dot(x_ref, z_axis)) * z_axis
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-15)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-15)
    R = np.column_stack([x_axis, y_axis, z_axis])
    if float(np.linalg.det(R)) < 0:
        R[:, 1] *= -1.0
    return R


def estimate_plane_pose_rectified(
    depth: np.ndarray,
    u: int,
    v: int,
    p1: np.ndarray,
    *,
    patch_half_m: float = 0.015,
    min_points: int = 24,
) -> Dict[str, Any]:
    """
    在校正左目系用中心点周围约 patch_half_m*2 方矩形邻域点拟合法线与姿态。

    默认 patch_half_m=0.015 → 约 3cm×3cm。
    先得到朝向相机的表面外法向，再取 **局部 Z = 其反向**（与相机光轴 +Z 同向、指向场景/表面内侧），
    以便 TCP 姿态 Z 与相机观察方向一致。
    """
    fx = float(p1[0, 0])
    fy = float(p1[1, 1])
    cx = float(p1[0, 2])
    cy = float(p1[1, 2])
    h, w = int(depth.shape[0]), int(depth.shape[1])
    z0 = float(depth[v, u])
    if not np.isfinite(z0) or z0 <= 0.0:
        return {"success": False, "error": "中心点无有效深度"}

    p0 = _backproject_uvz(u, v, z0, fx, fy, cx, cy)
    half_u = int(max(1, min(120, np.ceil(patch_half_m * fx / z0))))
    half_v = int(max(1, min(120, np.ceil(patch_half_m * fy / z0))))

    pts: List[np.ndarray] = []
    u0 = max(0, u - half_u)
    u1 = min(w - 1, u + half_u)
    v0 = max(0, v - half_v)
    v1 = min(h - 1, v + half_v)
    for vv in range(v0, v1 + 1):
        for uu in range(u0, u1 + 1):
            zz = float(depth[vv, uu])
            if not np.isfinite(zz) or zz <= 0.0:
                continue
            p = _backproject_uvz(uu, vv, zz, fx, fy, cx, cy)
            if abs(float(p[0] - p0[0])) <= patch_half_m and abs(float(p[1] - p0[1])) <= patch_half_m:
                pts.append(p)

    if len(pts) < min_points:
        return {
            "success": False,
            "error": f"邻域有效点不足（{len(pts)}<{min_points}），无法估计法向",
            "plane_point_count": len(pts),
            "patch_half_m": patch_half_m,
        }

    cloud = np.stack(pts, axis=0)
    center = cloud.mean(axis=0)
    _, singular, vh = np.linalg.svd(cloud - center, full_matrices=False)
    n_out = vh[-1].astype(np.float64)
    # 表面外法向：朝向相机光心（与中心点反向）
    if float(np.dot(n_out, center)) > 0.0:
        n_out = -n_out
    n_out = n_out / (np.linalg.norm(n_out) + 1e-15)
    # 姿态 Z：与相机 +Z 同向（指向场景），即外法向取反
    n_z = -n_out
    R = _orthonormal_frame_from_normal(n_z)
    residuals = cloud @ n_out - float(np.dot(center, n_out))
    rmse = float(np.sqrt(np.mean(residuals * residuals)))

    return {
        "success": True,
        "plane_point_count": int(cloud.shape[0]),
        "plane_rmse_m": rmse,
        "patch_half_m": patch_half_m,
        "pixel_half_u": half_u,
        "pixel_half_v": half_v,
        "normal_outward_rect": [float(n_out[0]), float(n_out[1]), float(n_out[2])],
        "normal_rect": [float(n_z[0]), float(n_z[1]), float(n_z[2])],
        "R_rect_row_major": R.tolist(),
        "centroid_rect": [float(center[0]), float(center[1]), float(center[2])],
        "singular_values": [float(s) for s in singular],
    }


def pixel_to_camera_xyz(
    output_dir: Union[str, Path],
    u: float,
    v: float,
    *,
    patch_half_m: float = 0.015,
) -> Dict[str, Any]:
    """
    校正左图像素 (u,v) → 校正左目 3D，再经 R1^T 得到原始左目 OpenCV 系（手眼 p_cam）。

    同时在约 3cm×3cm（默认）邻域拟合法线，构造局部姿态（Z 与相机 +Z 同向）。
    """
    out = Path(output_dir)
    depth_path = out / "depth.npy"
    npz_path = out / "stereo_rectify.npz"
    if not depth_path.is_file():
        return {"success": False, "error": f"缺少 depth.npy: {depth_path}"}
    if not npz_path.is_file():
        return {"success": False, "error": f"缺少 stereo_rectify.npz: {npz_path}"}

    depth = np.load(depth_path)
    npz = np.load(npz_path)
    if "P1" not in npz:
        return {"success": False, "error": "stereo_rectify.npz 缺少 P1"}
    p1 = npz["P1"]
    h, w = int(depth.shape[0]), int(depth.shape[1])
    ui = int(round(float(u)))
    vi = int(round(float(v)))
    if ui < 0 or vi < 0 or ui >= w or vi >= h:
        return {
            "success": False,
            "error": f"像素越界 ({ui},{vi})，图像尺寸 {w}x{h}",
            "u": ui,
            "v": vi,
            "image_size": [w, h],
        }

    z = float(depth[vi, ui])
    if not np.isfinite(z) or z <= 0.0:
        return {
            "success": False,
            "error": "该点无有效深度（视差无效或被遮挡）",
            "u": ui,
            "v": vi,
            "image_size": [w, h],
        }

    fx = float(p1[0, 0])
    fy = float(p1[1, 1])
    cx = float(p1[0, 2])
    cy = float(p1[1, 2])
    if abs(fx) < 1e-9 or abs(fy) < 1e-9:
        return {"success": False, "error": f"P1 焦距异常 fx={fx} fy={fy}"}

    p_rect = _backproject_uvz(ui, vi, z, fx, fy, cx, cy)

    r1_assumed = False
    if "R1" in npz:
        r1 = np.asarray(npz["R1"], dtype=np.float64).reshape(3, 3)
    else:
        r1 = np.eye(3, dtype=np.float64)
        r1_assumed = True
    p_cam = r1.T @ p_rect

    plane = estimate_plane_pose_rectified(
        depth, ui, vi, p1, patch_half_m=float(patch_half_m)
    )
    result: Dict[str, Any] = {
        "success": True,
        "u": ui,
        "v": vi,
        "x": float(p_rect[0]),
        "y": float(p_rect[1]),
        "z": float(p_rect[2]),
        "frame": "opencv_rectified_left",
        "x_cam": float(p_cam[0]),
        "y_cam": float(p_cam[1]),
        "z_cam": float(p_cam[2]),
        "frame_cam": "opencv_left_camera",
        "r1_assumed_identity": r1_assumed,
        "units": "m",
        "image_size": [w, h],
        "to_three_left_camera": [float(p_rect[0]), float(-p_rect[1]), float(-p_rect[2])],
    }

    if not plane.get("success"):
        result["plane_ok"] = False
        result["plane_error"] = plane.get("error", "法向估计失败")
        result["plane_point_count"] = plane.get("plane_point_count", 0)
        return result

    R_rect = np.asarray(plane["R_rect_row_major"], dtype=np.float64).reshape(3, 3)
    R_cam = r1.T @ R_rect
    n_rect = np.asarray(plane["normal_rect"], dtype=np.float64)
    n_cam = r1.T @ n_rect
    rx_r, ry_r, rz_r = _rpy_deg_zyx_from_R(R_rect)
    rx_c, ry_c, rz_c = _rpy_deg_zyx_from_R(R_cam)

    result.update(
        {
            "plane_ok": True,
            "plane_point_count": plane["plane_point_count"],
            "plane_rmse_m": plane["plane_rmse_m"],
            "patch_half_m": plane["patch_half_m"],
            "pixel_half_u": plane["pixel_half_u"],
            "pixel_half_v": plane["pixel_half_v"],
            "normal_rect": [float(n_rect[0]), float(n_rect[1]), float(n_rect[2])],
            "normal_cam": [float(n_cam[0]), float(n_cam[1]), float(n_cam[2])],
            "normal_outward_rect": plane.get("normal_outward_rect"),
            "R_rect_row_major": R_rect.tolist(),
            "R_cam_row_major": R_cam.tolist(),
            "rpy_rect_deg": [rx_r, ry_r, rz_r],
            "rpy_cam_deg": [rx_c, ry_c, rz_c],
        }
    )
    return result


def _load_T_matrix_yaml(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not doc:
        raise ValueError(f"YAML 为空: {path}")
    rows = doc.get("T_matrix_row_major")
    if rows is None:
        rows = doc.get("T")
    if rows is None:
        raise ValueError(f"{path} 缺少 T_matrix_row_major")
    return np.asarray(rows, dtype=np.float64).reshape(4, 4)


def _rpy_deg_zyx_from_R(R: np.ndarray) -> Tuple[float, float, float]:
    """R = Rz·Ry·Rx，返回 (rx, ry, rz) 度。"""
    r20 = float(R[2, 0])
    pitch = float(np.arcsin(np.clip(-r20, -1.0, 1.0)))
    if abs(float(np.cos(pitch))) > 1e-8:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    return float(np.rad2deg(roll)), float(np.rad2deg(pitch)), float(np.rad2deg(yaw))


def apply_scheme_a_hand_eye(
    pixel_result: Dict[str, Any],
    *,
    hand_eye_yaml: Union[str, Path],
    tcp_pose_yaml: Union[str, Path],
) -> Dict[str, Any]:
    """
    方案 A：T_cam（含位置与可选平面姿态）→ T_tcp / T_base。

    有平面姿态时：T_base = T_base_tcp @ T_cam2tcp @ T_cam_point（姿态来自法向）；
    否则仅变换位置，姿态保持拍照时 TCP。
    """
    out = dict(pixel_result)
    if not pixel_result.get("success"):
        return out

    he_path = Path(hand_eye_yaml)
    tcp_path = Path(tcp_pose_yaml)
    if not he_path.is_file():
        out["success"] = False
        out["error"] = f"缺少手眼结果: {he_path}"
        return out
    if not tcp_path.is_file():
        out["success"] = False
        out["error"] = f"缺少拍照时 TCP 位姿: {tcp_path}（请重新双目拍照）"
        return out

    with he_path.open("r", encoding="utf-8") as f:
        he_doc = yaml.safe_load(f) or {}
    setup = str(he_doc.get("setup", "eye_in_hand")).lower()
    if setup in ("eye_to_hand", "eye2hand", "fixed_camera"):
        out["success"] = False
        out["error"] = "当前定位运动仅支持 eye_in_hand（眼在手上）"
        return out

    T_cam2tcp = _load_T_matrix_yaml(he_path)
    T_base_tcp = _load_T_matrix_yaml(tcp_path)

    T_cam = np.eye(4, dtype=np.float64)
    T_cam[0, 3] = float(pixel_result["x_cam"])
    T_cam[1, 3] = float(pixel_result["y_cam"])
    T_cam[2, 3] = float(pixel_result["z_cam"])
    pose_from_plane = bool(pixel_result.get("plane_ok") and pixel_result.get("R_cam_row_major"))

    if pose_from_plane:
        T_cam[:3, :3] = np.asarray(pixel_result["R_cam_row_major"], dtype=np.float64).reshape(3, 3)
        T_tcp = T_cam2tcp @ T_cam
        T_base = T_base_tcp @ T_tcp
        rx, ry, rz = _rpy_deg_zyx_from_R(T_base[:3, :3])
        semantics = "p/R: T_base=T_base_tcp@T_cam2tcp@T_cam；Z∥相机+Z（指向场景）"
    else:
        p_cam_h = np.array(
            [
                float(pixel_result["x_cam"]),
                float(pixel_result["y_cam"]),
                float(pixel_result["z_cam"]),
                1.0,
            ],
            dtype=np.float64,
        )
        p_tcp_h = T_cam2tcp @ p_cam_h
        p_base_h = T_base_tcp @ p_tcp_h
        T_tcp = np.eye(4, dtype=np.float64)
        T_tcp[:3, 3] = p_tcp_h[:3]
        T_base = np.eye(4, dtype=np.float64)
        T_base[:3, :3] = T_base_tcp[:3, :3]
        T_base[:3, 3] = p_base_h[:3]
        rx, ry, rz = _rpy_deg_zyx_from_R(T_base_tcp[:3, :3])
        semantics = "仅位置手眼；姿态=拍照时 TCP（法向估计失败）"

    out.update(
        {
            "x_tcp": float(T_tcp[0, 3]),
            "y_tcp": float(T_tcp[1, 3]),
            "z_tcp": float(T_tcp[2, 3]),
            "x_base": float(T_base[0, 3]),
            "y_base": float(T_base[1, 3]),
            "z_base": float(T_base[2, 3]),
            "tcp_target": {
                "x": float(T_base[0, 3]),
                "y": float(T_base[1, 3]),
                "z": float(T_base[2, 3]),
                "rxDeg": rx,
                "ryDeg": ry,
                "rzDeg": rz,
            },
            "T_base_row_major": T_base.tolist(),
            "pose_from_plane": pose_from_plane,
            "hand_eye_path": str(he_path.resolve()),
            "tcp_pose_path": str(tcp_path.resolve()),
            "scheme": "A",
            "semantics": semantics,
        }
    )
    return out


def pixel_to_base_via_hand_eye(
    output_dir: Union[str, Path],
    u: float,
    v: float,
    *,
    hand_eye_yaml: Union[str, Path],
    tcp_pose_yaml: Union[str, Path],
) -> Dict[str, Any]:
    """像素 → 校正/原始相机点 → 手眼方案 A → 基座目标。"""
    pix = pixel_to_camera_xyz(output_dir, u, v)
    return apply_scheme_a_hand_eye(
        pix,
        hand_eye_yaml=hand_eye_yaml,
        tcp_pose_yaml=tcp_pose_yaml,
    )
