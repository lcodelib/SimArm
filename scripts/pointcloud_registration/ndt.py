#!/usr/bin/env python3
"""
三维 NDT（Normal Distributions Transform）配准核心。

参考 Magnusson 2009 / PCL NormalDistributionsTransform：
  1. 将目标点云按体素划分，每格拟合高斯 (均值 μ, 协方差 Σ)
  2. 用源点云在当前刚体变换下的似然和作为目标函数
  3. 牛顿法迭代优化 6 自由度位姿，使源点云对齐到目标

坐标系约定：输出 4×4 齐次矩阵 T，满足 p_b ≈ T @ p_a（将 a 变换到 b）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# 体素内最少点数，少于此则不建高斯单元
_MIN_POINTS_PER_CELL = 5
# 协方差正则项，避免奇异
_COV_EPSILON = 1e-3


@dataclass
class NdtParams:
    """NDT 超参数。"""

    # 体素边长（与点云单位一致，立体深度 PLY 一般为米）
    resolution: float = 0.05
    # 牛顿/线搜索初始步长上界（当前优化器为 L-BFGS，保留兼容）
    step_size: float = 1.0
    # 异常点比例（保留字段，与 PCL 接口对齐；当前似然为简化高斯）
    outlier_ratio: float = 0.55
    # 最大迭代次数
    max_iterations: int = 50
    # 变换增量范数小于该阈值则判定收敛
    epsilon: float = 1e-6
    # 多尺度：由粗到细的 resolution 列表；空则仅用 resolution
    # 例：[0.2, 0.1, 0.05]
    resolutions: Optional[Tuple[float, ...]] = None


@dataclass
class NdtResult:
    """NDT 优化结果。"""

    transformation: np.ndarray  # 4x4，a → b
    iterations: int
    converged: bool
    final_score: float


def _voxel_key(points: np.ndarray, resolution: float) -> np.ndarray:
    """将点坐标映射为整数体素索引，shape (N, 3)。"""
    return np.floor(points / resolution).astype(np.int64)


def build_ndt_map(
    target: np.ndarray,
    resolution: float,
) -> Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]]:
    """
    由目标点云构建 NDT 体素高斯图。

    Args:
        target: (N, 3) 目标点
        resolution: 体素边长

    Returns:
        dict: key=(ix,iy,iz) → (mean(3,), inv_cov(3,3))
    """
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(f"target 须为 (N,3)，实际 {target.shape}")
    if resolution <= 0:
        raise ValueError("resolution 必须 > 0")

    keys = _voxel_key(target, resolution)
    # 用结构化视图做分组，避免 Python 层逐点 append 过慢
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_sorted = keys[order]
    pts_sorted = target[order]

    ndt_map: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}
    n = len(pts_sorted)
    i = 0
    while i < n:
        j = i + 1
        while j < n and np.array_equal(keys_sorted[j], keys_sorted[i]):
            j += 1
        cell_pts = pts_sorted[i:j]
        if cell_pts.shape[0] >= _MIN_POINTS_PER_CELL:
            mean = cell_pts.mean(axis=0)
            centered = cell_pts - mean
            cov = (centered.T @ centered) / float(cell_pts.shape[0] - 1)
            cov = cov + np.eye(3) * _COV_EPSILON
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov)
            key = (int(keys_sorted[i, 0]), int(keys_sorted[i, 1]), int(keys_sorted[i, 2]))
            ndt_map[key] = (mean.astype(np.float64), inv_cov.astype(np.float64))
        i = j

    logger.info("NDT 地图：体素数=%d，目标点数=%d，resolution=%.4f", len(ndt_map), n, resolution)
    if not ndt_map:
        raise RuntimeError(
            "NDT 地图为空：请增大点云密度或减小 resolution，"
            f"当前 resolution={resolution}，每格至少 {_MIN_POINTS_PER_CELL} 点"
        )
    return ndt_map


def _params_to_matrix(x: np.ndarray) -> np.ndarray:
    """6 维位姿 [tx,ty,tz,rx,ry,rz]（旋转向量 rad）→ 4×4。"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_rotvec(x[3:6]).as_matrix()
    T[:3, 3] = x[:3]
    return T


def _matrix_to_params(T: np.ndarray) -> np.ndarray:
    """4×4 → 6 维位姿向量。"""
    R = T[:3, :3]
    t = T[:3, 3]
    rvec = Rotation.from_matrix(R).as_rotvec()
    return np.concatenate([t, rvec]).astype(np.float64)


def _skew(v: np.ndarray) -> np.ndarray:
    """三维向量反对称矩阵 [v]_×。"""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """对 (N,3) 点施加刚体变换。"""
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def _compute_score(
    source: np.ndarray,
    T: np.ndarray,
    ndt_map: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]],
    resolution: float,
) -> float:
    """
    计算 NDT 得分（无梯度）。

    按体素单元聚合源点，用 einsum 批量算马氏距离，避免逐点 Python 循环。
    """
    transformed = _transform_points(T, source)
    keys = _voxel_key(transformed, resolution)
    score = 0.0
    for key, (mean, inv_cov) in ndt_map.items():
        mask = (
            (keys[:, 0] == key[0])
            & (keys[:, 1] == key[1])
            & (keys[:, 2] == key[2])
        )
        if not np.any(mask):
            continue
        pts = transformed[mask]
        err = pts - mean
        # mahalanobis = e^T Σ^{-1} e
        mahal = np.einsum("ni,ij,nj->n", err, inv_cov, err)
        score += float(np.exp(-0.5 * mahal).sum())
    return float(score)


def _compute_score_derivatives(
    source: np.ndarray,
    T: np.ndarray,
    ndt_map: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]],
    resolution: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    计算当前变换下的 NDT 得分、梯度与近似海森。

    得分 S = Σ exp(-0.5 * mahalanobis)，越大越好。
    雅可比按左乘增量：J = [I | -[x]_×]。牛顿方程：hess @ delta = -grad。
    """
    transformed = _transform_points(T, source)
    keys = _voxel_key(transformed, resolution)

    score = 0.0
    grad = np.zeros(6, dtype=np.float64)
    hess = np.zeros((6, 6), dtype=np.float64)
    for i in range(transformed.shape[0]):
        key = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
        cell = ndt_map.get(key)
        if cell is None:
            continue
        mean, inv_cov = cell
        x = transformed[i]
        e = x - mean
        mahal = float(e @ inv_cov @ e)
        w = np.exp(-0.5 * mahal)
        if w < 1e-12:
            continue
        score += w

        # J = [I | -[x]_×]，J^T We = [We; x×We]
        We = inv_cov @ e
        jtwe = np.empty(6, dtype=np.float64)
        jtwe[:3] = We
        jtwe[3:] = np.cross(x, We)
        grad -= w * jtwe

        # J = [I, -[x]_×] → J^T W J 分块：
        # [ W , -W[x]_× ; [x]_× W , [x]_× W [x]_× ]
        W = inv_cov
        sx = _skew(x)
        JT_W_J = np.zeros((6, 6), dtype=np.float64)
        JT_W_J[:3, :3] = W
        JT_W_J[:3, 3:] = -W @ sx
        JT_W_J[3:, :3] = sx @ W
        JT_W_J[3:, 3:] = sx @ W @ sx
        hess += w * np.outer(jtwe, jtwe) - w * JT_W_J

    return float(score), grad, hess


def _optimize_at_resolution(
    source: np.ndarray,
    target: np.ndarray,
    resolution: float,
    T: np.ndarray,
    params: NdtParams,
) -> Tuple[np.ndarray, float, int, bool]:
    """
    在单一 resolution 下用 L-BFGS 最大化 NDT 得分。

    优化变量为绝对位姿 6 维 [tx,ty,tz,rx,ry,rz]（旋转向量），
    避免增量参数中平移/旋转量纲混杂导致线搜索失败。
    """
    from scipy.optimize import minimize

    ndt_map = build_ndt_map(target, resolution)
    x0 = _matrix_to_params(T)

    eval_count = {"n": 0}
    last = {"score": 0.0}

    def objective(x: np.ndarray) -> float:
        eval_count["n"] += 1
        Tc = _params_to_matrix(x)
        score = _compute_score(source, Tc, ndt_map, resolution)
        last["score"] = score
        return -score  # 最小化负得分 = 最大化得分

    # 平移边界放宽；旋转向量约 ±π
    bounds = [
        (None, None),
        (None, None),
        (None, None),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
    ]

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": int(params.max_iterations),
            "ftol": 1e-10,
            "gtol": float(params.epsilon),
            "maxls": 20,
        },
    )

    T_out = _params_to_matrix(result.x)
    score, grad, _ = _compute_score_derivatives(source, T_out, ndt_map, resolution)
    converged = bool(result.success) or float(np.linalg.norm(grad)) < 1e-2
    it_done = int(result.nit) if result.nit is not None else eval_count["n"]

    logger.info(
        "NDT resolution=%.4f score=%.6f iters=%d success=%s evals=%d",
        resolution,
        score,
        it_done,
        result.success,
        eval_count["n"],
    )
    if not result.success:
        logger.debug("L-BFGS 消息: %s", result.message)

    return T_out, float(score), it_done, converged


def register_ndt(
    source: np.ndarray,
    target: np.ndarray,
    params: Optional[NdtParams] = None,
    init_transform: Optional[np.ndarray] = None,
) -> NdtResult:
    """
    用 NDT 将 source 对齐到 target。

    Args:
        source: (Ns, 3) 源点云 a
        target: (Nt, 3) 目标点云 b
        params: NDT 参数
        init_transform: 可选初值 4×4；默认单位阵

    Returns:
        NdtResult
    """
    params = params or NdtParams()
    if source.size == 0 or target.size == 0:
        raise ValueError("源/目标点云不能为空")

    T = (
        np.eye(4, dtype=np.float64)
        if init_transform is None
        else np.asarray(init_transform, dtype=np.float64).reshape(4, 4).copy()
    )

    if params.resolutions:
        resolutions = tuple(float(r) for r in params.resolutions)
    else:
        # 默认粗→细两级，提升收敛盆地
        r = float(params.resolution)
        resolutions = (r * 2.0, r) if r > 0 else (0.05,)

    logger.info(
        "开始 NDT：source=%d, target=%d, resolutions=%s, max_iter/级=%d",
        source.shape[0],
        target.shape[0],
        resolutions,
        params.max_iterations,
    )

    total_iters = 0
    score = 0.0
    converged = False
    for res in resolutions:
        T, score, it_done, converged = _optimize_at_resolution(
            source, target, res, T, params
        )
        total_iters += it_done

    if not converged:
        logger.warning("NDT 未完全收敛：iters=%d, score=%.6f", total_iters, score)

    return NdtResult(
        transformation=T,
        iterations=total_iters,
        converged=converged,
        final_score=score,
    )
