#!/usr/bin/env python3
"""
手眼标定 CLI：config 与 hand_eye_solve.py 相同；stdout 仅一行紧凑 JSON，无其它日志。

成功：{"success":true,"T_matrix_row_major":[...],"visualization":{...},...}
失败：{"success":false,"error":"..."}

用法：python hand_eye_solve_cli.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# 与 hand_eye_solve.py 同目录，复用标定核心
import hand_eye_solve as he


def _configure_utf8_stdio() -> None:
    """管道输出统一 UTF-8，避免 Windows 下中文错误信息乱码。"""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


_configure_utf8_stdio()


@contextmanager
def _quiet_io() -> Iterator[None]:
    """屏蔽 hand_eye_solve 内部 print，保证 stdout 仅一行 JSON。"""
    devnull = open(os.devnull, "w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        devnull.close()


def _T_to_list(T: np.ndarray) -> List[List[float]]:
    return np.asarray(T, dtype=np.float64).reshape(4, 4).tolist()


def _subsume_indices(n: int, max_poses: int) -> List[int]:
    if n <= 0:
        return []
    max_poses = max(1, max_poses)
    if n <= max_poses:
        return list(range(n))
    return [int(round(i * (n - 1) / (max_poses - 1))) for i in range(max_poses)]


def _build_visualization_poses(
    *,
    eye_to_hand: bool,
    T_hand_eye: np.ndarray,
    T_base_tcp_list: List[np.ndarray],
    used_files: List[str],
    R_target2cam_list: List[np.ndarray],
    t_target2cam_list: List[np.ndarray],
    max_poses_draw: int,
) -> Dict[str, Any]:
    """提取与 _visualize_hand_eye_frames 中绘制的位姿一致的数据（不弹窗）。"""
    n = len(T_base_tcp_list)
    indices = _subsume_indices(n, max_poses_draw)
    frames: List[Dict[str, Any]] = []

    for idx in indices:
        name = used_files[idx]
        T_bt = T_base_tcp_list[idx]
        item: Dict[str, Any] = {
            "image": name,
            "T_base_tcp_row_major": _T_to_list(T_bt),
        }
        if eye_to_hand:
            # 眼在手外：固定相机 T_cam2base 在 hand_eye 顶层，此处仅 TCP
            pass
        else:
            T_bc = T_bt @ T_hand_eye
            item["T_base_camera_row_major"] = _T_to_list(T_bc)
            R_tc = R_target2cam_list[idx]
            t_tc = t_target2cam_list[idx].reshape(3)
            T_ct = he._T_from_Rt(R_tc, t_tc)
            T_board = T_bt @ T_hand_eye @ T_ct
            origin = T_board[:3, 3]
            item["T_base_board_row_major"] = _T_to_list(T_board)
            item["board_origin_position_base_m"] = [float(origin[0]), float(origin[1]), float(origin[2])]
        frames.append(item)

    out: Dict[str, Any] = {
        "T_base_row_major": _T_to_list(np.eye(4)),
        "frames": frames,
    }
    if eye_to_hand:
        out["T_base_camera_row_major"] = _T_to_list(T_hand_eye)
        out["hand_eye_label"] = "T_cam2base"
        out["hand_eye_semantics"] = "p_base = T @ p_cam"
    else:
        out["T_cam2tcp_row_major"] = _T_to_list(T_hand_eye)
        out["hand_eye_label"] = "T_cam2tcp"
        out["hand_eye_semantics"] = "p_tcp = T @ p_cam"
    return out


def _attach_viz_scale(out: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    viz = cfg.get("visualize_3d") or {}
    out["frame_scale_m"] = float(viz.get("frame_scale_m", 0.04))


def _emit(obj: Dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    # 直接写 UTF-8 字节，不依赖控制台 code page
    sys.stdout.buffer.write(line.encode("utf-8"))
    sys.stdout.buffer.flush()


def run_cli(cfg_path: Path) -> int:
    try:
        cfg = he._load_yaml(cfg_path)
        data_dir = Path(cfg["data_dir"]).expanduser().resolve()
        if not data_dir.is_dir():
            _emit({"success": False, "error": f"data_dir 不存在: {data_dir}"})
            return 1

        poses_rel = Path(cfg["poses_file"])
        poses_path = poses_rel if poses_rel.is_absolute() else (data_dir / poses_rel)
        if not poses_path.is_file():
            _emit({"success": False, "error": f"位姿文件不存在: {poses_path}"})
            return 1

        K, dist = he._camera_matrix(cfg)
        board = cfg["board"]
        inner_w, inner_h = int(board["inner_corners"][0]), int(board["inner_corners"][1])
        square_m = float(board["square_size_m"])
        pattern_size = (inner_w, inner_h)
        obj_points = he._chessboard_object_points(inner_w, inner_h, square_m)

        pose_fmt = cfg.get("pose_format", {})
        ptype = str(pose_fmt.get("type", "csv")).lower()
        if ptype == "csv":
            pose_map = he._load_poses_csv(data_dir, poses_path, pose_fmt)
        elif ptype == "json":
            pose_map = he._load_poses_json(data_dir, poses_path)
        else:
            _emit({"success": False, "error": f"不支持的 pose_format.type: {ptype}"})
            return 1

        glob_pat = cfg.get("images_glob", "*.png")
        images = he._gather_image_paths(data_dir, glob_pat)
        if len(images) < 3:
            _emit(
                {
                    "success": False,
                    "error": f"匹配图像少于 3 张（当前 {len(images)} 张）",
                }
            )
            return 1

        setup = str(cfg.get("setup", "eye_in_hand")).lower()
        eye_to_hand = setup in ("eye_to_hand", "eye2hand", "fixed_camera")

        R_target2cam: List[np.ndarray] = []
        t_target2cam: List[np.ndarray] = []
        T_base_tcp_used: List[np.ndarray] = []
        used_files: List[str] = []
        skipped: List[str] = []

        subpix = int(cfg.get("corner_subpix_win", 11))

        for img_path in images:
            name = img_path.name
            T_base_tcp = pose_map.get(name)
            if T_base_tcp is None:
                skipped.append(f"{name} (无位姿)")
                continue
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                skipped.append(f"{name} (无法读取图像)")
                continue

            ok_pose, R_tc, t_tc, _vis, detail_cn, _rms = he._detect_board_pose_with_vis(
                bgr, K, dist, pattern_size, obj_points, subpix
            )
            if not ok_pose:
                skipped.append(f"{name} ({detail_cn})")
                continue

            R_target2cam.append(R_tc)
            t_target2cam.append(t_tc.reshape(3, 1))
            T_base_tcp_used.append(T_base_tcp.astype(np.float64))
            used_files.append(name)

        n = len(used_files)
        if n < 3:
            _emit({"success": False, "error": "成功配对（角点+位姿）少于 3 组"})
            return 1

        raw_method = str(cfg.get("hand_eye_method", "AUTO"))
        pref = raw_method.upper()
        preferred_label = None if pref == "AUTO" else pref
        try_inv_cfg = cfg.get("hand_eye_try_pose_inverse", True)
        try_pose_inverse = try_inv_cfg is True or str(try_inv_cfg).lower() in ("1", "true", "yes")

        he._method_from_name(raw_method)

        with _quiet_io():
            T_result, method_used, _pose_tip, pose_var_used, sel_score = he._solve_hand_eye_autopick(
                eye_fixed_cam=eye_to_hand,
                T_base_tcp_used=T_base_tcp_used,
                R_target2cam=R_target2cam,
                t_target2cam=t_target2cam,
                preferred_label=preferred_label,
                try_pose_inverse=try_pose_inverse,
            )

        if he._is_near_identity_xyzrot(T_result[:3, :3], T_result[:3, 3], ang_tol_deg=0.02, t_tol_m=1e-5):
            _emit({"success": False, "error": "手眼解近似单位阵，标定退化"})
            return 1

        viz = cfg.get("visualize_3d") or {}
        max_draw = int(viz.get("max_poses_draw", 28))

        vis_poses = _build_visualization_poses(
            eye_to_hand=eye_to_hand,
            T_hand_eye=T_result,
            T_base_tcp_list=T_base_tcp_used,
            used_files=used_files,
            R_target2cam_list=R_target2cam,
            t_target2cam_list=t_target2cam,
            max_poses_draw=max_draw,
        )
        _attach_viz_scale(vis_poses, cfg)

        payload: Dict[str, Any] = {
            "success": True,
            "T_matrix_row_major": _T_to_list(T_result),
            "visualization": vis_poses,
            "hand_eye_method_used": method_used,
            "pose_variant": pose_var_used,
            "num_poses": n,
            "setup": setup,
            "used_images": used_files,
            "board_origin_rmse_m": float(sel_score) if not eye_to_hand else None,
        }

        if he._parse_bool_config(cfg.get("save_hand_eye_result"), True):
            out_override = cfg.get("output_result_yaml")
            if out_override is not None and str(out_override).strip() != "":
                outp = Path(str(out_override))
                if not outp.is_absolute():
                    outp = data_dir / outp
            else:
                outp = data_dir / "hand_eye_result.yaml"
            import yaml
            from datetime import datetime, timezone

            c = cfg["camera"]
            yaml_payload = {
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "setup": setup,
                "hand_eye_method_used": method_used,
                "pose_variant": pose_var_used,
                "eye_in_hand_board_origin_rmse_m": None if eye_to_hand else float(sel_score),
                "num_poses": n,
                "used_images": used_files,
                "T_matrix_row_major": _T_to_list(T_result),
                "output_semantics": vis_poses.get("hand_eye_semantics"),
                "camera_intrinsics": {
                    "fx": float(c["fx"]),
                    "fy": float(c["fy"]),
                    "cx": float(c["cx"]),
                    "cy": float(c["cy"]),
                    "dist_coeffs": [float(x) for x in np.asarray(c.get("dist_coeffs", [])).flatten().tolist()],
                },
                "board": {
                    "inner_corners": list(board["inner_corners"]),
                    "square_size_m": float(board["square_size_m"]),
                },
            }
            outp.parent.mkdir(parents=True, exist_ok=True)
            with outp.open("w", encoding="utf-8") as f:
                yaml.safe_dump(yaml_payload, f, allow_unicode=True, default_flow_style=False)

        _emit(payload)
        return 0

    except Exception as e:
        _emit({"success": False, "error": str(e)})
        return 1


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    if not args.config.is_file():
        _emit({"success": False, "error": f"找不到配置文件: {args.config}"})
        sys.exit(1)
    sys.exit(run_cli(args.config))


if __name__ == "__main__":
    main()
