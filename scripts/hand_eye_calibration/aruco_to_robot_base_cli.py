#!/usr/bin/env python3
"""
ArUco → 机器人基座 CLI：参数与 aruco_to_robot_base.py 相同；stdout 仅一行紧凑 JSON。

成功：{"success":true,"setup":"...","markers":[...]}
失败：{"success":false,"error":"..."}

用法：
  python aruco_to_robot_base_cli.py --image cap.png --hand-eye hand_eye_result.yaml --marker-length 0.044
  python aruco_to_robot_base_cli.py ... --tcp-pose tcp_pose.yaml --dict DICT_4X4_50 --marker-id 5
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")

import aruco_to_robot_base as arb


def _configure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


_configure_utf8_stdio()


def _emit(obj: Dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    sys.stdout.buffer.write(line.encode("utf-8"))
    sys.stdout.buffer.flush()


def _T_to_list(T: np.ndarray) -> List[List[float]]:
    return np.asarray(T, dtype=np.float64).reshape(4, 4).tolist()


def run_cli(args: argparse.Namespace) -> int:
    try:
        img_path = Path(args.image).expanduser().resolve()
        he_path = Path(args.hand_eye).expanduser().resolve()
        if not img_path.is_file():
            _emit({"success": False, "error": f"找不到图像: {img_path}"})
            return 1
        if not he_path.is_file():
            _emit({"success": False, "error": f"找不到手眼文件: {he_path}"})
            return 1

        T_he, setup_kind, (K, dist) = arb._load_hand_eye(he_path)
        T_base_tcp: Optional[np.ndarray] = None
        if args.tcp_pose:
            T_base_tcp = arb._load_tcp_pose(Path(args.tcp_pose).expanduser().resolve())

        if setup_kind == "eye_in_hand" and T_base_tcp is None:
            _emit({"success": False, "error": "eye_in_hand 需提供 --tcp-pose（当前 T_base_tcp）"})
            return 1

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            _emit({"success": False, "error": f"无法读取图像: {img_path}"})
            return 1
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        corners, ids = arb._detect_aruco_markers(cv2, gray, args.dict)
        if ids is None or len(ids) == 0:
            _emit({"success": False, "error": "未检测到 ArUco 标码"})
            return 1

        marker_len = float(args.marker_length)
        want_id = args.marker_id
        detected_ids = [int(ids[i][0]) for i in range(len(ids))]

        markers_out: List[Dict[str, Any]] = []
        vis = bgr.copy() if args.viz_out else None
        panel_y = 12

        for i in range(len(ids)):
            mid = int(ids[i][0])
            if want_id is not None and mid != want_id:
                continue
            R_cm, t_cm = arb._estimate_marker_pose_cam(cv2, corners[i], marker_len, K, dist)
            T_cm = arb._T_from_rt(R_cm, t_cm)
            T_base_m = arb._T_marker_to_robot_base(T_cm, T_he, setup_kind, T_base_tcp)
            pose_extra = arb._pose_summary_from_T(T_base_m)
            markers_out.append(
                {
                    "marker_id": mid,
                    "T_cam_marker_row_major": _T_to_list(T_cm),
                    "T_base_marker_row_major": _T_to_list(T_base_m),
                    "position_base_m": pose_extra["position_base_m"],
                    "rpy_deg": pose_extra["rpy_deg"],
                }
            )
            if vis is not None:
                rvec, _ = cv2.Rodrigues(R_cm)
                tvec = t_cm.reshape(3, 1)
                arb._draw_axis(cv2, vis, K, dist, rvec, tvec, marker_len * 0.6)
                pt = tuple(np.asarray(corners[i][0, 0], dtype=np.int32))
                cv2.putText(vis, f"id={mid}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                lines = arb._format_T_base_marker_lines(mid, T_base_m)
                arb._annotate_pose_panel(cv2, vis, lines, (12, panel_y))
                panel_y += 22 * len(lines) + 14

        if not markers_out:
            _emit(
                {
                    "success": False,
                    "error": f"未找到 marker_id={want_id}",
                    "detected_ids": detected_ids,
                }
            )
            return 1

        if args.viz_out and vis is not None:
            outp = Path(args.viz_out)
            outp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(outp), vis)

        payload = {
            "success": True,
            "setup": setup_kind,
            "markers": markers_out,
        }
        if args.out_result:
            outj = Path(args.out_result)
            outj.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            outj.write_text(line + "\n", encoding="utf-8")

        _emit(payload)
        return 0

    except Exception as e:
        _emit({"success": False, "error": str(e)})
        return 1


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--hand-eye", type=Path, required=True)
    ap.add_argument("--marker-length", type=float, required=True)
    ap.add_argument("--dict", dest="dict", default="DICT_4X4_50")
    ap.add_argument("--marker-id", type=int, default=None)
    ap.add_argument("--tcp-pose", type=Path, default=None)
    ap.add_argument("--viz-out", type=Path, default=None)
    ap.add_argument("--out-result", type=Path, default=None)
    args = ap.parse_args()
    sys.exit(run_cli(args))


if __name__ == "__main__":
    main()
