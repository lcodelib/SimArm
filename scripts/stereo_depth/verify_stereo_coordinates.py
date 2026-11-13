#!/usr/bin/env python3
"""检查 stereo_output 深度与点云坐标系是否与 OpenCV 约定一致。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import stereo_depth as sd


def verify(output_dir: Path) -> dict:
    out = {}
    depth_path = output_dir / "depth.npy"
    disp_path = output_dir / "disparity.npy"
    npz_path = output_dir / "stereo_rectify.npz"
    ply_path = output_dir / "pointcloud.ply"

    if not all(p.is_file() for p in (depth_path, disp_path, npz_path)):
        raise FileNotFoundError(f"缺少 depth/disparity/npz，请先运行 stereo_depth_cli.py -o {output_dir}")

    depth = np.load(depth_path)
    disp = np.load(disp_path)
    npz = np.load(npz_path)
    Q = npz["Q"]

    mask = (depth > 0) & (disp > 0)
    if not np.any(mask):
        out["ok"] = False
        out["error"] = "无有效深度像素"
        return out

    ys, xs = np.where(mask)
    i = len(xs) // 2
    u, v = int(xs[i]), int(ys[i])
    d_val = float(disp[v, u])
    z_depth = float(depth[v, u])
    p_q = sd.reproject_disparity_to_points(np.array([[d_val]], np.float32), Q)[0, 0]
    z_q = float(p_q[2])

    out["sample_pixel"] = [u, v]
    out["disparity_px"] = d_val
    out["depth_npy_m"] = z_depth
    out["reproject_z_m"] = z_q
    out["depth_matches_reproject"] = abs(z_depth - z_q) < 0.002

    # PLY 抽样
    if ply_path.is_file():
        zs = []
        with open(ply_path, encoding="utf-8") as f:
            header = True
            for line in f:
                if header:
                    if line.strip() == "end_header":
                        header = False
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    zs.append(float(parts[2]))
                if len(zs) >= 500:
                    break
        if zs:
            med_z = float(np.median(zs))
            out["ply_median_z_m"] = med_z
            out["ply_z_positive"] = med_z > 0
            out["ply_matches_depth"] = abs(med_z - z_depth) < 0.05

    out["z_forward_positive"] = z_q > 0 and z_depth > 0
    out["ok"] = bool(
        out.get("z_forward_positive")
        and out.get("depth_matches_reproject", True)
        and out.get("ply_matches_depth", True)
    )
    meta_path = output_dir / "coordinate_frame.json"
    if meta_path.is_file():
        out["frame_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="验证立体输出坐标系")
    ap.add_argument("-o", "--output", default="E:/SimArm/stereo_output", help="输出目录")
    args = ap.parse_args()
    try:
        report = verify(Path(args.output))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report.get("ok"):
            print("坐标检查未通过，请确认 extrinsic.T 为 [-baseline,0,0] 并重新生成。", file=sys.stderr)
            return 1
        print("坐标检查通过：Z 为光轴前方正深度，depth.npy 与点云一致。")
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
