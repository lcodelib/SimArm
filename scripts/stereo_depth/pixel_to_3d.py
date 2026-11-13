#!/usr/bin/env python3
"""
将立体输出目录中校正左图上的像素 (u,v) 转为 3D 点；可选方案 A 手眼到基座。

用法：
  python pixel_to_3d.py --output-dir .../out --u 640 --v 360 --json
  python pixel_to_3d.py --output-dir .../out --u 640 --v 360 \\
      --hand-eye .../hand_eye_result.yaml --tcp-pose .../tcp_pose.yaml --json
"""

from __future__ import annotations

import argparse
import json
import sys

import stereo_depth as sd


def _configure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


_configure_utf8_stdio()


def main() -> int:
    ap = argparse.ArgumentParser(description="立体深度：像素 → 相机系 / 手眼基座")
    ap.add_argument("--output-dir", required=True, help="stereo_depth 输出目录（含 depth.npy）")
    ap.add_argument("--u", type=float, required=True, help="图像列（像素）")
    ap.add_argument("--v", type=float, required=True, help="图像行（像素）")
    ap.add_argument("--hand-eye", default="", help="左目 hand_eye_result.yaml（可选，启用方案 A）")
    ap.add_argument("--tcp-pose", default="", help="拍照时 tcp_pose.yaml（与 --hand-eye 同用）")
    ap.add_argument("--json", action="store_true", help="仅向 stdout 打印一行 JSON")
    args = ap.parse_args()

    if bool(args.hand_eye) != bool(args.tcp_pose):
        print(
            json.dumps(
                {"success": False, "error": "--hand-eye 与 --tcp-pose 须同时提供"},
                ensure_ascii=False,
            )
        )
        return 1

    if args.hand_eye and args.tcp_pose:
        result = sd.pixel_to_base_via_hand_eye(
            args.output_dir,
            args.u,
            args.v,
            hand_eye_yaml=args.hand_eye,
            tcp_pose_yaml=args.tcp_pose,
        )
    else:
        result = sd.pixel_to_camera_xyz(args.output_dir, args.u, args.v)

    line = json.dumps(result, ensure_ascii=False)
    if args.json:
        print(line)
    else:
        if result.get("success"):
            print(
                f"像素 ({result['u']},{result['v']}) rect=({result['x']:.6f},{result['y']:.6f},{result['z']:.6f}) "
                f"cam=({result.get('x_cam', 0):.6f},{result.get('y_cam', 0):.6f},{result.get('z_cam', 0):.6f}) m"
            )
            if "x_base" in result:
                print(
                    f"方案A base=({result['x_base']:.6f},{result['y_base']:.6f},{result['z_base']:.6f}) m"
                )
        else:
            print(result.get("error", "失败"), file=sys.stderr)
            print(line)
            return 1
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
