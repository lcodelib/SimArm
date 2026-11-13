#!/usr/bin/env python3
"""
双目深度与点云 CLI。

流程：立体校正 → 极线校正图 → 视差(SGBM/BM) → 深度图 → PLY 点云。

用法：
  python stereo_depth_cli.py --config config.example.yaml
  python stereo_depth_cli.py --config config.yaml --json   # stdout 仅一行 JSON

也可通过 npz 传入矩阵（与 YAML 二选一）：
  python stereo_depth_cli.py --left L.png --right R.png --npz calib.npz --output out/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml

import stereo_depth as sd


def _configure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


_configure_utf8_stdio()


def _load_npz_calib(npz_path: Path) -> Dict[str, Any]:
    data = np.load(npz_path)
    required = ("K1", "K2", "R", "T", "D1", "D2")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"npz 缺少键: {missing}，需要 {required}")
    return {
        "left": {"camera_matrix": data["K1"], "dist_coeffs": data["D1"]},
        "right": {"camera_matrix": data["K2"], "dist_coeffs": data["D2"]},
        "extrinsic": {"R": data["R"], "T": data["T"]},
    }


def _build_cfg_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("--config 须为 YAML 字典")
        # 允许在 config 基础上覆盖左右图与输出目录（供 SimArm 双目拍照调用）
        if args.left:
            cfg["left_image"] = args.left
        if args.right:
            cfg["right_image"] = args.right
        if args.output:
            cfg["output_dir"] = args.output
        return cfg

    if not args.left or not args.right:
        raise ValueError("未指定 --config 时须提供 --left 与 --right 图像路径")

    cfg: Dict[str, Any] = {
        "left_image": args.left,
        "right_image": args.right,
        "output_dir": args.output or "stereo_output",
    }

    if args.npz:
        cfg.update(_load_npz_calib(Path(args.npz)))
    else:
        if not all([args.K1, args.K2, args.R, args.T, args.D1, args.D2]):
            raise ValueError("未使用 --npz 时须提供 --K1 --K2 --R --T --D1 --D2（JSON 字符串或 npz 文件路径）")

        def _parse_json_matrix(s: str) -> Any:
            p = Path(s)
            if p.is_file():
                if p.suffix.lower() == ".npy":
                    return np.load(p).tolist()
                return json.loads(p.read_text(encoding="utf-8"))
            return json.loads(s)

        cfg["left"] = {
            "camera_matrix": _parse_json_matrix(args.K1),
            "dist_coeffs": _parse_json_matrix(args.D1),
        }
        cfg["right"] = {
            "camera_matrix": _parse_json_matrix(args.K2),
            "dist_coeffs": _parse_json_matrix(args.D2),
        }
        cfg["extrinsic"] = {"R": _parse_json_matrix(args.R), "T": _parse_json_matrix(args.T)}

    if args.output:
        cfg["output_dir"] = args.output
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="双目立体：校正、视差、深度、点云")
    parser.add_argument("--config", "-c", help="YAML 配置文件路径")
    parser.add_argument("--left", help="左相机图像")
    parser.add_argument("--right", help="右相机图像")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--npz", help="标定 npz：K1,K2,R,T,D1,D2")
    parser.add_argument("--K1", help="左内参 3x3 JSON")
    parser.add_argument("--K2", help="右内参 3x3 JSON")
    parser.add_argument("--D1", help="左畸变 JSON")
    parser.add_argument("--D2", help="右畸变 JSON")
    parser.add_argument("--R", help="旋转 3x3 JSON（右相对左）")
    parser.add_argument("--T", help="平移 [tx,ty,tz] JSON，单位米")
    parser.add_argument("--json", action="store_true", help="stdout 仅输出一行 JSON")
    args = parser.parse_args()

    try:
        cfg = _build_cfg_from_args(args)
        result = sd.run_from_config(cfg)
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print("立体处理完成。")
            print(f"  输出目录: {result['output_dir']}")
            print(f"  基线 (m): {result['baseline_m']:.6f}")
            for k, v in result["files"].items():
                print(f"  {k}: {v}")
        return 0
    except Exception as e:
        err = {"success": False, "error": str(e)}
        if args.json:
            print(json.dumps(err, ensure_ascii=False))
        else:
            print(f"错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
