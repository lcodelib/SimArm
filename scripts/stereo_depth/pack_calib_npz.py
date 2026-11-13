#!/usr/bin/env python3
"""将 K1,K2,D1,D2,R,T 打包为 stereo_depth_cli 可用的 calib.npz。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import stereo_depth as sd


def _load_matrix(path_or_json: str, name: str) -> np.ndarray:
    p = Path(path_or_json)
    if p.is_file():
        if p.suffix.lower() == ".npy":
            return np.load(p)
        text = p.read_text(encoding="utf-8")
    else:
        text = path_or_json
    arr = np.asarray(json.loads(text), dtype=np.float64)
    if name in ("K1", "K2"):
        return sd.parse_camera_matrix(arr)
    if name in ("D1", "D2"):
        return sd.parse_distortion(arr)
    if name == "R":
        return sd.parse_rotation(arr)
    if name == "T":
        return sd.parse_translation(arr).ravel()
    return arr


def main() -> int:
    parser = argparse.ArgumentParser(description="打包双目标定为 npz")
    parser.add_argument("--output", "-o", required=True, help="输出 calib.npz")
    parser.add_argument("--K1", required=True, help="左内参 JSON 或 .npy")
    parser.add_argument("--K2", required=True, help="右内参 JSON 或 .npy")
    parser.add_argument("--D1", required=True, help="左畸变 JSON")
    parser.add_argument("--D2", required=True, help="右畸变 JSON")
    parser.add_argument("--R", required=True, help="旋转 JSON（右相对左）")
    parser.add_argument("--T", required=True, help="平移 JSON [tx,ty,tz] 米")
    args = parser.parse_args()

    np.savez(
        args.output,
        K1=_load_matrix(args.K1, "K1"),
        K2=_load_matrix(args.K2, "K2"),
        D1=_load_matrix(args.D1, "D1"),
        D2=_load_matrix(args.D2, "D2"),
        R=_load_matrix(args.R, "R"),
        T=_load_matrix(args.T, "T"),
    )
    print(f"已写入: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
