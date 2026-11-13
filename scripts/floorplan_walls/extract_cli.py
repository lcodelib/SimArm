#!/usr/bin/env python3
"""
平面图墙体抽线 CLI（OpenCV，在 Blender 外运行）。

用法：
  cd scripts/floorplan_walls
  pip install -r requirements.txt
  python extract_cli.py --config config.yaml

或直接传参：
  python extract_cli.py --image floorplan.png --mpp 0.02 --output out/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from wall_extract import ExtractParams, params_from_config, run_extract, setup_logging


def _configure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


_configure_utf8_stdio()


def _build_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    """合并 YAML 与命令行覆盖项。"""
    cfg: Dict[str, Any] = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise ValueError("--config 须为 YAML 字典")
        cfg.update(loaded)

    if args.image:
        cfg["image"] = args.image
    if args.output:
        cfg["output_dir"] = args.output
    if args.mpp is not None:
        cfg["meters_per_pixel"] = args.mpp
    if args.thickness is not None:
        cfg["wall_thickness_m"] = args.thickness
    if args.height is not None:
        cfg["wall_height_m"] = args.height
    if args.log_level:
        cfg["log_level"] = args.log_level

    if "image" not in cfg:
        raise ValueError("须在 config 中指定 image，或使用 --image")
    if "output_dir" not in cfg:
        cfg["output_dir"] = "output/floorplan_walls"
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 2D 平面图提取墙体线段 → walls.json")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置路径")
    parser.add_argument("--image", type=str, default=None, help="平面图 PNG/JPG")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--mpp", type=float, default=None, help="米/像素（手填比例尺）")
    parser.add_argument("--thickness", type=float, default=None, help="墙厚（米）")
    parser.add_argument("--height", type=float, default=None, help="墙高（米）")
    parser.add_argument("--log-level", type=str, default=None, help="DEBUG/INFO/WARNING")
    args = parser.parse_args(argv)

    cfg = _build_cfg(args)
    setup_logging(str(cfg.get("log_level", "INFO")))
    params: ExtractParams = params_from_config(cfg)

    result = run_extract(
        image_path=Path(cfg["image"]),
        output_dir=Path(cfg["output_dir"]),
        params=params,
    )
    print(f"wall_count={len(result.walls)}")
    print(f"walls_json={result.debug_paths['walls_json']}")
    print(f"overlay={result.debug_paths['overlay']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
