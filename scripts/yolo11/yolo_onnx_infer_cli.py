#!/usr/bin/env python3
"""
YOLO11 ONNX 检测推理 CLI。

用法：
  python yolo_onnx_infer_cli.py --image path/to.jpg
  python yolo_onnx_infer_cli.py --image path/to/dir --conf 0.3 --iou 0.5
  python yolo_onnx_infer_cli.py --model best.onnx --image ../stereo_data/pair_001/left.png

默认模型为同目录下的 best.onnx，结果写入 output/。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from yolo_onnx_infer import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    YoloOnnxDetector,
    collect_images,
    save_result,
    setup_logging,
)

logger = logging.getLogger(__name__)


def _configure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="YOLO11 ONNX 图像检测推理")
    parser.add_argument(
        "--model",
        type=Path,
        default=script_dir / "best.onnx",
        help="ONNX 模型路径（默认: 同目录 best.onnx）",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="输入图像文件或目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "output",
        help="结果输出目录（默认: ./output）",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"置信度阈值（默认: {DEFAULT_CONF}）",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help=f"NMS IoU 阈值（默认: {DEFAULT_IOU}）",
    )
    parser.add_argument(
        "--no-txt",
        action="store_true",
        help="不保存 txt 标签文件",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="stdout 额外输出 JSON 摘要",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="调试日志",
    )
    return parser.parse_args()


def main() -> int:
    _configure_utf8_stdio()
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    try:
        images = collect_images(args.image)
        detector = YoloOnnxDetector(
            model_path=args.model,
            conf_thres=args.conf,
            iou_thres=args.iou,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("%s", exc)
        return 1

    summary = []
    for img_path in images:
        try:
            result = detector.infer_path(img_path)
            vis_path = save_result(result, args.output, save_txt=not args.no_txt)
        except (FileNotFoundError, ValueError, OSError) as exc:
            logger.error("处理失败 %s: %s", img_path, exc)
            return 1

        for det in result.detections:
            logger.info(
                "  %s conf=%.3f box=(%.1f,%.1f,%.1f,%.1f)",
                det.cls_name,
                det.conf,
                *det.xyxy,
            )
        summary.append(
            {
                "image": str(img_path),
                "vis": str(vis_path),
                "num_detections": len(result.detections),
                "detections": [
                    {
                        "cls_id": d.cls_id,
                        "cls_name": d.cls_name,
                        "conf": round(d.conf, 4),
                        "xyxy": [round(v, 2) for v in d.xyxy],
                    }
                    for d in result.detections
                ],
            }
        )

    if args.json:
        # 单行 JSON，便于前端 / Tauri 解析；单图时摊平首项字段
        payload: dict = {"success": True, "results": summary}
        if len(summary) == 1:
            payload.update(summary[0])
        print(json.dumps(payload, ensure_ascii=False))
    logger.info("完成：%d 张图像 → %s", len(images), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
