#!/usr/bin/env python3
"""
YOLO11 ONNX 检测推理模块。

负责：加载 best.onnx、letterbox 预处理、推理、NMS 后处理、绘制与保存结果。
类名优先从 ONNX metadata 读取（与导出时 train.yaml 一致）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# 默认与导出模型一致：输入 1280x1280，输出 [1, 4+nc, N]
DEFAULT_IMGSZ = 1280
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45


@dataclass
class Detection:
    """单条检测结果（原图像素坐标）。"""

    xyxy: Tuple[float, float, float, float]  # x1, y1, x2, y2
    conf: float
    cls_id: int
    cls_name: str


@dataclass
class InferResult:
    """单张图推理结果。"""

    image_path: Path
    image_bgr: np.ndarray
    detections: List[Detection]
    # letterbox 还原参数：缩放比与左右/上下 padding
    ratio: float
    pad: Tuple[float, float]  # (pad_w, pad_h)


def setup_logging(level: int = logging.INFO) -> None:
    """配置根日志（CLI 入口可调用）。"""
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_names_metadata(raw: str) -> Dict[int, str]:
    """
    解析 Ultralytics 写入的 names 元数据。
    形如: "{0: '__background__', 1: 'a', 2: 'b', ...}"
    """
    # ast.literal_eval 无法直接解析单引号 key，这里用安全替换后 eval 字典字面量
    import ast
    import re

    text = raw.strip()
    # 将 {0: 'a'} 转为合法 Python dict 字面量
    text = re.sub(r"(\d+)\s*:", r"\1:", text)
    try:
        obj = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"无法解析 ONNX names 元数据: {raw!r}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"ONNX names 元数据不是字典: {type(obj)}")
    return {int(k): str(v) for k, v in obj.items()}


def letterbox(
    image_bgr: np.ndarray,
    new_shape: Tuple[int, int] = (DEFAULT_IMGSZ, DEFAULT_IMGSZ),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    保持宽高比缩放并填充到 new_shape（与 YOLO 导出预处理一致）。

    Returns:
        padded: 填充后的 BGR 图
        ratio: 缩放比例（相对原图）
        pad: (pad_w, pad_h) 左右/上下各半的填充像素（浮点）
    """
    h, w = image_bgr.shape[:2]
    new_h, new_w = new_shape
    ratio = min(new_h / h, new_w / w)
    resized_w = int(round(w * ratio))
    resized_h = int(round(h * ratio))
    img = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (new_w - resized_w) / 2.0
    pad_h = (new_h - resized_h) / 2.0
    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, ratio, (pad_w, pad_h)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """中心点宽高 → 左上右下。"""
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2.0
    y[:, 1] = x[:, 1] - x[:, 3] / 2.0
    y[:, 2] = x[:, 0] + x[:, 2] / 2.0
    y[:, 3] = x[:, 1] + x[:, 3] / 2.0
    return y


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """对一组 xyxy 框做 NMS，返回保留下标。"""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_thres]
    return keep


class YoloOnnxDetector:
    """基于 onnxruntime 的 YOLO11 检测器。"""

    def __init__(
        self,
        model_path: Union[str, Path],
        conf_thres: float = DEFAULT_CONF,
        iou_thres: float = DEFAULT_IOU,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"模型不存在: {self.model_path}")

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        avail = ort.get_available_providers()
        if providers is None:
            # 优先 CUDA，否则 CPU
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in avail
                else ["CPUExecutionProvider"]
            )
        self.session = ort.InferenceSession(
            str(self.model_path), providers=list(providers)
        )
        self.input_name = self.session.get_inputs()[0].name
        in_shape = self.session.get_inputs()[0].shape
        # [1, 3, H, W]
        self.imgsz = int(in_shape[2]) if isinstance(in_shape[2], int) else DEFAULT_IMGSZ

        self.class_names = self._load_class_names()
        logger.info(
            "已加载模型 %s | input=%s imgsz=%s providers=%s classes=%s",
            self.model_path.name,
            self.input_name,
            self.imgsz,
            self.session.get_providers(),
            self.class_names,
        )

    def _load_class_names(self) -> Dict[int, str]:
        """从 ONNX metadata 读取类名；失败则按输出通道推断占位名。"""
        # onnxruntime ModelMetadata.custom_metadata_map 为 dict[str, str]
        meta = dict(self.session.get_modelmeta().custom_metadata_map)
        if "names" in meta:
            names = _parse_names_metadata(meta["names"])
            logger.debug("从 ONNX metadata 读取类名: %s", names)
            return names

        out_shape = self.session.get_outputs()[0].shape
        # [1, 4+nc, N]
        nc = int(out_shape[1]) - 4 if isinstance(out_shape[1], int) else 0
        logger.warning("ONNX 无 names 元数据，使用 cls_%d 占位", nc)
        return {i: f"cls_{i}" for i in range(max(nc, 0))}

    def preprocess(
        self, image_bgr: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """BGR 图 → NCHW float32（0~1），并返回 letterbox 参数。"""
        padded, ratio, pad = letterbox(image_bgr, (self.imgsz, self.imgsz))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]  # 1x3xHxW
        return np.ascontiguousarray(blob), ratio, pad

    def postprocess(
        self,
        output: np.ndarray,
        ratio: float,
        pad: Tuple[float, float],
        orig_shape: Tuple[int, int],
    ) -> List[Detection]:
        """
        将原始输出还原到原图坐标并做置信度过滤 + 按类 NMS。

        YOLO11 导出（nms=False）输出形状: [1, 4+nc, N]，前 4 维为 cx,cy,w,h（letterbox 坐标系）。
        """
        # [1, 4+nc, N] → [N, 4+nc]
        pred = np.squeeze(output, axis=0).T
        if pred.ndim != 2 or pred.shape[1] < 5:
            logger.error("意外的输出形状: %s", getattr(output, "shape", None))
            return []

        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]
        cls_ids = np.argmax(cls_scores, axis=1)
        confs = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

        mask = confs >= self.conf_thres
        boxes_xywh = boxes_xywh[mask]
        confs = confs[mask]
        cls_ids = cls_ids[mask]
        if boxes_xywh.size == 0:
            return []

        boxes = xywh2xyxy(boxes_xywh)
        pad_w, pad_h = pad
        # 去掉 letterbox padding 再按 ratio 还原到原图
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes /= ratio

        oh, ow = orig_shape
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh)

        detections: List[Detection] = []
        for c in np.unique(cls_ids):
            idx = np.where(cls_ids == c)[0]
            keep = nms_xyxy(boxes[idx], confs[idx], self.iou_thres)
            for k in keep:
                i = int(idx[k])
                x1, y1, x2, y2 = (float(v) for v in boxes[i])
                cid = int(cls_ids[i])
                detections.append(
                    Detection(
                        xyxy=(x1, y1, x2, y2),
                        conf=float(confs[i]),
                        cls_id=cid,
                        cls_name=self.class_names.get(cid, f"cls_{cid}"),
                    )
                )

        detections.sort(key=lambda d: d.conf, reverse=True)
        return detections

    def infer_bgr(self, image_bgr: np.ndarray, image_path: Optional[Path] = None) -> InferResult:
        """对已读入的 BGR 图像做推理。"""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("输入图像为空")
        blob, ratio, pad = self.preprocess(image_bgr)
        outputs = self.session.run(None, {self.input_name: blob})
        dets = self.postprocess(outputs[0], ratio, pad, image_bgr.shape[:2])
        return InferResult(
            image_path=image_path or Path("memory.jpg"),
            image_bgr=image_bgr,
            detections=dets,
            ratio=ratio,
            pad=pad,
        )

    def infer_path(self, image_path: Union[str, Path]) -> InferResult:
        """从文件路径读图并推理。"""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"图像不存在: {path}")
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"无法读取图像: {path}")
        logger.info("推理: %s (%dx%d)", path.name, image_bgr.shape[1], image_bgr.shape[0])
        return self.infer_bgr(image_bgr, path)


def draw_detections(
    image_bgr: np.ndarray,
    detections: Sequence[Detection],
    line_thickness: int = 2,
) -> np.ndarray:
    """在图上绘制检测框与标签，返回副本。"""
    vis = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
        label = f"{det.cls_name} {det.conf:.2f}"
        color = _color_for_class(det.cls_id)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, line_thickness)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        ty1 = max(0, y1 - th - baseline - 4)
        cv2.rectangle(vis, (x1, ty1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            vis,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def _color_for_class(cls_id: int) -> Tuple[int, int, int]:
    """按类 id 生成稳定的 BGR 颜色。"""
    rng = np.random.default_rng(cls_id + 17)
    color = rng.integers(64, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def save_result(
    result: InferResult,
    output_dir: Union[str, Path],
    save_txt: bool = True,
) -> Path:
    """
    保存可视化图；可选保存 YOLO 风格 txt（cls cx cy w h conf，相对原图宽高归一化）。

    Returns:
        可视化图路径
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = result.image_path.stem
    vis_path = out_dir / f"{stem}_det.jpg"
    vis = draw_detections(result.image_bgr, result.detections)
    if not cv2.imwrite(str(vis_path), vis):
        raise OSError(f"保存失败: {vis_path}")
    logger.info("已保存可视化: %s (%d 个目标)", vis_path, len(result.detections))

    if save_txt:
        txt_path = out_dir / f"{stem}_det.txt"
        h, w = result.image_bgr.shape[:2]
        lines: List[str] = []
        for det in result.detections:
            x1, y1, x2, y2 = det.xyxy
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(
                f"{det.cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {det.conf:.6f}"
            )
        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        logger.info("已保存标签: %s", txt_path)

    return vis_path


def collect_images(path: Union[str, Path]) -> List[Path]:
    """收集单文件或目录下的常见图像。"""
    p = Path(path)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if p.is_file():
        if p.suffix.lower() not in exts:
            raise ValueError(f"不支持的图像后缀: {p.suffix}")
        return [p]
    if p.is_dir():
        images = sorted(
            f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts
        )
        if not images:
            raise FileNotFoundError(f"目录中无图像: {p}")
        return images
    raise FileNotFoundError(f"路径不存在: {p}")
