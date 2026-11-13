/**
 * 调用 yolo_onnx_infer_cli.py 对图像做检测推理。
 */
import { normalizeCliErrorMessage } from "./subprocessText";

export type YoloDetection = {
  cls_id: number;
  cls_name: string;
  conf: number;
  xyxy: [number, number, number, number];
};

export type YoloInferResult = {
  success: boolean;
  image?: string;
  vis?: string;
  num_detections?: number;
  detections?: YoloDetection[];
  results?: unknown[];
  error?: string;
};

function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

async function invokeTauri<T>(cmd: string, args: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(cmd, args);
}

function parseYoloJson(raw: string): YoloInferResult {
  const line = raw.trim().split(/\r?\n/).filter(Boolean).pop() ?? raw.trim();
  const parsed = JSON.parse(line) as YoloInferResult;
  if (!parsed.success && parsed.error) {
    parsed.error = normalizeCliErrorMessage(parsed.error);
  }
  return parsed;
}

/**
 * 对指定图像路径运行 YOLO ONNX 推理。
 * @param imagePath 绝对路径（通常为校正左图）
 * @param outputDir 可选输出目录；默认写到图像同级 yolo/
 */
export async function runYoloOnnxInfer(
  imagePath: string,
  outputDir?: string,
): Promise<YoloInferResult> {
  if (isTauriRuntime()) {
    const raw = await invokeTauri<string>("yolo_onnx_infer", {
      imagePath,
      outputDir: outputDir ?? null,
    });
    return parseYoloJson(raw);
  }
  const res = await fetch("/__yolo/infer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ imagePath, outputDir }),
  });
  const text = await res.text().catch(() => "");
  if (!res.ok) {
    throw new Error(text || `HTTP ${res.status}`);
  }
  return parseYoloJson(text);
}

/** 将绝对路径转为 stereo_data 相对路径（供 asset 接口加载可视化图） */
export function stereoDataRelFromAbs(absPath: string): string | null {
  const norm = absPath.replace(/\\/g, "/");
  const marker = "/stereo_data/";
  const i = norm.toLowerCase().lastIndexOf(marker);
  if (i < 0) return null;
  return norm.slice(i + marker.length);
}
