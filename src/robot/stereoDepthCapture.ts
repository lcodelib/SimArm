/**
 * 双目拍照：保存左右图、拍照 TCP 位姿，并调用 stereo_depth_cli.py。
 */
import { normalizeCliErrorMessage } from "./subprocessText";

export const STEREO_DATA_DIR_REL = "stereo_data";
export const STEREO_DEPTH_CONFIG_REL = "scripts/stereo_depth/config.hand_eye.yaml";
/** 每对图旁保存的拍照瞬间 T_base_tcp */
export const STEREO_TCP_POSE_YAML = "tcp_pose.yaml";

export type StereoDepthCaptureResult = {
  success: boolean;
  pair_id?: string;
  left_image?: string;
  right_image?: string;
  /** 拍照时写入的 tcp_pose.yaml 绝对路径 */
  tcp_pose?: string;
  output_dir?: string;
  files?: Record<string, string>;
  baseline_m?: number;
  error?: string;
};

function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

async function invokeTauri<T>(cmd: string, args: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(cmd, args);
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const s = String(reader.result ?? "");
      const i = s.indexOf(",");
      resolve(i >= 0 ? s.slice(i + 1) : s);
    };
    reader.onerror = () => reject(reader.error ?? new Error("读取 Blob 失败"));
    reader.readAsDataURL(blob);
  });
}

function parseStereoDepthJson(raw: string): StereoDepthCaptureResult {
  const line = raw.trim().split(/\r?\n/).filter(Boolean).pop() ?? raw.trim();
  const parsed = JSON.parse(line) as StereoDepthCaptureResult;
  if (!parsed.success && parsed.error) {
    parsed.error = normalizeCliErrorMessage(parsed.error);
  }
  return parsed;
}

/**
 * 保存左右 PNG（及可选 TCP YAML）并运行立体深度 CLI。
 * 数据落在 stereo_data/pair_NNN/，结果输出到该目录下 out/。
 */
export async function captureStereoAndRunDepth(
  leftBlob: Blob,
  rightBlob: Blob,
  tcpPoseYaml?: string,
): Promise<StereoDepthCaptureResult> {
  const leftPngBase64 = await blobToBase64(leftBlob);
  const rightPngBase64 = await blobToBase64(rightBlob);

  let rawJson: string;
  if (isTauriRuntime()) {
    rawJson = await invokeTauri<string>("stereo_depth_capture_and_run", {
      leftPngBase64,
      rightPngBase64,
      tcpPoseYaml: tcpPoseYaml ?? null,
    });
  } else {
    const res = await fetch("/__stereo_depth/capture-and-run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ leftPngBase64, rightPngBase64, tcpPoseYaml }),
    });
    const text = await res.text().catch(() => "");
    if (!res.ok) {
      throw new Error(text || `HTTP ${res.status}`);
    }
    rawJson = text;
  }

  return parseStereoDepthJson(rawJson);
}
