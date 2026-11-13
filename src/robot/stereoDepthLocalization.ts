/**
 * 双目定位测试：校正左图像素 → OpenCV 左目 / 方案 A 手眼到基座，以及预览图加载。
 */
import type { TcpPoseBase } from "./auboC5Kinematics";
import { HAND_EYE_RESULT_YAML } from "./handEyeArucoVerify";
import { handEyeCameraRelPath } from "./handEyeCalibration";
import { normalizeCliErrorMessage } from "./subprocessText";
import type { StereoDepthCaptureResult } from "./stereoDepthCapture";
import { STEREO_TCP_POSE_YAML } from "./stereoDepthCapture";

export type StereoTcpTarget = TcpPoseBase;

export type StereoPixelTo3dResult = {
  success: boolean;
  u?: number;
  v?: number;
  /** 校正左目 OpenCV */
  x?: number;
  y?: number;
  z?: number;
  frame?: string;
  /** 方案 A：R1^T 后的原始左目 OpenCV（手眼 p_cam） */
  x_cam?: number;
  y_cam?: number;
  z_cam?: number;
  frame_cam?: string;
  r1_assumed_identity?: boolean;
  x_tcp?: number;
  y_tcp?: number;
  z_tcp?: number;
  x_base?: number;
  y_base?: number;
  z_base?: number;
  tcp_target?: StereoTcpTarget;
  /** 3cm 邻域平面 */
  plane_ok?: boolean;
  plane_error?: string;
  plane_point_count?: number;
  plane_rmse_m?: number;
  patch_half_m?: number;
  normal_rect?: [number, number, number];
  normal_cam?: [number, number, number];
  R_rect_row_major?: number[][];
  R_cam_row_major?: number[][];
  rpy_rect_deg?: [number, number, number];
  pose_from_plane?: boolean;
  units?: string;
  image_size?: [number, number];
  to_three_left_camera?: [number, number, number];
  scheme?: string;
  semantics?: string;
  error?: string;
};

function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

async function invokeTauri<T>(cmd: string, args: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(cmd, args);
}

function parseJsonLine(raw: string): StereoPixelTo3dResult {
  const line = raw.trim().split(/\r?\n/).filter(Boolean).pop() ?? raw.trim();
  const parsed = JSON.parse(line) as StereoPixelTo3dResult;
  if (!parsed.success && parsed.error) {
    parsed.error = normalizeCliErrorMessage(parsed.error);
  }
  return parsed;
}

/** 相对 stereo_data 的校正左图路径 */
export function stereoRectifiedLeftRel(pairId: string): string {
  return `${pairId}/out/rectified_left.png`;
}

/** 左目手眼结果相对工程根路径 */
export function leftHandEyeResultRel(): string {
  return handEyeCameraRelPath("left", HAND_EYE_RESULT_YAML);
}

/** 解析工程根目录（Vite / Tauri） */
export async function resolveProjectRootAbs(): Promise<string> {
  if (isTauriRuntime()) {
    return invokeTauri<string>("hand_eye_project_root", {});
  }
  const res = await fetch("/__hand_eye/project-root");
  if (!res.ok) throw new Error("无法获取工程根目录");
  const doc = (await res.json()) as { root: string };
  return doc.root.replace(/\\/g, "/");
}

/** 左目手眼 YAML 绝对路径 */
export async function resolveLeftHandEyeAbs(): Promise<string> {
  const root = await resolveProjectRootAbs();
  return `${root.replace(/\/+$/, "")}/${leftHandEyeResultRel()}`;
}

/**
 * OpenCV 校正左目点 → Three.js TcpCameraLeft 局部坐标。
 */
export function opencvCamToThreeLocal(x: number, y: number, z: number): [number, number, number] {
  return [x, -y, -z];
}

/** 由双目结果推断拍照 tcp_pose.yaml 绝对路径 */
export function tcpPosePathFromStereoResult(result: StereoDepthCaptureResult): string | null {
  if (result.tcp_pose) return result.tcp_pose.replace(/\\/g, "/");
  if (result.left_image) {
    const left = result.left_image.replace(/\\/g, "/");
    const i = left.lastIndexOf("/");
    if (i >= 0) return `${left.slice(0, i)}/${STEREO_TCP_POSE_YAML}`;
  }
  if (result.output_dir) {
    const out = result.output_dir.replace(/\\/g, "/").replace(/\/+$/, "");
    if (out.toLowerCase().endsWith("/out")) {
      return `${out.slice(0, -4)}/${STEREO_TCP_POSE_YAML}`;
    }
  }
  return null;
}

/**
 * 像素 → 3D；传入 handEyeAbs + tcpPoseAbs 时走方案 A（到基座 TCP 目标）。
 */
export async function stereoPixelToCameraXyz(
  outputDir: string,
  u: number,
  v: number,
  opts?: { handEyePath?: string; tcpPosePath?: string },
): Promise<StereoPixelTo3dResult> {
  const handEyePath = opts?.handEyePath;
  const tcpPosePath = opts?.tcpPosePath;

  if (isTauriRuntime()) {
    const raw = await invokeTauri<string>("stereo_depth_pixel_to_3d", {
      outputDir,
      u,
      v,
      handEyePath: handEyePath ?? null,
      tcpPosePath: tcpPosePath ?? null,
    });
    return parseJsonLine(raw);
  }
  const res = await fetch("/__stereo_depth/pixel-to-3d", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ outputDir, u, v, handEyePath, tcpPosePath }),
  });
  const text = await res.text().catch(() => "");
  if (!res.ok) {
    throw new Error(text || `HTTP ${res.status}`);
  }
  return parseJsonLine(text);
}

/**
 * 加载双目对中的资源为 Blob URL（须 revokeObjectURL）。
 */
export async function loadStereoDataAssetBlobUrl(relPath: string): Promise<string> {
  const normalized = relPath.replace(/\\/g, "/").replace(/^\/+/, "");
  if (normalized.includes("..")) {
    throw new Error("非法资源路径");
  }

  if (isTauriRuntime()) {
    const b64 = await invokeTauri<string>("stereo_depth_read_asset_base64", {
      relPath: normalized,
    });
    const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const blob = new Blob([bin], { type: "image/png" });
    return URL.createObjectURL(blob);
  }

  const res = await fetch(`/__stereo_depth/asset?rel=${encodeURIComponent(normalized)}`);
  if (!res.ok) {
    const t = await res.text().catch(() => "");
    throw new Error(t || `加载资源失败 HTTP ${res.status}`);
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

/** 从双目结果取校正左图绝对路径 */
export function rectifiedLeftAbsFromResult(result: StereoDepthCaptureResult): string | null {
  const fromFiles = result.files?.rectified_left;
  if (fromFiles) return fromFiles.replace(/\\/g, "/");
  if (result.output_dir) {
    const out = result.output_dir.replace(/\\/g, "/").replace(/\/+$/, "");
    return `${out}/rectified_left.png`;
  }
  return null;
}

/** 从最近一次双目结果取校正左图相对路径；失败返回 null */
export function rectifiedLeftRelFromResult(result: StereoDepthCaptureResult): string | null {
  if (result.pair_id) {
    return stereoRectifiedLeftRel(result.pair_id);
  }
  const abs = result.files?.rectified_left;
  if (!abs) return null;
  const norm = abs.replace(/\\/g, "/");
  const marker = "/stereo_data/";
  const i = norm.toLowerCase().lastIndexOf(marker);
  if (i >= 0) return norm.slice(i + marker.length);
  return null;
}
