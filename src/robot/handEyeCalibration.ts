import * as THREE from "three";
import type { TcpPoseBase } from "./auboC5Kinematics";
import { fkTcpMatrixInBase } from "./auboC5Kinematics";
import type { ToolOffset } from "./toolOffset";
import { getPnpCheckerboardConfig } from "./pnpCheckerboard";
import {
  intrinsicsFromTcpCameraSpec,
  type TcpCameraIntrinsics,
  type TcpStereoEye,
} from "./tcpCameraConfig";

const { degToRad } = THREE.MathUtils;

/** 手眼数据根目录（相对工程根） */
export const HAND_EYE_DATA_DIR_REL = "hand_eye_data";
export const HAND_EYE_LEFT_CAMERA_DIR = "left_camera";
export const HAND_EYE_RIGHT_CAMERA_DIR = "right_camera";

/** 手眼校验：固定文件名，每次覆盖 */
export const HAND_EYE_VERIFY_IMAGE = "verify.png";
export const HAND_EYE_VERIFY_POSE_YAML = "tcp_pose.yaml";
export const HAND_EYE_CONFIG_FILENAME = "config.yaml";

/** 标定脚本默认仍可读此路径；与左目 config 内容同步导出 */
export const HAND_EYE_CONFIG_REL = "scripts/hand_eye_calibration/config.yaml";

export function cameraDirForStereoEye(eye: TcpStereoEye): string {
  return eye === "left" ? HAND_EYE_LEFT_CAMERA_DIR : HAND_EYE_RIGHT_CAMERA_DIR;
}

export function handEyeCameraRelPath(eye: TcpStereoEye, ...parts: string[]): string {
  return [HAND_EYE_DATA_DIR_REL, cameraDirForStereoEye(eye), ...parts].join("/");
}

export type HandEyePoseCsvRow = {
  image: string;
  tx: number;
  ty: number;
  tz: number;
  qx: number;
  qy: number;
  qz: number;
  qw: number;
};

/** R = Rz(rz)·Ry(ry)·Rx(rx)，四元数 scalar_last (qx,qy,qz,qw) */
export function rpyDegZyxToQuaternion(rxDeg: number, ryDeg: number, rzDeg: number): THREE.Quaternion {
  const r = degToRad(rxDeg);
  const p = degToRad(ryDeg);
  const y = degToRad(rzDeg);
  const Rx = new THREE.Matrix4().makeRotationX(r);
  const Ry = new THREE.Matrix4().makeRotationY(p);
  const Rz = new THREE.Matrix4().makeRotationZ(y);
  const R = new THREE.Matrix4().copy(Rz).multiply(Ry).multiply(Rx);
  return new THREE.Quaternion().setFromRotationMatrix(R);
}

export function tcpPoseToCsvRow(image: string, pose: TcpPoseBase): HandEyePoseCsvRow {
  const q = rpyDegZyxToQuaternion(pose.rxDeg, pose.ryDeg, pose.rzDeg);
  return {
    image,
    tx: pose.x,
    ty: pose.y,
    tz: pose.z,
    qx: q.x,
    qy: q.y,
    qz: q.z,
    qw: q.w,
  };
}

export function formatPoseCsvLine(row: HandEyePoseCsvRow): string {
  const f = (v: number) => Number(v.toFixed(6));
  return `${row.image},${f(row.tx)},${f(row.ty)},${f(row.tz)},${f(row.qx)},${f(row.qy)},${f(row.qz)},${f(row.qw)}`;
}

export const POSES_CSV_HEADER = "image,tx,ty,tz,qx,qy,qz,qw";

function matrix4ToRowMajor16ForYaml(m: THREE.Matrix4): number[][] {
  const e = m.elements;
  const r = (v: number) => Number(v.toFixed(9));
  return [
    [r(e[0]), r(e[4]), r(e[8]), r(e[12])],
    [r(e[1]), r(e[5]), r(e[9]), r(e[13])],
    [r(e[2]), r(e[6]), r(e[10]), r(e[14])],
    [r(e[3]), r(e[7]), r(e[11]), r(e[15])],
  ];
}

export function buildTcpPoseExampleYaml(T: THREE.Matrix4, imageBasename: string): string {
  const rows = matrix4ToRowMajor16ForYaml(T);
  return [
    "# 眼在手上时，与某张 ArUco 图对应的「当前」末端位姿 T_base_tcp（p_base = T @ p_tcp）",
    "# 与手眼标定时 CSV 中矩阵含义一致",
    `# SimArm 手眼校验快照 | 配套图像: ${imageBasename}`,
    "",
    "T_matrix_row_major:",
    `  - [${rows[0].join(", ")}]`,
    `  - [${rows[1].join(", ")}]`,
    `  - [${rows[2].join(", ")}]`,
    `  - [${rows[3].join(", ")}]`,
    "",
  ].join("\n");
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      const base64 = dataUrl.split(",")[1];
      if (!base64) reject(new Error("读取 PNG 失败"));
      else resolve(base64);
    };
    reader.onerror = () => reject(reader.error ?? new Error("FileReader 错误"));
    reader.readAsDataURL(blob);
  });
}

function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

async function invokeTauri<T>(cmd: string, args: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(cmd, args);
}

async function postDevApi(path: string, body: unknown): Promise<void> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `HTTP ${res.status}`);
  }
}

/** 指定相机目录下下一张采集图，如 capture_001.png */
export async function nextCaptureImageName(stereoEye: TcpStereoEye): Promise<string> {
  if (isTauriRuntime()) {
    const idx = await invokeTauri<number>("hand_eye_next_capture_index", { stereoEye });
    return `capture_${String(idx).padStart(3, "0")}.png`;
  }
  const res = await fetch(`/__hand_eye/next-index?eye=${stereoEye}`);
  if (!res.ok) throw new Error("无法获取采集序号");
  const { index } = (await res.json()) as { index: number };
  return `capture_${String(index).padStart(3, "0")}.png`;
}

export type HandEyeVerifySaveResult = {
  imageName: string;
  yamlFileName: string;
  cameraDir: string;
};

/** 手眼校验：写入对应相机目录，固定 verify.png + tcp_pose.yaml（覆盖） */
export async function saveHandEyeVerifySnapshot(
  pngBlob: Blob,
  jointAnglesDeg: readonly number[],
  stereoEye: TcpStereoEye,
  toolOffset?: ToolOffset,
): Promise<HandEyeVerifySaveResult> {
  const imageName = HAND_EYE_VERIFY_IMAGE;
  const yamlFileName = HAND_EYE_VERIFY_POSE_YAML;
  const T = fkTcpMatrixInBase(jointAnglesDeg, toolOffset);
  const yamlContent = buildTcpPoseExampleYaml(T, imageName);
  const pngBase64 = await blobToBase64(pngBlob);
  const cameraDir = cameraDirForStereoEye(stereoEye);
  if (isTauriRuntime()) {
    await invokeTauri("hand_eye_save_verify", { pngBase64, stereoEye, yamlContent });
  } else {
    await postDevApi("/__hand_eye/verify", { pngBase64, stereoEye, yamlContent });
  }
  return { imageName, yamlFileName, cameraDir };
}

/** 拍照采集：写入 hand_eye_data/{left|right}_camera/，递增 capture_NNN.png + poses.csv */
export async function saveHandEyeCapture(
  pngBlob: Blob,
  pose: TcpPoseBase,
  stereoEye: TcpStereoEye,
): Promise<string> {
  const imageName = await nextCaptureImageName(stereoEye);
  const pngBase64 = await blobToBase64(pngBlob);
  const csvLine = formatPoseCsvLine(tcpPoseToCsvRow(imageName, pose));
  if (isTauriRuntime()) {
    await invokeTauri("hand_eye_save_capture", { pngBase64, imageName, csvLine, stereoEye });
  } else {
    await postDevApi("/__hand_eye/capture", { pngBase64, imageName, csvLine, stereoEye });
  }
  return imageName;
}

export function buildHandEyeConfigYaml(
  projectRootAbs: string,
  stereoEye: TcpStereoEye,
  intrinsics: TcpCameraIntrinsics = intrinsicsFromTcpCameraSpec(),
): string {
  const board = getPnpCheckerboardConfig();
  const dataDir = `${projectRootAbs.replace(/\\/g, "/")}/${handEyeCameraRelPath(stereoEye)}`;
  const [icx, icy] = board.innerCorners;
  const [k1, k2, p1, p2] = intrinsics.distCoeffs;

  return `# 手眼标定配置（由 SimArm 导出 · ${stereoEye} 相机）

# 本配置对应光路：${stereoEye}（支架局部 ${stereoEye === "left" ? "−X" : "+X"}，相机 UV 中 u 向右为 +X）

# 数据目录：内含棋盘格照片、位姿与校验快照
data_dir: "${dataDir}"

# 照片 glob（相对 data_dir，不含 verify.png）
images_glob: "capture_*.png"

# 位姿文件（相对 data_dir）
poses_file: "poses.csv"

# 手眼单次校验（相对 data_dir，覆盖写入）
verify_image: "${HAND_EYE_VERIFY_IMAGE}"
verify_tcp_pose: "${HAND_EYE_VERIFY_POSE_YAML}"

# 相机内参（与 OpenCV 标定/采集一致）
camera:
  fx: ${intrinsics.fx.toFixed(4)}
  fy: ${intrinsics.fy.toFixed(4)}
  cx: ${intrinsics.cx.toFixed(4)}
  cy: ${intrinsics.cy.toFixed(4)}
  dist_coeffs: [${k1}, ${k2}, ${p1}, ${p2}]

board:
  inner_corners: [${icx}, ${icy}]
  square_size_m: ${board.squareSizeM}

setup: eye_in_hand

pose_format:
  type: csv
  image_column: image
  translation_m: [tx, ty, tz]
  quaternion: [qx, qy, qz, qw]
  quaternion_order: scalar_last

hand_eye_method: TSAI

corner_subpix_win: 11
`;
}

export type HandEyeConfigExportResult = {
  leftPath: string;
  rightPath: string;
  legacyScriptPath: string;
};

/** 左右相机 config 写入 hand_eye_data/{left|right}_camera/config.yaml，并同步左目到 scripts/.../config.yaml */
export async function exportHandEyeConfigYaml(
  intrinsics: TcpCameraIntrinsics = intrinsicsFromTcpCameraSpec(),
): Promise<HandEyeConfigExportResult> {
  let projectRoot: string;
  if (isTauriRuntime()) {
    projectRoot = await invokeTauri<string>("hand_eye_project_root", {});
  } else {
    const res = await fetch("/__hand_eye/project-root");
    if (!res.ok) throw new Error("无法获取工程根目录");
    const doc = (await res.json()) as { root: string };
    projectRoot = doc.root;
  }
  const leftYaml = buildHandEyeConfigYaml(projectRoot, "left", intrinsics);
  const rightYaml = buildHandEyeConfigYaml(projectRoot, "right", intrinsics);
  if (isTauriRuntime()) {
    const paths = await invokeTauri<{ left_path: string; right_path: string; legacy_script_path: string }>(
      "hand_eye_export_config",
      { leftYamlContent: leftYaml, rightYamlContent: rightYaml },
    );
    return {
      leftPath: paths.left_path,
      rightPath: paths.right_path,
      legacyScriptPath: paths.legacy_script_path,
    };
  }
  const res = await fetch("/__hand_eye/export-config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ leftYamlContent: leftYaml, rightYamlContent: rightYaml }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `HTTP ${res.status}`);
  }
  const doc = (await res.json()) as HandEyeConfigExportResult;
  return doc;
}

export type HandEyeClearResult = {
  deletedImages: number;
};

function mapClearResult(raw: { deleted_images?: number; deletedImages?: number }): HandEyeClearResult {
  return { deletedImages: raw.deletedImages ?? raw.deleted_images ?? 0 };
}

import { parseHandEyeSolveCliJson, type HandEyeSolveCliResponse } from "./handEyeSolveTypes";

export type HandEyeSolveRunResult = {
  rawJson: string;
  parsed: HandEyeSolveCliResponse;
  resultRelPath?: string;
};

/** 对当前相机目录运行 hand_eye_solve_cli.py；返回完整 JSON（成功/失败均由弹窗展示） */
export async function runHandEyeSolve(stereoEye: TcpStereoEye): Promise<HandEyeSolveRunResult> {
  let rawJson: string;
  if (isTauriRuntime()) {
    rawJson = await invokeTauri<string>("hand_eye_run_solve", { stereoEye });
  } else {
    const res = await fetch("/__hand_eye/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stereoEye }),
    });
    const text = await res.text().catch(() => "");
    if (!res.ok) {
      throw new Error(text || `HTTP ${res.status}`);
    }
    rawJson = text;
  }
  const parsed = parseHandEyeSolveCliJson(rawJson);
  const displayJson = parsed.success ? rawJson : JSON.stringify(parsed, null, 2);
  const resultRelPath = parsed.success
    ? `${handEyeCameraRelPath(stereoEye)}/hand_eye_result.yaml`
    : undefined;
  return { rawJson: displayJson, parsed, resultRelPath };
}

/** 清除左右相机目录下全部 capture_*.png 并重置各自 poses.csv */
export async function clearHandEyeCaptureData(): Promise<HandEyeClearResult> {
  if (isTauriRuntime()) {
    const raw = await invokeTauri<{ deleted_images: number }>("hand_eye_clear_data", {});
    return mapClearResult(raw);
  }
  const res = await fetch("/__hand_eye/clear", { method: "POST" });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `HTTP ${res.status}`);
  }
  return mapClearResult((await res.json()) as { deletedImages: number });
}
