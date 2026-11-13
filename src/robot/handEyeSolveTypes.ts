/** hand_eye_solve_cli.py  stdout 单行 JSON（成功/失败） */

import { normalizeCliErrorMessage } from "./subprocessText";

export type HandEyeSolveFrame = {
  image: string;
  T_base_tcp_row_major: number[][];
  T_base_camera_row_major?: number[][];
  T_base_board_row_major?: number[][];
  board_origin_position_base_m?: [number, number, number];
};

export type HandEyeSolveVisualization = {
  T_base_row_major: number[][];
  frames: HandEyeSolveFrame[];
  T_cam2tcp_row_major?: number[][];
  T_base_camera_row_major?: number[][];
  hand_eye_label?: string;
  hand_eye_semantics?: string;
  /** 与 hand_eye_solve.py visualize_3d.frame_scale_m 一致 */
  frame_scale_m?: number;
};

/** 坐标轴 RGB：X / Y / Z（与 matplotlib 手眼 3D 图一致） */
export const HAND_EYE_AXIS_COLORS = {
  tcp: [0xe74c3c, 0x2ecc71, 0x3498db] as const,
  base: [0x8b0000, 0x006400, 0x00008b] as const,
  camera: [0xe67e22, 0x27ae60, 0x9b59b6] as const,
  cameraFixed: [0xdc143c, 0x32cd32, 0x4169e1] as const,
};

/** 手眼 3D 每组位姿的区分色（HSL 色环） */
export function handEyeGroupColorHex(index: number, count: number): number {
  const hue = count <= 1 ? 0.58 : (index / count) * 0.92;
  return hslToHex(hue, 0.72, 0.5);
}

function hslToHex(h: number, s: number, l: number): number {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h * 6) % 2) - 1));
  const m = l - c / 2;
  let r = 0;
  let g = 0;
  let b = 0;
  const seg = Math.floor(h * 6) % 6;
  if (seg === 0) {
    r = c;
    g = x;
  } else if (seg === 1) {
    r = x;
    g = c;
  } else if (seg === 2) {
    g = c;
    b = x;
  } else if (seg === 3) {
    g = x;
    b = c;
  } else if (seg === 4) {
    r = x;
    b = c;
  } else {
    r = c;
    b = x;
  }
  const ri = Math.round((r + m) * 255);
  const gi = Math.round((g + m) * 255);
  const bi = Math.round((b + m) * 255);
  return (ri << 16) | (gi << 8) | bi;
}

export type HandEyeSolveSuccessPayload = {
  success: true;
  T_matrix_row_major: number[][];
  visualization: HandEyeSolveVisualization;
  board_origin_rmse_m?: number | null;
  hand_eye_method_used?: string;
  pose_variant?: string | number;
  num_poses?: number;
  setup?: string;
  used_images?: string[];
};

export type HandEyeSolveFailPayload = {
  success: false;
  error: string;
};

export type HandEyeSolveCliResponse = HandEyeSolveSuccessPayload | HandEyeSolveFailPayload;

export function parseHandEyeSolveCliJson(raw: string): HandEyeSolveCliResponse {
  const line = raw.trim().split(/\r?\n/).filter(Boolean).pop() ?? raw.trim();
  const parsed = JSON.parse(line) as HandEyeSolveCliResponse;
  if (!parsed.success && parsed.error) {
    parsed.error = normalizeCliErrorMessage(parsed.error);
  }
  return parsed;
}

export function isEyeToHandVisualization(vis: HandEyeSolveVisualization, setup?: string): boolean {
  if (setup === "eye_to_hand") return true;
  if (setup === "eye_in_hand") return false;
  const first = vis.frames[0];
  return Boolean(vis.T_base_camera_row_major && !first?.T_base_camera_row_major);
}

export function boardOriginPointsFromVisualization(
  vis: HandEyeSolveVisualization,
): { image: string; position: [number, number, number] }[] {
  const out: { image: string; position: [number, number, number] }[] = [];
  for (const f of vis.frames) {
    const p = f.board_origin_position_base_m;
    if (p && p.length === 3) {
      out.push({ image: f.image, position: [p[0], p[1], p[2]] });
    }
  }
  return out;
}
