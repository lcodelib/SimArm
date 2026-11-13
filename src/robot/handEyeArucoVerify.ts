/** 手眼校验：aruco_to_robot_base_cli.py JSON 与运动 */

import * as THREE from "three";
import { rpyDegZYXFromRotationMatrix, type TcpPoseBase } from "./auboC5Kinematics";
import { rowMajor16ToMatrix4 } from "./auboC5JointKin";
import { getSceneArucoConfig } from "./arucoSceneConfig";
import {
  HAND_EYE_VERIFY_IMAGE,
  HAND_EYE_VERIFY_POSE_YAML,
  handEyeCameraRelPath,
} from "./handEyeCalibration";
import { normalizeCliErrorMessage } from "./subprocessText";
import type { TcpStereoEye } from "./tcpCameraConfig";

export const HAND_EYE_RESULT_YAML = "hand_eye_result.yaml";

export type ArucoVerifyMarker = {
  marker_id: number;
  T_cam_marker_row_major: number[][];
  T_base_marker_row_major: number[][];
  position_base_m: [number, number, number];
};

export type ArucoVerifySuccessPayload = {
  success: true;
  setup: string;
  markers: ArucoVerifyMarker[];
};

export type ArucoVerifyFailPayload = {
  success: false;
  error: string;
  detected_ids?: number[];
};

export type ArucoVerifyCliResponse = ArucoVerifySuccessPayload | ArucoVerifyFailPayload;

export function matrix4FromRowMajor4(T: number[][]): THREE.Matrix4 {
  const a = new Float64Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      a[r * 4 + c] = T[r][c] ?? 0;
    }
  }
  return rowMajor16ToMatrix4(a);
}

/** 标码在基座系下的位姿 → 末端目标位姿（对齐标码坐标系） */
export function tcpPoseFromBaseMarkerRowMajor(T: number[][]): TcpPoseBase {
  const M = matrix4FromRowMajor4(T);
  const e = M.elements;
  const rpy = rpyDegZYXFromRotationMatrix(M);
  return {
    x: e[12],
    y: e[13],
    z: e[14],
    rxDeg: rpy.rxDeg,
    ryDeg: rpy.ryDeg,
    rzDeg: rpy.rzDeg,
  };
}

export function parseArucoVerifyCliJson(raw: string): ArucoVerifyCliResponse {
  const line = raw.trim().split(/\r?\n/).filter(Boolean).pop() ?? raw.trim();
  const parsed = JSON.parse(line) as ArucoVerifyCliResponse;
  if (!parsed.success && parsed.error) {
    parsed.error = normalizeCliErrorMessage(parsed.error);
  }
  return parsed;
}

export function sortMarkersById(markers: ArucoVerifyMarker[]): ArucoVerifyMarker[] {
  return [...markers].sort((a, b) => a.marker_id - b.marker_id);
}

function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

async function invokeTauri<T>(cmd: string, args: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(cmd, args);
}

export type ArucoVerifyRunParams = {
  stereoEye: TcpStereoEye;
  markerLengthM: number;
  dict: string;
};

/** 调用 aruco_to_robot_base_cli（需已存在 verify.png、tcp_pose.yaml、hand_eye_result.yaml） */
export async function runHandEyeArucoVerify(params: ArucoVerifyRunParams): Promise<ArucoVerifyCliResponse> {
  const { stereoEye, markerLengthM, dict } = params;
  let rawJson: string;
  if (isTauriRuntime()) {
    rawJson = await invokeTauri<string>("hand_eye_run_aruco_verify", {
      stereoEye,
      markerLengthM,
      dictName: dict,
    });
  } else {
    const res = await fetch("/__hand_eye/aruco-verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stereoEye, markerLengthM, dict }),
    });
    const text = await res.text().catch(() => "");
    if (!res.ok) {
      throw new Error(text || `HTTP ${res.status}`);
    }
    rawJson = text;
  }
  return parseArucoVerifyCliJson(rawJson);
}

export function verifyInputPathsHint(eye: TcpStereoEye): string {
  const base = handEyeCameraRelPath(eye);
  return `${base}/${HAND_EYE_VERIFY_IMAGE}、${HAND_EYE_VERIFY_POSE_YAML}、${base}/${HAND_EYE_RESULT_YAML}`;
}

export { getSceneArucoConfig };
