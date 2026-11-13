/** 与 RobotScene 中 ArUco 标码一致（DICT_4X4_50，id 0–3） */

import { PNP_CHECKER_SQUARE_M } from "./pnpCheckerboard";

export const ARUCO_DICT_NAME = "DICT_4X4_50";

/** 与 RobotScene ARCUCO_MARKER_EDGE_SCALE 一致 */
export const ARCUCO_MARKER_EDGE_SCALE = 0.82;
/** 与 RobotScene ARCUCO_MARKER_SIZE_MULT 一致（3 ≈ 边长约 44 mm） */
export const ARCUCO_MARKER_SIZE_MULT = 3;

/** 标码边长（米），供 aruco_to_robot_base_cli --marker-length */
export function getSceneArucoMarkerLengthM(): number {
  return PNP_CHECKER_SQUARE_M * ARCUCO_MARKER_EDGE_SCALE * ARCUCO_MARKER_SIZE_MULT;
}

export function getSceneArucoConfig(): { dict: string; markerLengthM: number } {
  return {
    dict: ARUCO_DICT_NAME,
    markerLengthM: getSceneArucoMarkerLengthM(),
  };
}
