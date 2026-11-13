/**
 * 工具坐标系相对法兰（J6 / wrist3 连杆系）的固定偏移。
 * 约定：T_base_tool = T_base_flange · T_flange_tool；姿态 R = Rz(rz)·Ry(ry)·Rx(rx)。
 */

import * as THREE from "three";

const { degToRad } = THREE.MathUtils;

/** 法兰系下的工具偏移：位置 m，姿态 ° */
export type ToolOffset = {
  x: number;
  y: number;
  z: number;
  rxDeg: number;
  ryDeg: number;
  rzDeg: number;
};

/** 零偏移：工具系与法兰系重合（数值全 0；与 IDENTITY_TOOL_OFFSET 默认值可不同） */
export const ZERO_TOOL_OFFSET: Readonly<ToolOffset> = Object.freeze({
  x: 0,
  y: 0,
  z: 0,
  rxDeg: 0,
  ryDeg: 0,
  rzDeg: 0,
});

/**
 * 默认工具偏移（不参与末端位姿面板控制；仅场景箭头、手眼/双目记录、定位运行时等代码侧使用）。
 * 当前沿法兰局部 +Z 前移 5 cm。
 */
export const IDENTITY_TOOL_OFFSET: Readonly<ToolOffset> = Object.freeze({
  x: 0,
  y: 0,
  z: 0.05,
  rxDeg: 0,
  ryDeg: 0,
  rzDeg: 0,
});

/**
 * 定位测试「运动到该点」使用的默认运行时 TCP 偏移（相对法兰）。
 * 仅影响该动作的逆解执行 TCP，与面板基坐标控制独立。
 */
export const DEFAULT_RUNTIME_TOOL_OFFSET: Readonly<ToolOffset> = Object.freeze({
  ...IDENTITY_TOOL_OFFSET,
});

/**
 * 由工具偏移构造法兰→工具齐次矩阵 T_flange_tool。
 */
export function toolOffsetToMatrix4(offset: ToolOffset): THREE.Matrix4 {
  const r = degToRad(Number(offset.rxDeg) || 0);
  const p = degToRad(Number(offset.ryDeg) || 0);
  const y = degToRad(Number(offset.rzDeg) || 0);
  const Rx = new THREE.Matrix4().makeRotationX(r);
  const Ry = new THREE.Matrix4().makeRotationY(p);
  const Rz = new THREE.Matrix4().makeRotationZ(y);
  const M = new THREE.Matrix4().copy(Rz).multiply(Ry).multiply(Rx);
  M.setPosition(Number(offset.x) || 0, Number(offset.y) || 0, Number(offset.z) || 0);
  return M;
}

/** 数值是否均可有限（用于输入校验） */
export function toolOffsetFieldsAllFinite(offset: ToolOffset): boolean {
  const keys: (keyof ToolOffset)[] = ["x", "y", "z", "rxDeg", "ryDeg", "rzDeg"];
  for (const k of keys) {
    const v = offset[k];
    if (typeof v !== "number" || !Number.isFinite(v)) return false;
  }
  return true;
}

/** 是否可视为单位工具偏移（全零） */
export function isIdentityToolOffset(offset: ToolOffset, eps = 1e-12): boolean {
  return (
    Math.abs(offset.x) <= eps &&
    Math.abs(offset.y) <= eps &&
    Math.abs(offset.z) <= eps &&
    Math.abs(offset.rxDeg) <= eps &&
    Math.abs(offset.ryDeg) <= eps &&
    Math.abs(offset.rzDeg) <= eps
  );
}
