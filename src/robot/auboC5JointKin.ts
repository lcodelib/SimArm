/**
 * AUBO 改进 D-H 解析正/逆运动学（与参考 C++ Joint_Kin / IKFast 参数一致）。
 * 齐次矩阵 T 为行主序 16 元：与参考代码 T[row*4+col] 相同。
 */

import * as THREE from "three";

const M_PI = Math.PI;
const ZERO_THRESH = 1e-8;

/** 与参考代码一致（示教器/连杆长度） */
export const JOINT_KIN_D1 = 0.122;
export const JOINT_KIN_A2 = 0.408;
export const JOINT_KIN_A3 = 0.376;
export const JOINT_KIN_D2 = 0.1215;
export const JOINT_KIN_D5 = 0.1025;
export const JOINT_KIN_D6 = 0.094;

/** 参考头文件中的工具偏置常数（当前 forward 公式未使用，保留备查） */
export const JOINT_OFFSET = 0.0235;

const d1 = JOINT_KIN_D1;
const a2 = JOINT_KIN_A2;
const a3 = JOINT_KIN_A3;
const d2 = JOINT_KIN_D2;
const d5 = JOINT_KIN_D5;
const d6 = JOINT_KIN_D6;

function SIGN(x: number): number {
  return x > 0 ? 1 : x < 0 ? -1 : 0;
}

function antiSinCos(sA: number, cA: number): number {
  const eps = 1e-8;
  if (Math.abs(sA) < eps && Math.abs(cA) < eps) return 0;
  if (Math.abs(cA) < eps) return (M_PI / 2) * SIGN(sA);
  if (Math.abs(sA) < eps) return SIGN(cA) === 1 ? 0 : M_PI;
  return Math.atan2(sA, cA);
}

/** 输入关节角（度），输出 16 元行主序齐次矩阵 */
export function jointKinForwardRowMajor16(qDeg: readonly number[]): Float64Array {
  const q = qDeg.map((v) => (v * M_PI) / 180);
  const q1 = q[0] ?? 0;
  const q2 = q[1] ?? 0;
  const q3 = q[2] ?? 0;
  const q4 = q[3] ?? 0;
  const q5 = q[4] ?? 0;
  const q6 = q[5] ?? 0;

  const C1 = Math.cos(q1);
  const C2 = Math.cos(q2);
  const C4 = Math.cos(q4);
  const C5 = Math.cos(q5);
  const C6 = Math.cos(q6);
  const C23 = Math.cos(q2 - q3);
  const C234 = Math.cos(q2 - q3 + q4);
  const C2345 = Math.cos(q2 - q3 + q4 - q5);
  const C2345p = Math.cos(q2 - q3 + q4 + q5);
  const S1 = Math.sin(q1);
  const S2 = Math.sin(q2);
  const S4 = Math.sin(q4);
  const S5 = Math.sin(q5);
  const S6 = Math.sin(q6);
  const S23 = Math.sin(q2 - q3);
  const S234 = Math.sin(q2 - q3 + q4);

  const T = new Float64Array(16);
  T[0] = -C6 * S1 * S5 + C1 * (C234 * C5 * C6 - S234 * S6);
  T[1] = S1 * S5 * S6 - C1 * (C4 * C6 * S23 + C23 * C6 * S4 + C234 * C5 * S6);
  T[2] = C5 * S1 + C1 * C234 * S5;
  T[3] =
    (d2 + C5 * d6) * S1 -
    C1 * (a2 * S2 + (a3 + C4 * d5) * S23 + C23 * d5 * S4 - C234 * d6 * S5);

  T[4] = C234 * C5 * C6 * S1 + C1 * C6 * S5 - S1 * S234 * S6;
  T[5] = -C6 * S1 * S234 - (C234 * C5 * S1 + C1 * S5) * S6;
  T[6] = -C1 * C5 + C234 * S1 * S5;
  T[7] =
    -C1 * (d2 + C5 * d6) -
    S1 * (a2 * S2 + (a3 + C4 * d5) * S23 + C23 * d5 * S4 - C234 * d6 * S5);

  T[8] = C5 * C6 * S234 + C234 * S6;
  T[9] = C234 * C6 - C5 * S234 * S6;
  T[10] = S234 * S5;
  T[11] = d1 + a2 * C2 + a3 * C23 + d5 * C234 + (d6 * C2345) / 2 - (d6 * C2345p) / 2;
  T[12] = 0;
  T[13] = 0;
  T[14] = 0;
  T[15] = 1;
  return T;
}

/** 行主序 16 → Three.js Matrix4（列主序） */
export function rowMajor16ToMatrix4(T: Float64Array): THREE.Matrix4 {
  const m = new THREE.Matrix4();
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      m.elements[c * 4 + r] = T[r * 4 + c];
    }
  }
  return m;
}

export function matrix4ToRowMajor16(m: THREE.Matrix4): Float64Array {
  const e = m.elements;
  const T = new Float64Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      T[r * 4 + c] = e[c * 4 + r];
    }
  }
  return T;
}

export type JointKinInverseResult = {
  numSols: number;
  /** 每组 6 个关节角（弧度），与参考 inverse 写入 q_sols 一致 */
  qSols: Float64Array;
};

/**
 * 解析逆解：输入行主序 4×4 末端位姿（与 forward 输出布局一致）。
 * 无解时 numSols=0。
 */
export function jointKinInverse(T: Float64Array | number[]): JointKinInverseResult {
  const nx = T[0];
  const ox = T[1];
  const ax = T[2];
  const px = T[3];
  const ny = T[4];
  const oy = T[5];
  const ay = T[6];
  const py = T[7];
  const nz = T[8];
  const oz = T[9];
  const az = T[10];
  const pz = T[11];

  const qSols: number[] = [];
  const q1 = [0, 0];

  const A1 = d6 * ay - py;
  const B1 = d6 * ax - px;
  const R1 = A1 * A1 + B1 * B1 - d2 * d2;

  if (R1 < 0) {
    return { numSols: 0, qSols: new Float64Array(0) };
  }

  const R12 = Math.sqrt(R1);
  q1[0] = antiSinCos(A1, B1) - antiSinCos(d2, R12);
  q1[1] = antiSinCos(A1, B1) - antiSinCos(d2, -R12);
  for (let i = 0; i < 2; i++) {
    let a = q1[i];
    while (a > M_PI) a -= 2 * M_PI;
    while (a < -M_PI) a += 2 * M_PI;
    q1[i] = a;
  }

  const q5: number[][] = [
    [0, 0],
    [0, 0],
  ];
  for (let i = 0; i < 2; i++) {
    const C1 = Math.cos(q1[i]);
    const S1 = Math.sin(q1[i]);
    const B5 = -ay * C1 + ax * S1;
    const M5 = -ny * C1 + nx * S1;
    const N5 = -oy * C1 + ox * S1;
    const R5 = Math.sqrt(M5 * M5 + N5 * N5);
    q5[i][0] = antiSinCos(R5, B5);
    q5[i][1] = antiSinCos(-R5, B5);
  }

  const q3 = [0, 0];
  const q2 = [0, 0];
  const q4 = [0, 0];

  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      const C1 = Math.cos(q1[i]);
      const S1 = Math.sin(q1[i]);
      const S5 = Math.sin(q5[i][j]);

      const A6 = -oy * C1 + ox * S1;
      const B6 = ny * C1 - nx * S1;

      if (Math.abs(S5) < ZERO_THRESH) {
        break;
      }

      const q6 = antiSinCos(A6 * S5, B6 * S5);
      const C6 = Math.cos(q6);
      const S6 = Math.sin(q6);

      const pp1 =
        C1 * (ax * d6 - px + d5 * ox * C6 + d5 * nx * S6) +
        S1 * (ay * d6 - py + d5 * oy * C6 + d5 * ny * S6);
      const pp2 = -d1 - az * d6 + pz - d5 * oz * C6 - d5 * nz * S6;
      const B3 = (pp1 * pp1 + pp2 * pp2 - a2 * a2 - a3 * a3) / (2 * a2 * a3);

      if (1 - B3 * B3 < ZERO_THRESH) {
        continue;
      }

      const Sin3 = Math.sqrt(1 - B3 * B3);
      q3[0] = antiSinCos(Sin3, B3);
      q3[1] = antiSinCos(-Sin3, B3);

      for (let k = 0; k < 2; k++) {
        const C3 = Math.cos(q3[k]);
        const S3 = Math.sin(q3[k]);
        const A2 = pp1 * (a2 + a3 * C3) + pp2 * (a3 * S3);
        const B2 = pp2 * (a2 + a3 * C3) - pp1 * (a3 * S3);

        q2[k] = antiSinCos(A2, B2);
        const C2 = Math.cos(q2[k]);
        const S2 = Math.sin(q2[k]);

        const A4 = -C1 * (ox * C6 + nx * S6) - S1 * (oy * C6 + ny * S6);
        const B4 = oz * C6 + nz * S6;
        const A41 = pp1 - a2 * S2;
        const B41 = pp2 - a2 * C2;

        q4[k] = antiSinCos(A4, B4) - antiSinCos(A41, B41);

        while (q4[k] > M_PI) q4[k] -= 2 * M_PI;
        while (q4[k] < -M_PI) q4[k] += 2 * M_PI;

        qSols.push(q1[i], q2[k], q3[k], q4[k], q5[i][j], q6);
      }
    }
  }

  return { numSols: qSols.length / 6, qSols: new Float64Array(qSols) };
}
