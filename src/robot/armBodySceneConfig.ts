/**
 * 机械臂底座座体 armbody.glb：挂在 base_link 下、贴于基座底面。
 */
import * as THREE from "three";

/** public/ 下相对路径 */
export const ARMBODY_GLB_REL = "armbody.glb";

/**
 * 单位换算到场景米制（与 aubo 连杆常见 mm→m 一致用 `0.001`；若已是米则改为 `1`）。
 */
export const ARMBODY_MESH_LOAD_SCALE = 0.1;

/**
 * 相对底座局部系的欧拉角（度，XYZ）。
 * 若 CAD 轴向与 URDF 底座不一致可改此处。
 */
export const ARMBODY_LOCAL_EULER_DEG = { x: 90, y: -90, z: 0 } as const;

/**
 * 相对基座原点的额外平移（米，底座局部系）。
 * 贴底后再加此偏移；「底座下」默认 z 向由包围盒自动贴齐。
 */
export const ARMBODY_LOCAL_OFFSET_M = new THREE.Vector3(0.3, 0, 0.1);

/**
 * 将已缩放模型贴到底座原点下方：包围盒顶面与局部 z=0 对齐，再加 offset。
 * 底座局部 +Z 为 URDF 上方向（经 root.rotation.x 后对应世界 +Y）。
 */
export function placeArmBodyUnderBase(
  obj: THREE.Object3D,
  eulerDeg: { x: number; y: number; z: number },
  offsetM: THREE.Vector3,
): void {
  const { degToRad } = THREE.MathUtils;
  obj.rotation.set(degToRad(eulerDeg.x), degToRad(eulerDeg.y), degToRad(eulerDeg.z), "XYZ");
  obj.position.set(0, 0, 0);
  obj.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(obj);
  const cx = (box.min.x + box.max.x) * 0.5;
  const cy = (box.min.y + box.max.y) * 0.5;
  // 顶面贴 z=0，整体在底座下方
  obj.position.set(offsetM.x - cx, offsetM.y - cy, offsetM.z - box.max.z);
}

/**
 * 抬高机械臂根节点，使 armbody 底面落在世界地面（y=0），基座仍压在 armbody 顶上。
 * 须在 armbody 已加入 root（与 yawGroup 并列）之后调用。仅改场景显示位姿，不影响基座系正逆解数值。
 */
export function liftRobotRootSoArmBodyOnGround(robotRoot: THREE.Object3D, armBody: THREE.Object3D): void {
  robotRoot.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(armBody);
  if (!Number.isFinite(box.min.y)) return;
  robotRoot.position.y -= box.min.y;
}
