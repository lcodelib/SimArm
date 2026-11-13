import * as THREE from "three";
import { createRgbFrameArrows } from "./rgbFrameArrows";
import type { AuboC5LinkMeshes } from "./loadAuboC5VisualMeshes";

const PI = Math.PI;
const { degToRad } = THREE.MathUtils;

/**
 * 末端相机挂载点相对 **J6 旋转连杆系原点**（与 TCP 三色箭头同一父节点、同一局部坐标系）的平移，单位 m。
 * 沿该连杆局部 X/Y/Z 偏移，例如 `{ x: 0, y: 0, z: 0.02 }` 为沿局部 +Z 前移 2cm。
 */
export const EE_CAMERA_MOUNT_OFFSET_M = { x: 0, y: -0.05, z: 0 } as const;

/**
 * 挂载点相对上述原点的姿态微调：欧拉 **XYZ**、单位度（与 Three `Object3D.rotation` 一致）。
 * 默认绕 X 为 180°：使透视相机视线大致沿连杆 +Z（相机默认朝局部 −Z 看）。
 */
export const EE_CAMERA_MOUNT_EULER_OFFSET_DEG = { x: 180, y: 0, z: 0 } as const;

/**
 * 机械臂（不含 armbody）绕**世界竖直轴 +Y** 的偏航角（度）。
 * 实现为 root（已 rx=-90°）下 yawGroup 的局部 -Y 旋转。
 */
export const ROBOT_BASE_YAW_WORLD_DEG = -45;

/** 关节原点 xyz(米)、rpy(弧度)，与 AuboRobot/aubo_description 中 aubo_C5.urdf 一致 */
export type JointSpec = {
  name: string;
  minDeg: number;
  maxDeg: number;
  urdfOrigin: {
    xyz: [number, number, number];
    rpy: [number, number, number];
  };
};

/** 转角范围（度）来自 AUBO-C5 技术手册 Joint Motion Parameters */
export const AUBO_C5_JOINT_SPECS: JointSpec[] = [
  {
    name: "J1 基座",
    minDeg: -360,
    maxDeg: 360,
    urdfOrigin: { xyz: [0, 0, 0.122], rpy: [0, 0, PI] },
  },
  {
    name: "J2 肩部",
    minDeg: -360,
    maxDeg: 360,
    urdfOrigin: { xyz: [0, 0.1215, 0], rpy: [-PI / 2, -PI / 2, 0] },
  },
  {
    name: "J3 肘部",
    minDeg: -162,
    maxDeg: 162,
    urdfOrigin: { xyz: [0.408, 0, 0], rpy: [-PI, 0, 0] },
  },
  {
    name: "J4 腕部 1",
    minDeg: -360,
    maxDeg: 360,
    urdfOrigin: { xyz: [0.376, 0, 0], rpy: [PI, 0, PI / 2] },
  },
  {
    name: "J5 腕部 2",
    minDeg: -360,
    maxDeg: 360,
    urdfOrigin: { xyz: [0, 0.1025, 0], rpy: [-PI / 2, 0, 0] },
  },
  {
    name: "J6 法兰",
    minDeg: -360,
    maxDeg: 360,
    urdfOrigin: { xyz: [0, -0.094, 0], rpy: [PI / 2, 0, 0] },
  },
];

/**
 * URDF 关节/连杆 origin 的 rpy：绕**父连杆固定坐标系**的 X→Y→Z 依次旋转（外旋，与 ROS urdfdom 一致）。
 * 等价旋转矩阵 R = Rz(yaw)*Ry(pitch)*Rx(roll)，其中 roll=rpy[0], pitch=rpy[1], yaw=rpy[2]。
 */
function applyUrdfOrigin(group: THREE.Group, spec: JointSpec["urdfOrigin"]): void {
  const [px, py, pz] = spec.xyz;
  const [roll, pitch, yaw] = spec.rpy;
  group.position.set(px, py, pz);
  const Rx = new THREE.Matrix4().makeRotationX(roll);
  const Ry = new THREE.Matrix4().makeRotationY(pitch);
  const Rz = new THREE.Matrix4().makeRotationZ(yaw);
  const R = new THREE.Matrix4().copy(Rz).multiply(Ry).multiply(Rx);
  group.quaternion.setFromRotationMatrix(R);
}

function linkMaterial(color: number): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color,
    metalness: 0.25,
    roughness: 0.55,
  });
}

/** 圆柱几何默认沿 +Y，旋转后沿 +Z，与 URDF revolute axis 0 0 1 一致 */
function alignMeshCylinderToLocalZ(mesh: THREE.Mesh): void {
  mesh.rotation.x = PI / 2;
}

/** 无 `public/aubo_C5/collision` 网格时的占位几何 */
function addLinkGeometryFallback(linkIndex: number, parent: THREE.Group): void {
  const arm = linkMaterial(0x3d5a80);
  const joint = linkMaterial(0xe76f51);
  const accent = linkMaterial(0x2a9d8f);

  switch (linkIndex) {
    case 0: {
      const g = new THREE.CylinderGeometry(0.07, 0.075, 0.11, 28);
      const m = new THREE.Mesh(g, joint);
      alignMeshCylinderToLocalZ(m);
      m.castShadow = true;
      parent.add(m);
      break;
    }
    case 1: {
      const len = 0.408;
      const g = new THREE.BoxGeometry(len, 0.09, 0.09);
      const m = new THREE.Mesh(g, arm);
      m.position.set(len / 2, 0, 0);
      m.castShadow = true;
      parent.add(m);
      break;
    }
    case 2: {
      const len = 0.376;
      const g = new THREE.BoxGeometry(len, 0.08, 0.08);
      const m = new THREE.Mesh(g, arm);
      m.position.set(len / 2, 0, 0);
      m.castShadow = true;
      parent.add(m);
      break;
    }
    case 3: {
      const h = 0.1025;
      const g = new THREE.BoxGeometry(0.07, h, 0.07);
      const m = new THREE.Mesh(g, accent);
      m.position.set(0, h / 2, 0);
      m.castShadow = true;
      parent.add(m);
      break;
    }
    case 4: {
      const h = 0.094;
      const g = new THREE.BoxGeometry(0.06, h, 0.06);
      const m = new THREE.Mesh(g, accent);
      m.position.set(0, -h / 2, 0);
      m.castShadow = true;
      parent.add(m);
      break;
    }
    case 5: {
      const g = new THREE.CylinderGeometry(0.038, 0.038, 0.045, 24);
      const m = new THREE.Mesh(g, joint);
      alignMeshCylinderToLocalZ(m);
      m.castShadow = true;
      parent.add(m);
      const tip = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 16, 16),
        linkMaterial(0xf4a261),
      );
      tip.position.set(0, 0, 0);
      tip.castShadow = true;
      parent.add(tip);
      // 工具系箭头由 createAuboC5Robot 的 toolFrameMount 统一挂载
      break;
    }
    default:
      break;
  }
}

function addTcpMarkers(parent: THREE.Group, withProbeSphere: boolean): void {
  if (withProbeSphere) {
    const tip = new THREE.Mesh(
      new THREE.SphereGeometry(0.018, 16, 16),
      linkMaterial(0xf4a261),
    );
    tip.position.set(0, 0, 0);
    tip.castShadow = true;
    parent.add(tip);
  }
  // 法兰原点小箭头（工具零偏移时与工具系重合）
  const flangeFrame = createRgbFrameArrows(0.04, 0.014, 0.01, new THREE.Vector3(0, 0, 0), "FlangeFrameXYZ");
  parent.add(flangeFrame);
}

/**
 * 将工具偏移应用到挂载点（法兰局部系 → 工具系，与运动学 T_flange_tool 一致）。
 */
export function applyToolOffsetToMount(
  mount: THREE.Object3D,
  offset: {
    x: number;
    y: number;
    z: number;
    rxDeg: number;
    ryDeg: number;
    rzDeg: number;
  },
): void {
  const r = degToRad(offset.rxDeg);
  const p = degToRad(offset.ryDeg);
  const y = degToRad(offset.rzDeg);
  const Rx = new THREE.Matrix4().makeRotationX(r);
  const Ry = new THREE.Matrix4().makeRotationY(p);
  const Rz = new THREE.Matrix4().makeRotationZ(y);
  const R = new THREE.Matrix4().copy(Rz).multiply(Ry).multiply(Rx);
  mount.quaternion.setFromRotationMatrix(R);
  mount.position.set(offset.x, offset.y, offset.z);
}

/**
 * @param linkMeshes `public/aubo_C5/collision/` 下 link0…link6；`null` 项用占位几何。缺省或全 null 则全占位。
 */
export function createAuboC5Robot(linkMeshes?: AuboC5LinkMeshes | null): {
  root: THREE.Group;
  jointRotationGroups: THREE.Group[];
  /** 基座连杆（base_link）；在 yawGroup 下，会随 {@link ROBOT_BASE_YAW_WORLD_DEG} 转动 */
  baseLink: THREE.Group;
  /** 偏航组（不含 armbody）；armbody 应挂在 root 下与本节点并列 */
  yawGroup: THREE.Group;
  /** 末端相机挂载点（相对 J6 连杆系，见 `EE_CAMERA_MOUNT_*` 常量） */
  eeCameraMount: THREE.Object3D;
  /** 工具坐标系挂载点（相对 J6 法兰系，由工具偏移驱动） */
  toolFrameMount: THREE.Object3D;
} {
  const meshes = linkMeshes ?? null;

  const root = new THREE.Group();
  root.name = "AuboC5Root";
  // URDF Z-up → Three Y-up
  root.rotation.x = -PI / 2;

  /** 绕世界 +Y：在 root 局部即绕 -Y */
  const yawGroup = new THREE.Group();
  yawGroup.name = "RobotYawGroup";
  yawGroup.rotation.y = degToRad(ROBOT_BASE_YAW_WORLD_DEG);
  root.add(yawGroup);

  const jointRotationGroups: THREE.Group[] = [];
  const eeCameraMount = new THREE.Object3D();
  eeCameraMount.name = "EeCameraMount";
  eeCameraMount.position.set(EE_CAMERA_MOUNT_OFFSET_M.x, EE_CAMERA_MOUNT_OFFSET_M.y, EE_CAMERA_MOUNT_OFFSET_M.z);
  eeCameraMount.rotation.set(
    degToRad(EE_CAMERA_MOUNT_EULER_OFFSET_DEG.x),
    degToRad(EE_CAMERA_MOUNT_EULER_OFFSET_DEG.y),
    degToRad(EE_CAMERA_MOUNT_EULER_OFFSET_DEG.z),
    "XYZ",
  );

  /** 工具 TCP 箭头挂载：相对法兰的 T_flange_tool（R=Rz·Ry·Rx） */
  const toolFrameMount = new THREE.Object3D();
  toolFrameMount.name = "ToolFrameMount";
  const toolFrame = createRgbFrameArrows(0.07, 0.025, 0.016, new THREE.Vector3(0, 0, 0), "TcpFrameXYZ");
  toolFrameMount.add(toolFrame);

  const baseLink = new THREE.Group();
  baseLink.name = "BaseLink";
  yawGroup.add(baseLink);

  const link0 = meshes?.[0];
  if (link0) {
    baseLink.add(link0);
  } else {
    const baseCyl = new THREE.Mesh(
      new THREE.CylinderGeometry(0.09, 0.095, 0.08, 32),
      linkMaterial(0x264653),
    );
    alignMeshCylinderToLocalZ(baseCyl);
    baseCyl.castShadow = true;
    baseLink.add(baseCyl);
  }

  const baseFrame = createRgbFrameArrows(
    0.38,
    0.068,
    0.032,
    new THREE.Vector3(0, 0, 0.052),
    "RobotBaseFrameXYZ",
  );
  baseLink.add(baseFrame);

  let parent: THREE.Object3D = baseLink;

  for (let i = 0; i < 6; i++) {
    const spec = AUBO_C5_JOINT_SPECS[i];
    const originGroup = new THREE.Group();
    applyUrdfOrigin(originGroup, spec.urdfOrigin);
    parent.add(originGroup);

    const rotGroup = new THREE.Group();
    originGroup.add(rotGroup);
    jointRotationGroups.push(rotGroup);

    const linkMesh = meshes?.[i + 1] ?? null;
    if (linkMesh) {
      rotGroup.add(linkMesh);
      if (i === 5) {
        addTcpMarkers(rotGroup, false);
      }
    } else {
      addLinkGeometryFallback(i, rotGroup);
    }

    /* 各关节连杆坐标系箭头（按需取消注释）
    rotGroup.add(
      createRgbFrameArrows(0.042, 0.014, 0.009, new THREE.Vector3(0, 0, 0), `Joint${i + 1}LinkFrame`),
    );
    */

    if (i === 5) {
      rotGroup.add(eeCameraMount);
      rotGroup.add(toolFrameMount);
    }

    parent = rotGroup;
  }

  return { root, jointRotationGroups, baseLink, yawGroup, eeCameraMount, toolFrameMount };
}

export function setJointAnglesDeg(
  jointRotationGroups: THREE.Group[],
  anglesDeg: readonly number[],
): void {
  for (let i = 0; i < jointRotationGroups.length; i++) {
    const deg = anglesDeg[i] ?? 0;
    /**
     * 约定：沿关节局部 +Z 轴、从 +Z 指向原点看去，角度 + 为顺时针，− 为逆时针。
     * Three 绕 +Z 的右手正角在该视角下为逆时针，故此处取负与界面约定一致。
     */
    jointRotationGroups[i].rotation.z = -THREE.MathUtils.degToRad(deg);
  }
}
