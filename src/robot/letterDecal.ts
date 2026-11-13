/**
 * 在已有 Mesh 表面投射字母 Decal（Three.js DecalGeometry）。
 * position / orientation 使用世界坐标系（与官方 decals 示例一致）。
 */
import * as THREE from "three";
import { DecalGeometry } from "three/examples/jsm/geometries/DecalGeometry.js";

/** Canvas 绘制带圆圈的单个大写字母贴图（透明底） */
export function createLetterTexture(letter: string, px = 256): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = px;
  canvas.height = px;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("无法创建 2D 画布用于字母贴图");
  }
  ctx.clearRect(0, 0, px, px);

  const cx = px / 2;
  const cy = px / 2;
  const ringR = px * 0.42;
  /** 圆圈线宽（加粗） */
  const strokeW = Math.max(6, Math.floor(px * 0.09));
  /** 字母描边，略粗于普通字但不至于过重 */
  const letterStrokeW = Math.max(2, Math.floor(px * 0.022));

  // 外圈蓝色圆环（圈内保持透明，不铺白底）
  ctx.beginPath();
  ctx.arc(cx, cy, ringR, 0, Math.PI * 2);
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = strokeW;
  ctx.stroke();

  // 圈内蓝色大写字母（填充 + 轻描边）
  const ch = letter.slice(0, 1).toUpperCase();
  const ty = cy + px * 0.03;
  ctx.font = `bold ${Math.floor(px * 0.52)}px Arial, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  ctx.miterLimit = 2;
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = letterStrokeW;
  ctx.strokeText(ch, cx, ty);
  ctx.fillStyle = "#2563eb";
  ctx.fillText(ch, cx, ty);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

function collectMeshes(root: THREE.Object3D): THREE.Mesh[] {
  const meshes: THREE.Mesh[] = [];
  root.traverse((o) => {
    const m = o as THREE.Mesh;
    if (m.isMesh && m.geometry) meshes.push(m);
  });
  return meshes;
}

/**
 * 从 worldOrigin 沿 worldDir 射线击中 root 下第一个网格，返回命中信息。
 */
export function raycastSurface(
  root: THREE.Object3D,
  worldOrigin: THREE.Vector3,
  worldDir: THREE.Vector3,
): { mesh: THREE.Mesh; point: THREE.Vector3; normal: THREE.Vector3 } | null {
  root.updateMatrixWorld(true);
  const meshes = collectMeshes(root);
  if (meshes.length === 0) return null;

  const raycaster = new THREE.Raycaster(worldOrigin, worldDir.clone().normalize());
  const hits = raycaster.intersectObjects(meshes, false);
  if (hits.length === 0) return null;

  const hit = hits[0];
  const face = hit.face;
  if (!face) return null;
  const mesh = hit.object as THREE.Mesh;
  const normal = face.normal.clone().transformDirection(mesh.matrixWorld).normalize();
  return { mesh, point: hit.point.clone(), normal };
}

/**
 * 在命中点生成贴图 Decal Mesh（顶点在世界系，应 scene.add）。
 * @param rollRad 绕表面法向（贴花局部 Z）的扭转角（弧度），用于旋转字母朝向
 */
export function createTextureDecalMesh(
  targetMesh: THREE.Mesh,
  worldPoint: THREE.Vector3,
  worldNormal: THREE.Vector3,
  texture: THREE.Texture,
  sizeXyM: { x: number; y: number },
  name = "TextureDecal",
  rollRad = 0,
): THREE.Mesh {
  targetMesh.updateMatrixWorld(true);

  const helper = new THREE.Object3D();
  helper.position.copy(worldPoint);
  helper.lookAt(worldPoint.clone().add(worldNormal));
  if (Math.abs(rollRad) > 1e-9) {
    helper.rotateZ(rollRad);
  }
  const orientation = helper.rotation.clone();

  const depth = Math.max(sizeXyM.x, sizeXyM.y) * 0.35;
  const geometry = new DecalGeometry(
    targetMesh,
    worldPoint,
    orientation,
    new THREE.Vector3(sizeXyM.x, sizeXyM.y, depth),
  );

  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;

  const material = new THREE.MeshStandardMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -4,
    polygonOffsetUnits: -4,
    roughness: 0.85,
    metalness: 0,
    side: THREE.DoubleSide,
  });

  const decal = new THREE.Mesh(geometry, material);
  decal.name = name;
  decal.renderOrder = 1;
  return decal;
}

/**
 * 在命中点生成字母 Decal Mesh（顶点已在世界系，应直接 scene.add，勿再挂到变形父节点下）。
 * @param rollRad 绕法向扭转（弧度）
 */
export function createLetterDecalMesh(
  targetMesh: THREE.Mesh,
  worldPoint: THREE.Vector3,
  worldNormal: THREE.Vector3,
  letter: string,
  sizeM: number,
  rollRad = 0,
): THREE.Mesh {
  return createTextureDecalMesh(
    targetMesh,
    worldPoint,
    worldNormal,
    createLetterTexture(letter),
    { x: sizeM, y: sizeM },
    `LetterDecal_${letter}`,
    rollRad,
  );
}

/**
 * 对人体：自包围盒上方沿 -法向射向躯干，返回命中；失败 null。
 */
export function raycastHumanSurface(
  humanRoot: THREE.Object3D,
  approachNormalWorld: THREE.Vector3,
  aimOffsetWorld: THREE.Vector3 = new THREE.Vector3(),
): { mesh: THREE.Mesh; point: THREE.Vector3; normal: THREE.Vector3 } | null {
  humanRoot.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(humanRoot);
  if (box.isEmpty()) return null;

  const center = box.getCenter(new THREE.Vector3()).add(aimOffsetWorld);
  const n = approachNormalWorld.clone().normalize();
  const span = box.getSize(new THREE.Vector3()).length();
  const origin = center.clone().add(n.clone().multiplyScalar(Math.max(span * 0.6, 0.3)));
  const dir = n.clone().negate();
  return raycastSurface(humanRoot, origin, dir);
}

/**
 * 对人体：自包围盒上方沿 -法向射向躯干，成功则贴一个字母；失败返回 null。
 * @param approachNormalWorld 贴花“压入”方向的反方向（表面朝外法向，如棋盘 +Z 世界方向）
 * @param aimOffsetWorld 瞄准点相对包围盒中心的世界系偏移（米），用于挪贴图位置
 * @param rollRad 绕法向扭转（弧度）
 */
export function tryAttachLetterDecalOnHuman(
  humanRoot: THREE.Object3D,
  letter: string,
  sizeM: number,
  approachNormalWorld: THREE.Vector3,
  aimOffsetWorld: THREE.Vector3 = new THREE.Vector3(),
  rollRad = 0,
): THREE.Mesh | null {
  const hit = raycastHumanSurface(humanRoot, approachNormalWorld, aimOffsetWorld);
  if (!hit) {
    console.warn(`[SimArm] 字母 Decal「${letter}」射线未命中人体表面`);
    return null;
  }

  return createLetterDecalMesh(hit.mesh, hit.point, hit.normal, letter, sizeM, rollRad);
}

/**
 * 在人体表面按网格贴多个字母（相对 aimOffsetWorld 居中）。
 * colAxisWorld：同行方向；rowAxisWorld：换行方向（与法向垂直）。
 * @param rollRads 每个字母绕法向的扭转角（弧度）；缺省为 0；长度不足时其余为 0
 */
export function tryAttachLetterDecalsOnHuman(
  humanRoot: THREE.Object3D,
  letters: string,
  sizeM: number,
  colSpacingM: number,
  rowSpacingM: number,
  colsPerRow: number,
  approachNormalWorld: THREE.Vector3,
  colAxisWorld: THREE.Vector3,
  rowAxisWorld: THREE.Vector3,
  aimOffsetWorld: THREE.Vector3 = new THREE.Vector3(),
  rollRads?: readonly number[],
): THREE.Mesh[] {
  const chars = [...letters].map((c) => c.toUpperCase()).filter((c) => /[A-Z]/.test(c));
  if (chars.length === 0) return [];

  const cols = Math.max(1, Math.floor(colsPerRow));
  const rows = Math.ceil(chars.length / cols);

  const n = approachNormalWorld.clone().normalize();

  const projectOnPlane = (v: THREE.Vector3): THREE.Vector3 => {
    const out = v.clone();
    out.addScaledVector(n, -out.dot(n));
    if (out.lengthSq() < 1e-10) {
      out.crossVectors(n, Math.abs(n.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0));
    }
    return out.normalize();
  };

  const colAxis = projectOnPlane(colAxisWorld);
  let rowAxis = projectOnPlane(rowAxisWorld);
  // 若行列几乎平行，用法向×列向补一行方向
  if (Math.abs(rowAxis.dot(colAxis)) > 0.95) {
    rowAxis = new THREE.Vector3().crossVectors(n, colAxis).normalize();
  }

  const out: THREE.Mesh[] = [];
  const midCol = (cols - 1) * 0.5;
  const midRow = (rows - 1) * 0.5;

  for (let i = 0; i < chars.length; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const aim = aimOffsetWorld
      .clone()
      .addScaledVector(colAxis, (col - midCol) * colSpacingM)
      .addScaledVector(rowAxis, (row - midRow) * rowSpacingM);
    const roll = rollRads?.[i] ?? 0;
    const decal = tryAttachLetterDecalOnHuman(
      humanRoot,
      chars[i],
      sizeM,
      approachNormalWorld,
      aim,
      roll,
    );
    if (decal) out.push(decal);
  }
  return out;
}

/** 为每个字母生成 [0, 2π) 均匀随机扭转角 */
export function randomLetterRollRads(letterCount: number): number[] {
  const n = Math.max(0, Math.floor(letterCount));
  return Array.from({ length: n }, () => Math.random() * Math.PI * 2);
}
