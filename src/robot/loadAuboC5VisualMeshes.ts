import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

/**
 * 从文件加载的连杆网格在加入场景前的统一缩放。
 * 例如 STL 以毫米为单位、仿真以米为单位时使用 `0.001`；
 * 若 glb 已是米制，可改为 `1`。
 */
export const AUBO_C5_MESH_LOAD_SCALE = 0.001;

const { degToRad } = THREE.MathUtils;

/**
 * 加载后施加在连杆根节点上的局部欧拉角（度），顺序 **XYZ**（与 Three `Object3D.rotation` 一致）。
 * 用于修正 CAD/STL 导出轴与 URDF 不一致等情况。
 */
export const AUBO_C5_MESH_LOAD_EULER_DEG = { x: 90, y: 0, z: 0 } as const;

/** link0 = base_link，link1…link6 = 各连杆（与 URDF collision 命名一致） */
export const AUBO_C5_LINK_MESH_COUNT = 7;

/** @deprecated 使用 AUBO_C5_LINK_MESH_COUNT */
export const AUBO_C5_VISUAL_COUNT = AUBO_C5_LINK_MESH_COUNT;

export type AuboC5LinkMeshes = (THREE.Object3D | null)[];

/** @deprecated 使用 AuboC5LinkMeshes */
export type AuboC5VisualMeshes = AuboC5LinkMeshes;

function collisionBaseUrl(): string {
  const base = import.meta.env.BASE_URL;
  const prefix = base.endsWith("/") ? base : `${base}/`;
  return `${prefix}aubo_C5/visual/`;
}

/**
 * Vite SPA 下缺失静态资源常返回 200 + HTML；仅靠 Loader 会误解析。
 * 用 HEAD/GET 确认是真实二进制资源后再交给 Three Loader。
 */
async function assetExists(url: string): Promise<boolean> {
  try {
    let res = await fetch(url, { method: "HEAD", cache: "force-cache" });
    if (res.status === 405 || res.status === 501) {
      res = await fetch(url, { method: "GET", cache: "force-cache" });
    }
    if (!res.ok) return false;
    const ct = (res.headers.get("content-type") ?? "").toLowerCase();
    if (ct.includes("text/html")) return false;
    return true;
  } catch {
    return false;
  }
}

function prepareForScene(root: THREE.Object3D): THREE.Object3D {
  const obj = root.clone(true);
  obj.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const m of mats) {
      const mat = m as THREE.MeshStandardMaterial & { map?: THREE.Texture };
      if (mat.map) mat.map.colorSpace = THREE.SRGBColorSpace;
    }
  });
  return obj;
}

async function tryLoadStl(url: string): Promise<THREE.Object3D | null> {
  if (!(await assetExists(url))) return null;
  try {
    const geometry = await new STLLoader().loadAsync(url);
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({
      color: 0x6b7c93,
      metalness: 0.25,
      roughness: 0.55,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    const g = new THREE.Group();
    g.name = "stl_link";
    g.add(mesh);
    return g;
  } catch {
    return null;
  }
}

async function tryLoadGltf(url: string): Promise<THREE.Object3D | null> {
  if (!(await assetExists(url))) return null;
  try {
    const gltf = await new GLTFLoader().loadAsync(url);
    return prepareForScene(gltf.scene);
  } catch {
    return null;
  }
}

function applyAuboC5MeshLoadTransform(obj: THREE.Object3D): void {
  obj.scale.setScalar(AUBO_C5_MESH_LOAD_SCALE);
  obj.rotation.set(
    degToRad(AUBO_C5_MESH_LOAD_EULER_DEG.x),
    degToRad(AUBO_C5_MESH_LOAD_EULER_DEG.y),
    degToRad(AUBO_C5_MESH_LOAD_EULER_DEG.z),
    "XYZ",
  );
}

/** 优先 glb/gltf，其次 stl；不再尝试 Collada（.dae） */
async function loadOneLink(linkIndex: number): Promise<THREE.Object3D | null> {
  const base = `${collisionBaseUrl()}link${linkIndex}`;
  const candidates: [string, (u: string) => Promise<THREE.Object3D | null>][] = [
    [".glb", tryLoadGltf],
    [".gltf", tryLoadGltf],
    [".stl", tryLoadStl],
    [".STL", tryLoadStl],
  ];
  for (const [ext, loader] of candidates) {
    const obj = await loader(base + ext);
    if (obj) {
      obj.name = `link${linkIndex}_collision`;
      applyAuboC5MeshLoadTransform(obj);
      return obj;
    }
  }
  return null;
}

/**
 * 从 `public/aubo_C5/visual/` 加载 link0 … link6；应用 {@link AUBO_C5_MESH_LOAD_SCALE} 与 {@link AUBO_C5_MESH_LOAD_EULER_DEG}。
 * 缺失文件对应项为 `null`，由 `createAuboC5Robot` 回退到占位几何。
 */
export async function loadAuboC5CollisionMeshes(): Promise<AuboC5LinkMeshes> {
  const tasks = Array.from({ length: AUBO_C5_LINK_MESH_COUNT }, (_, i) => loadOneLink(i));
  return Promise.all(tasks);
}

/** @deprecated 使用 loadAuboC5CollisionMeshes */
export async function loadAuboC5VisualMeshes(): Promise<AuboC5LinkMeshes> {
  return loadAuboC5CollisionMeshes();
}
