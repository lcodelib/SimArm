/**
 * 场景 PLY 加载（如 skin_surface.ply：三角网格或点云）。
 */
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

/**
 * 从 URL 加载 PLY：有 index 则为 Mesh，否则为 Points。
 * 失败返回 null。
 */
export async function loadScenePly(url: string): Promise<THREE.Object3D | null> {
  try {
    const geometry = await new PLYLoader().loadAsync(url);
    if (!geometry) return null;

    geometry.computeVertexNormals();

    const root = new THREE.Group();
    root.name = "PlyRoot";

    const hasIndex = geometry.index != null && geometry.index.count > 0;
    if (hasIndex) {
      const mat = new THREE.MeshStandardMaterial({
        color: 0xc4a484,
        metalness: 0.05,
        roughness: 0.75,
        side: THREE.DoubleSide,
        flatShading: false,
      });
      const mesh = new THREE.Mesh(geometry, mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.name = "PlyMesh";
      root.add(mesh);
    } else {
      const mat = new THREE.PointsMaterial({
        color: 0xc4a484,
        size: 0.002,
        sizeAttenuation: true,
      });
      const points = new THREE.Points(geometry, mat);
      points.name = "PlyPoints";
      root.add(points);
    }
    return root;
  } catch (err) {
    console.warn("[SimArm] 加载 PLY 失败:", url, err);
    return null;
  }
}
