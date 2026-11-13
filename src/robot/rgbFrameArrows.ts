import * as THREE from "three";

/**
 * 坐标系箭头专用层。末端离屏相机不要 `enable` 该层，则所有箭头不会出现在末端画面中。
 */
export const FRAME_ARROWS_RENDER_LAYER = 2;

/** 右手系：X 红、Y 绿、Z 蓝 */
export function createRgbFrameArrows(
  axisLength: number,
  headLength: number,
  headRadius: number,
  origin?: THREE.Vector3,
  name = "FrameXYZ",
  renderLayer: number = FRAME_ARROWS_RENDER_LAYER,
): THREE.Group {
  const o = origin ?? new THREE.Vector3(0, 0, 0);
  const group = new THREE.Group();
  group.name = name;
  const specs: readonly [THREE.Vector3, number][] = [
    [new THREE.Vector3(1, 0, 0), 0xff3344],
    [new THREE.Vector3(0, 1, 0), 0x33dd55],
    [new THREE.Vector3(0, 0, 1), 0x3399ff],
  ];
  for (const [dir, color] of specs) {
    group.add(
      new THREE.ArrowHelper(
        dir.clone().normalize(),
        o.clone(),
        axisLength,
        color,
        headLength,
        headRadius,
      ),
    );
  }
  group.traverse((obj) => {
    obj.layers.set(renderLayer);
  });
  return group;
}
