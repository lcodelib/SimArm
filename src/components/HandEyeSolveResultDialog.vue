<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from "vue";
import * as THREE from "three";
import { ArrowHelper } from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  boardOriginPointsFromVisualization,
  HAND_EYE_AXIS_COLORS,
  handEyeGroupColorHex,
  isEyeToHandVisualization,
  type HandEyeSolveCliResponse,
  type HandEyeSolveVisualization,
} from "../robot/handEyeSolveTypes";
import { normalizeCliErrorMessage } from "../robot/subprocessText";

const props = defineProps<{
  open: boolean;
  parsed: HandEyeSolveCliResponse | null;
  rawJson: string;
}>();

const emit = defineEmits<{
  close: [];
}>();

const canvasRef = ref<HTMLCanvasElement | null>(null);

let renderer: THREE.WebGLRenderer | null = null;
let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let controls: OrbitControls | null = null;
let vizGroup: THREE.Group | null = null;
let raf = 0;

const successPayload = computed(() =>
  props.parsed?.success === true ? props.parsed : null,
);

const isSuccess = computed(() => successPayload.value != null);

const failMessage = computed(() => {
  const p = props.parsed;
  if (!p || p.success) return "未知错误";
  return normalizeCliErrorMessage(p.error);
});

const rmseM = computed(() => {
  const p = successPayload.value;
  if (!p) return null;
  const v = p.board_origin_rmse_m;
  return v == null || Number.isNaN(Number(v)) ? null : Number(v);
});

const visualization = computed((): HandEyeSolveVisualization | null => successPayload.value?.visualization ?? null);

const boardPoints = computed(() => {
  const vis = visualization.value;
  if (!vis) return [];
  return boardOriginPointsFromVisualization(vis);
});

const rmseMm = computed(() => (rmseM.value == null ? null : rmseM.value * 1000));

const eyeToHand = computed(() => {
  const p = successPayload.value;
  if (!p?.visualization) return false;
  return isEyeToHandVisualization(p.visualization, p.setup);
});

const poseFrameCount = computed(() => visualization.value?.frames.length ?? 0);

const frameScaleM = computed(() => {
  const v = visualization.value?.frame_scale_m;
  return v != null && Number.isFinite(v) && v > 0 ? v : 0.04;
});

const frameLegend = computed(() => {
  const vis = visualization.value;
  if (!vis) return [];
  const n = vis.frames.length;
  return vis.frames.map((f, k) => ({
    image: f.image,
    color: `#${handEyeGroupColorHex(k, n).toString(16).padStart(6, "0")}`,
    index: k + 1,
  }));
});

function originFromT(T: number[][]): THREE.Vector3 {
  return new THREE.Vector3(T[0][3], T[1][3], T[2][3]);
}

/** 在 T 原点处绘制 X/Y/Z 三色箭头（列向量为轴方向） */
function addRgbFrameAxes(
  parent: THREE.Object3D,
  T: number[][],
  axisLen: number,
  colors: readonly [number, number, number],
  opacity: number,
): THREE.Vector3 {
  const origin = originFromT(T);
  const headLen = axisLen * 0.2;
  const headW = axisLen * 0.12;
  for (let col = 0; col < 3; col++) {
    const dir = new THREE.Vector3(T[0][col], T[1][col], T[2][col]);
    if (dir.lengthSq() < 1e-12) continue;
    dir.normalize();
    const arrow = new ArrowHelper(dir, origin, axisLen, colors[col], headLen, headW);
    const lineMat = arrow.line.material as THREE.LineBasicMaterial;
    lineMat.transparent = true;
    lineMat.opacity = opacity;
    lineMat.depthTest = true;
    const coneMat = arrow.cone.material as THREE.MeshBasicMaterial;
    coneMat.transparent = true;
    coneMat.opacity = opacity;
    parent.add(arrow);
  }
  return origin;
}

function addOriginSphere(
  parent: THREE.Object3D,
  position: THREE.Vector3,
  color: number,
  radius: number,
): void {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 18, 18),
    new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.22,
      metalness: 0.15,
      roughness: 0.42,
    }),
  );
  mesh.position.copy(position);
  parent.add(mesh);
}

function collectBoundsPoints(vis: HandEyeSolveVisualization, eyeToHandMode: boolean): THREE.Vector3[] {
  const pts: THREE.Vector3[] = [new THREE.Vector3(0, 0, 0)];
  const n = vis.frames.length;
  for (let k = 0; k < n; k++) {
    const f = vis.frames[k];
    if (f.T_base_tcp_row_major) pts.push(originFromT(f.T_base_tcp_row_major));
    if (!eyeToHandMode && f.T_base_camera_row_major) {
      pts.push(originFromT(f.T_base_camera_row_major));
    }
    const p = f.board_origin_position_base_m;
    if (p?.length === 3) pts.push(new THREE.Vector3(p[0], p[1], p[2]));
  }
  if (eyeToHandMode && vis.T_base_camera_row_major) {
    pts.push(originFromT(vis.T_base_camera_row_major));
  }
  return pts;
}

function disposeVizGroup(): void {
  if (!vizGroup) return;
  vizGroup.traverse((obj) => {
    if (obj instanceof ArrowHelper) {
      obj.dispose();
      return;
    }
    if (obj instanceof THREE.Mesh || obj instanceof THREE.Points || obj instanceof THREE.Line) {
      obj.geometry?.dispose();
      const mat = obj.material;
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
      else mat?.dispose();
    }
  });
  scene?.remove(vizGroup);
  vizGroup = null;
}

function buildVisualization(): void {
  if (!scene || !isSuccess.value || !visualization.value) return;
  disposeVizGroup();
  vizGroup = new THREE.Group();

  const vis = visualization.value;
  const eyeToHandMode = eyeToHand.value;
  const fs = frameScaleM.value;
  const frames = vis.frames;
  const n = frames.length;

  addRgbFrameAxes(vizGroup, vis.T_base_row_major, fs * 2, HAND_EYE_AXIS_COLORS.base, 1);

  const tcpSphereR = Math.max(fs * 0.14, 0.005);
  const camSphereR = Math.max(fs * 0.1, 0.004);

  for (let k = 0; k < n; k++) {
    const alpha = 0.35 + 0.5 * (n <= 1 ? 1 : k / (n - 1));
    const f = frames[k];
    const groupColor = handEyeGroupColorHex(k, n);

    if (f.T_base_tcp_row_major) {
      addRgbFrameAxes(vizGroup, f.T_base_tcp_row_major, fs * 1.1, HAND_EYE_AXIS_COLORS.tcp, alpha);
      addOriginSphere(vizGroup, originFromT(f.T_base_tcp_row_major), groupColor, tcpSphereR);
    }
    if (!eyeToHandMode && f.T_base_camera_row_major) {
      const camAlpha = 0.4 + 0.45 * (n <= 1 ? 1 : k / (n - 1));
      addRgbFrameAxes(vizGroup, f.T_base_camera_row_major, fs * 0.95, HAND_EYE_AXIS_COLORS.camera, camAlpha);
      addOriginSphere(vizGroup, originFromT(f.T_base_camera_row_major), groupColor, camSphereR);
    }
  }

  if (eyeToHandMode && vis.T_base_camera_row_major) {
    addRgbFrameAxes(
      vizGroup,
      vis.T_base_camera_row_major,
      fs * 1.5,
      HAND_EYE_AXIS_COLORS.cameraFixed,
      1,
    );
    addOriginSphere(
      vizGroup,
      originFromT(vis.T_base_camera_row_major),
      0xffc107,
      Math.max(fs * 0.16, 0.006),
    );
  }

  const boardPts = boardPoints.value;
  if (boardPts.length > 0) {
    const positions = new Float32Array(boardPts.length * 3);
    for (let i = 0; i < boardPts.length; i++) {
      positions[i * 3] = boardPts[i].position[0];
      positions[i * 3 + 1] = boardPts[i].position[1];
      positions[i * 3 + 2] = boardPts[i].position[2];
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const cloud = new THREE.Points(
      geom,
      new THREE.PointsMaterial({
        color: 0x8b4513,
        size: Math.max(fs * 0.35, 0.01),
        sizeAttenuation: true,
      }),
    );
    vizGroup.add(cloud);
  }

  scene.add(vizGroup);

  const boundsPts = collectBoundsPoints(vis, eyeToHandMode);
  if (boundsPts.length > 0 && camera && controls) {
    const box = new THREE.Box3();
    for (const p of boundsPts) box.expandByPoint(p);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    const dist = Math.max(size * 1.8, 0.35);
    camera.position.set(center.x + dist * 0.6, center.y + dist * 0.45, center.z + dist * 0.65);
    controls.target.copy(center);
    controls.update();
  }
}

function initScene(): void {
  const canvas = canvasRef.value;
  if (!canvas || !props.open || !isSuccess.value) return;

  const w = canvas.clientWidth || 480;
  const h = canvas.clientHeight || 320;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf1f5f9);

  camera = new THREE.PerspectiveCamera(45, w / h, 0.001, 50);
  camera.position.set(0.45, 0.35, 0.5);

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w, h, false);

  const hemi = new THREE.HemisphereLight(0xffffff, 0xcbd5e1, 0.85);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff, 0.9);
  dir.position.set(2, 3, 2);
  scene.add(dir);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;

  buildVisualization();
  tick();
}

function tick(): void {
  raf = requestAnimationFrame(tick);
  controls?.update();
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
}

function teardownScene(): void {
  cancelAnimationFrame(raf);
  disposeVizGroup();
  controls?.dispose();
  renderer?.dispose();
  controls = null;
  renderer = null;
  scene = null;
  camera = null;
}

function onOverlayClick(e: MouseEvent): void {
  if ((e.target as HTMLElement).classList.contains("he-dialog-overlay")) {
    emit("close");
  }
}

watch(
  () => [props.open, props.parsed] as const,
  ([open]) => {
    teardownScene();
    if (open && isSuccess.value) {
      requestAnimationFrame(() => initScene());
    }
  },
);

onUnmounted(() => {
  teardownScene();
});
</script>

<template>
  <Teleport to="body">
    <div v-if="open" class="he-dialog-overlay" @click="onOverlayClick">
      <div class="he-dialog" role="dialog" aria-modal="true" aria-labelledby="he-dialog-title" @click.stop>
        <header class="he-dialog-head">
          <h2 id="he-dialog-title">{{ isSuccess ? "手眼标定结果" : "手眼标定失败" }}</h2>
          <button type="button" class="he-close" aria-label="关闭" @click="emit('close')">×</button>
        </header>

        <div v-if="!isSuccess" class="he-fail">
          <p class="he-fail-msg">{{ failMessage }}</p>
          <details class="he-json-details" open>
            <summary>JSON 响应</summary>
            <pre class="he-json-pre">{{ rawJson }}</pre>
          </details>
        </div>

        <div v-else class="he-success">
          <div class="he-metrics">
            <div class="he-metric he-metric-rmse">
              <span class="he-metric-label">棋盘原点 RMSE</span>
              <span class="he-metric-value">
                <template v-if="rmseM != null">{{ rmseM.toExponential(3) }} m（{{ rmseMm?.toFixed(3) }} mm）</template>
                <template v-else>—</template>
              </span>
            </div>
            <div class="he-metric">
              <span class="he-metric-label">算法</span>
              <span class="he-metric-value">{{ successPayload?.hand_eye_method_used ?? "—" }}</span>
            </div>
            <div class="he-metric">
              <span class="he-metric-label">位姿组数</span>
              <span class="he-metric-value">{{ successPayload?.num_poses ?? poseFrameCount }}</span>
            </div>
          </div>

          <p class="he-viz-hint">
            RGB 轴为坐标系方向；同色圆球：大球 = TCP 原点，小球 = 相机原点（同组同色）；
            <template v-if="eyeToHand">固定相机为金色大球</template>
            <template v-else>棋盘原点为棕色点</template>
          </p>
          <canvas ref="canvasRef" class="he-viz-canvas" />

          <ul v-if="frameLegend.length" class="he-legend-list">
            <li v-for="item in frameLegend" :key="item.image">
              <span class="he-swatch" :style="{ background: item.color }" />
              #{{ item.index }} {{ item.image }}
              <span class="he-legend-sub">TCP 大球 · 相机小球</span>
            </li>
          </ul>

          <ul v-if="boardPoints.length" class="he-point-list">
            <li v-for="(p, i) in boardPoints" :key="p.image">
              #{{ i + 1 }} 棋盘 {{ p.image }} — ({{ p.position.map((v) => v.toFixed(4)).join(", ") }}) m
            </li>
          </ul>

          <details class="he-json-details">
            <summary>JSON 响应</summary>
            <pre class="he-json-pre">{{ rawJson }}</pre>
          </details>
        </div>

        <footer class="he-dialog-foot">
          <button type="button" class="he-ok" @click="emit('close')">关闭</button>
        </footer>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.he-dialog-overlay {
  position: fixed;
  inset: 0;
  z-index: 2000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
  background: rgba(15, 23, 42, 0.45);
}

.he-dialog {
  width: min(720px, 100%);
  max-height: min(92vh, 900px);
  display: flex;
  flex-direction: column;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 20px 50px rgba(15, 23, 42, 0.25);
  overflow: hidden;
}

.he-dialog-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.85rem 1rem;
  border-bottom: 1px solid #e2e8f0;
}

.he-dialog-head h2 {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: #0f172a;
}

.he-close {
  border: none;
  background: transparent;
  font-size: 1.4rem;
  line-height: 1;
  color: #64748b;
  cursor: pointer;
  padding: 0 0.25rem;
}

.he-fail,
.he-success {
  padding: 0.85rem 1rem;
  overflow: auto;
  flex: 1;
}

.he-fail-msg {
  margin: 0 0 0.75rem;
  color: #b91c1c;
  font-size: 0.9rem;
  line-height: 1.45;
}

.he-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.5rem;
  margin-bottom: 0.65rem;
}

.he-metric {
  padding: 0.45rem 0.55rem;
  border-radius: 8px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
}

.he-metric-rmse {
  background: #eff6ff;
  border-color: #bfdbfe;
}

.he-metric-label {
  display: block;
  font-size: 0.68rem;
  color: #64748b;
  margin-bottom: 0.15rem;
}

.he-metric-value {
  font-size: 0.82rem;
  font-weight: 600;
  color: #0f172a;
  word-break: break-all;
}

.he-viz-hint {
  margin: 0 0 0.35rem;
  font-size: 0.72rem;
  color: #64748b;
}

.he-viz-canvas {
  display: block;
  width: 100%;
  height: 320px;
  border-radius: 8px;
  border: 1px solid #cbd5e1;
  touch-action: none;
}

.he-legend-list {
  margin: 0.5rem 0 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 0.75rem;
  font-size: 0.68rem;
  color: #334155;
  max-height: 88px;
  overflow: auto;
}

.he-legend-list li {
  display: flex;
  align-items: center;
  gap: 0.35rem;
}

.he-swatch {
  width: 0.65rem;
  height: 0.65rem;
  border-radius: 50%;
  border: 1px solid rgba(15, 23, 42, 0.2);
  flex-shrink: 0;
}

.he-legend-sub {
  color: #94a3b8;
  font-size: 0.62rem;
}

.he-point-list {
  margin: 0.55rem 0 0;
  padding-left: 1.1rem;
  font-size: 0.68rem;
  color: #475569;
  max-height: 120px;
  overflow: auto;
}

.he-json-details {
  margin-top: 0.65rem;
  font-size: 0.75rem;
}

.he-json-details summary {
  cursor: pointer;
  color: #0369a1;
  font-weight: 600;
}

.he-json-pre {
  margin: 0.4rem 0 0;
  padding: 0.5rem;
  max-height: 200px;
  overflow: auto;
  background: #0f172a;
  color: #e2e8f0;
  border-radius: 6px;
  font-size: 0.68rem;
  line-height: 1.35;
}

.he-dialog-foot {
  padding: 0.65rem 1rem;
  border-top: 1px solid #e2e8f0;
  display: flex;
  justify-content: flex-end;
}

.he-ok {
  padding: 0.4rem 1rem;
  border-radius: 8px;
  border: 1px solid #0284c7;
  background: #0ea5e9;
  color: #fff;
  font-size: 0.85rem;
  cursor: pointer;
}

.he-ok:hover {
  background: #0284c7;
}
</style>
