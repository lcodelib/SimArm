<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from "vue";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { createAuboC5Robot, setJointAnglesDeg, applyToolOffsetToMount } from "../robot/auboC5Robot";
import type { ToolOffset } from "../robot/toolOffset";
import { IDENTITY_TOOL_OFFSET } from "../robot/toolOffset";
import { createRgbFrameArrows, FRAME_ARROWS_RENDER_LAYER } from "../robot/rgbFrameArrows";
import { loadAuboC5CollisionMeshes } from "../robot/loadAuboC5VisualMeshes";
import {
  ARCUCO_MARKER_EDGE_SCALE,
  ARCUCO_MARKER_SIZE_MULT,
} from "../robot/arucoSceneConfig";
import {
  PNP_CHECKER_COLS,
  PNP_CHECKER_ROWS,
  PNP_CHECKER_SQUARE_M,
} from "../robot/pnpCheckerboard";
import { createAruco4x4_50Texture } from "../robot/arucoDict4x4_50";
import {
  TCP_CAM_FOV_DEG,
  TCP_CAM_RT_HEIGHT,
  TCP_CAM_RT_WIDTH,
  TCP_STEREO_BASELINE_M,
  intrinsicsFromTcpCameraSpec,
  type TcpCameraIntrinsics,
  type TcpStereoEye,
} from "../robot/tcpCameraConfig";
import {
  FEMALE_GLB_REL,
  FEMALE_WORLD_EULER_DEG,
  FEMALE_WORLD_POSITION_M,
  HUMAN_DECAL_AIM_OFFSET_M,
  HUMAN_DECAL_COLS,
  HUMAN_DECAL_LETTERS,
  HUMAN_DECAL_ROW_SPACING_M,
  HUMAN_DECAL_SIZE_M,
  HUMAN_DECAL_SPACING_M,
  HUMAN_GLB_REL,
  HUMAN_STANDING_HEIGHT_M,
  HUMAN_WORLD_EULER_DEG,
  HUMAN_WORLD_POSITION_M,
  REALMAN_GLB_REL,
  REALMAN_MESH_LOAD_SCALE,
  REALMAN_ON_BOARD_EULER_DEG,
  REALMAN_ON_BOARD_OFFSET_M,
  REALMAN_ON_BOARD_Z_LIFT_M,
  REALMAN_STANDING_HEIGHT_M,
} from "../robot/humanSceneConfig";
import {
  BED_GLB_REL,
  BED_STANDING_HEIGHT_M,
  BED_WORLD_EULER_DEG,
  BED_WORLD_POSITION_M,
  placeBedInWorld,
} from "../robot/bedSceneConfig";
import {
  ARMBODY_GLB_REL,
  ARMBODY_LOCAL_EULER_DEG,
  ARMBODY_LOCAL_OFFSET_M,
  ARMBODY_MESH_LOAD_SCALE,
  liftRobotRootSoArmBodyOnGround,
  placeArmBodyUnderBase,
} from "../robot/armBodySceneConfig";
import {
  autofitStandingHeight,
  loadSceneGltf,
  placeOnGroundInWorld,
  placeProneOnXyBoard,
} from "../robot/loadSceneGltf";
import { loadScenePly } from "../robot/loadScenePly";
import { createGroundFloorMesh } from "../robot/groundSceneConfig";
import {
  SKIN_SURFACE_MESH_LOAD_SCALE,
  SKIN_SURFACE_ON_BOARD_EULER_DEG,
  SKIN_SURFACE_ON_BOARD_OFFSET_M,
  SKIN_SURFACE_ON_BOARD_Z_LIFT_M,
  SKIN_SURFACE_PLY_REL,
} from "../robot/skinSurfaceSceneConfig";
import { tryAttachLetterDecalsOnHuman, randomLetterRollRads } from "../robot/letterDecal";

/** 主画布上画中画显示尺寸（CSS 像素，16:9） */
const TCP_PIP_DISPLAY_W = 240;
const TCP_PIP_MARGIN = 10;
/** 双目两路画中画之间的间距（CSS 像素） */
const TCP_PIP_STEREO_GAP = 8;

const { degToRad } = THREE.MathUtils;

/**
 * 末端透视相机相对 `eeCameraMount` 的平移，单位 m（沿挂载点局部轴）。
 * 例如沿视线后退：`z` 略小于 0（相机默认朝局部 −Z 看）。
 */
const TCP_CAMERA_OFFSET_M = { x: 0, y: 0, z: 0 } as const;

/** 末端相机相对挂载点的姿态微调：欧拉 XYZ、单位度。 */
const TCP_CAMERA_EULER_OFFSET_DEG = { x: 0, y: 0, z: 0 } as const;

/** `public/camera.stl` 加载后包围盒最大边缩放到该尺寸（米），便于 CAD 单位不一；≤0 则不做自动缩放 */
const TCP_CAMERA_STL_AUTOFIT_MAX_M = 0.072;

/** 在自动缩放之后再乘的倍率 */
const TCP_CAMERA_STL_SCALE = 1;

/** 机身 STL 相对透视相机局部系绕 X 轴旋转（弧度），与镜头/模型导出轴向对齐 */
const TCP_CAMERA_STL_ROTATION_X = Math.PI;

/**
 * 末端相机 STL 机身所在层（非箭头）。末端离屏相机不启用该层，机身不会挡小窗画面。
 * 主场景相机需 `layers.enable(TCP_CAM_STL_LAYER)`。
 */
const TCP_CAM_STL_LAYER = 1;

/** Canvas 每格像素，越大越清晰、显存越大 */
const PNP_CHECKER_CELL_PX = 48;
/**
 * 棋盘中心在世界坐标系中的位置（米），与地板、世界 RGB 箭头同一空间，
 * 不随机械臂或末端相机移动。请按实际标定需求微调，使板面落在末端镜头视野内。
 */
const PNP_CHECKER_WORLD_POSITION = new THREE.Vector3(-0.35, 0.25, 0.1);
/** 棋盘在世界系中的姿态：欧拉 XYZ、单位度（板默认在局部 XY，法向 +Z） */
const PNP_CHECKER_WORLD_EULER_DEG = { x: 90, y: 180, z: 90 } as const;

/** 相对棋盘面向前（+Z 法向）略微抬起，减轻与棋盘 Z 冲突 */
const ARCUCO_Z_LIFT_M = 0.0004;

const props = withDefaults(
  defineProps<{
    jointAnglesDeg: number[];
    /** 世界系棋盘格（PnP 平面）显示 */
    showCheckerboard?: boolean;
    /** 四角 DICT_4X4_50（id 0–3）贴图与局部坐标系一并显示 */
    showAruco?: boolean;
    /** 趴在棋盘位姿上的人体模型显示 */
    showHuman?: boolean;
    /** 趴在棋盘位姿上的女性人体模型显示（与 human 同位姿） */
    showFemale?: boolean;
    /** 趴在棋盘位姿上的真人模型显示（与 human 同位姿） */
    showRealman?: boolean;
    /** CT 皮肤表面 PLY（与真人同位姿） */
    showSkinSurface?: boolean;
    /** 趴在棋盘位姿上的床模型显示 */
    showBed?: boolean;
    /** 手眼采集/导出/画中画高亮所用的双目光路 */
    activeTcpEye?: TcpStereoEye;
    /** 工具系相对法兰的偏移（驱动场景中 TcpFrame 位置） */
    toolOffset?: ToolOffset;
  }>(),
  {
    showCheckerboard: false,
    showAruco: false,
    showHuman: false,
    showFemale: false,
    showRealman: false,
    showSkinSurface: false,
    showBed: false,
    activeTcpEye: "left",
    toolOffset: () => ({ ...IDENTITY_TOOL_OFFSET }),
  },
);

const canvasRef = ref<HTMLCanvasElement | null>(null);
let loadAborted = false;

let renderer: THREE.WebGLRenderer | null = null;
let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let controls: OrbitControls | null = null;
let jointRotationGroups: THREE.Group[] = [];
/** 工具坐标系挂载点（相对 J6 法兰） */
let toolFrameMount: THREE.Object3D | null = null;
/** 双目支架：中心挂一个 camera.stl，两路透视相机对称在 ±基线/2 */
let tcpStereoRig: THREE.Group | null = null;
let tcpCamera: THREE.PerspectiveCamera | null = null;
let tcpCameraStereo: THREE.PerspectiveCamera | null = null;
let tcpRenderTarget: THREE.WebGLRenderTarget | null = null;
let tcpRenderTargetStereo: THREE.WebGLRenderTarget | null = null;
let hudCamera: THREE.OrthographicCamera | null = null;
let hudScene: THREE.Scene | null = null;
let pipMesh: THREE.Mesh | null = null;
let pipMeshStereo: THREE.Mesh | null = null;
let pipMaterial: THREE.MeshBasicMaterial | null = null;
let pipMaterialStereo: THREE.MeshBasicMaterial | null = null;
let raf = 0;
let onResize: (() => void) | null = null;
let alive = false;
let calibrationCheckerMesh: THREE.Mesh | null = null;
let calibrationArucoGroup: THREE.Group | null = null;
/** 四角 ArUco 局部坐标系箭头组 */
let calibrationArucoPoseGroup: THREE.Group | null = null;
/** 场景人体 `public/human.glb` */
let humanRoot: THREE.Object3D | null = null;
/** 场景女性人体 `public/female.glb`（与 human 相同缩放/趴姿） */
let femaleRoot: THREE.Object3D | null = null;
/** 场景真人 `public/realman.glb`（与 human 相同缩放/趴姿） */
let realmanRoot: THREE.Object3D | null = null;
/** 场景皮肤表面 `public/skin_surface.ply`（与真人同位姿） */
let skinSurfaceRoot: THREE.Object3D | null = null;
/** 场景床 `public/bed.glb` */
let bedRoot: THREE.Object3D | null = null;
/** 标定板装配体（人体父节点，用于取板面法向/轴向） */
let calibBoardAssembly: THREE.Group | null = null;
/** 人体表面字母 Decal 列表（世界系几何，与 human 同显隐） */
let humanLetterDecals: THREE.Mesh[] = [];
/** 定位测试：世界系固定位姿组（红球 + 三色坐标轴，不随臂/相机运动） */
let stereoLocGroup: THREE.Group | null = null;
let stereoLocMarker: THREE.Mesh | null = null;
let stereoLocAxes: THREE.Group | null = null;
/** 双目点云（世界系固定，拍照瞬间左相机位姿钉住） */
let stereoPointCloud: THREE.Points | null = null;
/**
 * 双目拍照瞬间左相机世界矩阵（Three.js）。
 * 点选 2D→相机系 3D 后，用此矩阵转到世界系放置；与之后机械臂/相机运动无关。
 */
let stereoCaptureLeftCamWorld: THREE.Matrix4 | null = null;

function publicAssetUrl(path: string): string {
  const base = import.meta.env.BASE_URL;
  const prefix = base.endsWith("/") ? base : `${base}/`;
  const p = path.startsWith("/") ? path.slice(1) : path;
  return `${prefix}${p}`;
}

/** 将 OrbitControls 旋转中心设为机械臂基座（base_link）世界坐标 */
function setOrbitTargetToRobotBase(baseLink: THREE.Object3D): void {
  if (!controls || !camera) return;
  baseLink.updateMatrixWorld(true);
  const target = new THREE.Vector3();
  baseLink.getWorldPosition(target);
  controls.target.copy(target);
  controls.update();
}

function setObjectLayerRecursive(root: THREE.Object3D, layer: number): void {
  root.traverse((o) => {
    o.layers.set(layer);
  });
}

/** 生成黑白棋盘格纹理（角点检测友好：左上为黑格） */
function createPnpCheckerboardTexture(
  squaresX: number,
  squaresY: number,
  cellPx: number,
): THREE.CanvasTexture {
  const cw = squaresX * cellPx;
  const ch = squaresY * cellPx;
  const c = document.createElement("canvas");
  c.width = cw;
  c.height = ch;
  const ctx = c.getContext("2d");
  if (!ctx) {
    throw new Error("PNP checkerboard: 2D context unavailable");
  }
  for (let row = 0; row < squaresY; row++) {
    for (let col = 0; col < squaresX; col++) {
      const black = (col + row) % 2 === 0;
      ctx.fillStyle = black ? "#050505" : "#fafafa";
      ctx.fillRect(col * cellPx, row * cellPx, cellPx, cellPx);
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  return tex;
}

/** 创建 PnP 用棋盘格网格（未设置世界位姿；由调用方加入 scene 并设定 position/rotation） */
function createPnpCheckerboardMesh(): THREE.Mesh {
  const boardW = PNP_CHECKER_COLS * PNP_CHECKER_SQUARE_M;
  const boardH = PNP_CHECKER_ROWS * PNP_CHECKER_SQUARE_M;
  const map = createPnpCheckerboardTexture(PNP_CHECKER_COLS, PNP_CHECKER_ROWS, PNP_CHECKER_CELL_PX);
  const mat = new THREE.MeshStandardMaterial({
    map,
    roughness: 0.88,
    metalness: 0.02,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
  const geom = new THREE.PlaneGeometry(boardW, boardH);
  const board = new THREE.Mesh(geom, mat);
  board.name = "PnpCheckerboardWorld";
  board.castShadow = true;
  board.receiveShadow = true;
  return board;
}

/** 棋盘 + 四角 ArUco + 标码位姿轴（同一父节点，与 {@link PNP_CHECKER_WORLD_POSITION} 对齐） */
function createCalibrationBoardAssembly(): {
  group: THREE.Group;
  checker: THREE.Mesh;
  arucoGroup: THREE.Group;
  arucoPoseGroup: THREE.Group;
} {
  const group = new THREE.Group();
  group.name = "CalibrationBoardAssembly";

  const checker = createPnpCheckerboardMesh();
  group.add(checker);

  const boardW = PNP_CHECKER_COLS * PNP_CHECKER_SQUARE_M;
  const boardH = PNP_CHECKER_ROWS * PNP_CHECKER_SQUARE_M;
  const halfW = boardW / 2;
  const halfH = boardH / 2;
  const edge = PNP_CHECKER_SQUARE_M * ARCUCO_MARKER_EDGE_SCALE * ARCUCO_MARKER_SIZE_MULT;
  /** 位姿轴略短于标码边长，避免互相遮挡 */
  const axisLen = edge * 0.72;
  const axisHead = axisLen * 0.28;
  const axisRadius = axisLen * 0.08;

  const arucoGroup = new THREE.Group();
  arucoGroup.name = "ArucoCorners";
  const arucoPoseGroup = new THREE.Group();
  arucoPoseGroup.name = "ArucoPoseFrames";

  /** 顺序：NW→NE→SE→SW，id 0–3；标码局部：u→+X、v→+Y、法向 +Z */
  const corners: readonly { readonly x: number; readonly y: number; readonly id: number }[] = [
    { x: -halfW, y: halfH, id: 0 },
    { x: halfW, y: halfH, id: 1 },
    { x: halfW, y: -halfH, id: 2 },
    { x: -halfW, y: -halfH, id: 3 },
  ];

  for (const { x, y, id } of corners) {
    const tex = createAruco4x4_50Texture(id);
    const mat = new THREE.MeshStandardMaterial({
      map: tex,
      roughness: 0.82,
      metalness: 0.04,
      polygonOffset: true,
      polygonOffsetFactor: -0.5,
      polygonOffsetUnits: 1,
    });
    const plaque = new THREE.Mesh(new THREE.PlaneGeometry(edge, edge), mat);
    plaque.name = `Aruco4x4_50_id${id}`;
    plaque.position.set(x, y, ARCUCO_Z_LIFT_M);
    plaque.castShadow = true;
    plaque.receiveShadow = true;
    arucoGroup.add(plaque);

    const poseFrame = createRgbFrameArrows(
      axisLen,
      axisHead,
      axisRadius,
      new THREE.Vector3(x, y, ARCUCO_Z_LIFT_M + 0.001),
      `ArucoId${id}FrameXYZ`,
    );
    arucoPoseGroup.add(poseFrame);
  }
  group.add(arucoGroup);
  group.add(arucoPoseGroup);

  return { group, checker, arucoGroup, arucoPoseGroup };
}

function createTcpPerspectiveCamera(): THREE.PerspectiveCamera {
  const cam = new THREE.PerspectiveCamera(
    TCP_CAM_FOV_DEG,
    TCP_CAM_RT_WIDTH / TCP_CAM_RT_HEIGHT,
    0.02,
    40,
  );
  cam.layers.disableAll();
  cam.layers.enable(0);
  return cam;
}

function attachTcpStereoEyeFrame(cam: THREE.PerspectiveCamera, frameName: string): void {
  const frame = createRgbFrameArrows(0.056, 0.019, 0.013, new THREE.Vector3(0, 0, 0), frameName);
  cam.add(frame);
}

/** 在双目支架中心加载单个 camera.stl；两路相机仅作光心，不重复机身 */
async function loadTcpStereoRigBody(rig: THREE.Group): Promise<void> {
  const url = publicAssetUrl("camera.stl");
  let geometry: THREE.BufferGeometry | null = null;
  try {
    geometry = await new STLLoader().loadAsync(url);
    geometry.computeVertexNormals();
  } catch (err) {
    console.warn("[SimArm] 未加载到 public/camera.stl，末端相机无机身模型", err);
  }

  if (loadAborted) {
    geometry?.dispose();
    return;
  }

  if (!geometry) return;

  geometry.computeBoundingBox();
  const bb = geometry.boundingBox;
  if (bb) {
    const cx = (bb.min.x + bb.max.x) * 0.5;
    const cy = (bb.min.y + bb.max.y) * 0.5;
    const cz = (bb.min.z + bb.max.z) * 0.5;
    geometry.translate(-cx, -cy, -cz);
  }
  geometry.computeBoundingBox();
  const b2 = geometry.boundingBox;
  let fitScale = TCP_CAMERA_STL_SCALE;
  if (b2 && TCP_CAMERA_STL_AUTOFIT_MAX_M > 0) {
    const sx = b2.max.x - b2.min.x;
    const sy = b2.max.y - b2.min.y;
    const sz = b2.max.z - b2.min.z;
    const maxDim = Math.max(sx, sy, sz);
    if (maxDim > 1e-9) fitScale = (TCP_CAMERA_STL_AUTOFIT_MAX_M / maxDim) * TCP_CAMERA_STL_SCALE;
  }

  if (loadAborted) {
    geometry.dispose();
    return;
  }

  const mat = new THREE.MeshStandardMaterial({
    color: 0x4a5568,
    metalness: 0.35,
    roughness: 0.45,
  });
  const mesh = new THREE.Mesh(geometry, mat);
  mesh.name = "TcpStereoCameraStl";
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.scale.setScalar(fitScale);
  mesh.rotation.x = TCP_CAMERA_STL_ROTATION_X;
  setObjectLayerRecursive(mesh, TCP_CAM_STL_LAYER);
  rig.add(mesh);
}

function createTcpStereoRig(): THREE.Group {
  const half = TCP_STEREO_BASELINE_M * 0.5;
  const rig = new THREE.Group();
  rig.name = "TcpStereoRig";
  rig.position.set(TCP_CAMERA_OFFSET_M.x, TCP_CAMERA_OFFSET_M.y, TCP_CAMERA_OFFSET_M.z);
  rig.rotation.set(
    degToRad(TCP_CAMERA_EULER_OFFSET_DEG.x),
    degToRad(TCP_CAMERA_EULER_OFFSET_DEG.y),
    degToRad(TCP_CAMERA_EULER_OFFSET_DEG.z),
    "XYZ",
  );

  // 相机 UV：u 向右为 +X → 左目 −X、右目 +X（与侧栏「左/右相机」及采集一致）
  tcpCamera = createTcpPerspectiveCamera();
  tcpCamera.name = "TcpCameraLeft";
  tcpCamera.position.set(-half, 0, 0);
  attachTcpStereoEyeFrame(tcpCamera, "TcpCameraLeftFrameXYZ");
  rig.add(tcpCamera);

  tcpCameraStereo = createTcpPerspectiveCamera();
  tcpCameraStereo.name = "TcpCameraRight";
  tcpCameraStereo.position.set(half, 0, 0);
  attachTcpStereoEyeFrame(tcpCameraStereo, "TcpCameraRightFrameXYZ");
  rig.add(tcpCameraStereo);

  return rig;
}

function layoutPipHud(canvasW: number, canvasH: number): void {
  if (!hudCamera || !pipMesh) return;
  const pipH = (TCP_PIP_DISPLAY_W * TCP_CAM_RT_HEIGHT) / TCP_CAM_RT_WIDTH;
  hudCamera.left = 0;
  hudCamera.right = canvasW;
  hudCamera.top = canvasH;
  hudCamera.bottom = 0;
  hudCamera.updateProjectionMatrix();
  const y = canvasH - TCP_PIP_MARGIN - pipH / 2;
  const pipRightX = canvasW - TCP_PIP_MARGIN - TCP_PIP_DISPLAY_W / 2;
  const pipLeftX = pipRightX - TCP_PIP_DISPLAY_W - TCP_PIP_STEREO_GAP;
  const active = props.activeTcpEye;
  const activeW = TCP_PIP_DISPLAY_W + 6;
  const activeH = pipH + 6;
  const idleW = TCP_PIP_DISPLAY_W;
  const idleH = pipH;

  // pipMesh=左目(−X) 在屏左；pipMeshStereo=右目(+X) 在屏右，与 UV 及采集一致
  pipMesh.position.set(pipLeftX, y, 0);
  pipMeshStereo?.position.set(pipRightX, y, 0);
  if (active === "left") {
    pipMesh.scale.set(activeW, activeH, 1);
    pipMeshStereo?.scale.set(idleW, idleH, 1);
  } else {
    pipMesh.scale.set(idleW, idleH, 1);
    pipMeshStereo?.scale.set(activeW, activeH, 1);
  }
  updatePipHighlight();
}

function updatePipHighlight(): void {
  const active = props.activeTcpEye;
  const activeColor = new THREE.Color(0xffffff);
  const idleColor = new THREE.Color(0x94a3b8);
  if (pipMaterial) pipMaterial.color.copy(active === "left" ? activeColor : idleColor);
  if (pipMaterialStereo) pipMaterialStereo.color.copy(active === "right" ? activeColor : idleColor);
}

/** 左目(−X) → tcpRenderTarget；右目(+X) → tcpRenderTargetStereo */
function getActiveTcpRenderTarget(): THREE.WebGLRenderTarget | null {
  return props.activeTcpEye === "right" ? tcpRenderTargetStereo : tcpRenderTarget;
}

function renderTcpCameraPairToTargets(): void {
  if (!renderer || !scene || !tcpCamera || !tcpRenderTarget) return;
  const prevRt = renderer.getRenderTarget();
  const aspect = TCP_CAM_RT_WIDTH / TCP_CAM_RT_HEIGHT;

  renderer.setRenderTarget(tcpRenderTarget);
  renderer.setViewport(0, 0, TCP_CAM_RT_WIDTH, TCP_CAM_RT_HEIGHT);
  tcpCamera.aspect = aspect;
  tcpCamera.fov = TCP_CAM_FOV_DEG;
  tcpCamera.updateProjectionMatrix();
  renderer.render(scene, tcpCamera);

  if (tcpCameraStereo && tcpRenderTargetStereo) {
    renderer.setRenderTarget(tcpRenderTargetStereo);
    renderer.setViewport(0, 0, TCP_CAM_RT_WIDTH, TCP_CAM_RT_HEIGHT);
    tcpCameraStereo.aspect = aspect;
    tcpCameraStereo.fov = TCP_CAM_FOV_DEG;
    tcpCameraStereo.updateProjectionMatrix();
    renderer.render(scene, tcpCameraStereo);
  }

  renderer.setRenderTarget(prevRt);
}

/** 读取指定离屏 RT 为 PNG Blob（含 Y 翻转） */
async function renderTargetToPngBlob(rt: THREE.WebGLRenderTarget): Promise<Blob> {
  if (!renderer) throw new Error("场景未就绪，无法采集末端相机图像");
  const w = TCP_CAM_RT_WIDTH;
  const h = TCP_CAM_RT_HEIGHT;
  const buffer = new Uint8Array(w * h * 4);
  renderer.readRenderTargetPixels(rt, 0, 0, w, h, buffer);
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const srcOff = (h - 1 - y) * w * 4;
    rgba.set(buffer.subarray(srcOff, srcOff + w * 4), y * w * 4);
  }
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("无法创建 2D 画布");
  ctx.putImageData(new ImageData(rgba, w, h), 0, 0);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("PNG 编码失败"));
    }, "image/png");
  });
}

/** 读取当前选中光路的离屏 RT 为 PNG Blob（1280×720） */
async function captureTcpCameraPngBlob(): Promise<Blob> {
  const rt = getActiveTcpRenderTarget();
  if (!rt) {
    throw new Error("场景未就绪，无法采集末端相机图像");
  }
  renderTcpCameraPairToTargets();
  return renderTargetToPngBlob(rt);
}

/** 同帧采集左右相机 PNG（先渲染双目 RT，再分别读出），并冻结左相机世界位姿供定位点使用 */
async function captureTcpStereoPairPngBlobs(): Promise<{ left: Blob; right: Blob }> {
  if (!tcpRenderTarget || !tcpRenderTargetStereo) {
    throw new Error("双目离屏目标未就绪，无法采集");
  }
  renderTcpCameraPairToTargets();
  // 与图像同帧：记录左相机世界矩阵，后续点选只用此快照转到世界系
  if (tcpCamera) {
    tcpCamera.updateMatrixWorld(true);
    stereoCaptureLeftCamWorld = tcpCamera.matrixWorld.clone();
  }
  const left = await renderTargetToPngBlob(tcpRenderTarget);
  const right = await renderTargetToPngBlob(tcpRenderTargetStereo);
  return { left, right };
}

function getTcpCameraIntrinsics(): TcpCameraIntrinsics {
  return intrinsicsFromTcpCameraSpec(TCP_CAM_RT_WIDTH, TCP_CAM_RT_HEIGHT, TCP_CAM_FOV_DEG);
}

function getActiveTcpEye(): TcpStereoEye {
  return props.activeTcpEye;
}

/**
 * 红球逻辑：点选 2D → 相机系 3D → 用拍照瞬间左相机世界位姿转到世界系放置。
 * 挂在 scene 下固定，之后不与机械臂运动、相机运动产生关系。
 * @param opencvXyz 校正左目 OpenCV 坐标（米）
 * @param opencvRRowMajor 可选：校正左目下局部姿态 R（行主序 3×3），Z∥相机+Z
 */
function showStereoLocalizationMarker(
  opencvXyz: { x: number; y: number; z: number },
  opencvRRowMajor?: number[][],
): void {
  if (!scene) return;

  // 优先用双目拍照时冻结的左相机世界矩阵；无快照时退化为当前左相机（仅兜底）
  let camWorld = stereoCaptureLeftCamWorld;
  if (!camWorld) {
    if (!tcpCamera) return;
    tcpCamera.updateMatrixWorld(true);
    camWorld = tcpCamera.matrixWorld;
    console.warn("[SimArm] 无拍照时相机快照，改用当前左相机位姿放置定位点");
  }

  // OpenCV → Three 相机局部：p' = S p，R' = S R S，S=diag(1,-1,-1)
  const S = new THREE.Matrix4().makeScale(1, -1, -1);
  const pThree = new THREE.Vector3(opencvXyz.x, opencvXyz.y, opencvXyz.z).applyMatrix4(S);

  const localMat = new THREE.Matrix4();
  if (opencvRRowMajor && opencvRRowMajor.length === 3) {
    const R = new THREE.Matrix4();
    R.set(
      opencvRRowMajor[0][0],
      opencvRRowMajor[0][1],
      opencvRRowMajor[0][2],
      0,
      opencvRRowMajor[1][0],
      opencvRRowMajor[1][1],
      opencvRRowMajor[1][2],
      0,
      opencvRRowMajor[2][0],
      opencvRRowMajor[2][1],
      opencvRRowMajor[2][2],
      0,
      0,
      0,
      0,
      1,
    );
    const Rthree = new THREE.Matrix4().multiplyMatrices(S, R).multiply(S);
    localMat.compose(pThree, new THREE.Quaternion().setFromRotationMatrix(Rthree), new THREE.Vector3(1, 1, 1));
  } else {
    localMat.setPosition(pThree);
  }

  const worldMat = new THREE.Matrix4().multiplyMatrices(camWorld, localMat);
  const worldPos = new THREE.Vector3();
  const worldQuat = new THREE.Quaternion();
  worldMat.decompose(worldPos, worldQuat, new THREE.Vector3());

  if (!stereoLocGroup) {
    stereoLocGroup = new THREE.Group();
    stereoLocGroup.name = "StereoLocalizationGroup";
    const geo = new THREE.SphereGeometry(0.006, 20, 16);
    const mat = new THREE.MeshBasicMaterial({ color: 0xf43f5e, depthTest: true });
    stereoLocMarker = new THREE.Mesh(geo, mat);
    stereoLocMarker.name = "StereoLocalizationMarker";
    stereoLocGroup.add(stereoLocMarker);
    stereoLocAxes = createRgbFrameArrows(0.04, 0.012, 0.004, new THREE.Vector3(0, 0, 0), "StereoLocFrameXYZ");
    stereoLocGroup.add(stereoLocAxes);
    scene.add(stereoLocGroup);
  } else if (stereoLocGroup.parent !== scene) {
    stereoLocGroup.removeFromParent();
    scene.add(stereoLocGroup);
  }

  stereoLocGroup.position.copy(worldPos);
  stereoLocGroup.quaternion.copy(worldQuat);
  stereoLocGroup.scale.set(1, 1, 1);
  if (stereoLocAxes) {
    stereoLocAxes.visible = Boolean(opencvRRowMajor && opencvRRowMajor.length === 3);
  }
  stereoLocGroup.visible = true;
}

function clearStereoLocalizationMarker(): void {
  if (!stereoLocGroup) return;
  stereoLocGroup.visible = false;
}

/**
 * 在世界系显示双目点云（测试用）。
 * @param xyzThreeLocal 左相机 Three 局部系交错 xyz（已由 OpenCV [X,-Y,-Z] 转换）
 * @param rgb01 可选 0–1 RGB
 */
function showStereoPointCloud(xyzThreeLocal: Float32Array, rgb01?: Float32Array | null): void {
  if (!scene) return;

  let camWorld = stereoCaptureLeftCamWorld;
  if (!camWorld) {
    if (!tcpCamera) return;
    tcpCamera.updateMatrixWorld(true);
    camWorld = tcpCamera.matrixWorld;
    console.warn("[SimArm] 无拍照时相机快照，点云改用当前左相机位姿");
  }

  clearStereoPointCloud();

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(xyzThreeLocal, 3));
  if (rgb01 && rgb01.length === xyzThreeLocal.length) {
    geo.setAttribute("color", new THREE.BufferAttribute(rgb01, 3));
  }
  geo.computeBoundingSphere();

  const mat = new THREE.PointsMaterial({
    size: 0.004,
    sizeAttenuation: true,
    vertexColors: Boolean(rgb01 && rgb01.length === xyzThreeLocal.length),
    color: rgb01 ? 0xffffff : 0x38bdf8,
    depthTest: true,
  });

  stereoPointCloud = new THREE.Points(geo, mat);
  stereoPointCloud.name = "StereoPointCloud";

  const worldPos = new THREE.Vector3();
  const worldQuat = new THREE.Quaternion();
  camWorld.decompose(worldPos, worldQuat, new THREE.Vector3());
  stereoPointCloud.position.copy(worldPos);
  stereoPointCloud.quaternion.copy(worldQuat);
  stereoPointCloud.scale.set(1, 1, 1);
  stereoPointCloud.frustumCulled = true;
  scene.add(stereoPointCloud);
}

function clearStereoPointCloud(): void {
  if (!stereoPointCloud) return;
  stereoPointCloud.removeFromParent();
  stereoPointCloud.geometry.dispose();
  (stereoPointCloud.material as THREE.PointsMaterial).dispose();
  stereoPointCloud = null;
}

function setStereoPointCloudVisible(visible: boolean): void {
  if (stereoPointCloud) stereoPointCloud.visible = visible;
}

/** 释放并移除现有字母 Decal */
function disposeHumanLetterDecals(): void {
  for (const decal of humanLetterDecals) {
    decal.removeFromParent();
    decal.geometry?.dispose();
    const mat = decal.material as THREE.MeshStandardMaterial;
    mat.map?.dispose();
    mat.dispose();
  }
  humanLetterDecals = [];
}

/**
 * 在人体表面重建字母贴纸。
 * 人体已挂世界系：自上而下（世界 +Y）逼近，列沿 +X、行沿 +Z。
 * @param randomRoll 为 true 时每个字母绕法向随机扭转
 */
function rebuildHumanLetterDecals(randomRoll = false): void {
  if (!scene || !humanRoot || !HUMAN_DECAL_LETTERS) return;

  disposeHumanLetterDecals();
  humanRoot.updateMatrixWorld(true);
  const approachNormalWorld = new THREE.Vector3(0, 1, 0);
  const colAxisWorld = new THREE.Vector3(1, 0, 0);
  const rowAxisWorld = new THREE.Vector3(0, 0, 1);

  const letterCount = [...HUMAN_DECAL_LETTERS].filter((c) => /[A-Za-z]/.test(c)).length;
  const rollRads = randomRoll ? randomLetterRollRads(letterCount) : undefined;

  const decals = tryAttachLetterDecalsOnHuman(
    humanRoot,
    HUMAN_DECAL_LETTERS,
    HUMAN_DECAL_SIZE_M,
    HUMAN_DECAL_SPACING_M,
    HUMAN_DECAL_ROW_SPACING_M,
    HUMAN_DECAL_COLS,
    approachNormalWorld,
    colAxisWorld,
    rowAxisWorld,
    HUMAN_DECAL_AIM_OFFSET_M,
    rollRads,
  );
  for (const decal of decals) {
    decal.visible = props.showHuman;
    scene.add(decal);
  }
  humanLetterDecals = decals;
}

/** 随机旋转人体表面各字母贴纸（绕表面法向） */
function randomizeLetterDecalRotations(): void {
  if (!humanRoot) {
    console.warn("[SimArm] 人体未就绪，无法旋转字母贴纸");
    return;
  }
  rebuildHumanLetterDecals(true);
}

defineExpose({
  captureTcpCameraPngBlob,
  captureTcpStereoPairPngBlobs,
  getTcpCameraIntrinsics,
  getActiveTcpEye,
  showStereoLocalizationMarker,
  clearStereoLocalizationMarker,
  showStereoPointCloud,
  clearStereoPointCloud,
  setStereoPointCloudVisible,
  randomizeLetterDecalRotations,
});

function tick(): void {
  if (!alive) return;
  raf = requestAnimationFrame(tick);
  controls?.update();
  if (!renderer || !scene || !camera) return;

  const canvas = renderer.domElement;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;

  if (tcpRenderTarget && tcpCamera) {
    renderTcpCameraPairToTargets();
  }

  renderer.setViewport(0, 0, w, h);
  renderer.setScissorTest(false);
  renderer.render(scene, camera);

  if (hudScene && hudCamera && pipMesh) {
    layoutPipHud(w, h);
    const prevAuto = renderer.autoClear;
    renderer.autoClear = false;
    renderer.setViewport(0, 0, w, h);
    renderer.clearDepth();
    renderer.render(hudScene, hudCamera);
    renderer.autoClear = prevAuto;
  }
}

onMounted(() => {
  const canvas = canvasRef.value;
  if (!canvas) return;

  const width = canvas.clientWidth;
  const height = canvas.clientHeight;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xeef2f7);

  camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 50);
  // 初始朝向；机械臂加载后会把轨道中心改到基座世界坐标
  camera.position.set(1.35, 0.55, 1.35);
  camera.lookAt(0, 0.35, 0);
  camera.layers.enable(TCP_CAM_STL_LAYER);
  camera.layers.enable(FRAME_ARROWS_RENDER_LAYER);

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(width, height, false);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 0.35, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;

  const hemi = new THREE.HemisphereLight(0xffffff, 0xd4dce8, 0.78);
  scene.add(hemi);
  const fill = new THREE.DirectionalLight(0xf8fafc, 0.35);
  fill.position.set(-2.5, 3, -1.5);
  scene.add(fill);

  const dir = new THREE.DirectionalLight(0xfff8f0, 1.18);
  dir.position.set(3.2, 7, 4.2);
  dir.castShadow = true;
  dir.shadow.mapSize.set(2048, 2048);
  dir.shadow.camera.near = 0.5;
  dir.shadow.camera.far = 20;
  dir.shadow.camera.left = -3;
  dir.shadow.camera.right = 3;
  dir.shadow.camera.top = 3;
  dir.shadow.camera.bottom = -3;
  scene.add(dir);

  // 地板贴图（public/ground.png）
  scene.add(createGroundFloorMesh());

  scene.add(
    createRgbFrameArrows(0.48, 0.07, 0.036, new THREE.Vector3(-0.5, 0, -0.5), "WorldFrameXYZ"),
  );

  const { group: calibAssembly, checker: calibChecker, arucoGroup: calibAruco, arucoPoseGroup: calibArucoPose } =
    createCalibrationBoardAssembly();
  calibrationCheckerMesh = calibChecker;
  calibrationArucoGroup = calibAruco;
  calibrationArucoPoseGroup = calibArucoPose;
  calibBoardAssembly = calibAssembly;
  calibChecker.visible = props.showCheckerboard;
  calibAruco.visible = props.showAruco;
  calibArucoPose.visible = props.showAruco;
  calibAssembly.position.copy(PNP_CHECKER_WORLD_POSITION);
  calibAssembly.rotation.set(
    degToRad(PNP_CHECKER_WORLD_EULER_DEG.x),
    degToRad(PNP_CHECKER_WORLD_EULER_DEG.y),
    degToRad(PNP_CHECKER_WORLD_EULER_DEG.z),
    "XYZ",
  );
  scene.add(calibAssembly);

  const makeTcpRenderTarget = () => {
    const rt = new THREE.WebGLRenderTarget(TCP_CAM_RT_WIDTH, TCP_CAM_RT_HEIGHT, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: true,
    });
    rt.texture.colorSpace = THREE.SRGBColorSpace;
    return rt;
  };
  tcpRenderTarget = makeTcpRenderTarget();
  tcpRenderTargetStereo = makeTcpRenderTarget();

  tcpStereoRig = createTcpStereoRig();

  hudCamera = new THREE.OrthographicCamera(0, width, height, 0, -50, 50);
  hudCamera.position.set(0, 0, 10);
  hudCamera.lookAt(0, 0, 0);

  hudScene = new THREE.Scene();
  pipMaterial = new THREE.MeshBasicMaterial({
    map: tcpRenderTarget.texture,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });
  const pipGeom = new THREE.PlaneGeometry(1, 1);
  pipMesh = new THREE.Mesh(pipGeom, pipMaterial);
  pipMesh.frustumCulled = false;
  hudScene.add(pipMesh);

  pipMaterialStereo = new THREE.MeshBasicMaterial({
    map: tcpRenderTargetStereo.texture,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });
  pipMeshStereo = new THREE.Mesh(pipGeom.clone(), pipMaterialStereo);
  pipMeshStereo.frustumCulled = false;
  hudScene.add(pipMeshStereo);

  void (async () => {
    const tcpVisualPromise = loadTcpStereoRigBody(tcpStereoRig!);
    const humanPromise = loadSceneGltf(publicAssetUrl(HUMAN_GLB_REL));
    const femalePromise = loadSceneGltf(publicAssetUrl(FEMALE_GLB_REL));
    const realmanPromise = loadSceneGltf(publicAssetUrl(REALMAN_GLB_REL));
    const skinPromise = loadScenePly(publicAssetUrl(SKIN_SURFACE_PLY_REL));
    const bedPromise = loadSceneGltf(publicAssetUrl(BED_GLB_REL));
    const armBodyPromise = loadSceneGltf(publicAssetUrl(ARMBODY_GLB_REL));
    let meshes;
    try {
      meshes = await loadAuboC5CollisionMeshes();
    } catch (err) {
      console.warn("[SimArm] 加载 aubo_C5/collision 失败，使用占位几何", err);
      meshes = Array.from({ length: 7 }, () => null);
    }
    if (loadAborted || !scene || !tcpStereoRig || !tcpCamera || !tcpCameraStereo) return;

    const { root, jointRotationGroups: jg, baseLink, eeCameraMount, toolFrameMount: tfm } =
      createAuboC5Robot(meshes);
    jointRotationGroups = jg;
    toolFrameMount = tfm;
    applyToolOffsetToMount(toolFrameMount, props.toolOffset);
    eeCameraMount.add(tcpStereoRig);
    scene.add(root);
    setJointAnglesDeg(jointRotationGroups, props.jointAnglesDeg);
    await tcpVisualPromise;

    const [human, female, realman, skinSurface, bed, armBody] = await Promise.all([
      humanPromise,
      femalePromise,
      realmanPromise,
      skinPromise,
      bedPromise,
      armBodyPromise,
    ]);
    if (loadAborted || !scene) return;

    // 座体挂在 base_link 下，顶面贴基座原点平面
    if (armBody) {
      armBody.name = "ArmBodyGlb";
      armBody.scale.setScalar(ARMBODY_MESH_LOAD_SCALE);
      placeArmBodyUnderBase(armBody, ARMBODY_LOCAL_EULER_DEG, ARMBODY_LOCAL_OFFSET_M);
      // 与 yawGroup 并列挂在 root：不随机械臂绕竖直轴偏航
      root.add(armBody);
      // 座体贴地，整机（含基座）抬到座体之上
      liftRobotRootSoArmBodyOnGround(root, armBody);
    } else {
      console.warn("[SimArm] 未加载到 public/armbody.glb");
    }

    setOrbitTargetToRobotBase(baseLink);

    // 床：世界系放置（不挂棋盘装配体）
    if (bed) {
      bed.name = "BedGlb";
      autofitStandingHeight(bed, BED_STANDING_HEIGHT_M);
      placeBedInWorld(bed, BED_WORLD_EULER_DEG, BED_WORLD_POSITION_M);
      scene.add(bed);
      bed.visible = props.showBed;
      bedRoot = bed;
    } else {
      console.warn("[SimArm] 未加载到 public/bed.glb");
    }

    // 男性：世界系放置（不挂棋盘装配体）
    if (human) {
      human.name = "HumanGlb";
      autofitStandingHeight(human, HUMAN_STANDING_HEIGHT_M);
      placeOnGroundInWorld(human, HUMAN_WORLD_EULER_DEG, HUMAN_WORLD_POSITION_M);
      scene.add(human);
      human.visible = props.showHuman;
      humanRoot = human;
      // 试贴 ABCDEF：世界系自上而下，列 +X / 行 +Z
      rebuildHumanLetterDecals(false);
    } else {
      console.warn("[SimArm] 未加载到 public/human.glb");
    }

    // 女性：世界系放置（与男性相同逻辑，位姿可单独配置）
    if (female) {
      female.name = "FemaleGlb";
      autofitStandingHeight(female, HUMAN_STANDING_HEIGHT_M);
      placeOnGroundInWorld(female, FEMALE_WORLD_EULER_DEG, FEMALE_WORLD_POSITION_M);
      scene.add(female);
      female.visible = props.showFemale;
      femaleRoot = female;
    } else {
      console.warn("[SimArm] 未加载到 public/female.glb");
    }

    // 真人：与 human 相同缩放与棋盘局部趴姿/偏移
    if (realman) {
      realman.name = "RealmanGlb";
      realman.scale.setScalar(REALMAN_MESH_LOAD_SCALE);
      autofitStandingHeight(realman, REALMAN_STANDING_HEIGHT_M);
      placeProneOnXyBoard(
        realman,
        REALMAN_ON_BOARD_EULER_DEG,
        REALMAN_ON_BOARD_OFFSET_M,
        REALMAN_ON_BOARD_Z_LIFT_M,
      );
      calibAssembly.add(realman);
      realman.visible = props.showRealman;
      realmanRoot = realman;
    } else {
      console.warn("[SimArm] 未加载到 public/realman.glb");
    }

    // 皮肤表面 PLY：与真人相同棋盘局部位姿
    if (skinSurface) {
      skinSurface.name = "SkinSurfacePly";
      skinSurface.scale.setScalar(SKIN_SURFACE_MESH_LOAD_SCALE);
      placeProneOnXyBoard(
        skinSurface,
        SKIN_SURFACE_ON_BOARD_EULER_DEG,
        SKIN_SURFACE_ON_BOARD_OFFSET_M,
        SKIN_SURFACE_ON_BOARD_Z_LIFT_M,
      );
      calibAssembly.add(skinSurface);
      skinSurface.visible = props.showSkinSurface;
      skinSurfaceRoot = skinSurface;
    } else {
      console.warn("[SimArm] 未加载到 public/skin_surface.ply");
    }
  })();

  onResize = (): void => {
    if (!canvas || !renderer || !camera) return;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
    if (hudCamera) {
      hudCamera.left = 0;
      hudCamera.right = w;
      hudCamera.top = h;
      hudCamera.bottom = 0;
      hudCamera.updateProjectionMatrix();
    }
    layoutPipHud(w, h);
  };
  window.addEventListener("resize", onResize);

  alive = true;
  tick();
});

onUnmounted(() => {
  alive = false;
  loadAborted = true;
  if (onResize) window.removeEventListener("resize", onResize);
  cancelAnimationFrame(raf);
  controls?.dispose();
  const disposeObjectTree = (root: THREE.Object3D) => {
    root.traverse((obj) => {
      if (obj instanceof THREE.Mesh) {
        obj.geometry?.dispose();
        const mat = obj.material;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else mat?.dispose();
      } else if (obj instanceof THREE.Line || obj instanceof THREE.LineSegments) {
        obj.geometry?.dispose();
        const mat = obj.material;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else (mat as THREE.Material | undefined)?.dispose?.();
      }
    });
    root.clear();
    root.removeFromParent();
  };
  if (stereoLocGroup) {
    disposeObjectTree(stereoLocGroup);
    stereoLocGroup = null;
    stereoLocMarker = null;
    stereoLocAxes = null;
  }
  clearStereoPointCloud();
  stereoCaptureLeftCamWorld = null;
  if (tcpStereoRig) {
    disposeObjectTree(tcpStereoRig);
    tcpStereoRig = null;
    tcpCamera = null;
    tcpCameraStereo = null;
  }
  tcpRenderTarget?.dispose();
  tcpRenderTarget = null;
  tcpRenderTargetStereo?.dispose();
  tcpRenderTargetStereo = null;
  pipMesh?.geometry.dispose();
  pipMaterial?.dispose();
  pipMesh = null;
  pipMaterial = null;
  pipMeshStereo?.geometry.dispose();
  pipMaterialStereo?.dispose();
  pipMeshStereo = null;
  pipMaterialStereo = null;
  hudScene = null;
  hudCamera = null;
  renderer?.dispose();
  scene?.traverse((obj) => {
    if (obj instanceof THREE.Mesh) {
      obj.geometry?.dispose();
      const mat = obj.material;
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
      else mat?.dispose();
    } else if (obj instanceof THREE.Line || obj instanceof THREE.LineSegments) {
      obj.geometry?.dispose();
      const mat = obj.material;
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
      else (mat as THREE.Material | undefined)?.dispose?.();
    }
  });
  renderer = null;
  scene = null;
  camera = null;
  controls = null;
  jointRotationGroups = [];
  calibrationCheckerMesh = null;
  calibrationArucoGroup = null;
  calibrationArucoPoseGroup = null;
  humanRoot = null;
  femaleRoot = null;
  realmanRoot = null;
  skinSurfaceRoot = null;
  bedRoot = null;
  calibBoardAssembly = null;
  toolFrameMount = null;
  disposeHumanLetterDecals();
  onResize = null;
});

watch(
  () => props.jointAnglesDeg,
  (angles) => {
    setJointAnglesDeg(jointRotationGroups, angles);
  },
  { deep: true },
);

watch(
  () => props.toolOffset,
  (offset) => {
    if (toolFrameMount) applyToolOffsetToMount(toolFrameMount, offset);
  },
  { deep: true },
);

watch(
  () => props.showCheckerboard,
  (v) => {
    if (calibrationCheckerMesh) calibrationCheckerMesh.visible = v;
  },
);

watch(
  () => props.showAruco,
  (v) => {
    if (calibrationArucoGroup) calibrationArucoGroup.visible = v;
    if (calibrationArucoPoseGroup) calibrationArucoPoseGroup.visible = v;
  },
);

watch(
  () => props.showHuman,
  (v) => {
    if (humanRoot) humanRoot.visible = v;
    for (const decal of humanLetterDecals) decal.visible = v;
  },
);

watch(
  () => props.showFemale,
  (v) => {
    if (femaleRoot) femaleRoot.visible = v;
  },
);

watch(
  () => props.showRealman,
  (v) => {
    if (realmanRoot) realmanRoot.visible = v;
  },
);

watch(
  () => props.showSkinSurface,
  (v) => {
    if (skinSurfaceRoot) skinSurfaceRoot.visible = v;
  },
);

watch(
  () => props.showBed,
  (v) => {
    if (bedRoot) bedRoot.visible = v;
  },
);

watch(
  () => props.activeTcpEye,
  () => {
    if (renderer?.domElement) {
      layoutPipHud(renderer.domElement.clientWidth, renderer.domElement.clientHeight);
    }
  },
);
</script>

<template>
  <div class="scene-wrap">
    <canvas ref="canvasRef" class="scene-canvas" />
    <div class="pip-label" aria-hidden="true">双目 · 基线 {{ (TCP_STEREO_BASELINE_M * 100).toFixed(0) }} cm · 画中画 基于UV 左|右 · 采集：{{ activeTcpEye === "left" ? "左" : "右" }} · {{ TCP_CAM_RT_WIDTH }}×{{ TCP_CAM_RT_HEIGHT }}</div>
  </div>
</template>

<style scoped>
.scene-wrap {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.scene-canvas {
  display: block;
  width: 100%;
  height: 100%;
  touch-action: none;
}

.pip-label {
  position: absolute;
  top: 8px;
  right: 8px;
  width: calc(240px * 2 + 8px + 16px);
  max-width: calc(100% - 16px);
  text-align: center;
  pointer-events: none;
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  color: #1e293b;
  text-shadow: 0 0 6px #f8fafc, 0 0 2px #f8fafc;
  box-sizing: border-box;
  padding-right: 2px;
}
</style>
