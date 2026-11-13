<script setup lang="ts">
import { nextTick, onUnmounted, reactive, ref, watch } from "vue";
import HandEyeSolveResultDialog from "./components/HandEyeSolveResultDialog.vue";
import RobotScene from "./components/RobotScene.vue";
import StereoLocalizationDialog from "./components/StereoLocalizationDialog.vue";
import { AUBO_C5_JOINT_SPECS } from "./robot/auboC5Robot";
import {
  applyDeltaInToolFrame,
  fkTcpMatrixInBase,
  forwardKinematicsTcp,
  inverseKinematicsTcp,
  type TcpPoseBase,
} from "./robot/auboC5Kinematics";
import {
  DEFAULT_RUNTIME_TOOL_OFFSET,
  IDENTITY_TOOL_OFFSET,
  ZERO_TOOL_OFFSET,
  toolOffsetFieldsAllFinite,
  type ToolOffset,
} from "./robot/toolOffset";
import {
  getSceneArucoConfig,
  runHandEyeArucoVerify,
  sortMarkersById,
  tcpPoseFromBaseMarkerRowMajor,
  verifyInputPathsHint,
} from "./robot/handEyeArucoVerify";
import {
  buildTcpPoseExampleYaml,
  clearHandEyeCaptureData,
  exportHandEyeConfigYaml,
  handEyeCameraRelPath,
  runHandEyeSolve,
  saveHandEyeCapture,
  saveHandEyeVerifySnapshot,
} from "./robot/handEyeCalibration";
import type { HandEyeSolveCliResponse } from "./robot/handEyeSolveTypes";
import {
  captureStereoAndRunDepth,
  type StereoDepthCaptureResult,
} from "./robot/stereoDepthCapture";
import {
  loadStereoDataAssetText,
  opencvXyzBufferToThreeLocal,
  parseAsciiPlyCloud,
  pointCloudRelFromStereoResult,
} from "./robot/stereoPointCloud";
import type { StereoPixelTo3dResult } from "./robot/stereoDepthLocalization";
import { TCP_STEREO_EYE_LABEL, type TcpStereoEye } from "./robot/tcpCameraConfig";

const defaultJoints = AUBO_C5_JOINT_SPECS.map(() => 0);

/** 示教器记录的初始末端位姿（基坐标，m / °） */
const PRESET_INITIAL_TEACH_POSE: TcpPoseBase = {
  x: -0.400319,
  y: -0.121499,
  z: 0.547598,
  rxDeg: 179.9996,
  ryDeg: -0.000081,
  rzDeg: -89.99964,
};

/** 关节角（度），与场景一致 */
const jointAnglesDeg = reactive<number[]>([...defaultJoints]);

/**
 * 工具系相对法兰的偏移（m / °）：不参与末端位姿面板的 FK/IK，仅场景箭头等代码侧使用。
 * 改 `IDENTITY_TOOL_OFFSET`（toolOffset.ts）或此处初值即可。
 */
const toolOffset = reactive<ToolOffset>({ ...IDENTITY_TOOL_OFFSET });

/**
 * 定位测试「运动到该点」专用的运行时 TCP 偏移（相对法兰）。
 * 仅该动作逆解时在代码中施加，不改面板基坐标控制与场景箭头。
 */
const runtimeToolOffset = reactive<ToolOffset>({ ...DEFAULT_RUNTIME_TOOL_OFFSET });

/** 末端（法兰）在基坐标下位姿：位置 m；姿态 °（R = Rz·Ry·Rx）；控制不经工具偏移 */
const tcpPose = reactive(forwardKinematicsTcp(defaultJoints, ZERO_TOOL_OFFSET));

/**
 * 工具坐标系增量（右乘到当前工具系）。松手/确认后应用并归零。
 * 平移 m，姿态 °。
 */
const toolFrameJog = reactive({
  x: 0,
  y: 0,
  z: 0,
  rxDeg: 0,
  ryDeg: 0,
  rzDeg: 0,
});

let syncingFromJoints = false;
let syncingFromPose = false;
let syncingToolJogReset = false;

const ikHint = ref("");
let ikHintTimer = 0;

const sceneRef = ref<{
  captureTcpCameraPngBlob: () => Promise<Blob>;
  captureTcpStereoPairPngBlobs: () => Promise<{ left: Blob; right: Blob }>;
  getTcpCameraIntrinsics: () => import("./robot/tcpCameraConfig").TcpCameraIntrinsics;
  getActiveTcpEye: () => TcpStereoEye;
  showStereoLocalizationMarker: (
    opencvXyz: { x: number; y: number; z: number },
    opencvRRowMajor?: number[][],
  ) => void;
  clearStereoLocalizationMarker: () => void;
  showStereoPointCloud: (xyzThreeLocal: Float32Array, rgb01?: Float32Array | null) => void;
  clearStereoPointCloud: () => void;
  setStereoPointCloudVisible: (visible: boolean) => void;
  randomizeLetterDecalRotations: () => void;
} | null>(null);
const handEyeBusy = ref(false);
const stereoBusy = ref(false);
/** 最近一次成功的双目深度结果（供定位测试） */
const lastStereoResult = ref<StereoDepthCaptureResult | null>(null);
const stereoLocDialogOpen = ref(false);
const stereoLocMoveBusy = ref(false);
let verifyTourAbort = false;

const handEyeSolveDialog = ref<{
  open: boolean;
  parsed: HandEyeSolveCliResponse | null;
  rawJson: string;
}>({
  open: false,
  parsed: null,
  rawJson: "",
});

/** 手眼采集/校验/导出使用的双目光路，默认左相机 */
const activeHandEyeEye = ref<TcpStereoEye>("left");

function activeHandEyeEyeLabel(): string {
  return TCP_STEREO_EYE_LABEL[activeHandEyeEye.value];
}

/** 世界系 PnP 棋盘与四角 ArUco 显示（默认关闭） */
const showCalibChecker = ref(false);
const showCalibAruco = ref(false);
/** 趴在棋盘位姿上的男性人体模型（默认关闭） */
const showHuman = ref(false);
/** 趴在棋盘位姿上的女性人体模型（与男性同位姿，默认关闭） */
const showFemale = ref(false);
/** 趴在棋盘位姿上的真人模型（与男性同位姿，默认关闭） */
const showRealman = ref(false);
/** CT 皮肤表面 PLY（与真人同位姿，默认关闭） */
const showSkinSurface = ref(false);
/** 趴在棋盘位姿上的床模型（默认关闭） */
const showBed = ref(false);
/** 双目点云上景显示（拍照成功后自动加载） */
const showStereoCloud = ref(true);

function flashIkHint(msg: string): void {
  ikHint.value = msg;
  window.clearTimeout(ikHintTimer);
  ikHintTimer = window.setTimeout(() => {
    ikHint.value = "";
  }, 2600);
}

onUnmounted(() => {
  window.clearTimeout(ikHintTimer);
  verifyTourAbort = true;
});

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

async function moveTcpToPoseTarget(
  target: TcpPoseBase,
  options?: { /** 代码侧临时工具偏移（如定位测试）；缺省按法兰基坐标逆解 */ toolOffsetOverride?: ToolOffset },
): Promise<boolean> {
  const seed = jointAnglesDeg.map((v) => Number(v));
  const execOffset = options?.toolOffsetOverride ?? ZERO_TOOL_OFFSET;
  const res = inverseKinematicsTcp(target, seed, { toolOffset: { ...execOffset } });
  if (!res) return false;
  syncingFromPose = true;
  for (let i = 0; i < 6; i++) {
    jointAnglesDeg[i] = res.jointAnglesDeg[i] ?? 0;
  }
  // 面板读数始终为法兰基坐标（与控制一致）
  const p = forwardKinematicsTcp(jointAnglesDeg, ZERO_TOOL_OFFSET);
  tcpPose.x = p.x;
  tcpPose.y = p.y;
  tcpPose.z = p.z;
  tcpPose.rxDeg = p.rxDeg;
  tcpPose.ryDeg = p.ryDeg;
  tcpPose.rzDeg = p.rzDeg;
  await nextTick();
  syncingFromPose = false;
  return true;
}

async function runArucoVerifyMotionTour(markers: ReturnType<typeof sortMarkersById>): Promise<void> {
  verifyTourAbort = false;
  const ordered = sortMarkersById(markers);
  for (let i = 0; i < ordered.length; i++) {
    if (verifyTourAbort) {
      flashIkHint("手眼校验运动已取消");
      return;
    }
    const m = ordered[i];
    const target = tcpPoseFromBaseMarkerRowMajor(m.T_base_marker_row_major);
    const ok = await moveTcpToPoseTarget(target);
    if (!ok) {
      flashIkHint(`ArUco id=${m.marker_id} 逆解失败，校验运动中断`);
      return;
    }
    flashIkHint(`校验 ${i + 1}/${ordered.length}：ArUco id=${m.marker_id}，3s 后下一目标…`);
    if (i < ordered.length - 1) {
      await sleep(3000);
    }
  }
  flashIkHint(`手眼校验完成：已依次到达 ${ordered.length} 个 ArUco 位姿（${activeHandEyeEyeLabel()}）`);
}

/** 关节 → 面板末端位姿（法兰 · 基坐标） */
function fkToPose(): void {
  const p = forwardKinematicsTcp(jointAnglesDeg, ZERO_TOOL_OFFSET);
  syncingFromJoints = true;
  tcpPose.x = p.x;
  tcpPose.y = p.y;
  tcpPose.z = p.z;
  tcpPose.rxDeg = p.rxDeg;
  tcpPose.ryDeg = p.ryDeg;
  tcpPose.rzDeg = p.rzDeg;
  void nextTick(() => {
    syncingFromJoints = false;
  });
}

/** 面板末端位姿 → 关节（法兰 · 基坐标逆解，不经工具偏移） */
function ikToJoints(): void {
  const target = {
    x: Number(tcpPose.x),
    y: Number(tcpPose.y),
    z: Number(tcpPose.z),
    rxDeg: Number(tcpPose.rxDeg),
    ryDeg: Number(tcpPose.ryDeg),
    rzDeg: Number(tcpPose.rzDeg),
  };
  const seed = jointAnglesDeg.map((v) => Number(v));
  const res = inverseKinematicsTcp(target, seed, { toolOffset: { ...ZERO_TOOL_OFFSET } });
  if (!res) {
    flashIkHint("逆解未收敛或奇异，位姿已按当前关节还原");
    fkToPose();
    return;
  }
  syncingFromPose = true;
  for (let i = 0; i < 6; i++) {
    jointAnglesDeg[i] = res.jointAnglesDeg[i] ?? 0;
  }
  void nextTick(() => {
    syncingFromPose = false;
  });
}

/** 工具系增量是否全接近 0 */
function toolFrameJogNearZero(eps = 1e-9): boolean {
  return (
    Math.abs(toolFrameJog.x) <= eps &&
    Math.abs(toolFrameJog.y) <= eps &&
    Math.abs(toolFrameJog.z) <= eps &&
    Math.abs(toolFrameJog.rxDeg) <= eps &&
    Math.abs(toolFrameJog.ryDeg) <= eps &&
    Math.abs(toolFrameJog.rzDeg) <= eps
  );
}

function resetToolFrameJog(): void {
  syncingToolJogReset = true;
  toolFrameJog.x = 0;
  toolFrameJog.y = 0;
  toolFrameJog.z = 0;
  toolFrameJog.rxDeg = 0;
  toolFrameJog.ryDeg = 0;
  toolFrameJog.rzDeg = 0;
  void nextTick(() => {
    syncingToolJogReset = false;
  });
}

/**
 * 将 toolFrameJog 作为当前工具系下的增量右乘到末端，再逆解；成功后滑块归零。
 * 绑定滑块/数字框的 change（松手或确认输入）时调用。
 */
function commitToolFrameJog(): void {
  if (syncingToolJogReset) return;
  if (toolFrameJogNearZero()) return;
  const delta: TcpPoseBase = {
    x: Number(toolFrameJog.x),
    y: Number(toolFrameJog.y),
    z: Number(toolFrameJog.z),
    rxDeg: Number(toolFrameJog.rxDeg),
    ryDeg: Number(toolFrameJog.ryDeg),
    rzDeg: Number(toolFrameJog.rzDeg),
  };
  if (![delta.x, delta.y, delta.z, delta.rxDeg, delta.ryDeg, delta.rzDeg].every((v) => Number.isFinite(v))) {
    return;
  }

  const target = applyDeltaInToolFrame(jointAnglesDeg, delta, toolOffset);
  const seed = jointAnglesDeg.map((v) => Number(v));
  const res = inverseKinematicsTcp(target, seed, { toolOffset: { ...toolOffset } });
  if (!res) {
    flashIkHint("工具系运动逆解失败，已保持原位");
    resetToolFrameJog();
    return;
  }

  syncingFromPose = true;
  for (let i = 0; i < 6; i++) {
    jointAnglesDeg[i] = res.jointAnglesDeg[i] ?? 0;
  }
  // 面板仍显示法兰基坐标
  const p = forwardKinematicsTcp(jointAnglesDeg, ZERO_TOOL_OFFSET);
  tcpPose.x = p.x;
  tcpPose.y = p.y;
  tcpPose.z = p.z;
  tcpPose.rxDeg = p.rxDeg;
  tcpPose.ryDeg = p.ryDeg;
  tcpPose.rzDeg = p.rzDeg;
  void nextTick(() => {
    syncingFromPose = false;
  });
  resetToolFrameJog();
}

/** 末端位姿数字输入合法时再尝试逆解，避免输入中途的空串/NaN 误触发 */
function tcpPoseFieldsAllFinite(): boolean {
  const keys: (keyof TcpPoseBase)[] = ["x", "y", "z", "rxDeg", "ryDeg", "rzDeg"];
  for (const k of keys) {
    const v = tcpPose[k];
    if (typeof v !== "number" || !Number.isFinite(v)) return false;
  }
  return true;
}

watch(
  jointAnglesDeg,
  () => {
    if (syncingFromPose) return;
    fkToPose();
  },
  { deep: true, immediate: true },
);

watch(
  tcpPose,
  () => {
    if (syncingFromJoints) return;
    if (!tcpPoseFieldsAllFinite()) return;
    ikToJoints();
  },
  { deep: true },
);

watch(showStereoCloud, (v) => {
  sceneRef.value?.setStereoPointCloudVisible(v);
});

function resetPose(): void {
  defaultJoints.forEach((v, i) => {
    jointAnglesDeg[i] = v;
  });
}

/** 清零工具偏移（关节不动，末端读数回到法兰） */
function resetToolOffset(): void {
  toolOffset.x = 0;
  toolOffset.y = 0;
  toolOffset.z = 0;
  toolOffset.rxDeg = 0;
  toolOffset.ryDeg = 0;
  toolOffset.rzDeg = 0;
}

/** 运动到预设初始位姿（法兰基坐标示教值逆解，面板读数同步为法兰） */
function goToInitialTeachPose(): void {
  const seed = jointAnglesDeg.map((v) => Number(v));
  const res = inverseKinematicsTcp(PRESET_INITIAL_TEACH_POSE, seed, {
    logProgress: true,
    maxIter: 220,
    toolOffset: { ...ZERO_TOOL_OFFSET },
  });
  if (!res) {
    flashIkHint("初始位姿逆解失败，详情见控制台");
    return;
  }
  syncingFromPose = true;
  for (let i = 0; i < 6; i++) {
    jointAnglesDeg[i] = res.jointAnglesDeg[i] ?? 0;
  }
  const p = forwardKinematicsTcp(jointAnglesDeg, ZERO_TOOL_OFFSET);
  tcpPose.x = p.x;
  tcpPose.y = p.y;
  tcpPose.z = p.z;
  tcpPose.rxDeg = p.rxDeg;
  tcpPose.ryDeg = p.ryDeg;
  tcpPose.rzDeg = p.rzDeg;
  void nextTick(() => {
    syncingFromPose = false;
  });
  flashIkHint(res.converged ? "已运动到初始示教位姿" : "接近初始位姿（未完全收敛）");
  console.log(
    "[IK] 初始位姿应用完成 converged=",
    res.converged,
    "residual=",
    res.residual,
    "关节(°)=",
    res.jointAnglesDeg.map((d) => d.toFixed(4)),
  );
}

async function captureStereoAndProcess(): Promise<void> {
  if (stereoBusy.value || handEyeBusy.value) return;
  const scene = sceneRef.value;
  if (!scene) {
    flashIkHint("场景未就绪，请稍候再试");
    return;
  }
  stereoBusy.value = true;
  try {
    flashIkHint("双目拍照中…");
    const { left, right } = await scene.captureTcpStereoPairPngBlobs();
    const tcpYaml = buildTcpPoseExampleYaml(
      fkTcpMatrixInBase(jointAnglesDeg, toolOffset),
      "left.png",
    );
    flashIkHint("已采集左右图，正在立体深度 / 点云计算…");
    const result = await captureStereoAndRunDepth(left, right, tcpYaml);
    if (!result.success) {
      flashIkHint(`立体深度失败：${result.error ?? "未知错误"}`);
      return;
    }
    lastStereoResult.value = result;
    const ply = result.files?.pointcloud ?? "";
    // 测试：将点云钉到拍照瞬间左相机世界位姿，便于与场景模型对比
    try {
      const rel = pointCloudRelFromStereoResult(result);
      if (rel) {
        flashIkHint("正在加载点云到场景…");
        const plyText = await loadStereoDataAssetText(rel);
        const cloud = parseAsciiPlyCloud(plyText, 200_000);
        const xyzThree = opencvXyzBufferToThreeLocal(cloud.xyzOpencv);
        scene.showStereoPointCloud(xyzThree, cloud.rgb01);
        scene.setStereoPointCloudVisible(showStereoCloud.value);
        flashIkHint(
          `双目完成 ${result.pair_id ?? ""} · 点云 ${cloud.count} 点已显示${ply ? ` · ${ply}` : ""}`,
        );
      } else {
        flashIkHint(
          `双目完成 ${result.pair_id ?? ""} → ${result.output_dir ?? ""}${ply ? ` · 点云 ${ply}` : ""}（未找到 PLY 相对路径）`,
        );
      }
    } catch (cloudErr) {
      console.warn("[Stereo] 点云上景失败", cloudErr);
      flashIkHint(
        `双目完成，但点云上景失败：${cloudErr instanceof Error ? cloudErr.message : String(cloudErr)}`,
      );
    }
  } catch (err) {
    console.error("[Stereo] 双目拍照失败", err);
    flashIkHint(`双目拍照失败：${err instanceof Error ? err.message : String(err)}`);
  } finally {
    stereoBusy.value = false;
  }
}

function openStereoLocalizationTest(): void {
  if (!lastStereoResult.value?.success || !lastStereoResult.value.output_dir) {
    flashIkHint("请先完成「双目拍照」后再做定位测试");
    return;
  }
  stereoLocDialogOpen.value = true;
}

/** 随机扭转人体表面各字母贴纸朝向 */
function randomizeLetterStickers(): void {
  if (!showHuman.value) {
    flashIkHint("请先勾选「显示男性」");
    return;
  }
  sceneRef.value?.randomizeLetterDecalRotations();
  flashIkHint("已随机旋转字母贴纸");
}

function onStereoLocPointPicked(payload: {
  rect: { x: number; y: number; z: number; u: number; v: number };
  hit: StereoPixelTo3dResult;
}): void {
  const { rect, hit } = payload;
  sceneRef.value?.showStereoLocalizationMarker(
    { x: rect.x, y: rect.y, z: rect.z },
    hit.plane_ok ? hit.R_rect_row_major : undefined,
  );
  if (hit.x_base != null) {
    const n = hit.normal_rect;
    const nTxt = n ? ` n=(${n[0].toFixed(2)},${n[1].toFixed(2)},${n[2].toFixed(2)})` : "";
    flashIkHint(
      `基座 (${hit.x_base.toFixed(4)}, ${hit.y_base.toFixed(4)}, ${hit.z_base.toFixed(4)})` +
        (hit.pose_from_plane ? ` · 法向姿态${nTxt}` : " · 无平面姿态"),
    );
  } else {
    flashIkHint(
      `定位 (${rect.u},${rect.v}) → rect (${rect.x.toFixed(4)}, ${rect.y.toFixed(4)}, ${rect.z.toFixed(4)}) m`,
    );
  }
}

async function onStereoLocMoveToTarget(target: TcpPoseBase): Promise<void> {
  if (stereoLocMoveBusy.value) return;
  if (!toolOffsetFieldsAllFinite(runtimeToolOffset)) {
    flashIkHint("运行时偏移含非法数值，未运动");
    return;
  }
  stereoLocMoveBusy.value = true;
  try {
    // 定位运动：执行 TCP = 运行时偏移（非面板工具偏移）
    const ok = await moveTcpToPoseTarget(target, {
      toolOffsetOverride: { ...runtimeToolOffset },
    });
    if (!ok) {
      flashIkHint("定位点逆解失败，未运动");
      return;
    }
    flashIkHint(
      `已按运行时TCP到达 (${target.x.toFixed(4)}, ${target.y.toFixed(4)}, ${target.z.toFixed(4)}) m`,
    );
  } finally {
    stereoLocMoveBusy.value = false;
  }
}

async function captureHandEyeSample(): Promise<void> {
  if (handEyeBusy.value) return;
  const scene = sceneRef.value;
  if (!scene) {
    flashIkHint("场景未就绪，请稍候再试");
    return;
  }
  handEyeBusy.value = true;
  try {
    const blob = await scene.captureTcpCameraPngBlob();
    const pose: TcpPoseBase = {
      x: Number(tcpPose.x),
      y: Number(tcpPose.y),
      z: Number(tcpPose.z),
      rxDeg: Number(tcpPose.rxDeg),
      ryDeg: Number(tcpPose.ryDeg),
      rzDeg: Number(tcpPose.rzDeg),
    };
    const imageName = await saveHandEyeCapture(blob, pose, activeHandEyeEye.value);
    flashIkHint(
      `已保存 ${imageName}（${activeHandEyeEyeLabel()}）→ ${handEyeCameraRelPath(activeHandEyeEye.value)}/`,
    );
  } catch (err) {
    console.error("[HandEye] 采集失败", err);
    flashIkHint(`采集失败：${err instanceof Error ? err.message : String(err)}`);
  } finally {
    handEyeBusy.value = false;
  }
}

async function captureHandEyeVerifySnapshot(): Promise<void> {
  if (handEyeBusy.value) return;
  const scene = sceneRef.value;
  if (!scene) {
    flashIkHint("场景未就绪，请稍候再试");
    return;
  }
  const eye = activeHandEyeEye.value;
  const aruco = getSceneArucoConfig();
  handEyeBusy.value = true;
  try {
    const blob = await scene.captureTcpCameraPngBlob();
    const { imageName, yamlFileName } = await saveHandEyeVerifySnapshot(
      blob,
      jointAnglesDeg,
      eye,
      { ...toolOffset },
    );
    flashIkHint(`已保存 ${handEyeCameraRelPath(eye)}/${imageName} + ${yamlFileName}，正在 ArUco 检测…`);

    const parsed = await runHandEyeArucoVerify({
      stereoEye: eye,
      markerLengthM: aruco.markerLengthM,
      dict: aruco.dict,
    });

    if (!parsed.success) {
      const extra =
        parsed.detected_ids?.length ? `（已检测 id: ${parsed.detected_ids.join(", ")}）` : "";
      flashIkHint(`ArUco 校验失败：${parsed.error}${extra}`);
      return;
    }

    const ids = parsed.markers.map((m) => m.marker_id).join(", ");
    flashIkHint(`检测到 ${parsed.markers.length} 个标码 (id: ${ids})，开始按基座位姿运动…`);
    await runArucoVerifyMotionTour(parsed.markers);
  } catch (err) {
    console.error("[HandEye] 校验失败", err);
    const msg = err instanceof Error ? err.message : String(err);
    flashIkHint(`校验失败：${msg}（需 ${verifyInputPathsHint(eye)}）`);
  } finally {
    handEyeBusy.value = false;
  }
}

async function runHandEyeCalibration(): Promise<void> {
  if (handEyeBusy.value) return;
  handEyeBusy.value = true;
  try {
    const { rawJson, parsed, resultRelPath } = await runHandEyeSolve(activeHandEyeEye.value);
    handEyeSolveDialog.value = { open: true, parsed, rawJson };
    if (parsed.success) {
      flashIkHint(
        resultRelPath
          ? `手眼标定完成（${activeHandEyeEyeLabel()}）→ ${resultRelPath}`
          : `手眼标定完成（${activeHandEyeEyeLabel()}）`,
      );
    }
  } catch (err) {
    console.error("[HandEye] 标定失败", err);
    const msg = err instanceof Error ? err.message : String(err);
    handEyeSolveDialog.value = {
      open: true,
      parsed: { success: false, error: msg },
      rawJson: JSON.stringify({ success: false, error: msg }, null, 2),
    };
  } finally {
    handEyeBusy.value = false;
  }
}

async function exportHandEyeConfig(): Promise<void> {
  if (handEyeBusy.value) return;
  const scene = sceneRef.value;
  if (!scene) {
    flashIkHint("场景未就绪，请稍候再试");
    return;
  }
  handEyeBusy.value = true;
  try {
    const intrinsics = scene.getTcpCameraIntrinsics();
    const { leftPath, rightPath } = await exportHandEyeConfigYaml(intrinsics);
    flashIkHint(`已导出左右 config.yaml\n${leftPath}\n${rightPath}`);
  } catch (err) {
    console.error("[HandEye] 导出配置失败", err);
    flashIkHint(`导出失败：${err instanceof Error ? err.message : String(err)}`);
  } finally {
    handEyeBusy.value = false;
  }
}

async function clearHandEyeData(): Promise<void> {
  if (handEyeBusy.value) return;
  if (
    !window.confirm(
      "确定清除 hand_eye_data/left_camera 与 right_camera 下全部 capture_*.png，并重置各自 poses.csv？",
    )
  )
    return;
  handEyeBusy.value = true;
  try {
    const { deletedImages } = await clearHandEyeCaptureData();
    flashIkHint(
      deletedImages > 0
        ? `已清除 ${deletedImages} 张采集图，左右 poses.csv 已重置`
        : "无采集图片；左右 poses.csv 已重置",
    );
  } catch (err) {
    console.error("[HandEye] 清除失败", err);
    flashIkHint(`清除失败：${err instanceof Error ? err.message : String(err)}`);
  } finally {
    handEyeBusy.value = false;
  }
}
</script>

<template>
  <div class="app-shell">
    <aside class="panel">
      <header class="panel-head">
        <h1>仿真工具箱</h1>
      </header>

      <div class="sliders-scroll">
        <div class="sliders">
          <h2 class="section-title">关节角</h2>
          <div v-for="(spec, i) in AUBO_C5_JOINT_SPECS" :key="spec.name" class="row">
            <label :for="'j' + i">{{ spec.name }}</label>
            <div class="slider-line">
              <input
                :id="'j' + i"
                v-model.number="jointAnglesDeg[i]"
                type="range"
                :min="spec.minDeg"
                :max="spec.maxDeg"
                step="0.5"
              />
              <span class="deg">{{ jointAnglesDeg[i].toFixed(1) }}°</span>
            </div>
          </div>
        </div>

        <!--
          工具坐标偏移：界面隐藏，偏移改 IDENTITY_TOOL_OFFSET 或 App.vue 中 toolOffset 初值。
        <div class="pose-block">
          <h2 class="section-title">工具坐标偏移（相对法兰）</h2>
          <p class="pose-note">
            定义工具系相对 J6 法兰的固定变换：T_tool = T_flange · T_offset。调整后末端位姿读数与场景 Tcp
            箭头随之变化；关节角不变。姿态 °，R = Rz(rz)·Ry(ry)·Rx(rx)。
          </p>
          <div class="row pose-row-edit">
            <label for="tox">X</label>
            <div class="pose-slider-line">
              <input id="tox-slider" v-model.number="toolOffset.x" type="range" min="-0.3" max="0.3" step="0.001" />
              <input id="tox" v-model.number="toolOffset.x" type="number" step="0.001" class="pose-num" />
              <span class="pose-unit">m</span>
            </div>
          </div>
          <div class="row pose-row-edit">
            <label for="toy">Y</label>
            <div class="pose-slider-line">
              <input id="toy-slider" v-model.number="toolOffset.y" type="range" min="-0.3" max="0.3" step="0.001" />
              <input id="toy" v-model.number="toolOffset.y" type="number" step="0.001" class="pose-num" />
              <span class="pose-unit">m</span>
            </div>
          </div>
          <div class="row pose-row-edit">
            <label for="toz">Z</label>
            <div class="pose-slider-line">
              <input id="toz-slider" v-model.number="toolOffset.z" type="range" min="-0.3" max="0.3" step="0.001" />
              <input id="toz" v-model.number="toolOffset.z" type="number" step="0.001" class="pose-num" />
              <span class="pose-unit">m</span>
            </div>
          </div>
          <div class="row pose-row-edit">
            <label for="torx">Rx</label>
            <div class="pose-slider-line">
              <input id="torx-slider" v-model.number="toolOffset.rxDeg" type="range" min="-180" max="180" step="0.5" />
              <input id="torx" v-model.number="toolOffset.rxDeg" type="number" step="0.1" class="pose-num" />
              <span class="pose-unit">°</span>
            </div>
          </div>
          <div class="row pose-row-edit">
            <label for="tory">Ry</label>
            <div class="pose-slider-line">
              <input id="tory-slider" v-model.number="toolOffset.ryDeg" type="range" min="-180" max="180" step="0.5" />
              <input id="tory" v-model.number="toolOffset.ryDeg" type="number" step="0.1" class="pose-num" />
              <span class="pose-unit">°</span>
            </div>
          </div>
          <div class="row pose-row-edit">
            <label for="torz">Rz</label>
            <div class="pose-slider-line">
              <input id="torz-slider" v-model.number="toolOffset.rzDeg" type="range" min="-180" max="180" step="0.5" />
              <input id="torz" v-model.number="toolOffset.rzDeg" type="number" step="0.1" class="pose-num" />
              <span class="pose-unit">°</span>
            </div>
          </div>
          <button type="button" class="tool-reset" @click="resetToolOffset">重置工具偏移</button>
        </div>
        -->

        <div class="pose-block">
          <h2 class="section-title">末端位姿（基坐标）</h2>
          <p class="pose-note">
            法兰在基坐标下的位姿（不含工具偏移）。位置 m；姿态 °，R = Rz(rz)·Ry(ry)·Rx(rx)。滑块与数字框联动后逆解。
          </p>
          <div class="row pose-row-edit">
            <label for="px">X</label>
            <div class="pose-slider-line">
              <input id="px-slider" v-model.number="tcpPose.x" type="range" min="-1.2" max="1.2" step="0.001" />
              <input
                id="px"
                v-model.number="tcpPose.x"
                type="number"
                step="0.001"
                class="pose-num"
              />
              <span class="pose-unit">m</span>
            </div>
          </div>
          <div class="row pose-row-edit">
            <label for="py">Y</label>
            <div class="pose-slider-line">
              <input id="py-slider" v-model.number="tcpPose.y" type="range" min="-1.2" max="1.2" step="0.001" />
              <input
                id="py"
                v-model.number="tcpPose.y"
                type="number"
                step="0.001"
                class="pose-num"
              />
              <span class="pose-unit">m</span>
            </div>
          </div>
          <div class="row pose-row-edit">
            <label for="pz">Z</label>
            <div class="pose-slider-line">
              <input id="pz-slider" v-model.number="tcpPose.z" type="range" min="-1.2" max="1.2" step="0.001" />
              <input
                id="pz"
                v-model.number="tcpPose.z"
                type="number"
                step="0.001"
                class="pose-num"
              />
              <span class="pose-unit">m</span>
            </div>
          </div>
          <div class="row pose-row-edit">
            <label for="prx">Rx</label>
            <div class="pose-slider-line">
              <input id="prx-slider" v-model.number="tcpPose.rxDeg" type="range" min="-180" max="180" step="0.5" />
              <input
                id="prx"
                v-model.number="tcpPose.rxDeg"
                type="number"
                step="0.1"
                class="pose-num"
              />
              <span class="pose-unit">°</span>
            </div>
          </div>
          <div class="row pose-row-edit">
            <label for="pry">Ry</label>
            <div class="pose-slider-line">
              <input id="pry-slider" v-model.number="tcpPose.ryDeg" type="range" min="-180" max="180" step="0.5" />
              <input
                id="pry"
                v-model.number="tcpPose.ryDeg"
                type="number"
                step="0.1"
                class="pose-num"
              />
              <span class="pose-unit">°</span>
            </div>
          </div>
          <div class="row pose-row-edit">
            <label for="prz">Rz</label>
            <div class="pose-slider-line">
              <input id="prz-slider" v-model.number="tcpPose.rzDeg" type="range" min="-180" max="180" step="0.5" />
              <input
                id="prz"
                v-model.number="tcpPose.rzDeg"
                type="number"
                step="0.1"
                class="pose-num"
              />
              <span class="pose-unit">°</span>
            </div>
          </div>
        </div>

        <!--
          工具坐标系运动：界面隐藏，script 逻辑保留。
        <div class="pose-block">
          <h2 class="section-title">工具坐标系运动</h2>
          <p class="pose-note">
            相对<strong>当前工具系</strong>的增量：T′ = T · ΔT（右乘）。松手或确认数字后应用并归零。平移 m；姿态 °，R =
            Rz·Ry·Rx。
          </p>
          <div class="row pose-row-edit">
            <label for="tjx">ΔX</label>
            <div class="pose-slider-line">
              <input
                id="tjx-slider"
                v-model.number="toolFrameJog.x"
                type="range"
                min="-0.1"
                max="0.1"
                step="0.001"
                @change="commitToolFrameJog"
              />
              <input
                id="tjx"
                v-model.number="toolFrameJog.x"
                type="number"
                step="0.001"
                class="pose-num"
                @change="commitToolFrameJog"
              />
              <span class="pose-unit">m</span>
            </div>
          </div>
          <div class="row pose-row-edit">
            <label for="tjy">ΔY</label>
            <div class="pose-slider-line">
              <input
                id="tjy-slider"
                v-model.number="toolFrameJog.y"
                type="range"
                min="-0.1"
                max="0.1"
                step="0.001"
                @change="commitToolFrameJog"
              />
              <input
                id="tjy"
                v-model.number="toolFrameJog.y"
                type="number"
                step="0.001"
                class="pose-num"
                @change="commitToolFrameJog"
              />
              <span class="pose-unit">m</span>
            </div>
          </div>
          <div class="row pose-row-edit">
            <label for="tjz">ΔZ</label>
            <div class="pose-slider-line">
              <input
                id="tjz-slider"
                v-model.number="toolFrameJog.z"
                type="range"
                min="-0.1"
                max="0.1"
                step="0.001"
                @change="commitToolFrameJog"
              />
              <input
                id="tjz"
                v-model.number="toolFrameJog.z"
                type="number"
                step="0.001"
                class="pose-num"
                @change="commitToolFrameJog"
              />
              <span class="pose-unit">m</span>
            </div>
          </div>
          <div class="row pose-row-edit">
            <label for="tjrx">ΔRx</label>
            <div class="pose-slider-line">
              <input
                id="tjrx-slider"
                v-model.number="toolFrameJog.rxDeg"
                type="range"
                min="-30"
                max="30"
                step="0.5"
                @change="commitToolFrameJog"
              />
              <input
                id="tjrx"
                v-model.number="toolFrameJog.rxDeg"
                type="number"
                step="0.1"
                class="pose-num"
                @change="commitToolFrameJog"
              />
              <span class="pose-unit">°</span>
            </div>
          </div>
          <div class="row pose-row-edit">
            <label for="tjry">ΔRy</label>
            <div class="pose-slider-line">
              <input
                id="tjry-slider"
                v-model.number="toolFrameJog.ryDeg"
                type="range"
                min="-30"
                max="30"
                step="0.5"
                @change="commitToolFrameJog"
              />
              <input
                id="tjry"
                v-model.number="toolFrameJog.ryDeg"
                type="number"
                step="0.1"
                class="pose-num"
                @change="commitToolFrameJog"
              />
              <span class="pose-unit">°</span>
            </div>
          </div>
          <div class="row pose-row-edit">
            <label for="tjrz">ΔRz</label>
            <div class="pose-slider-line">
              <input
                id="tjrz-slider"
                v-model.number="toolFrameJog.rzDeg"
                type="range"
                min="-30"
                max="30"
                step="0.5"
                @change="commitToolFrameJog"
              />
              <input
                id="tjrz"
                v-model.number="toolFrameJog.rzDeg"
                type="number"
                step="0.1"
                class="pose-num"
                @change="commitToolFrameJog"
              />
              <span class="pose-unit">°</span>
            </div>
          </div>
          <button type="button" class="tool-reset" @click="commitToolFrameJog">应用增量</button>
          <button type="button" class="tool-reset" @click="resetToolFrameJog">归零增量</button>
        </div>
        -->
      </div>

      <p v-if="ikHint" class="ik-hint">{{ ikHint }}</p>

      <div class="btn-row">
        <button type="button" class="reset" @click="resetPose">恢复零位</button>
        <button type="button" class="preset" @click="goToInitialTeachPose">初始位姿</button>
      </div>

      <div class="calib-display-block">
        <h2 class="section-title">标定板显示</h2>
        <div class="calib-toggles">
          <label class="chk-line">
            <input v-model="showCalibChecker" type="checkbox" />
            显示棋盘格
          </label>
          <label class="chk-line">
            <input v-model="showCalibAruco" type="checkbox" />
            显示 ArUco
          </label>
          <label class="chk-line">
            <input v-model="showHuman" type="checkbox" />
            显示男性
          </label>
          <label class="chk-line">
            <input v-model="showFemale" type="checkbox" />
            显示女性
          </label>
          <label class="chk-line">
            <input v-model="showRealman" type="checkbox" />
            显示真人
          </label>
          <label class="chk-line">
            <input v-model="showSkinSurface" type="checkbox" />
            显示皮肤表面
          </label>
          <label class="chk-line">
            <input v-model="showBed" type="checkbox" />
            显示床
          </label>
          <label class="chk-line">
            <input v-model="showStereoCloud" type="checkbox" />
            显示双目点云
          </label>
          <button
            type="button"
            class="letter-roll"
            :disabled="!showHuman"
            @click="randomizeLetterStickers"
          >
            随机旋转字母
          </button>
        </div>
      </div>

      <div class="hand-eye-block">
        <div class="section-title-row">
          <h2 class="section-title">双目深度</h2>
          <details class="section-help">
            <summary>说明</summary>
            <p class="pose-note">
              同时保存左右末端相机图到 <code>stereo_data/pair_NNN/</code>，再调用
              <code>stereo_depth_cli.py</code>（配置
              <code>scripts/stereo_depth/config.hand_eye.yaml</code>）生成校正图、视差、深度与点云。
              「定位测试」在校正左图点选，经 R1⊤ 与左目手眼得到基座目标并可驱动运动。
            </p>
          </details>
        </div>
        <div class="btn-row">
          <button
            type="button"
            class="stereo-cap"
            :disabled="stereoBusy || handEyeBusy"
            @click="captureStereoAndProcess"
          >
            {{ stereoBusy ? "处理中…" : "双目拍照" }}
          </button>
          <button
            type="button"
            class="stereo-loc"
            :disabled="stereoBusy || handEyeBusy || !lastStereoResult?.success"
            @click="openStereoLocalizationTest"
          >
            定位测试
          </button>
        </div>
      </div>

      <div class="hand-eye-block">
        <div class="section-title-row">
          <h2 class="section-title">手眼标定采集</h2>
          <details class="section-help">
            <summary>说明</summary>
            <p class="pose-note">
              拍照写入 <code>hand_eye_data/left_camera</code> 或
              <code>right_camera</code>（递增 <code>capture_NNN.png</code> + <code>poses.csv</code>）。
              <strong>手眼校验</strong>：覆盖写入 <code>verify.png</code>、<code>tcp_pose.yaml</code>，再调用
              ArUco 解算（需已有 <code>hand_eye_result.yaml</code>）；成功则按基座位姿每隔 3s 运动至各标码。
              <strong>手眼标定</strong>对当前相机目录求解并生成 <code>hand_eye_result.yaml</code>。
            </p>
          </details>
        </div>
        <div class="camera-eye-row">
          <span class="eye-row-label">采集相机</span>
          <button
            type="button"
            class="eye-btn"
            :class="{ active: activeHandEyeEye === 'left' }"
            :disabled="handEyeBusy"
            @click="activeHandEyeEye = 'left'"
          >
            左相机
          </button>
          <button
            type="button"
            class="eye-btn"
            :class="{ active: activeHandEyeEye === 'right' }"
            :disabled="handEyeBusy"
            @click="activeHandEyeEye = 'right'"
          >
            右相机
          </button>
        </div>
        <div class="btn-row hand-eye-row-primary">
          <button type="button" class="capture" :disabled="handEyeBusy" @click="captureHandEyeSample">
            {{ handEyeBusy ? "处理中…" : "拍照" }}
          </button>
          <button type="button" class="export-cfg" :disabled="handEyeBusy" @click="exportHandEyeConfig">
            导出相机配置
          </button>
          <button type="button" class="hand-eye-solve" :disabled="handEyeBusy" @click="runHandEyeCalibration">
            {{ handEyeBusy ? "处理中…" : "手眼标定" }}
          </button>
        </div>
        <div class="btn-row hand-eye-row-secondary">
          <button type="button" class="verify-he" :disabled="handEyeBusy" @click="captureHandEyeVerifySnapshot">
            手眼校验
          </button>
          <button type="button" class="clear-data" :disabled="handEyeBusy" @click="clearHandEyeData">
            清除采集
          </button>
        </div>
      </div>
    </aside>

    <HandEyeSolveResultDialog
      :open="handEyeSolveDialog.open"
      :parsed="handEyeSolveDialog.parsed"
      :raw-json="handEyeSolveDialog.rawJson"
      @close="handEyeSolveDialog.open = false"
    />

    <StereoLocalizationDialog
      :open="stereoLocDialogOpen"
      :stereo-result="lastStereoResult"
      :move-busy="stereoLocMoveBusy"
      :runtime-tool-offset="runtimeToolOffset"
      @close="stereoLocDialogOpen = false"
      @point-picked="onStereoLocPointPicked"
      @move-to-target="onStereoLocMoveToTarget"
    />

    <main class="viewport">
      <RobotScene
        ref="sceneRef"
        :joint-angles-deg="jointAnglesDeg"
        :tool-offset="toolOffset"
        :show-checkerboard="showCalibChecker"
        :show-aruco="showCalibAruco"
        :show-human="showHuman"
        :show-female="showFemale"
        :show-realman="showRealman"
        :show-skin-surface="showSkinSurface"
        :show-bed="showBed"
        :active-tcp-eye="activeHandEyeEye"
      />
    </main>
  </div>
</template>

<style scoped>
.app-shell {
  display: flex;
  width: 100vw;
  height: 100vh;
  margin: 0;
  overflow: hidden;
  background: #f1f5f9;
  color: #0f172a;
}

.panel {
  width: 380px;
  flex-shrink: 0;
  padding: 1.1rem 1.15rem;
  border-right: 1px solid #e2e8f0;
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  box-shadow: 2px 0 12px rgba(15, 23, 42, 0.04);
}

.panel-head h1 {
  margin: 0 0 0.35rem;
  font-size: 1.15rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: #0f172a;
}

.sub {
  margin: 0;
  font-size: 0.78rem;
  line-height: 1.45;
  color: #475569;
}

.sub code {
  font-size: 0.72rem;
  color: #0369a1;
  background: #e0f2fe;
  padding: 0.1em 0.35em;
  border-radius: 4px;
}

.sub em {
  font-style: normal;
  color: #4338ca;
}

.sliders-scroll {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding-right: 0.25rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.section-title {
  margin: 0;
  font-size: 0.82rem;
  font-weight: 600;
  color: #334155;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.section-title-row {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.35rem 0.65rem;
  margin-bottom: 0.35rem;
}

.section-title-row .section-title {
  margin: 0;
}

.pose-note {
  margin: -0.25rem 0 0.5rem;
  font-size: 0.7rem;
  color: #64748b;
  line-height: 1.35;
}

.section-help {
  margin: 0;
  flex: 1 1 auto;
  min-width: 0;
}

.section-help > summary {
  cursor: pointer;
  user-select: none;
  list-style: none;
  font-size: 0.72rem;
  color: #0284c7;
  width: fit-content;
  line-height: 1.2;
}

.section-help > summary::-webkit-details-marker {
  display: none;
}

.section-help > summary::before {
  content: "▸ ";
  color: #64748b;
}

.section-help[open] > summary::before {
  content: "▾ ";
}

.section-help[open] {
  flex: 1 1 100%;
}

.section-help[open] > summary {
  margin-bottom: 0.25rem;
}

.section-help .pose-note {
  margin: 0;
}

.sliders,
.pose-block {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
}

.row label {
  display: block;
  font-size: 0.8rem;
  margin-bottom: 0.2rem;
  color: #334155;
}

.slider-line {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.pose-slider-line {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.pose-slider-line input[type="range"] {
  flex: 1;
  min-width: 0;
  accent-color: #0284c7;
}

.pose-slider-line input[type="number"].pose-num {
  flex: 0 0 4.75rem;
  width: 4.75rem;
  min-width: 4.75rem;
}

input[type="number"].pose-num {
  flex: 1;
  min-width: 0;
  padding: 0.35rem 0.5rem;
  border-radius: 6px;
  border: 1px solid #cbd5e1;
  font-size: 0.82rem;
  font-variant-numeric: tabular-nums;
  color: #0f172a;
  background: #fff;
}

input[type="number"].pose-num:focus {
  outline: none;
  border-color: #0284c7;
  box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.2);
}

.pose-unit {
  flex-shrink: 0;
  font-size: 0.72rem;
  color: #64748b;
  min-width: 1.25rem;
}

input[type="range"] {
  flex: 1;
  accent-color: #0284c7;
}

.deg,
.num {
  font-variant-numeric: tabular-nums;
  font-size: 0.78rem;
  min-width: 3.6rem;
  text-align: right;
  color: #64748b;
}

.ik-hint {
  margin: 0;
  font-size: 0.72rem;
  color: #b45309;
  line-height: 1.35;
}

.reset {
  margin-top: 0.1rem;
  padding: 0.45rem 0.75rem;
  border-radius: 8px;
  border: 1px solid #cbd5e1;
  background: #ffffff;
  color: #0f172a;
  font-size: 0.85rem;
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

.reset:hover {
  border-color: #0284c7;
  background: #f0f9ff;
}

.tool-reset {
  margin-top: 0.45rem;
  width: 100%;
  padding: 0.4rem 0.75rem;
  border-radius: 8px;
  border: 1px solid #cbd5e1;
  background: #f8fafc;
  color: #475569;
  font-size: 0.8rem;
  cursor: pointer;
}

.tool-reset:hover {
  border-color: #94a3b8;
  background: #f1f5f9;
}

.btn-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.1rem;
}

.btn-row .reset {
  margin-top: 0;
  flex: 1;
  min-width: 0;
}

.preset {
  flex: 1;
  min-width: 0;
  padding: 0.45rem 0.75rem;
  border-radius: 8px;
  border: 1px solid #86efac;
  background: #ecfdf5;
  color: #14532d;
  font-size: 0.85rem;
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

.preset:hover {
  border-color: #22c55e;
  background: #d1fae5;
}

.pose-note code {
  font-size: 0.68rem;
  color: #0369a1;
  background: #e0f2fe;
  padding: 0.08em 0.3em;
  border-radius: 4px;
}

.camera-eye-row {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  flex-wrap: wrap;
}

.eye-row-label {
  font-size: 0.78rem;
  font-weight: 600;
  color: #475569;
  flex: 0 0 auto;
}

.eye-btn {
  flex: 1;
  min-width: 0;
  padding: 0.38rem 0.6rem;
  border-radius: 8px;
  border: 1px solid #cbd5e1;
  background: #fff;
  color: #334155;
  font-size: 0.8rem;
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
}

.eye-btn:hover:not(:disabled) {
  border-color: #0284c7;
  background: #f0f9ff;
}

.eye-btn.active {
  border-color: #0284c7;
  background: #e0f2fe;
  color: #0c4a6e;
  font-weight: 600;
}

.eye-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.hand-eye-block {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}

.calib-display-block {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}

.calib-toggles {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.45rem 0.9rem;
}

.chk-line {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.8rem;
  color: #334155;
  cursor: pointer;
  white-space: nowrap;
}

.chk-line input {
  accent-color: #0284c7;
  width: 1rem;
  height: 1rem;
}

.letter-roll {
  flex: 1 1 100%;
  min-width: 0;
  margin-top: 0.15rem;
  padding: 0.4rem 0.75rem;
  border-radius: 8px;
  border: 1px solid #fdba74;
  background: #fff7ed;
  color: #9a3412;
  font-size: 0.85rem;
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

.letter-roll:hover:not(:disabled) {
  border-color: #fb923c;
  background: #ffedd5;
}

.letter-roll:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.capture {
  flex: 1;
  min-width: 0;
  padding: 0.45rem 0.75rem;
  border-radius: 8px;
  border: 1px solid #93c5fd;
  background: #eff6ff;
  color: #1e3a8a;
  font-size: 0.85rem;
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

.capture:hover:not(:disabled) {
  border-color: #3b82f6;
  background: #dbeafe;
}

.capture:disabled,
.stereo-cap:disabled,
.stereo-loc:disabled,
.export-cfg:disabled,
.hand-eye-solve:disabled,
.verify-he:disabled,
.clear-data:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.stereo-cap {
  flex: 1;
  min-width: 0;
  padding: 0.45rem 0.75rem;
  border-radius: 8px;
  border: 1px solid #5eead4;
  background: #f0fdfa;
  color: #115e59;
  font-size: 0.85rem;
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

.stereo-cap:hover:not(:disabled) {
  border-color: #14b8a6;
  background: #ccfbf1;
}

.stereo-loc {
  flex: 1;
  min-width: 0;
  padding: 0.45rem 0.75rem;
  border-radius: 8px;
  border: 1px solid #67e8f9;
  background: #ecfeff;
  color: #155e75;
  font-size: 0.85rem;
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

.stereo-loc:hover:not(:disabled) {
  border-color: #06b6d4;
  background: #cffafe;
}

.export-cfg {
  flex: 1;
  min-width: 0;
  padding: 0.45rem 0.75rem;
  border-radius: 8px;
  border: 1px solid #c4b5fd;
  background: #f5f3ff;
  color: #4c1d95;
  font-size: 0.85rem;
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

.export-cfg:hover:not(:disabled) {
  border-color: #8b5cf6;
  background: #ede9fe;
}

.verify-he {
  flex: 1;
  min-width: 0;
  padding: 0.45rem 0.75rem;
  border-radius: 8px;
  border: 1px solid #5eead4;
  background: #f0fdfa;
  color: #0f766e;
  font-size: 0.85rem;
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

.verify-he:hover:not(:disabled) {
  border-color: #14b8a6;
  background: #ccfbf1;
}

.hand-eye-row-primary .capture,
.hand-eye-row-primary .export-cfg,
.hand-eye-row-primary .hand-eye-solve,
.hand-eye-row-secondary .verify-he,
.hand-eye-row-secondary .clear-data {
  flex: 1;
  min-width: 0;
}

.hand-eye-solve {
  padding: 0.45rem 0.75rem;
  border-radius: 8px;
  border: 1px solid #fcd34d;
  background: #fffbeb;
  color: #92400e;
  font-size: 0.85rem;
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

.hand-eye-solve:hover:not(:disabled) {
  border-color: #f59e0b;
  background: #fef3c7;
}

.clear-data {
  min-width: 0;
  padding: 0.45rem 0.75rem;
  border-radius: 8px;
  border: 1px solid #fca5a5;
  background: #fef2f2;
  color: #991b1b;
  font-size: 0.85rem;
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

.clear-data:hover:not(:disabled) {
  border-color: #ef4444;
  background: #fee2e2;
}

.viewport {
  flex: 1;
  min-width: 0;
  position: relative;
}
</style>

<style>
html,
body {
  margin: 0;
  padding: 0;
  overflow: hidden;
  background: #f1f5f9;
}

#app {
  width: 100%;
  height: 100%;
}
</style>
