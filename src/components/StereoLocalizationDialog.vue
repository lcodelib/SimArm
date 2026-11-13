<script setup lang="ts">
/**
 * 定位测试：校正左图点选 → 方案 A 手眼；可选 YOLO 推理显示。
 */
import { computed, onUnmounted, ref, watch } from "vue";
import type { TcpPoseBase } from "../robot/auboC5Kinematics";
import type { ToolOffset } from "../robot/toolOffset";
import type { StereoDepthCaptureResult } from "../robot/stereoDepthCapture";
import {
  loadStereoDataAssetBlobUrl,
  rectifiedLeftAbsFromResult,
  rectifiedLeftRelFromResult,
  resolveLeftHandEyeAbs,
  stereoPixelToCameraXyz,
  tcpPosePathFromStereoResult,
  type StereoPixelTo3dResult,
} from "../robot/stereoDepthLocalization";
import {
  runYoloOnnxInfer,
  stereoDataRelFromAbs,
  type YoloDetection,
  type YoloInferResult,
} from "../robot/yoloOnnxInfer";

const props = defineProps<{
  open: boolean;
  stereoResult: StereoDepthCaptureResult | null;
  /** 运动中禁用「运动到该点」 */
  moveBusy?: boolean;
  /**
   * 定位「运动到该点」使用的运行时 TCP 偏移（相对法兰）。
   * 与面板工具偏移独立；直接编辑该对象字段即可。
   */
  runtimeToolOffset: ToolOffset;
}>();

const emit = defineEmits<{
  close: [];
  /** 校正左目点（场景小球）+ 方案 A 完整结果 */
  "point-picked": [
    payload: { rect: { x: number; y: number; z: number; u: number; v: number }; hit: StereoPixelTo3dResult },
  ];
  "move-to-target": [target: TcpPoseBase];
}>();

const imageUrl = ref<string | null>(null);
const loadError = ref("");
const lookupBusy = ref(false);
const yoloBusy = ref(false);
const lookupHint = ref("点击校正左图：3×3cm 拟合法向 → 姿态 → 手眼");
const lastPick = ref<StereoPixelTo3dResult | null>(null);
const yoloResult = ref<YoloInferResult | null>(null);
const imgRef = ref<HTMLImageElement | null>(null);
const handEyeAbs = ref<string | null>(null);
/** 当前预览是否为 YOLO 可视化图 */
const showingYoloVis = ref(false);

let blobUrlToRevoke: string | null = null;

const canMove = computed(() => {
  const t = lastPick.value?.tcp_target;
  return Boolean(
    t && lastPick.value?.success && !props.moveBusy && !lookupBusy.value && !yoloBusy.value,
  );
});

const canYolo = computed(() => {
  return Boolean(props.stereoResult?.success && imageUrl.value && !yoloBusy.value && !lookupBusy.value);
});

const yoloDetections = computed((): YoloDetection[] => yoloResult.value?.detections ?? []);

function revokePreview(): void {
  if (blobUrlToRevoke) {
    URL.revokeObjectURL(blobUrlToRevoke);
    blobUrlToRevoke = null;
  }
  imageUrl.value = null;
}

async function loadPreview(): Promise<void> {
  revokePreview();
  loadError.value = "";
  lastPick.value = null;
  yoloResult.value = null;
  showingYoloVis.value = false;
  handEyeAbs.value = null;
  lookupHint.value = "点击校正左图：3×3cm 拟合法向 → 姿态 → 手眼";
  const result = props.stereoResult;
  if (!result?.success || !result.output_dir) {
    loadError.value = "无有效双目结果，请先执行「双目拍照」";
    return;
  }
  const rel = rectifiedLeftRelFromResult(result);
  if (!rel) {
    loadError.value = "结果中缺少校正左图路径";
    return;
  }
  try {
    handEyeAbs.value = await resolveLeftHandEyeAbs();
    const url = await loadStereoDataAssetBlobUrl(rel);
    blobUrlToRevoke = url;
    imageUrl.value = url;
  } catch (err) {
    loadError.value = err instanceof Error ? err.message : String(err);
  }
}

watch(
  () => [props.open, props.stereoResult?.pair_id, props.stereoResult?.output_dir] as const,
  ([open]) => {
    if (open) void loadPreview();
    else revokePreview();
  },
);

onUnmounted(() => {
  revokePreview();
});

function onOverlayClick(e: MouseEvent): void {
  if ((e.target as HTMLElement).classList.contains("loc-dialog-overlay")) {
    emit("close");
  }
}

function formatHitHint(hit: StereoPixelTo3dResult): string {
  if (!hit.success) return hit.error ?? "定位失败";
  const lines = [
    `像素 (${hit.u}, ${hit.v})`,
    `rect (${hit.x?.toFixed(4)}, ${hit.y?.toFixed(4)}, ${hit.z?.toFixed(4)})`,
  ];
  if (hit.plane_ok && hit.normal_rect) {
    const n = hit.normal_rect;
    lines.push(
      `平面 n=${hit.plane_point_count}点 rmse=${(hit.plane_rmse_m ?? 0).toFixed(4)} ` +
        `n=(${n[0].toFixed(3)},${n[1].toFixed(3)},${n[2].toFixed(3)})`,
    );
  } else if (hit.plane_error) {
    lines.push(`法向失败: ${hit.plane_error}`);
  }
  if (hit.x_base != null) {
    lines.push(`base (${hit.x_base.toFixed(4)}, ${hit.y_base.toFixed(4)}, ${hit.z_base.toFixed(4)})`);
  }
  if (hit.r1_assumed_identity) {
    lines.push("（npz 无 R1，已按 I 处理）");
  }
  return lines.join(" · ");
}

async function onImageClick(e: MouseEvent): Promise<void> {
  const img = imgRef.value;
  const result = props.stereoResult;
  if (!img || !result?.output_dir || lookupBusy.value || yoloBusy.value) return;

  const rect = img.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
  const natW = img.naturalWidth || 1;
  const natH = img.naturalHeight || 1;
  const u = ((e.clientX - rect.left) / rect.width) * natW;
  const v = ((e.clientY - rect.top) / rect.height) * natH;

  const tcpPosePath = tcpPosePathFromStereoResult(result);
  if (!tcpPosePath) {
    lookupHint.value = "缺少拍照 TCP（tcp_pose.yaml），请重新执行「双目拍照」";
    return;
  }
  if (!handEyeAbs.value) {
    lookupHint.value = "无法解析左目 hand_eye_result.yaml 路径";
    return;
  }

  lookupBusy.value = true;
  lookupHint.value = `方案 A 查询 (${Math.round(u)}, ${Math.round(v)})…`;
  try {
    const hit = await stereoPixelToCameraXyz(result.output_dir, u, v, {
      handEyePath: handEyeAbs.value,
      tcpPosePath,
    });
    lastPick.value = hit;
    lookupHint.value = formatHitHint(hit);
    if (!hit.success || hit.x == null || hit.y == null || hit.z == null) {
      return;
    }
    emit("point-picked", {
      rect: {
        x: hit.x,
        y: hit.y,
        z: hit.z,
        u: hit.u ?? Math.round(u),
        v: hit.v ?? Math.round(v),
      },
      hit,
    });
  } catch (err) {
    lookupHint.value = err instanceof Error ? err.message : String(err);
  } finally {
    lookupBusy.value = false;
  }
}

async function onYoloClick(): Promise<void> {
  const result = props.stereoResult;
  if (!result || yoloBusy.value) return;
  const imageAbs = rectifiedLeftAbsFromResult(result);
  if (!imageAbs) {
    lookupHint.value = "无法解析校正左图路径";
    return;
  }
  const outDir = result.output_dir
    ? `${result.output_dir.replace(/\\/g, "/").replace(/\/+$/, "")}/yolo`
    : undefined;

  yoloBusy.value = true;
  lookupHint.value = "YOLO 推理中…";
  try {
    const infer = await runYoloOnnxInfer(imageAbs, outDir);
    yoloResult.value = infer;
    if (!infer.success && infer.error) {
      lookupHint.value = infer.error;
      return;
    }
    const n = infer.num_detections ?? infer.detections?.length ?? 0;
    lookupHint.value = `YOLO 完成：${n} 个目标`;
    if (infer.vis) {
      const rel = stereoDataRelFromAbs(infer.vis);
      if (rel) {
        const url = await loadStereoDataAssetBlobUrl(rel);
        revokePreview();
        blobUrlToRevoke = url;
        imageUrl.value = url;
        showingYoloVis.value = true;
      } else {
        lookupHint.value = `YOLO 完成，但可视化路径不在 stereo_data 下: ${infer.vis}`;
      }
    }
  } catch (err) {
    lookupHint.value = err instanceof Error ? err.message : String(err);
  } finally {
    yoloBusy.value = false;
  }
}

function onMoveClick(): void {
  const t = lastPick.value?.tcp_target;
  if (!t) return;
  emit("move-to-target", { ...t });
}
</script>

<template>
  <div v-if="open" class="loc-dialog-overlay" @click="onOverlayClick">
    <div class="loc-dialog" role="dialog" aria-modal="true" aria-labelledby="loc-dialog-title" @click.stop>
      <header class="loc-dialog-head">
        <h2 id="loc-dialog-title">定位测试</h2>
        <button type="button" class="loc-close" aria-label="关闭" @click="emit('close')">×</button>
      </header>
      <div class="loc-body">
        <p v-if="loadError" class="loc-error">{{ loadError }}</p>
        <template v-else-if="imageUrl">
          <button
            type="button"
            class="loc-img-wrap"
            :disabled="lookupBusy || yoloBusy"
            @click="onImageClick"
          >
            <img
              ref="imgRef"
              :src="imageUrl"
              :alt="showingYoloVis ? 'YOLO 可视化' : '校正左图'"
              draggable="false"
            />
          </button>
          <ul v-if="yoloDetections.length" class="loc-yolo-list">
            <li v-for="(d, i) in yoloDetections" :key="i">
              {{ d.cls_name }} · {{ d.conf.toFixed(3) }} · box
              ({{ d.xyxy.map((v) => v.toFixed(0)).join(", ") }})
            </li>
          </ul>
        </template>
        <p v-else class="loc-loading">加载图像中…</p>
      </div>
      <footer class="loc-dialog-foot">
        <p class="loc-hint">{{ lookupHint }}</p>
        <div class="loc-runtime">
          <p class="loc-runtime-title">运行时 TCP 偏移（仅「运动到该点」）</p>
          <p class="loc-runtime-note">
            相对法兰；逆解执行 TCP 用此偏移，不改面板工具偏移。姿态 R=Rz·Ry·Rx。
          </p>
          <div class="loc-runtime-grid">
            <label>
              X
              <input v-model.number="runtimeToolOffset.x" type="number" step="0.001" />
              <span>m</span>
            </label>
            <label>
              Y
              <input v-model.number="runtimeToolOffset.y" type="number" step="0.001" />
              <span>m</span>
            </label>
            <label>
              Z
              <input v-model.number="runtimeToolOffset.z" type="number" step="0.001" />
              <span>m</span>
            </label>
            <label>
              Rx
              <input v-model.number="runtimeToolOffset.rxDeg" type="number" step="0.1" />
              <span>°</span>
            </label>
            <label>
              Ry
              <input v-model.number="runtimeToolOffset.ryDeg" type="number" step="0.1" />
              <span>°</span>
            </label>
            <label>
              Rz
              <input v-model.number="runtimeToolOffset.rzDeg" type="number" step="0.1" />
              <span>°</span>
            </label>
          </div>
        </div>
        <button type="button" class="loc-yolo" :disabled="!canYolo" @click="onYoloClick">
          {{ yoloBusy ? "推理中…" : "YOLO 推理" }}
        </button>
        <button type="button" class="loc-move" :disabled="!canMove" @click="onMoveClick">
          {{ moveBusy ? "运动中…" : "运动到该点" }}
        </button>
        <button type="button" class="loc-ok" @click="emit('close')">关闭</button>
      </footer>
    </div>
  </div>
</template>

<style scoped>
.loc-dialog-overlay {
  position: fixed;
  inset: 0;
  z-index: 80;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(15, 23, 42, 0.45);
  padding: 1rem;
}

.loc-dialog {
  width: min(960px, 96vw);
  max-height: 92vh;
  display: flex;
  flex-direction: column;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 20px 50px rgba(15, 23, 42, 0.25);
  overflow: hidden;
}

.loc-dialog-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.85rem 1rem;
  border-bottom: 1px solid #e2e8f0;
}

.loc-dialog-head h2 {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 600;
  color: #0f172a;
}

.loc-close {
  border: none;
  background: transparent;
  font-size: 1.4rem;
  line-height: 1;
  color: #64748b;
  cursor: pointer;
  padding: 0.15rem 0.4rem;
}

.loc-body {
  flex: 1;
  min-height: 0;
  padding: 0.75rem 1rem;
  overflow: auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
}

.loc-img-wrap {
  display: block;
  padding: 0;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  background: #0f172a;
  cursor: crosshair;
  max-width: 100%;
}

.loc-img-wrap:disabled {
  cursor: wait;
  opacity: 0.85;
}

.loc-img-wrap img {
  display: block;
  max-width: 100%;
  max-height: min(62vh, 640px);
  height: auto;
  user-select: none;
  pointer-events: none;
}

.loc-error,
.loc-loading {
  margin: 1rem 0;
  font-size: 0.9rem;
  color: #b91c1c;
}

.loc-loading {
  color: #64748b;
}

.loc-dialog-foot {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  border-top: 1px solid #e2e8f0;
}

.loc-hint {
  flex: 1 1 100%;
  margin: 0;
  font-size: 0.72rem;
  line-height: 1.4;
  color: #334155;
  font-family: ui-monospace, Consolas, monospace;
}

.loc-runtime {
  flex: 1 1 100%;
  padding: 0.55rem 0.65rem;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #f8fafc;
}

.loc-runtime-title {
  margin: 0 0 0.2rem;
  font-size: 0.78rem;
  font-weight: 600;
  color: #0f172a;
}

.loc-runtime-note {
  margin: 0 0 0.45rem;
  font-size: 0.68rem;
  color: #64748b;
  line-height: 1.35;
}

.loc-runtime-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.35rem 0.5rem;
}

.loc-runtime-grid label {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  font-size: 0.72rem;
  color: #475569;
}

.loc-runtime-grid input {
  flex: 1;
  min-width: 0;
  padding: 0.2rem 0.3rem;
  border: 1px solid #cbd5e1;
  border-radius: 4px;
  font-size: 0.75rem;
  font-family: ui-monospace, Consolas, monospace;
}

.loc-runtime-grid span {
  flex-shrink: 0;
  width: 0.9rem;
  color: #94a3b8;
}

.loc-yolo-list {
  width: 100%;
  max-height: 7.5rem;
  overflow: auto;
  margin: 0.65rem 0 0;
  padding: 0.4rem 0.6rem;
  list-style: none;
  font-size: 0.75rem;
  font-family: ui-monospace, Consolas, monospace;
  color: #334155;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
}

.loc-yolo-list li {
  padding: 0.15rem 0;
  border-bottom: 1px solid #f1f5f9;
}

.loc-yolo-list li:last-child {
  border-bottom: none;
}

.loc-yolo {
  flex-shrink: 0;
  padding: 0.4rem 0.85rem;
  border-radius: 8px;
  border: 1px solid #fcd34d;
  background: #fffbeb;
  color: #92400e;
  font-size: 0.85rem;
  cursor: pointer;
}

.loc-yolo:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.loc-move {
  flex-shrink: 0;
  padding: 0.4rem 0.85rem;
  border-radius: 8px;
  border: 1px solid #67e8f9;
  background: #ecfeff;
  color: #155e75;
  font-size: 0.85rem;
  cursor: pointer;
}

.loc-move:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.loc-ok {
  flex-shrink: 0;
  padding: 0.4rem 0.9rem;
  border-radius: 8px;
  border: 1px solid #94a3b8;
  background: #f8fafc;
  color: #0f172a;
  font-size: 0.85rem;
  cursor: pointer;
}
</style>
