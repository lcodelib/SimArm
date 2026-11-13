import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { defineConfig, type Plugin } from "vite";
import vue from "@vitejs/plugin-vue";
import { decodeSubprocessOutput } from "./src/robot/subprocessText";

function pythonSpawnEnv(): NodeJS.ProcessEnv {
  return {
    ...process.env,
    PYTHONIOENCODING: "utf-8",
    PYTHONUTF8: "1",
  };
}

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;

const HAND_EYE_DATA = "hand_eye_data";
const HAND_EYE_LEFT = "left_camera";
const HAND_EYE_RIGHT = "right_camera";
const HAND_EYE_LEGACY_CONFIG = path.join("scripts", "hand_eye_calibration", "config.yaml");
const HAND_EYE_VERIFY_IMAGE = "verify.png";
const HAND_EYE_VERIFY_POSE_YAML = "tcp_pose.yaml";
const HAND_EYE_CONFIG_NAME = "config.yaml";
const POSES_HEADER = "image,tx,ty,tz,qx,qy,qz,qw\n";

function projectRoot(): string {
  return process.cwd();
}

function handEyeDataDir(): string {
  return path.join(projectRoot(), HAND_EYE_DATA);
}

function parseStereoEye(raw: string | null): "left" | "right" | null {
  if (raw === "left" || raw === "right") return raw;
  return null;
}

function cameraSubdir(eye: "left" | "right"): string {
  return path.join(handEyeDataDir(), eye === "left" ? HAND_EYE_LEFT : HAND_EYE_RIGHT);
}

function ensureCameraDir(eye: "left" | "right"): string {
  const dir = cameraSubdir(eye);
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

function nextCaptureIndexIn(dir: string): number {
  if (!fs.existsSync(dir)) return 1;
  let max = 0;
  for (const name of fs.readdirSync(dir)) {
    const m = /^capture_(\d+)\.png$/i.exec(name);
    if (m) max = Math.max(max, parseInt(m[1], 10));
  }
  return max + 1;
}

function clearCameraCaptureDir(dir: string): number {
  let deletedImages = 0;
  if (!fs.existsSync(dir)) return 0;
  for (const name of fs.readdirSync(dir)) {
    if (/^capture_\d+\.png$/i.test(name)) {
      fs.unlinkSync(path.join(dir, name));
      deletedImages += 1;
    }
  }
  fs.writeFileSync(path.join(dir, "poses.csv"), POSES_HEADER, "utf8");
  return deletedImages;
}

function clearHandEyeDataDir(): { deletedImages: number } {
  let deletedImages = 0;
  deletedImages += clearCameraCaptureDir(cameraSubdir("left"));
  deletedImages += clearCameraCaptureDir(cameraSubdir("right"));
  return { deletedImages };
}

const HAND_EYE_RESULT_YAML = "hand_eye_result.yaml";

function runArucoVerifyPy(
  eye: "left" | "right",
  markerLengthM: number,
  dict: string,
): Promise<string> {
  const dir = ensureCameraDir(eye);
  const image = path.join(dir, HAND_EYE_VERIFY_IMAGE);
  const handEye = path.join(dir, HAND_EYE_RESULT_YAML);
  const tcpPose = path.join(dir, HAND_EYE_VERIFY_POSE_YAML);
  if (!fs.existsSync(image)) {
    return Promise.reject(new Error(`未找到校验图像: ${image}`));
  }
  if (!fs.existsSync(handEye)) {
    return Promise.reject(new Error(`未找到 ${handEye}，请先完成手眼标定`));
  }
  if (!fs.existsSync(tcpPose)) {
    return Promise.reject(new Error(`未找到 ${tcpPose}`));
  }
  const scriptPath = path.join(projectRoot(), "scripts", "hand_eye_calibration", "aruco_to_robot_base_cli.py");
  return new Promise((resolve, reject) => {
    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];
    const child = spawn(
      "python",
      [
        scriptPath,
        "--image",
        image,
        "--hand-eye",
        handEye,
        "--tcp-pose",
        tcpPose,
        "--marker-length",
        String(markerLengthM),
        "--dict",
        dict,
      ],
      { cwd: projectRoot(), windowsHide: true, env: pythonSpawnEnv() },
    );
    child.stdout.on("data", (d: Buffer) => stdoutChunks.push(d));
    child.stderr.on("data", (d: Buffer) => stderrChunks.push(d));
    child.on("error", (err) => reject(err));
    child.on("close", (code) => {
      const stdout = decodeSubprocessOutput(new Uint8Array(Buffer.concat(stdoutChunks)));
      const stderr = decodeSubprocessOutput(new Uint8Array(Buffer.concat(stderrChunks)));
      const line = stdout.trim().split(/\r?\n/).filter(Boolean).pop() ?? "";
      if (!line) {
        reject(new Error(stderr || stdout || `ArUco 校验无 JSON 输出 (exit ${code})`));
        return;
      }
      try {
        JSON.parse(line);
        resolve(line);
      } catch {
        reject(new Error(stderr || stdout || `ArUco 校验 JSON 解析失败 (exit ${code})`));
      }
    });
  });
}

function runHandEyeSolvePy(eye: "left" | "right"): Promise<string> {
  const configPath = path.join(cameraSubdir(eye), HAND_EYE_CONFIG_NAME);
  if (!fs.existsSync(configPath)) {
    return Promise.reject(new Error(`未找到 ${configPath}，请先导出相机配置`));
  }
  const scriptPath = path.join(projectRoot(), "scripts", "hand_eye_calibration", "hand_eye_solve_cli.py");
  return new Promise((resolve, reject) => {
    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];
    const child = spawn("python", [scriptPath, "--config", configPath], {
      cwd: projectRoot(),
      windowsHide: true,
      env: pythonSpawnEnv(),
    });
    child.stdout.on("data", (d: Buffer) => {
      stdoutChunks.push(d);
    });
    child.stderr.on("data", (d: Buffer) => {
      stderrChunks.push(d);
    });
    child.on("error", (err) => reject(err));
    child.on("close", (code) => {
      const stdout = decodeSubprocessOutput(new Uint8Array(Buffer.concat(stdoutChunks)));
      const stderr = decodeSubprocessOutput(new Uint8Array(Buffer.concat(stderrChunks)));
      const line = stdout.trim().split(/\r?\n/).filter(Boolean).pop() ?? "";
      if (!line) {
        reject(new Error(stderr || stdout || `手眼标定无 JSON 输出 (exit ${code})`));
        return;
      }
      try {
        JSON.parse(line);
        resolve(line);
      } catch {
        reject(new Error(stderr || stdout || `手眼标定 JSON 解析失败 (exit ${code})`));
      }
    });
  });
}

/** 开发模式（npm run dev）下将手眼数据写入工程 hand_eye_data / config.yaml */
function handEyeDevApi(): Plugin {
  return {
    name: "hand-eye-dev-api",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = req.url?.split("?")[0] ?? "";
        if (url === "/__hand_eye/project-root" && req.method === "GET") {
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ root: projectRoot().replace(/\\/g, "/") }));
          return;
        }
        if (url.startsWith("/__hand_eye/next-index") && req.method === "GET") {
          const eyeParam = new URL(req.url ?? "", "http://local").searchParams.get("eye");
          const eye = parseStereoEye(eyeParam);
          if (!eye) {
            res.statusCode = 400;
            res.end("missing or invalid eye");
            return;
          }
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ index: nextCaptureIndexIn(ensureCameraDir(eye)) }));
          return;
        }
        if (url === "/__hand_eye/clear" && req.method === "POST") {
          try {
            const result = clearHandEyeDataDir();
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(result));
          } catch (e) {
            res.statusCode = 500;
            res.end(String(e));
          }
          return;
        }
        if (
          url !== "/__hand_eye/capture" &&
          url !== "/__hand_eye/export-config" &&
          url !== "/__hand_eye/verify" &&
          url !== "/__hand_eye/solve" &&
          url !== "/__hand_eye/aruco-verify"
        ) {
          next();
          return;
        }
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end("Method Not Allowed");
          return;
        }
        const chunks: Buffer[] = [];
        req.on("data", (c) => chunks.push(c));
        req.on("end", () => {
          try {
            const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, string>;
            if (url === "/__hand_eye/solve") {
              const eye = parseStereoEye(body.stereoEye ?? null);
              if (!eye) {
                res.statusCode = 400;
                res.end("missing or invalid stereoEye");
                return;
              }
              runHandEyeSolvePy(eye)
                .then((rawJsonLine) => {
                  res.setHeader("Content-Type", "application/json; charset=utf-8");
                  res.end(rawJsonLine);
                })
                .catch((e) => {
                  res.statusCode = 500;
                  res.end(String(e instanceof Error ? e.message : e));
                });
              return;
            }
            if (url === "/__hand_eye/aruco-verify") {
              const eye = parseStereoEye(body.stereoEye ?? null);
              const markerLengthM = Number(body.markerLengthM);
              const dict = String(body.dict ?? "DICT_4X4_50");
              if (!eye || !Number.isFinite(markerLengthM) || markerLengthM <= 0) {
                res.statusCode = 400;
                res.end("missing stereoEye or invalid markerLengthM");
                return;
              }
              runArucoVerifyPy(eye, markerLengthM, dict)
                .then((rawJsonLine) => {
                  res.setHeader("Content-Type", "application/json; charset=utf-8");
                  res.end(rawJsonLine);
                })
                .catch((e) => {
                  res.statusCode = 500;
                  res.end(String(e instanceof Error ? e.message : e));
                });
              return;
            }
            if (url === "/__hand_eye/verify") {
              const { pngBase64, stereoEye, yamlContent } = body;
              const eye = parseStereoEye(stereoEye ?? null);
              if (!pngBase64 || !eye || !yamlContent) {
                res.statusCode = 400;
                res.end("missing fields");
                return;
              }
              const dir = ensureCameraDir(eye);
              fs.writeFileSync(path.join(dir, HAND_EYE_VERIFY_IMAGE), Buffer.from(pngBase64, "base64"));
              fs.writeFileSync(path.join(dir, HAND_EYE_VERIFY_POSE_YAML), yamlContent, "utf8");
              res.setHeader("Content-Type", "application/json");
              res.end(
                JSON.stringify({
                  ok: true,
                  imageName: HAND_EYE_VERIFY_IMAGE,
                  yamlFileName: HAND_EYE_VERIFY_POSE_YAML,
                  cameraDir: eye === "left" ? HAND_EYE_LEFT : HAND_EYE_RIGHT,
                }),
              );
              return;
            }
            if (url === "/__hand_eye/capture") {
              const { pngBase64, imageName, csvLine, stereoEye } = body;
              const eye = parseStereoEye(stereoEye ?? null);
              if (!pngBase64 || !imageName || !csvLine || !eye) {
                res.statusCode = 400;
                res.end("missing fields");
                return;
              }
              const dir = ensureCameraDir(eye);
              fs.writeFileSync(path.join(dir, imageName), Buffer.from(pngBase64, "base64"));
              const csvPath = path.join(dir, "poses.csv");
              if (!fs.existsSync(csvPath)) fs.writeFileSync(csvPath, POSES_HEADER, "utf8");
              fs.appendFileSync(csvPath, `${csvLine}\n`, "utf8");
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ ok: true, imageName, cameraDir: eye === "left" ? HAND_EYE_LEFT : HAND_EYE_RIGHT }));
              return;
            }
            const leftYamlContent = body.leftYamlContent;
            const rightYamlContent = body.rightYamlContent;
            if (!leftYamlContent || !rightYamlContent) {
              res.statusCode = 400;
              res.end("missing yaml content");
              return;
            }
            const leftDir = ensureCameraDir("left");
            const rightDir = ensureCameraDir("right");
            const leftPath = path.join(leftDir, HAND_EYE_CONFIG_NAME);
            const rightPath = path.join(rightDir, HAND_EYE_CONFIG_NAME);
            fs.writeFileSync(leftPath, leftYamlContent, "utf8");
            fs.writeFileSync(rightPath, rightYamlContent, "utf8");
            const legacyPath = path.join(projectRoot(), HAND_EYE_LEGACY_CONFIG);
            fs.mkdirSync(path.dirname(legacyPath), { recursive: true });
            fs.writeFileSync(legacyPath, leftYamlContent, "utf8");
            res.setHeader("Content-Type", "application/json");
            res.end(
              JSON.stringify({
                ok: true,
                leftPath: leftPath.replace(/\\/g, "/"),
                rightPath: rightPath.replace(/\\/g, "/"),
                legacyScriptPath: legacyPath.replace(/\\/g, "/"),
              }),
            );
          } catch (e) {
            res.statusCode = 500;
            res.end(String(e));
          }
        });
      });
    },
  };
}

/** Vite 开发端口。勿用 1420：在 Win10+Hyper-V 上常落入系统保留段 1381–1480，会 EACCES。 */
const DEV_PORT = 5173;
const DEV_HMR_PORT = 5174;

const STEREO_DATA = "stereo_data";
const STEREO_DEPTH_CONFIG = path.join("scripts", "stereo_depth", "config.hand_eye.yaml");
const STEREO_DEPTH_CLI = path.join("scripts", "stereo_depth", "stereo_depth_cli.py");

function stereoDataDir(): string {
  return path.join(projectRoot(), STEREO_DATA);
}

function nextStereoPairIndex(): number {
  const dir = stereoDataDir();
  if (!fs.existsSync(dir)) return 1;
  let max = 0;
  for (const name of fs.readdirSync(dir)) {
    const m = /^pair_(\d+)$/i.exec(name);
    if (m) max = Math.max(max, parseInt(m[1], 10));
  }
  return max + 1;
}

const STEREO_PIXEL_TO_3D_CLI = path.join("scripts", "stereo_depth", "pixel_to_3d.py");

/** 将相对 stereo_data 的路径规范为绝对路径，禁止越界 */
function resolveStereoDataAsset(relPath: string): string {
  const cleaned = relPath.replace(/\\/g, "/").replace(/^\/+/, "");
  if (!cleaned || cleaned.includes("..")) {
    throw new Error("非法资源路径");
  }
  const abs = path.resolve(stereoDataDir(), cleaned);
  const root = path.resolve(stereoDataDir()) + path.sep;
  if (!abs.toLowerCase().startsWith(root.toLowerCase()) && abs.toLowerCase() !== path.resolve(stereoDataDir()).toLowerCase()) {
    throw new Error("资源路径超出 stereo_data");
  }
  if (!fs.existsSync(abs)) {
    throw new Error(`文件不存在: ${abs}`);
  }
  return abs;
}

/** 校正左图像素 → 相机系 / 可选方案 A 手眼到基座 */
function runStereoPixelTo3d(
  outputDir: string,
  u: number,
  v: number,
  handEyePath?: string,
  tcpPosePath?: string,
): Promise<string> {
  const scriptPath = path.join(projectRoot(), STEREO_PIXEL_TO_3D_CLI);
  if (!fs.existsSync(scriptPath)) {
    return Promise.reject(new Error(`未找到脚本: ${scriptPath}`));
  }
  const args = [scriptPath, "--output-dir", outputDir, "--u", String(u), "--v", String(v), "--json"];
  if (handEyePath && tcpPosePath) {
    const he = path.isAbsolute(handEyePath) ? handEyePath : path.join(projectRoot(), handEyePath);
    const tp = path.isAbsolute(tcpPosePath) ? tcpPosePath : path.join(projectRoot(), tcpPosePath);
    args.push("--hand-eye", he, "--tcp-pose", tp);
  }
  return new Promise((resolve, reject) => {
    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];
    const child = spawn("python", args, {
      cwd: path.join(projectRoot(), "scripts", "stereo_depth"),
      windowsHide: true,
      env: pythonSpawnEnv(),
    });
    child.stdout.on("data", (d: Buffer) => stdoutChunks.push(d));
    child.stderr.on("data", (d: Buffer) => stderrChunks.push(d));
    child.on("error", (err) => reject(err));
    child.on("close", (code) => {
      const stdout = decodeSubprocessOutput(new Uint8Array(Buffer.concat(stdoutChunks)));
      const stderr = decodeSubprocessOutput(new Uint8Array(Buffer.concat(stderrChunks)));
      const line = stdout.trim().split(/\r?\n/).filter(Boolean).pop() ?? "";
      if (!line) {
        reject(new Error(stderr || stdout || `像素定位无 JSON 输出 (exit ${code})`));
        return;
      }
      resolve(line);
    });
  });
}

/** 保存左右图并调用 stereo_depth_cli.py，返回 stdout 一行 JSON */
function runStereoDepthCaptureAndRun(
  leftPngBase64: string,
  rightPngBase64: string,
  tcpPoseYaml?: string,
): Promise<string> {
  const idx = nextStereoPairIndex();
  const pairId = `pair_${String(idx).padStart(3, "0")}`;
  const pairDir = path.join(stereoDataDir(), pairId);
  fs.mkdirSync(pairDir, { recursive: true });
  const leftPath = path.join(pairDir, "left.png");
  const rightPath = path.join(pairDir, "right.png");
  const tcpPosePath = path.join(pairDir, "tcp_pose.yaml");
  const outDir = path.join(pairDir, "out");
  fs.writeFileSync(leftPath, Buffer.from(leftPngBase64, "base64"));
  fs.writeFileSync(rightPath, Buffer.from(rightPngBase64, "base64"));
  if (tcpPoseYaml && tcpPoseYaml.trim()) {
    fs.writeFileSync(tcpPosePath, tcpPoseYaml, "utf8");
  }

  const configPath = path.join(projectRoot(), STEREO_DEPTH_CONFIG);
  const scriptPath = path.join(projectRoot(), STEREO_DEPTH_CLI);
  if (!fs.existsSync(configPath)) {
    return Promise.reject(new Error(`未找到立体配置: ${configPath}`));
  }
  if (!fs.existsSync(scriptPath)) {
    return Promise.reject(new Error(`未找到脚本: ${scriptPath}`));
  }

  return new Promise((resolve, reject) => {
    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];
    const child = spawn(
      "python",
      [
        scriptPath,
        "--config",
        configPath,
        "--left",
        leftPath,
        "--right",
        rightPath,
        "--output",
        outDir,
        "--json",
      ],
      { cwd: path.join(projectRoot(), "scripts", "stereo_depth"), windowsHide: true, env: pythonSpawnEnv() },
    );
    child.stdout.on("data", (d: Buffer) => stdoutChunks.push(d));
    child.stderr.on("data", (d: Buffer) => stderrChunks.push(d));
    child.on("error", (err) => reject(err));
    child.on("close", (code) => {
      const stdout = decodeSubprocessOutput(new Uint8Array(Buffer.concat(stdoutChunks)));
      const stderr = decodeSubprocessOutput(new Uint8Array(Buffer.concat(stderrChunks)));
      const line = stdout.trim().split(/\r?\n/).filter(Boolean).pop() ?? "";
      if (!line) {
        reject(new Error(stderr || stdout || `立体深度无 JSON 输出 (exit ${code})`));
        return;
      }
      try {
        const parsed = JSON.parse(line) as Record<string, unknown>;
        parsed.pair_id = pairId;
        parsed.left_image = leftPath.replace(/\\/g, "/");
        parsed.right_image = rightPath.replace(/\\/g, "/");
        if (fs.existsSync(tcpPosePath)) {
          parsed.tcp_pose = tcpPosePath.replace(/\\/g, "/");
        }
        resolve(JSON.stringify(parsed));
      } catch {
        reject(new Error(stderr || stdout || `立体深度 JSON 解析失败 (exit ${code})`));
      }
    });
  });
}

function stereoDepthDevApi(): Plugin {
  return {
    name: "stereo-depth-dev-api",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const rawUrl = req.url ?? "";
        const url = rawUrl.split("?")[0] ?? "";
        const qs = rawUrl.includes("?") ? new URL(rawUrl, "http://127.0.0.1").searchParams : null;

        if (url === "/__stereo_depth/asset" && req.method === "GET") {
          try {
            const rel = qs?.get("rel") ?? "";
            const abs = resolveStereoDataAsset(rel);
            const data = fs.readFileSync(abs);
            const ext = path.extname(abs).toLowerCase();
            const ctype =
              ext === ".png"
                ? "image/png"
                : ext === ".jpg" || ext === ".jpeg"
                  ? "image/jpeg"
                  : "application/octet-stream";
            res.setHeader("Content-Type", ctype);
            res.end(data);
          } catch (e) {
            res.statusCode = 404;
            res.end(String(e instanceof Error ? e.message : e));
          }
          return;
        }

        if (url === "/__stereo_depth/pixel-to-3d" && req.method === "POST") {
          const chunks: Buffer[] = [];
          req.on("data", (c) => chunks.push(c));
          req.on("end", () => {
            try {
              const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as {
                outputDir?: string;
                u?: number;
                v?: number;
                handEyePath?: string;
                tcpPosePath?: string;
              };
              if (!body.outputDir || body.u == null || body.v == null) {
                res.statusCode = 400;
                res.end("missing outputDir / u / v");
                return;
              }
              runStereoPixelTo3d(
                body.outputDir,
                Number(body.u),
                Number(body.v),
                body.handEyePath,
                body.tcpPosePath,
              )
                .then((rawJsonLine) => {
                  res.setHeader("Content-Type", "application/json; charset=utf-8");
                  res.end(rawJsonLine);
                })
                .catch((e) => {
                  res.statusCode = 500;
                  res.end(String(e instanceof Error ? e.message : e));
                });
            } catch (e) {
              res.statusCode = 400;
              res.end(String(e));
            }
          });
          return;
        }

        if (url !== "/__stereo_depth/capture-and-run") {
          next();
          return;
        }
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end("Method Not Allowed");
          return;
        }
        const chunks: Buffer[] = [];
        req.on("data", (c) => chunks.push(c));
        req.on("end", () => {
          try {
            const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as {
              leftPngBase64?: string;
              rightPngBase64?: string;
              tcpPoseYaml?: string;
            };
            if (!body.leftPngBase64 || !body.rightPngBase64) {
              res.statusCode = 400;
              res.end("missing leftPngBase64 or rightPngBase64");
              return;
            }
            runStereoDepthCaptureAndRun(body.leftPngBase64, body.rightPngBase64, body.tcpPoseYaml)
              .then((rawJsonLine) => {
                res.setHeader("Content-Type", "application/json; charset=utf-8");
                res.end(rawJsonLine);
              })
              .catch((e) => {
                res.statusCode = 500;
                res.end(String(e instanceof Error ? e.message : e));
              });
          } catch (e) {
            res.statusCode = 400;
            res.end(String(e));
          }
        });
      });
    },
  };
}

function yoloDevApi(): Plugin {
  const YOLO_CLI = path.join("scripts", "yolo11", "yolo_onnx_infer_cli.py");

  function runYoloInfer(imagePath: string, outputDir?: string): Promise<string> {
    const scriptPath = path.join(projectRoot(), YOLO_CLI);
    if (!fs.existsSync(scriptPath)) {
      return Promise.reject(new Error(`未找到脚本: ${scriptPath}`));
    }
    const img = path.isAbsolute(imagePath) ? imagePath : path.join(projectRoot(), imagePath);
    if (!fs.existsSync(img)) {
      return Promise.reject(new Error(`图像不存在: ${img}`));
    }
    const out =
      outputDir && outputDir.trim()
        ? path.isAbsolute(outputDir)
          ? outputDir
          : path.join(projectRoot(), outputDir)
        : path.join(path.dirname(img), "yolo");
    fs.mkdirSync(out, { recursive: true });

    return new Promise((resolve, reject) => {
      const stdoutChunks: Buffer[] = [];
      const stderrChunks: Buffer[] = [];
      const child = spawn(
        "python",
        [scriptPath, "--image", img, "--output", out, "--json"],
        { cwd: path.join(projectRoot(), "scripts", "yolo11"), windowsHide: true, env: pythonSpawnEnv() },
      );
      child.stdout.on("data", (d: Buffer) => stdoutChunks.push(d));
      child.stderr.on("data", (d: Buffer) => stderrChunks.push(d));
      child.on("error", (err) => reject(err));
      child.on("close", (code) => {
        const stdout = decodeSubprocessOutput(new Uint8Array(Buffer.concat(stdoutChunks)));
        const stderr = decodeSubprocessOutput(new Uint8Array(Buffer.concat(stderrChunks)));
        const line = stdout.trim().split(/\r?\n/).filter(Boolean).pop() ?? "";
        if (!line) {
          reject(new Error(stderr || stdout || `YOLO 无 JSON 输出 (exit ${code})`));
          return;
        }
        try {
          const parsed = JSON.parse(line) as Record<string, unknown>;
          if (parsed.success === undefined) parsed.success = true;
          resolve(JSON.stringify(parsed));
        } catch {
          reject(new Error(stderr || `YOLO JSON 解析失败: ${line}`));
        }
      });
    });
  }

  return {
    name: "yolo-onnx-dev-api",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = req.url?.split("?")[0] ?? "";
        if (url !== "/__yolo/infer") {
          next();
          return;
        }
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end("Method Not Allowed");
          return;
        }
        const chunks: Buffer[] = [];
        req.on("data", (c) => chunks.push(c));
        req.on("end", () => {
          try {
            const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as {
              imagePath?: string;
              outputDir?: string;
            };
            if (!body.imagePath) {
              res.statusCode = 400;
              res.end("missing imagePath");
              return;
            }
            runYoloInfer(body.imagePath, body.outputDir)
              .then((raw) => {
                res.setHeader("Content-Type", "application/json; charset=utf-8");
                res.end(raw);
              })
              .catch((e) => {
                res.statusCode = 500;
                res.end(String(e instanceof Error ? e.message : e));
              });
          } catch (e) {
            res.statusCode = 400;
            res.end(String(e));
          }
        });
      });
    },
  };
}

// https://vite.dev/config/
export default defineConfig(async () => ({
  plugins: [vue(), handEyeDevApi(), stereoDepthDevApi(), yoloDevApi()],

  clearScreen: false,
  server: {
    port: DEV_PORT,
    strictPort: true,
    // 显式 IPv4，避免 Node 绑 ::1 时在部分环境触发权限/保留端口问题
    host: host || "127.0.0.1",
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: DEV_HMR_PORT,
        }
      : undefined,
    watch: {
      ignored: ["**/src-tauri/**"],
    },
  },
}));
