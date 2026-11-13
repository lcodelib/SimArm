use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const HAND_EYE_DATA: &str = "hand_eye_data";
const LEFT_CAMERA: &str = "left_camera";
const RIGHT_CAMERA: &str = "right_camera";
const HAND_EYE_LEGACY_CONFIG: &str = "scripts/hand_eye_calibration/config.yaml";
const VERIFY_IMAGE: &str = "verify.png";
const VERIFY_POSE_YAML: &str = "tcp_pose.yaml";
const CONFIG_NAME: &str = "config.yaml";
const HAND_EYE_RESULT_YAML: &str = "hand_eye_result.yaml";
const POSES_HEADER: &str = "image,tx,ty,tz,qx,qy,qz,qw\n";
const STEREO_DATA: &str = "stereo_data";
const STEREO_DEPTH_CONFIG: &str = "scripts/stereo_depth/config.hand_eye.yaml";
const STEREO_DEPTH_CLI: &str = "scripts/stereo_depth/stereo_depth_cli.py";
const STEREO_PIXEL_TO_3D_CLI: &str = "scripts/stereo_depth/pixel_to_3d.py";
const YOLO_ONNX_CLI: &str = "scripts/yolo11/yolo_onnx_infer_cli.py";

fn decode_subprocess_bytes(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => String::from_utf8_lossy(bytes).into_owned(),
    }
}

fn hand_eye_python_command(script: &PathBuf, config: &PathBuf, root: &PathBuf) -> std::process::Command {
    let mut cmd = std::process::Command::new("python");
    cmd.arg(script)
        .arg("--config")
        .arg(config)
        .current_dir(root)
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1");
    cmd
}

fn python_with_utf8_env() -> std::process::Command {
    let mut cmd = std::process::Command::new("python");
    cmd.env("PYTHONIOENCODING", "utf-8").env("PYTHONUTF8", "1");
    cmd
}

fn last_json_line_from_output(stdout: &str, stderr: &str, exit_code: Option<i32>, err_label: &str) -> Result<String, String> {
    let line = stdout.trim().lines().last().unwrap_or("");
    if line.is_empty() {
        return Err(if stderr.is_empty() {
            format!("{err_label}无输出 (exit {})", exit_code.unwrap_or(-1))
        } else {
            stderr.to_string()
        });
    }
    Ok(line.to_string())
}

fn project_root() -> Result<PathBuf, String> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .ok_or_else(|| "无法定位工程根目录".to_string())
        .map(|p| p.to_path_buf())
}

fn hand_eye_data_dir() -> Result<PathBuf, String> {
    Ok(project_root()?.join(HAND_EYE_DATA))
}

fn parse_stereo_eye(eye: &str) -> Result<&'static str, String> {
    match eye {
        "left" => Ok(LEFT_CAMERA),
        "right" => Ok(RIGHT_CAMERA),
        _ => Err("stereo_eye 须为 left 或 right".to_string()),
    }
}

fn camera_dir(stereo_eye: &str) -> Result<PathBuf, String> {
    Ok(hand_eye_data_dir()?.join(parse_stereo_eye(stereo_eye)?))
}

fn ensure_camera_dir(stereo_eye: &str) -> Result<PathBuf, String> {
    let dir = camera_dir(stereo_eye)?;
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

fn next_capture_index_in(dir: &Path) -> Result<u32, String> {
    if !dir.exists() {
        return Ok(1);
    }
    let mut max_idx = 0u32;
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(cap) = name.strip_prefix("capture_").and_then(|s| s.strip_suffix(".png")) {
            if let Ok(n) = cap.parse::<u32>() {
                max_idx = max_idx.max(n);
            }
        }
    }
    Ok(max_idx + 1)
}

fn clear_camera_captures(dir: &Path) -> Result<u32, String> {
    let mut deleted_images = 0u32;
    if !dir.exists() {
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        fs::write(dir.join("poses.csv"), POSES_HEADER).map_err(|e| e.to_string())?;
        return Ok(0);
    }
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("capture_") && name.ends_with(".png") {
            fs::remove_file(&path).map_err(|e| e.to_string())?;
            deleted_images += 1;
        }
    }
    fs::write(dir.join("poses.csv"), POSES_HEADER).map_err(|e| e.to_string())?;
    Ok(deleted_images)
}

#[tauri::command]
fn hand_eye_project_root() -> Result<String, String> {
    let root = project_root()?;
    Ok(root.to_string_lossy().replace('\\', "/"))
}

#[tauri::command]
fn hand_eye_next_capture_index(stereo_eye: String) -> Result<u32, String> {
    let dir = ensure_camera_dir(&stereo_eye)?;
    next_capture_index_in(&dir)
}

#[tauri::command]
fn hand_eye_save_verify(png_base64: String, stereo_eye: String, yaml_content: String) -> Result<String, String> {
    let dir = ensure_camera_dir(&stereo_eye)?;
    let png_bytes = base64_decode(&png_base64)?;
    fs::write(dir.join(VERIFY_IMAGE), png_bytes).map_err(|e| e.to_string())?;
    fs::write(dir.join(VERIFY_POSE_YAML), yaml_content).map_err(|e| e.to_string())?;
    Ok(dir.to_string_lossy().replace('\\', "/"))
}

#[tauri::command]
fn hand_eye_save_capture(
    png_base64: String,
    image_name: String,
    csv_line: String,
    stereo_eye: String,
) -> Result<String, String> {
    if image_name.contains('/') || image_name.contains('\\') || image_name.contains("..") {
        return Err("非法图像文件名".to_string());
    }
    let dir = ensure_camera_dir(&stereo_eye)?;
    let png_bytes = base64_decode(&png_base64)?;
    let img_path = dir.join(&image_name);
    fs::write(&img_path, png_bytes).map_err(|e| e.to_string())?;

    let csv_path = dir.join("poses.csv");
    if !csv_path.exists() {
        fs::write(&csv_path, POSES_HEADER).map_err(|e| e.to_string())?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&csv_path)
        .map_err(|e| e.to_string())?;
    writeln!(file, "{}", csv_line).map_err(|e| e.to_string())?;

    Ok(img_path.to_string_lossy().replace('\\', "/"))
}

#[derive(serde::Serialize)]
struct HandEyeExportConfigResult {
    left_path: String,
    right_path: String,
    legacy_script_path: String,
}

#[tauri::command]
fn hand_eye_export_config(left_yaml_content: String, right_yaml_content: String) -> Result<HandEyeExportConfigResult, String> {
    let left_dir = ensure_camera_dir("left")?;
    let right_dir = ensure_camera_dir("right")?;
    let left_path = left_dir.join(CONFIG_NAME);
    let right_path = right_dir.join(CONFIG_NAME);
    fs::write(&left_path, left_yaml_content).map_err(|e| e.to_string())?;
    fs::write(&right_path, right_yaml_content).map_err(|e| e.to_string())?;

    let legacy_path = project_root()?.join(HAND_EYE_LEGACY_CONFIG);
    if let Some(parent) = legacy_path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(&legacy_path, &left_yaml_content).map_err(|e| e.to_string())?;

    Ok(HandEyeExportConfigResult {
        left_path: left_path.to_string_lossy().replace('\\', "/"),
        right_path: right_path.to_string_lossy().replace('\\', "/"),
        legacy_script_path: legacy_path.to_string_lossy().replace('\\', "/"),
    })
}

#[derive(serde::Serialize)]
struct HandEyeClearResult {
    deleted_images: u32,
}

#[tauri::command]
fn hand_eye_run_solve(stereo_eye: String) -> Result<String, String> {
    let dir = ensure_camera_dir(&stereo_eye)?;
    let cfg = dir.join(CONFIG_NAME);
    if !cfg.is_file() {
        return Err(format!("未找到 {}，请先导出相机配置", cfg.to_string_lossy()));
    }
    let root = project_root()?;
    let script = root.join("scripts").join("hand_eye_calibration").join("hand_eye_solve_cli.py");
    let output = hand_eye_python_command(&script, &cfg, &root)
        .output()
        .map_err(|e| format!("无法启动 python: {e}"))?;
    let stdout = decode_subprocess_bytes(&output.stdout);
    let stderr = decode_subprocess_bytes(&output.stderr);
    let line = last_json_line_from_output(&stdout, &stderr, output.status.code(), "手眼标定")?;
    serde_json::from_str::<serde_json::Value>(&line)
        .map_err(|e| format!("解析标定结果失败: {e}; stdout={line}"))?;
    Ok(line)
}

#[tauri::command]
fn hand_eye_run_aruco_verify(
    stereo_eye: String,
    marker_length_m: f64,
    dict_name: String,
) -> Result<String, String> {
    let dir = camera_dir(&stereo_eye)?;
    let image = dir.join(VERIFY_IMAGE);
    let hand_eye = dir.join(HAND_EYE_RESULT_YAML);
    let tcp_pose = dir.join(VERIFY_POSE_YAML);
    if !image.is_file() {
        return Err(format!("未找到校验图像: {}", image.to_string_lossy()));
    }
    if !hand_eye.is_file() {
        return Err(format!(
            "未找到 {}，请先完成手眼标定",
            hand_eye.to_string_lossy()
        ));
    }
    if !tcp_pose.is_file() {
        return Err(format!("未找到 {}", tcp_pose.to_string_lossy()));
    }
    let root = project_root()?;
    let script = root
        .join("scripts")
        .join("hand_eye_calibration")
        .join("aruco_to_robot_base_cli.py");
    let output = python_with_utf8_env()
        .arg(&script)
        .arg("--image")
        .arg(&image)
        .arg("--hand-eye")
        .arg(&hand_eye)
        .arg("--tcp-pose")
        .arg(&tcp_pose)
        .arg("--marker-length")
        .arg(format!("{marker_length_m}"))
        .arg("--dict")
        .arg(&dict_name)
        .current_dir(&root)
        .output()
        .map_err(|e| format!("无法启动 python: {e}"))?;
    let stdout = decode_subprocess_bytes(&output.stdout);
    let stderr = decode_subprocess_bytes(&output.stderr);
    let line = last_json_line_from_output(&stdout, &stderr, output.status.code(), "ArUco 校验")?;
    serde_json::from_str::<serde_json::Value>(&line)
        .map_err(|e| format!("解析 ArUco 校验结果失败: {e}; stdout={line}"))?;
    Ok(line)
}

#[tauri::command]
fn hand_eye_clear_data() -> Result<HandEyeClearResult, String> {
    let mut deleted_images = 0u32;
    deleted_images += clear_camera_captures(&camera_dir("left")?)?;
    deleted_images += clear_camera_captures(&camera_dir("right")?)?;
    Ok(HandEyeClearResult { deleted_images })
}

fn stereo_data_dir() -> Result<PathBuf, String> {
    Ok(project_root()?.join(STEREO_DATA))
}

fn next_stereo_pair_index(dir: &Path) -> u32 {
    let mut max = 0u32;
    if !dir.is_dir() {
        return 1;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return 1;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(rest) = name.strip_prefix("pair_") {
            if let Ok(n) = rest.parse::<u32>() {
                max = max.max(n);
            }
        }
    }
    max + 1
}

/// 保存左右相机 PNG 并调用 stereo_depth_cli.py，返回一行 JSON
#[tauri::command]
fn stereo_depth_capture_and_run(
    left_png_base64: String,
    right_png_base64: String,
    tcp_pose_yaml: Option<String>,
) -> Result<String, String> {
    let root = project_root()?;
    let data = stereo_data_dir()?;
    fs::create_dir_all(&data).map_err(|e| format!("创建 stereo_data 失败: {e}"))?;
    let idx = next_stereo_pair_index(&data);
    let pair_id = format!("pair_{idx:03}");
    let pair_dir = data.join(&pair_id);
    fs::create_dir_all(&pair_dir).map_err(|e| format!("创建 {pair_id} 失败: {e}"))?;
    let left_path = pair_dir.join("left.png");
    let right_path = pair_dir.join("right.png");
    let tcp_pose_path = pair_dir.join("tcp_pose.yaml");
    let out_dir = pair_dir.join("out");
    let left_bytes = base64_decode(&left_png_base64)?;
    let right_bytes = base64_decode(&right_png_base64)?;
    fs::write(&left_path, left_bytes).map_err(|e| format!("写入左图失败: {e}"))?;
    fs::write(&right_path, right_bytes).map_err(|e| format!("写入右图失败: {e}"))?;
    if let Some(yaml) = tcp_pose_yaml.as_ref().filter(|s| !s.trim().is_empty()) {
        fs::write(&tcp_pose_path, yaml.as_bytes()).map_err(|e| format!("写入 tcp_pose.yaml 失败: {e}"))?;
    }

    let config = root.join(STEREO_DEPTH_CONFIG);
    let script = root.join(STEREO_DEPTH_CLI);
    if !config.is_file() {
        return Err(format!("未找到立体配置: {}", config.to_string_lossy()));
    }
    if !script.is_file() {
        return Err(format!("未找到脚本: {}", script.to_string_lossy()));
    }
    let cwd = root.join("scripts").join("stereo_depth");
    let output = python_with_utf8_env()
        .arg(&script)
        .arg("--config")
        .arg(&config)
        .arg("--left")
        .arg(&left_path)
        .arg("--right")
        .arg(&right_path)
        .arg("--output")
        .arg(&out_dir)
        .arg("--json")
        .current_dir(&cwd)
        .output()
        .map_err(|e| format!("无法启动 python: {e}"))?;
    let stdout = decode_subprocess_bytes(&output.stdout);
    let stderr = decode_subprocess_bytes(&output.stderr);
    let line = last_json_line_from_output(&stdout, &stderr, output.status.code(), "立体深度")?;
    let mut value: serde_json::Value = serde_json::from_str(&line)
        .map_err(|e| format!("解析立体深度结果失败: {e}; stdout={line}"))?;
    if let Some(obj) = value.as_object_mut() {
        obj.insert("pair_id".into(), serde_json::Value::String(pair_id));
        obj.insert(
            "left_image".into(),
            serde_json::Value::String(left_path.to_string_lossy().replace('\\', "/")),
        );
        obj.insert(
            "right_image".into(),
            serde_json::Value::String(right_path.to_string_lossy().replace('\\', "/")),
        );
        if tcp_pose_path.is_file() {
            obj.insert(
                "tcp_pose".into(),
                serde_json::Value::String(tcp_pose_path.to_string_lossy().replace('\\', "/")),
            );
        }
    }
    Ok(value.to_string())
}

/// 校正左图像素 → 相机 / 可选方案 A 手眼基座（一行 JSON）
#[tauri::command]
fn stereo_depth_pixel_to_3d(
    output_dir: String,
    u: f64,
    v: f64,
    hand_eye_path: Option<String>,
    tcp_pose_path: Option<String>,
) -> Result<String, String> {
    let root = project_root()?;
    let script = root.join(STEREO_PIXEL_TO_3D_CLI);
    if !script.is_file() {
        return Err(format!("未找到脚本: {}", script.to_string_lossy()));
    }
    let cwd = root.join("scripts").join("stereo_depth");
    let mut cmd = python_with_utf8_env();
    cmd.arg(&script)
        .arg("--output-dir")
        .arg(&output_dir)
        .arg("--u")
        .arg(u.to_string())
        .arg("--v")
        .arg(v.to_string())
        .arg("--json");
    if let (Some(he), Some(tp)) = (hand_eye_path.as_ref(), tcp_pose_path.as_ref()) {
        if !he.is_empty() && !tp.is_empty() {
            let he_p = {
                let p = PathBuf::from(he);
                if p.is_absolute() {
                    p
                } else {
                    root.join(he)
                }
            };
            let tp_p = {
                let p = PathBuf::from(tp);
                if p.is_absolute() {
                    p
                } else {
                    root.join(tp)
                }
            };
            cmd.arg("--hand-eye").arg(&he_p).arg("--tcp-pose").arg(&tp_p);
        }
    }
    let output = cmd
        .current_dir(&cwd)
        .output()
        .map_err(|e| format!("无法启动 python: {e}"))?;
    let stdout = decode_subprocess_bytes(&output.stdout);
    let stderr = decode_subprocess_bytes(&output.stderr);
    last_json_line_from_output(&stdout, &stderr, output.status.code(), "像素定位")
}

/// YOLO ONNX 推理（一行 JSON）
#[tauri::command]
fn yolo_onnx_infer(image_path: String, output_dir: Option<String>) -> Result<String, String> {
    let root = project_root()?;
    let script = root.join(YOLO_ONNX_CLI);
    if !script.is_file() {
        return Err(format!("未找到脚本: {}", script.to_string_lossy()));
    }
    let img = {
        let p = PathBuf::from(&image_path);
        if p.is_absolute() {
            p
        } else {
            root.join(&image_path)
        }
    };
    if !img.is_file() {
        return Err(format!("图像不存在: {}", img.to_string_lossy()));
    }
    let out = match output_dir.as_ref().filter(|s| !s.trim().is_empty()) {
        Some(d) => {
            let p = PathBuf::from(d);
            if p.is_absolute() {
                p
            } else {
                root.join(d)
            }
        }
        None => img
            .parent()
            .map(|p| p.join("yolo"))
            .ok_or_else(|| "无法确定输出目录".to_string())?,
    };
    fs::create_dir_all(&out).map_err(|e| format!("创建输出目录失败: {e}"))?;
    let cwd = root.join("scripts").join("yolo11");
    let output = python_with_utf8_env()
        .arg(&script)
        .arg("--image")
        .arg(&img)
        .arg("--output")
        .arg(&out)
        .arg("--json")
        .current_dir(&cwd)
        .output()
        .map_err(|e| format!("无法启动 python: {e}"))?;
    let stdout = decode_subprocess_bytes(&output.stdout);
    let stderr = decode_subprocess_bytes(&output.stderr);
    let line = last_json_line_from_output(&stdout, &stderr, output.status.code(), "YOLO")?;
    let mut value: serde_json::Value = serde_json::from_str(&line)
        .map_err(|e| format!("解析 YOLO JSON 失败: {e}; stdout={line}"))?;
    if let Some(obj) = value.as_object_mut() {
        if !obj.contains_key("success") {
            obj.insert("success".into(), serde_json::Value::Bool(true));
        }
    }
    Ok(value.to_string())
}

/// 读取 stereo_data 下相对路径文件，返回 base64（供定位测试预览图）
#[tauri::command]
fn stereo_depth_read_asset_base64(rel_path: String) -> Result<String, String> {
    let cleaned = rel_path.replace('\\', "/").trim_start_matches('/').to_string();
    if cleaned.is_empty() || cleaned.contains("..") {
        return Err("非法资源路径".into());
    }
    let data_root = stereo_data_dir()?;
    let abs = data_root.join(&cleaned);
    let abs_canon = abs
        .canonicalize()
        .map_err(|e| format!("资源不存在: {e}"))?;
    let root_canon = data_root
        .canonicalize()
        .map_err(|e| format!("stereo_data 无效: {e}"))?;
    if !abs_canon.starts_with(&root_canon) {
        return Err("资源路径超出 stereo_data".into());
    }
    let bytes = fs::read(&abs_canon).map_err(|e| format!("读取失败: {e}"))?;
    Ok(base64_encode(&bytes))
}

fn base64_encode(input: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((input.len() + 2) / 3 * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let (b1, b2, pad) = match chunk.len() {
            1 => (0u32, 0u32, 2u8),
            2 => (chunk[1] as u32, 0u32, 1u8),
            _ => (chunk[1] as u32, chunk[2] as u32, 0u8),
        };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(if pad == 2 {
            '='
        } else {
            TABLE[((n >> 6) & 63) as usize] as char
        });
        out.push(if pad >= 1 {
            '='
        } else {
            TABLE[(n & 63) as usize] as char
        });
    }
    out
}

fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    const TABLE: &[u8; 256] = &{
        let mut t = [255u8; 256];
        let mut i = 0u8;
        while i < 64 {
            let c = match i {
                0..=25 => b'A' + i,
                26..=51 => b'a' + (i - 26),
                52..=61 => b'0' + (i - 52),
                62 => b'+',
                _ => b'/',
            };
            t[c as usize] = i;
            i += 1;
        }
        t
    };
    let input = input.trim();
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &b in input.as_bytes() {
        if b == b'=' {
            break;
        }
        if b.is_ascii_whitespace() {
            continue;
        }
        let v = TABLE[b as usize];
        if v == 255 {
            return Err("base64 解码失败".to_string());
        }
        buf = (buf << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            greet,
            hand_eye_project_root,
            hand_eye_next_capture_index,
            hand_eye_save_capture,
            hand_eye_save_verify,
            hand_eye_export_config,
            hand_eye_run_solve,
            hand_eye_run_aruco_verify,
            hand_eye_clear_data,
            stereo_depth_capture_and_run,
            stereo_depth_pixel_to_3d,
            stereo_depth_read_asset_base64,
            yolo_onnx_infer,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
