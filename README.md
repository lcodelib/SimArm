# SimArm

基于 **Three.js** 的机械臂简易仿真工具（Vue 3 + Vite，可选 Tauri 桌面壳）。

面向开发与算法联调：在浏览器里可视化机械臂、相机与场景，完成运动学、手眼与双目等仿真流程，并接入 YOLO 目标识别。

当前导入模型：**遨博 Aubo C5**。

## 功能概览

| 能力 | 说明 |
| --- | --- |
| 机械臂运动学仿真 | Aubo C5 正/逆运动学、关节驱动与场景联动 |
| 手眼仿真 | Eye-in-hand 采集、标定求解与校验（左右光路） |
| 双目成像仿真 | 双目拍照、深度/点云生成与基座位姿定位联调 |
| YOLO 识别 | 通过本地脚本 + ONNX 推理接入检测结果 |

## 技术栈

- 前端：Vue 3、TypeScript、Vite、Three.js
- 桌面（可选）：Tauri 2
- 算法脚本：Python（手眼标定、双目深度、YOLO ONNX 等，见 `scripts/`）

## 环境要求

- Node.js 18+（建议 LTS）
- 可选：Python 3.9+（手眼 / 双目 / YOLO 相关脚本）
- 可选：Rust 工具链（仅在使用 `tauri` 桌面端时需要）

## 快速开始

```bash
npm install
npm run dev
```

默认开发地址：`http://127.0.0.1:5173`

> Windows 若出现 `EACCES` / 端口无法绑定：本机 Hyper-V 等可能把 `5173` 划入保留段。可临时改用：
>
> ```bash
> npx vite --port 5200 --host 127.0.0.1 --strictPort
> ```

构建与预览：

```bash
npm run build
npm run preview
```

Tauri 桌面开发：

```bash
npm run tauri -- dev
```

## 资源与目录说明

```
SimArm/
├── public/
│   ├── aubo_C5/          # C5 碰撞 / 视觉网格
│   └── human.glb         # 场景人体模型
├── src/                  # Vue + Three.js 仿真主程序
├── scripts/              # 手眼、双目、YOLO 等 Python 工具
├── src-tauri/            # Tauri 桌面壳（可选）
├── hand_eye_data/        # 本地手眼采集数据
├── stereo_data/          # 本地双目成对数据
└── stereo_output/        # 本地双目输出
```

`public/` 中还可按需放置其他场景资产（床、相机、人体模型等）供本地加载。

### YOLO 模型

将 ONNX 权重放到 `scripts/yolo11/`（例如 `best.onnx`）。

依赖示例：

```bash
pip install -r scripts/yolo11/requirements.txt
```

手眼 / 双目等 Python 依赖见各子目录下的 `requirements.txt` 与 `*.example.yaml`。
