# 双目立体：校正 → 视差 → 深度 → 点云

输入左右相机内参 `K1`/`K2`、畸变 `D1`/`D2`、外参 `R`/`T` 与左右图像，经立体校正与 SGBM/BM 匹配得到视差、深度图与 PLY 点云。

## 约定

- **左相机** = camera1，**右相机** = camera2。
- 外参 `R`、`T` 表示**右相机相对左相机**：`p_right = R @ p_left + T`，`T` 单位为**米**。
- 畸变 `D` 为 OpenCV 顺序：`(k1, k2, p1, p2[, k3])`。

## 安装

```bash
cd scripts/stereo_depth
pip install -r requirements.txt
```

## 用法

### 1. YAML 配置（推荐）

**SimArm 手眼数据**：内参已写在 `hand_eye_data/left_camera/config.yaml` 与 `right_camera/config.yaml`，基线 5 cm：

```bash
python stereo_depth_cli.py --config config.hand_eye.yaml
```

将 `config.hand_eye.yaml` 中的 `left_image` / `right_image` 改为你的左右成对图像（如 `capture_001.png`）。

其它项目可复制 `config.example.yaml` 后修改。

### 2. NPZ 标定 + 图像路径

将矩阵打包为 `calib.npz`（键名须为 `K1,K2,D1,D2,R,T`）：

```bash
python pack_calib_npz.py --output calib.npz --K1 K1.json --K2 K2.json ...
python stereo_depth_cli.py --left left.png --right right.png --npz calib.npz -o out/
```

### 3. 命令行直接传 JSON 矩阵

```bash
python stereo_depth_cli.py --left L.png --right R.png -o out/ ^
  --K1 "[[800,0,640],[0,800,360],[0,0,1]]" ^
  --K2 "[[800,0,640],[0,800,360],[0,0,1]]" ^
  --D1 "[0,0,0,0]" --D2 "[0,0,0,0]" ^
  --R "[[1,0,0],[0,1,0],[0,0,1]]" ^
  --T "[0.05,0,0]"
```

### 4. Python 代码调用

```python
import cv2
import stereo_depth as sd

left_bgr = cv2.imread("left.png")
right_bgr = cv2.imread("right.png")

left = sd.StereoCameraParams(K=K1, D=D1)
right = sd.StereoCameraParams(K=K2, D=D2)
ext = sd.StereoExtrinsic(R=R, T=T.reshape(3, 1))

result = sd.run_stereo_depth(left_bgr, right_bgr, left, right, ext)
sd.save_stereo_outputs(result, "output/")
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `rectified_left.png` / `rectified_right.png` | 极线校正后的左右图 |
| `disparity.png` / `disparity.npy` | 视差（像素，SGBM 为 subpixel/16 后的浮点） |
| `depth.png` / `depth.npy` | 深度（米）；PNG 为 mm×1 的 16 位，0=无效 |
| `pointcloud.ply` | 彩色点云（左校正图着色） |
| `stereo_rectify.npz` | `Q`, `P1`, `P2`, ROI |

## 流程示意

```
左右原图 → stereoRectify + remap → 校正图
         → StereoSGBM/BM → 视差
         → reprojectImageTo3D(Q) → 深度 / 3D 点
         → 有效像素 → PLY 点云
```
