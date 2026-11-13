/** 末端离屏相机渲染与透视参数（与 RobotScene 一致） */
export const TCP_CAM_RT_WIDTH = 1280;
export const TCP_CAM_RT_HEIGHT = 720;
/** 竖直视场角（度），Three.js PerspectiveCamera.fov */
export const TCP_CAM_FOV_DEG = 52;

/** 双目基线：两路末端相机光心间距（米） */
export const TCP_STEREO_BASELINE_M = 0.05;

/**
 * 双目末端光路（相机 UV：u 向右为支架 +X）：
 * 左 = `TcpCameraPrimary`（−X），右 = `TcpCameraStereo`（+X）。
 */
export type TcpStereoEye = "left" | "right";

export const TCP_STEREO_EYE_LABEL: Record<TcpStereoEye, string> = {
  left: "左相机",
  right: "右相机",
};

export type TcpCameraIntrinsics = {
  width: number;
  height: number;
  fovDeg: number;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  distCoeffs: [number, number, number, number];
};

/**
 * 由理想针孔模型（无畸变、主点在中心、方形像素）从竖直 FOV 与分辨率推算 OpenCV 风格内参。
 *
 * 与 Three.js `PerspectiveCamera(fov, aspect, …)` 一致：水平视场满足
 * tan(fov_x/2) = aspect·tan(fov_y/2)，故 (W/2)/tan(fov_x/2) = (H/2)/tan(fov_y/2)，即 **fx = fy**。
 * 此前误写 fx = fy·aspect，会导致 PnP / 手眼与仿真画面几何不一致。
 */
export function intrinsicsFromTcpCameraSpec(
  width = TCP_CAM_RT_WIDTH,
  height = TCP_CAM_RT_HEIGHT,
  fovDeg = TCP_CAM_FOV_DEG,
): TcpCameraIntrinsics {
  const fovRad = (fovDeg * Math.PI) / 180;
  const fy = height / (2 * Math.tan(fovRad / 2));
  const fx = fy;
  return {
    width,
    height,
    fovDeg,
    fx,
    fy,
    cx: width / 2,
    cy: height / 2,
    distCoeffs: [0, 0, 0, 0],
  };
}
