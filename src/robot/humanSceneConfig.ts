/**
 * 场景人体模型：男性 human.glb / 女性 female.glb。
 * 挂在场景根下，位姿相对**世界坐标系**（Y 向上）；真人仍可挂棋盘（见 REALMAN_*）。
 */
import * as THREE from "three";

/** public/ 下相对路径 */
export const HUMAN_GLB_REL = "human.glb";

/** public/ 下女性人体模型 */
export const FEMALE_GLB_REL = "female.glb";

/** public/ 下真人人体模型 */
export const REALMAN_GLB_REL = "realman.glb";

/**
 * 站立身高（米），用于等比缩放。human / female 共用。
 * ≤0 表示保留模型原始单位。
 */
export const HUMAN_STANDING_HEIGHT_M = 1.7;

/**
 * 真人模型站立身高（米）。
 * ≤0：不做按身高自适应；仅应用 {@link REALMAN_MESH_LOAD_SCALE}。
 */
export const REALMAN_STANDING_HEIGHT_M = 0;

/**
 * 真人 glb 单位换算到场景米制。
 */
export const REALMAN_MESH_LOAD_SCALE = 0.001;

/**
 * 男性/女性世界系欧拉角（度，XYZ）。
 * 默认沿用原棋盘趴姿参数，便于躺姿；不对可改。
 */
export const HUMAN_WORLD_EULER_DEG = { x: -90, y: 0, z: 0 } as const;

/** 女性与男性相同姿态（可单独改） */
export const FEMALE_WORLD_EULER_DEG = HUMAN_WORLD_EULER_DEG;

/**
 * 男性世界系位置（米）。底面贴 y=0，再叠加本向量的 y。
 */
export const HUMAN_WORLD_POSITION_M = new THREE.Vector3(-0.6, 0.7, 0);

/**
 * 女性世界系位置（米）。默认与男性相同；可错开避免重叠。
 */
export const FEMALE_WORLD_POSITION_M = new THREE.Vector3(-0.6, 0.58, 0);

/**
 * 真人棋盘局部欧拉角（仍挂棋盘装配体时使用）。
 */
export const REALMAN_ON_BOARD_EULER_DEG = { x: 0, y: 0, z: 90 } as const;

export const REALMAN_ON_BOARD_OFFSET_M = new THREE.Vector3(0, 0, 0);

/** 真人相对板面法向 +Z 抬起 */
export const REALMAN_ON_BOARD_Z_LIFT_M = -0.22;

/** 人体表面字母串（空则不贴） */
export const HUMAN_DECAL_LETTERS = "ABCDEF";

export const HUMAN_DECAL_COLS = 3;
export const HUMAN_DECAL_SIZE_M = 0.04;
export const HUMAN_DECAL_SPACING_M = 0.05;
export const HUMAN_DECAL_ROW_SPACING_M = 0.05;

/**
 * 字母网格中心相对人体包围盒中心的瞄准偏移（世界系，米）。
 */
export const HUMAN_DECAL_AIM_OFFSET_M = new THREE.Vector3(0.08, 0, -0.25);
