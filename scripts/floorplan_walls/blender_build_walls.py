#!/usr/bin/env python3
"""
Blender 脚本：读取 OpenCV 抽线得到的 walls.json，挤出墙体网格。

运行方式（任选其一）：
  1) Blender → Scripting → 打开本文件 → 改下方「手填参数」→ Run Script
  2) 命令行：
     blender --background --python blender_build_walls.py

依赖：仅需 Blender 自带 bpy / 标准库；walls.json 由 extract_cli.py 生成。
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 手填参数（按需修改）
# ---------------------------------------------------------------------------

# walls.json 绝对或相对路径（相对 = 相对本脚本所在目录）
WALLS_JSON = r"output/floorplan_walls/walls.json"

# 若 >0，覆盖 JSON 中每段的墙高（米）；<=0 则用 JSON 内 height_m
OVERRIDE_WALL_HEIGHT_M = 2.8

# 若 >0，覆盖 JSON 中每段的墙厚（米）；<=0 则用 JSON 内 thickness_m
OVERRIDE_WALL_THICKNESS_M = 0.0

# 生成后的集合与物体名前缀
COLLECTION_NAME = "FloorplanWalls"
OBJECT_PREFIX = "Wall"

# 是否清空同名集合内旧物体后再生成
CLEAR_EXISTING = True

# 导出（可选）：留空字符串则不导出
EXPORT_FBX = r""  # 例如 r"E:\SimArm\scripts\floorplan_walls\output\walls.fbx"
EXPORT_OBJ = r""

# 日志级别：DEBUG / INFO / WARNING
LOG_LEVEL = "INFO"

# ---------------------------------------------------------------------------

logger = logging.getLogger("blender_build_walls")


def setup_logging(level: str = "INFO") -> None:
    """配置日志（Blender 控制台可见）。"""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _script_dir() -> Path:
    """本脚本所在目录；在 Blender 文本块中运行时可能无 __file__。"""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def resolve_walls_json(path_str: str) -> Path:
    """解析 walls.json 路径。"""
    p = Path(path_str)
    if not p.is_absolute():
        p = _script_dir() / p
    return p.resolve()


def load_walls_json(path: Path) -> Dict[str, Any]:
    """加载抽线结果 JSON。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到 walls.json: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "walls" not in data or not isinstance(data["walls"], list):
        raise ValueError("walls.json 缺少 walls 数组")
    return data


def _ensure_bpy():
    """延迟导入 bpy，便于在 Blender 外做语法检查。"""
    try:
        import bpy
        from mathutils import Vector
    except ImportError as exc:
        raise RuntimeError("本脚本须在 Blender 内运行（需要 bpy）") from exc
    return bpy, Vector


def get_or_create_collection(bpy, name: str):
    """获取或创建集合，并链到场景主集合。"""
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def clear_collection_objects(bpy, col) -> int:
    """删除集合内全部物体，返回删除数量。"""
    removed = 0
    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
        removed += 1
    return removed


def create_wall_box(
    bpy,
    Vector,
    *,
    name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    thickness: float,
    height: float,
    collection,
) -> Optional[Any]:
    """
    沿中心线创建墙体长方体。

    坐标系：XY 为平面，Z 向上；墙底落在 Z=0，顶面 Z=height。
    """
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        logger.warning("跳过零长度墙段: %s", name)
        return None
    if thickness <= 0 or height <= 0:
        logger.warning("跳过非法尺寸墙段: %s (t=%.3f h=%.3f)", name, thickness, height)
        return None

    # 默认立方体边长 2，先缩放再旋转平移
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.name = name

    # 局部：X=长度，Y=厚度，Z=高度；底面贴地 → 中心在 Z=height/2
    obj.scale = (length / 2.0, thickness / 2.0, height / 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    yaw = math.atan2(dy, dx)
    obj.rotation_euler = (0.0, 0.0, yaw)
    mid_x = 0.5 * (x1 + x2)
    mid_y = 0.5 * (y1 + y2)
    obj.location = Vector((mid_x, mid_y, height / 2.0))

    # 移入目标集合
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def build_walls_from_data(
    data: Dict[str, Any],
    *,
    override_height: float,
    override_thickness: float,
    collection_name: str,
    object_prefix: str,
    clear_existing: bool,
) -> Tuple[int, List[str]]:
    """根据 JSON 数据在场景中生成墙体，返回 (成功数, 物体名列表)。"""
    bpy, Vector = _ensure_bpy()

    col = get_or_create_collection(bpy, collection_name)
    if clear_existing:
        n = clear_collection_objects(bpy, col)
        logger.info("已清空集合 '%s' 中 %d 个旧物体", collection_name, n)

    created_names: List[str] = []
    walls = data["walls"]
    for i, w in enumerate(walls):
        x1 = float(w["x1"])
        y1 = float(w["y1"])
        x2 = float(w["x2"])
        y2 = float(w["y2"])
        height = override_height if override_height > 0 else float(w.get("height_m", 2.8))
        thickness = (
            override_thickness
            if override_thickness > 0
            else float(w.get("thickness_m", 0.2))
        )
        name = f"{object_prefix}_{i:04d}"
        obj = create_wall_box(
            bpy,
            Vector,
            name=name,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            thickness=thickness,
            height=height,
            collection=col,
        )
        if obj is not None:
            created_names.append(obj.name)

    logger.info("成功生成 %d / %d 段墙体", len(created_names), len(walls))
    return len(created_names), created_names


def maybe_export(bpy, fbx_path: str, obj_path: str) -> None:
    """可选导出 FBX / OBJ。"""
    if fbx_path:
        out = Path(fbx_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.export_scene.fbx(filepath=str(out), use_selection=False)
        logger.info("已导出 FBX: %s", out)
    if obj_path:
        out = Path(obj_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Blender 4.x: wm.obj_export；旧版 export_scene.obj
        if hasattr(bpy.ops.wm, "obj_export"):
            bpy.ops.wm.obj_export(filepath=str(out))
        else:
            bpy.ops.export_scene.obj(filepath=str(out))
        logger.info("已导出 OBJ: %s", out)


def main() -> int:
    setup_logging(LOG_LEVEL)
    json_path = resolve_walls_json(WALLS_JSON)
    logger.info("读取 %s", json_path)
    data = load_walls_json(json_path)

    count, _names = build_walls_from_data(
        data,
        override_height=OVERRIDE_WALL_HEIGHT_M,
        override_thickness=OVERRIDE_WALL_THICKNESS_M,
        collection_name=COLLECTION_NAME,
        object_prefix=OBJECT_PREFIX,
        clear_existing=CLEAR_EXISTING,
    )

    bpy, _Vector = _ensure_bpy()
    maybe_export(bpy, EXPORT_FBX, EXPORT_OBJ)
    logger.info("完成，墙体数量=%d", count)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Blender 建墙失败")
        raise
