"""固定障碍物构造：箱体（凹形碰撞体）+ 桌面 + 相机碰撞体。

⛔ 本项目与 ``yanjiuyuan`` **不互通**：箱体几何 / CDPrim / 桌面 / 相机碰撞体的
构造代码全部落在本文件（从 ``yanjiuyuan/obstacle.py`` 抄入并在此固定），
不 import yanjiuyuan 的任何模块。``wrs`` 是公共库，照常使用。

坐标系：三者都在 **sim 世界系**（``UR7EDH76().fk()`` 的输出系）；箱位姿由
``config.load_box_homomat()`` 给出，桌面 STL 位于 wrs 的 ur7e/meshes 下。

几何唯一真源
    ``box_collider_slabs()`` 给出 5 块板（底板 + 四壁）的中心与半长；碰撞 CDPrim
    （``make_hollow_box_cdprim``）、展示厚板（``box_panel_slabs``）、箱内腔 /
    壁顶 / 底板顶的反算全部读它，改一处即可全局跟随。壁厚 43 mm / 底板 5 mm。

仿真展示
    ``attach_obstacles_to_base(base)`` 无条件挂箱体 5 块厚板 + 桌面 + 相机碰撞体，
    ⛔ 不叠外廓棱线、不设展示开关、不设画法选择。
    ⛔ 只影响画面：碰撞检测请照旧用 ``build_obstacles()``，两者不可互替。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
for _root in (str(PROJECT_DIR), str(REPO_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import config as cfg  # noqa: E402
from panda3d.core import CollisionBox, CollisionNode, Point3  # noqa: E402
import modeling.collision_model as mcm  # noqa: E402
import modeling.geometric_model as mgm  # noqa: E402
import visualization.panda.world as wd
import robot_sim.robots.ur7e.ur7e_withouttable_dh76 as u76  # noqa: E402


# ==================== 箱体碰撞体几何（本项目唯一真源，不再跨项目取数） ====================
# 抄自 yanjiuyuan/obstacle.py:41-60 / 179-320 / 438-480，并在此固定。
# 两个项目不互通：以后改箱体几何只改这一段，不要去改 yanjiuyuan 的那一份。
DEFAULT_PANEL_EX_RADIUS = 0.001
PANEL_ALPHA = 0.34
BOX_CDPRIM_LENGTH = 0.6 + 0.05
BOX_CDPRIM_WIDTH = 0.4 + 0.05
BOX_CDPRIM_WALL_HEIGHT = 0.23 + 0.02
# GOMP 高度场的壁高 = **真实壁顶**（与 BOX_CDPRIM_WALL_HEIGHT 同源，11-16 统一）。
# 历史值 0.23 + 0.10 = 0.33 是「加高壁调试对照」：高度场比真实碰撞体高 8 cm，
# SQP 会要求比 min_lift_to_clear_wall 再多抬 8 cm，与实际抬升口径打架。
GOMP_BOX_WALL_HEIGHT = BOX_CDPRIM_WALL_HEIGHT
BOX_CDPRIM_WALL_THICKNESS = 0.02 + 0.023
BOX_CDPRIM_BOTTOM_THICKNESS = 0.005
TABLE_MESH_PATH = (REPO_ROOT / "robot_sim" / "robots" / "ur7e"
                   / "meshes" / "wholetable.STL")


def box_collider_slabs(wall_height: float | None = None) -> list:
    """箱体碰撞体 5 块板在**箱局系**的 ``(name, center, half)``。

    本函数是几何唯一真源：``make_hollow_box_cdprim`` 逐块 ``addSolid``、
    ``box_panel_slabs`` 反算展示厚板、箱内腔 / 壁顶 / 底板顶都读它。
    外扩半径固定取 ``DEFAULT_PANEL_EX_RADIUS``，不是输入量。

    Args:
        wall_height: 壁高（箱局系 z，米）。``None`` = 真实碰撞体壁高
            ``BOX_CDPRIM_WALL_HEIGHT``（0.25 m）；``GOMP_BOX_WALL_HEIGHT`` 自
            11-16 起与它**同值**（高度场壁高 = 真实壁顶）。xy 尺寸不受影响。

    Returns:
        list[tuple]: 5 项，顺序 = CDPrim 的 ``addSolid`` 顺序
        （``bottom`` / ``front`` / ``back`` / ``left`` / ``right``）；每项为
        ``(name: str, center: (3,) ndarray, half: (3,) ndarray)``，单位米、箱局系，
        ``half`` 是三轴半长（已含 1 mm 外扩）。
    """
    ex = float(DEFAULT_PANEL_EX_RADIUS)
    length = float(BOX_CDPRIM_LENGTH)
    width = float(BOX_CDPRIM_WIDTH)
    # 不能叫 wall_height（会 shadow 掉入参）
    _wh = float(BOX_CDPRIM_WALL_HEIGHT if wall_height is None else wall_height)
    wall_t = float(BOX_CDPRIM_WALL_THICKNESS)
    bottom_t = float(BOX_CDPRIM_BOTTOM_THICKNESS)
    half_length = 0.5 * length + ex
    half_width = 0.5 * width + ex
    half_bottom_t = 0.5 * bottom_t + ex
    half_wall_t = 0.5 * wall_t + ex
    half_wall_height = 0.5 * _wh + ex
    return [
        ("bottom", np.array([0.0, 0.0, 0.5 * bottom_t]),
         np.array([half_length, half_width, half_bottom_t])),
        ("front", np.array([0.0, 0.5 * width - 0.5 * wall_t, 0.5 * _wh]),
         np.array([half_length, half_wall_t, half_wall_height])),
        ("back", np.array([0.0, -0.5 * width + 0.5 * wall_t, 0.5 * _wh]),
         np.array([half_length, half_wall_t, half_wall_height])),
        ("left", np.array([-0.5 * length + 0.5 * wall_t, 0.0, 0.5 * _wh]),
         np.array([half_wall_t, half_width, half_wall_height])),
        ("right", np.array([0.5 * length - 0.5 * wall_t, 0.0, 0.5 * _wh]),
         np.array([half_wall_t, half_width, half_wall_height])),
    ]


def make_hollow_box_cdprim(name="hollow_box_cdprim", **_cdprim_kwargs):
    """构造箱体凹形 CDPrim（底板 + 四壁，5 个 ``CollisionBox``）。

    Args:
        name: CDPrim 节点名前缀（str）。wrs 回调会以 ``name=...`` 传入。
        **_cdprim_kwargs: 仅为兼容 wrs 的 ``userdefined_cdprimitive_fn(name, radius)``
            回调签名；内容**一律忽略**，外扩半径固定由 ``box_collider_slabs()`` 决定。

    Returns:
        CollisionNode: 含 5 个 ``CollisionBox`` 的 Panda3D 碰撞节点，几何逐块对应
        ``box_collider_slabs()``。
    """
    pdcnd = CollisionNode(name + "_cnode")
    for _slab_name, center, half in box_collider_slabs():
        pdcnd.addSolid(
            CollisionBox(
                Point3(float(center[0]), float(center[1]), float(center[2])),
                x=float(half[0]),
                y=float(half[1]),
                z=float(half[2]),
            )
        )
    return pdcnd


def make_concave_box_collision_obstacles(box_homomat: np.ndarray,
                                         show_cdprim: bool = True) -> list:
    """构造箱体碰撞模型：STL 外观 + USER_DEFINED 凹形 CDPrim。

    Args:
        box_homomat: (4,4) 箱位姿（sim 世界系），直接作为模型的 ``homomat``。
        show_cdprim: 是否调用 ``show_cdprimit()`` 把 5 个 AABB 画出来（仅调试）。
            默认 True 与原实现一致；生产链路 ``build_obstacles`` 传 False。

    Returns:
        list[CollisionModel]: 恒为 1 个元素（箱体），调用方会再追加桌面与相机。
    """
    box = mcm.CollisionModel(
        initor=str(cfg.CONFIG["path"]["box_stl_path"]),
        name="detected_hollow_box_collision",
        cdprimit_type="user_defined",
        expand_radius=DEFAULT_PANEL_EX_RADIUS,
        userdefined_cdprimitive_fn=make_hollow_box_cdprim,
    )
    box.set_rgba([0.15, 0.42, 0.95, PANEL_ALPHA])
    box.set_homomat(box_homomat)
    if show_cdprim:
        box.show_cdprimit()
    return [box]


def make_box_wall_cdprim(name="box_wall_cdprim", **_cdprim_kwargs):
    """箱体**四壁**的 CDPrim（跳过底板），几何逐块对应 ``box_collider_slabs()[1:]``。

    为什么要有它：抓取那一刻瓶底就贴在箱底板上（``box_inner_bounds()`` 的
    ``z_min`` = 0.006 m），拿含底板的凹箱去查「持物 vs 障碍」会**恒报碰撞**；
    底板本来就被机器人自身碰撞体覆盖，持物检查不需要它。

    Args:
        name: CDPrim 节点名前缀（str），wrs 回调会以 ``name=...`` 传入。
        **_cdprim_kwargs: 仅为兼容 wrs 的 ``userdefined_cdprimitive_fn(name, radius)``
            回调签名；内容**一律忽略**，外扩半径固定由 ``box_collider_slabs()`` 决定。

    Returns:
        CollisionNode: 含 4 个 ``CollisionBox`` 的 Panda3D 碰撞节点
        （front / back / left / right）。
    """
    pdcnd = CollisionNode(name + "_cnode")
    for _slab_name, center, half in box_collider_slabs()[1:]:
        pdcnd.addSolid(CollisionBox(
            Point3(float(center[0]), float(center[1]), float(center[2])),
            x=float(half[0]),
            y=float(half[1]),
            z=float(half[2]),
        ))
    return pdcnd


def make_box_wall_collision_obstacle(box_homomat: np.ndarray):
    """构造「只有四壁」的箱体碰撞模型，专供持物（瓶子）碰撞检查。

    Args:
        box_homomat: (4,4) 箱位姿（sim 世界系），直接作为模型的 ``homomat``。

    Returns:
        CollisionModel: 4 个 ``CollisionBox`` 的 USER_DEFINED 模型，名字
        ``box_wall_only``；不含底板、不含桌面、不含相机。
    """
    wall = mcm.CollisionModel(
        initor=str(cfg.CONFIG["path"]["box_stl_path"]),
        name="box_wall_only",
        cdprimit_type="user_defined",
        # 回调忽略 radius（外扩已含在 box_collider_slabs 的 half 里），显式写 0
        # 而不是走本机默认 0.002，避免语义歧义。
        expand_radius=0.0,
        userdefined_cdprimitive_fn=make_box_wall_cdprim,
    )
    wall.set_homomat(np.asarray(box_homomat, dtype=float).reshape(4, 4))
    return wall


def make_table_obstacle():
    """构造桌面碰撞体（AABB，几何来自 wrs 公共库 STL）。

    Returns:
        CollisionModel: 由 ``wholetable.STL`` 构造的桌面模型，``expand_radius``
        = 0.01 m、灰色半透明（alpha=0.65），位姿取 STL 自带（不额外平移）。

    Raises:
        FileNotFoundError: ``TABLE_MESH_PATH`` 不存在。
    """
    if not TABLE_MESH_PATH.exists():
        raise FileNotFoundError(f"Table mesh not found: {TABLE_MESH_PATH}")
    table = mcm.CollisionModel(
        initor=str(TABLE_MESH_PATH),
        name="table",
        cdprimit_type="box",
        expand_radius=0.01,
    )
    table.set_rgba([0.55, 0.55, 0.55, 0.65])
    return table


def make_camera_collision_obstacle(show_cdprim: bool = False):
    """构造梅卡曼德相机占位碰撞体（世界系 (0.7, 0, 1.3) 的 1.30x1.3x0.1 m 板）。

    Args:
        show_cdprim: 是否调用 ``show_cdprimit()`` 把 AABB 画出来（仅调试）。

    Returns:
        CollisionModel: AABB 类型、``expand_radius``=0.005 m、橙色半透明的相机占位盒，
        位姿 ``((0.7, 0, 1.3), I)``。
    """
    camera_pos = np.array([0.7, 0.0, 1.3], dtype=float)
    box_sgm = mgm.gen_box(extent=np.array([1.30, 1.3, 0.1], dtype=float))
    camera_box = mcm.CollisionModel(
        box_sgm,
        name="camera_collision",
        cdprimit_type="box",
        expand_radius=0.005,
    )
    camera_box.set_rgba([1.0, 0.62, 0.2, 0.28])
    camera_box.set_pos(camera_pos)
    camera_box.set_rotmat(np.eye(3))
    if show_cdprim:
        camera_box.show_cdprimit()
    return camera_box


def _slab_lookup(wall_height: float | None = None) -> dict:
    """把 ``box_collider_slabs()`` 转成 ``{name: (center, half)}``。

    Args:
        wall_height: 壁高（箱局系 z，米）。``None`` = 真实碰撞体壁高；
            ``GOMP_BOX_WALL_HEIGHT`` 自 11-16 起与它同值。

    Returns:
        dict: 键为 ``bottom/front/back/left/right``，值为 ``(center(3,), half(3,))``
        numpy 数组，单位米、箱局系。
    """
    return {
        name: (np.asarray(center, float), np.asarray(half, float))
        for name, center, half in box_collider_slabs(wall_height=wall_height)
    }


def box_inner_bounds(wall_height: float | None = None) -> np.ndarray:
    """箱内腔（箱局系）的 min/max 边界。

    Args:
        wall_height: 壁高（箱局系 z，米）。``None`` = 真实碰撞体
            （``BOX_CDPRIM_WALL_HEIGHT`` = 0.25）；``GOMP_BOX_WALL_HEIGHT`` 自
            11-16 起与它同值，传了不再抬高内腔。

    Returns:
        np.ndarray: shape (2,3) 的 ``[[xmin, ymin, zmin], [xmax, ymax, zmax]]``，
        单位米、箱局系。实测真实碰撞体为 x±0.281 / y±0.181 / z∈[0.006, 0.251]。
    """
    slabs = _slab_lookup(wall_height=wall_height)
    left_center, left_half = slabs["left"]
    right_center, right_half = slabs["right"]
    back_center, back_half = slabs["back"]
    front_center, front_half = slabs["front"]
    bottom_center, bottom_half = slabs["bottom"]

    x_min = float(left_center[0] + left_half[0])
    x_max = float(right_center[0] - right_half[0])
    y_min = float(back_center[1] + back_half[1])
    y_max = float(front_center[1] - front_half[1])
    z_min = float(bottom_center[2] + bottom_half[2])
    z_max = float(front_center[2] + front_half[2])
    return np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]], dtype=float)


def box_wall_top_local_z(wall_height: float | None = None) -> float:
    """箱壁顶的箱局系 z（米）。

    Args:
        wall_height: 壁高（箱局系 z，米）。``None`` = 真实碰撞体（返回 0.251）；
            ``GOMP_BOX_WALL_HEIGHT`` 自 11-16 起与它同值（0.251）。

    Returns:
        float: 四壁顶面在箱局系下的 z 坐标（米）。抓取抬升段必须越过它，
        否则手爪会撞壁顶。
    """
    return float(box_inner_bounds(wall_height=wall_height)[1, 2])


def box_floor_top_local_z() -> float:
    """箱底板顶面（= 物体放置面）的箱局系 z（米）。

    Args:
        无。

    Returns:
        float: 底板顶面 z，实测 0.006 m（底板厚 0.005 + expand_radius 0.001）。
        物体 CAD 原点就在这个面上，所以物体抓点世界高度 =
        ``box_homomat[2, 3] + 0.006 + OBJECT_GRASP_HEIGHT_LOCAL_M``。
    """
    return float(box_inner_bounds()[0, 2])


def box_outer_bounds() -> np.ndarray:
    """箱体外廓（箱局系）的 min/max 边界。

    ⚠️ 与 ``box_inner_bounds`` 的区别：内腔是**开口**范围，外廓是**外壁**范围。
    判「物体是否在箱子正上方的投影里」要用外廓 —— 物体落在内外壁之间同样会撞壁。

    Args:
        无。

    Returns:
        np.ndarray: shape (2,3) 的 ``[[xmin, ymin, zmin], [xmax, ymax, zmax]]``，
        单位米、箱局系。实测 x±0.326 / y±0.226 / z∈[-0.001, 0.251]（比内腔每边
        厚 45 mm，这就是壁厚；``grasp_plan`` 的越壁检查也取这组值）。
    """
    slabs = _slab_lookup()
    left_center, left_half = slabs["left"]
    right_center, right_half = slabs["right"]
    back_center, back_half = slabs["back"]
    front_center, front_half = slabs["front"]
    bottom_center, bottom_half = slabs["bottom"]
    return np.array(
        [
            [
                float(left_center[0] - left_half[0]),
                float(back_center[1] - back_half[1]),
                float(bottom_center[2] - bottom_half[2]),
            ],
            [
                float(right_center[0] + right_half[0]),
                float(front_center[1] + front_half[1]),
                float(box_wall_top_local_z()),
            ],
        ],
        dtype=float,
    )


def wall_top_world_z(box_homomat: "np.ndarray | None" = None) -> float:
    """箱壁顶在 sim 世界系下的 z（米）。

    Args:
        box_homomat: (4,4) 箱体位姿（sim 世界系）；``None`` = ``cfg.load_box_homomat()``。

    Returns:
        float: 壁顶世界 z，实测 0.248458 m。
    """
    box = cfg.load_box_homomat() if box_homomat is None else np.asarray(box_homomat, float)
    return float((box @ np.array([0.0, 0.0, box_wall_top_local_z(), 1.0]))[2])


def object_corners_world(object_pose: "np.ndarray") -> np.ndarray:
    """物体 bbox 的 8 个角点在世界系下的坐标。

    Args:
        object_pose: (4,4) 物体位姿（sim 世界系，原点 = CAD 原点 = 底面中心）。

    Returns:
        np.ndarray: shape (8,3)，顺序为 ``(±x, ±y, 0|高度)`` 的三重笛卡尔积，
        单位米、世界系。物体倾斜时最低点不在原点正下方，所以必须取角点最小值。
    """
    obj = np.asarray(object_pose, dtype=float)
    assert obj.shape == (4, 4), "object_pose 必须是 4x4，实际 {}".format(obj.shape)
    hx = cfg.CONFIG["object"]["bbox_xy_m"][0] / 2.0
    hy = cfg.CONFIG["object"]["bbox_xy_m"][1] / 2.0
    local = np.array(
        [
            [sx * hx, sy * hy, z]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for z in (0.0, cfg.CONFIG["object"]["height_m"])
        ],
        dtype=float,
    )
    return (obj[:3, :3] @ local.T).T + obj[:3, 3]


def object_lowest_world_z(object_pose: "np.ndarray") -> float:
    """物体最低点的世界 z（米）。

    Args:
        object_pose: (4,4) 物体位姿（sim 世界系）。

    Returns:
        float: 8 个 bbox 角点里最小的世界 z。瓶子直立时等于 ``object_pose[2, 3]``
        （原点就是底面中心）；倾斜时会低于它。
    """
    return float(object_corners_world(object_pose)[:, 2].min())


def min_lift_to_clear_wall(
    object_pose: "np.ndarray",
    clearance: "float | None" = None,
    box_homomat: "np.ndarray | None" = None,
) -> float:
    """让**被夹物体**最低点升到「箱壁顶 + clearance」所需的最小竖直抬升。

    为什么需要它（实测）：抓点在物体系 z=0.204，瓶子 220 mm 高 ⇒ 整只瓶身垂在
    夹爪下方 204 mm。只按手爪本体算「高出壁顶 47 mm」是不够的，抬升不足时
    转运会让瓶底蹭着壁顶拖过去。

    Args:
        object_pose: (4,4) 物体位姿（sim 世界系，原点 = 底面中心）。
        clearance: 瓶底高出壁顶的余量（米）；``None`` = ``cfg.CONFIG["motion"]["transit_clearance_m"]``。
        box_homomat: (4,4) 箱体位姿；``None`` = ``cfg.load_box_homomat()``。

    Returns:
        float: 需要的最小竖直抬升（米）。当前样本实测 0.2950 m。可能为负
        （物体本来就高于壁顶时），调用方应自带下限 ``cfg.CONFIG["motion"]["lift_distance_m"]``。
    """
    margin = cfg.CONFIG["motion"]["transit_clearance_m"] if clearance is None else float(clearance)
    return wall_top_world_z(box_homomat) + margin - object_lowest_world_z(object_pose)


def build_obstacles(
    *,
    box_homomat: np.ndarray | None = None,
    show_cdprim: bool = False,
) -> list:
    """构造规划用的障碍物列表。

    Args:
        box_homomat: 4x4 箱位姿（sim 世界系）。``None`` = 调 ``cfg.load_box_homomat()``
            读固定文件。
        show_cdprim: 是否调用各模型的 ``show_cdprimit()`` 做可视化调试。默认 False
            （无 DISPLAY 的远程环境不需要）。

    Returns:
        list: ``CollisionModel`` 列表，可直接传给
        ``robot.is_collided(obstacle_list=...)``。元素 0 是箱体（凹形 5 板），
        其后依次追加桌面与相机碰撞体（无条件包含）。
    """
    homomat = (
        cfg.load_box_homomat()
        if box_homomat is None
        else np.asarray(box_homomat, dtype=float).reshape(4, 4)
    )
    obstacles = list(make_concave_box_collision_obstacles(homomat, show_cdprim=show_cdprim))
    obstacles.append(make_table_obstacle())
    obstacles.append(_camera_obstacle(show_cdprim=show_cdprim))
    return obstacles


def _camera_obstacle(show_cdprim: bool = False):
    """构造相机碰撞体，并把名字固定成 ``camera_collision``。

    wrs 的 ``CollisionModel.name`` 是只读属性，用几何体（``gen_box``）初始化时会被
    几何体名覆盖成 ``"sgm"`` ⇒ 日志/``describe_obstacles`` 里看不出这是相机。
    这里补一次 ``_name``（同 ``make_hollow_box_cdprim`` 显式命名的做法）。

    Args:
        show_cdprim: 是否调 ``show_cdprimit()`` 把 AABB 也画出来（仅调试）。

    Returns:
        CollisionModel: 已摆好 ``pose = ((0.7, 0, 1.3), I)`` 的相机占位盒。
    """
    model = make_camera_collision_obstacle(show_cdprim=show_cdprim)
    model._name = "camera_collision"
    return model


def _homomat(pos, rot) -> np.ndarray:
    """把位置 + 旋转矩阵拼成 4x4 齐次矩阵。

    Args:
        pos: 长度 3 的位置（米）。
        rot: (3,3) 旋转矩阵（正交，det=+1）。

    Returns:
        np.ndarray: shape (4,4) 齐次矩阵，末行 ``[0, 0, 0, 1]``。
    """
    homomat = np.eye(4, dtype=float)
    homomat[:3, :3] = np.asarray(rot, dtype=float).reshape(3, 3)
    homomat[:3, 3] = np.asarray(pos, dtype=float).reshape(3)
    return homomat


def box_panel_slabs(
    box_homomat: "np.ndarray | None" = None,
    wall_height: float | None = None,
) -> list:
    """箱体 5 块板（底板 + 四壁）在**世界系**的位姿与尺寸（= 碰撞 CDPrim 的几何）。

    ⛔ 几何不在这里算：中心/半长全部来自本模块的 ``box_collider_slabs()``，
    也就是 ``make_hollow_box_cdprim`` 逐块 ``addSolid`` 用的同一组数。
    所以「画面上的厚板」与「碰撞体」逐块重合，不存在画一套、算一套。

    实测壁厚 ``BOX_CDPRIM_WALL_THICKNESS`` = 43 mm（四壁）、底板 5 mm
    （``BOX_CDPRIM_BOTTOM_THICKNESS``），壁高 250 mm —— 这就是「有厚度」的来源；
    每块还各向外扩 ``DEFAULT_PANEL_EX_RADIUS`` = 1 mm。

    Args:
        box_homomat: (4,4) 箱体位姿（sim 世界系）；``None`` = ``cfg.load_box_homomat()``。
        wall_height: 壁高（箱局系 z，米）；``None`` = 真实碰撞体壁高。
            ``GOMP_BOX_WALL_HEIGHT`` 自 11-16 起与它同值。

    Returns:
        list[dict]: 5 项，顺序 = ``box_collider_slabs()`` 的 ``addSolid`` 顺序
        （``bottom`` / ``front`` / ``back`` / ``left`` / ``right``）。每项键为
        ``name``（str）、``center``（世界系 3 维，米）、``rotmat``（3x3，等于箱体
        旋转）、``lengths``（3 维**三边全长**，米 = 2×half，可直接喂 ``gen_box``）。
    """
    homomat = (
        cfg.load_box_homomat()
        if box_homomat is None
        else np.asarray(box_homomat, dtype=float).reshape(4, 4)
    )
    rotmat = homomat[:3, :3]
    slabs = []
    for name, center, half in box_collider_slabs(wall_height=wall_height):
        slabs.append(
            {
                "name": name,
                "center": rotmat @ np.asarray(center, dtype=float) + homomat[:3, 3],
                "rotmat": rotmat.copy(),
                "lengths": 2.0 * np.asarray(half, dtype=float),
            }
        )
    return slabs


def box_panel_models(
    box_homomat: "np.ndarray | None" = None,
    *,
    alpha: float = 0.34,
    rgb=(0.15, 0.42, 0.95),
    wall_height: float | None = None,
) -> list:
    """把箱体 5 块板画成**有厚度的实体盒子**（可视化模型，已摆好位姿，未 attach）。

    ⛔ 为什么不用整箱 ``box.STL``：那个模型是**薄壳**，看起来只有一层皮，看不到壁厚；
    这里按 ``box_panel_slabs()`` 逐块生成长方体，四壁 43 mm、底板 5 mm 一眼可见，
    而且与碰撞 CDPrim 完全重合。

    Args:
        box_homomat: (4,4) 箱体位姿（sim 世界系）；``None`` = ``cfg.load_box_homomat()``。
        alpha: 板的不透明度（0~1），默认 0.34（能看见壁厚又能看见箱内轨迹）。
        rgb: 长度 3 的颜色（0~1），默认 (0.15, 0.42, 0.95)。
        wall_height: 壁高（箱局系 z，米）；``None`` = 真实碰撞体壁高。

    Returns:
        list: 5 个 ``mcm.CollisionModel``（顺序同 ``box_panel_slabs()``），
        ``.name`` 为 ``box_panel_<slab>``，``.homomat`` 已写好。调用方
        ``model.attach_to(base)`` 即可显示。
    """
    alpha = float(alpha)
    rgb = np.asarray(rgb, dtype=float).reshape(3)
    rgba = [rgb[0], rgb[1], rgb[2], alpha]
    models = []
    for slab in box_panel_slabs(box_homomat, wall_height=wall_height):
        sgm = mgm.gen_box(extent=slab["lengths"], rgba=rgba)
        model = mcm.CollisionModel(
            sgm,
            name="box_panel_{}".format(slab["name"]),
            cdprimit_type="box",
            expand_radius=0.0,  # 外扩已含在 box_collider_slabs 的 half 里
        )
        model.set_rgba(rgba)
        # wrs 的 GeometricModel.name 是只读属性，且用几何体初始化时会被几何体名（"sgm"）
        # 覆盖 ⇒ 必须像 make_hollow_box_cdprim 一样显式写 _name，
        # 否则 describe_obstacles / 日志里 5 块板全叫 "sgm"，无法区分。
        model._name = "box_panel_{}".format(slab["name"])
        model.set_homomat(_homomat(slab["center"], slab["rotmat"]))
        models.append(model)
    return models


def _make_obstacle_model(kind: str):
    """按种类构造单个障碍物的**可视化模型**（不 attach，不参与碰撞）。

    ``attach_obstacles_to_base`` 用本函数构造桌面与相机的可视化模型。

    Args:
        kind: ``"table"`` / ``"camera"``。箱体走 ``box_panel_models``（5 块厚板），
            不经过本函数。

    Returns:
        CollisionModel: 对应障碍物的可视化模型。

    Raises:
        ValueError: ``kind`` 不在上述两种之内。
    """
    if kind == "table":
        return make_table_obstacle()
    if kind == "camera":
        return _camera_obstacle()
    raise ValueError(f"未知障碍物种类 {kind!r}，只支持 'table' / 'camera'")

def get_gomp_finger_cdprim(name="dh76_gomp_finger", radius=0.0):
    """GOMP（SQP）用的手爪外接盒 —— 张开 0.041 时的**保守包络**，比真实指尖大。

    逐项照抄上游新版 wrs 的 ``Dh76._gomp_finger_cdprim``：中心 (0, 0, 0.038)、
    半长 (0.035, 0.012, 0.040)（+ ``radius``）。本机扁平 wrs 的 dh76 只有两只
    手指各自的 CDPrim，没有这个「合起来的外接盒」，故在此复刻，不改公共库。

    调用方只用**半长**（``read_boxes_from_cdprim(...)[0][1]`` 拼 ``AttachedObject``），
    盒心相对 TCP 的偏移是另一处常量（-0.0255 m），中心值仍与上游保持一致以便核对。

    Args:
        name: CDPrim 节点名前缀。
        radius: 三轴外扩半径（米）。

    Returns:
        CollisionNode: 含 1 个 ``CollisionBox`` 的 Panda3D 碰撞节点。
    """
    pdcnd = CollisionNode(name + "_cnode")
    pdcnd.addSolid(CollisionBox(Point3(0, 0, .038),
                                x=.035 + radius,
                                y=.012 + radius,
                                z=.040 + radius))
    return pdcnd


def get_tea_bottle_sample_cdprim(name="bottle_cdprim", radius=0):
    """冰红茶瓶子的两段式 USER_DEFINED CDPrim（瓶身 + 瓶盖）。

    几何口径：瓶身中心 z=0.10 m、半长 0.034/0.034/0.10 m；瓶盖中心 z=0.212 m、
    半长 0.020/0.020/0.008 m。对应 220 mm 高的 ``tea_bottle.STL``，比整只瓶的
    单 AABB 更贴合，专供「持物 vs 障碍」碰撞检查使用。

    Args:
        name: CDPrim 节点名前缀（str），wrs 回调会以 ``name=...`` 传入。
        radius: 三轴外扩半径（米），逐段叠加到两个盒子的半长上；默认 0。

    Returns:
        CollisionNode: 含 2 个 ``CollisionBox`` 的 Panda3D 碰撞节点。
    """
    pdcnd = CollisionNode(name + "_cnode")
    pdcnd.addSolid(CollisionBox(
        Point3(0, 0, 0.10),
        x=.034 + radius, y=.034 + radius, z=.10 + radius))
    pdcnd.addSolid(CollisionBox(
        Point3(0, 0, 0.212),
        x=.02 + radius, y=.02 + radius, z=.008 + radius))
    return pdcnd


def make_tea_bottle_obstacle():
    """构造冰红茶瓶子的碰撞模型（**备用**，尚未接入规划链路）。

    外观用 ``cfg.CONFIG["path"]["object_stl_path"]``，碰撞用 ``get_tea_bottle_sample_cdprim()``
    的两段式 USER_DEFINED CDPrim；位姿固定在世界原点（与原草稿一致），
    半透明红色便于画面核对。

    Returns:
        CollisionModel: 位姿为原点的瓶子碰撞模型，``pos=(0,0,0)``、
        ``rgba=(1,0,0,0.3)``。调用方需自行摆到持物位姿或已放置位姿。

    Note:
        ⛔ 本函数只提供模型工厂：``build_obstacles()`` / ``plan_place`` 目前都不
        调用它，因此**不改变任何现有碰撞检测结果**。接进放置路径前需单独确认
        「持物 vs obstacle_list」的判定时机与豁免区间。
    """
    bottle = mcm.CollisionModel(
        initor=str(cfg.CONFIG["path"]["object_stl_path"]),
        cdprimit_type="user_defined",
        # 该回调把 radius 叠加到盒半长上，本机默认 0.002 会让瓶身整体大 2 mm，
        # 与上游新版默认 0 一致地显式传 0。
        expand_radius=0.0,
        userdefined_cdprimitive_fn=get_tea_bottle_sample_cdprim,
    )
    bottle.set_pos(np.array([0.0, 0.0, 0.0]))
    bottle.set_rgba([1.0, 0.0, 0.0, 0.3])
    return bottle


def get_fanta_bottle_sample_cdprim(name="bottle_cdprim", radius=0):
    """芬达瓶子的两段式 USER_DEFINED CDPrim（瓶身 + 瓶颈）。

    几何口径（STL 沿 z 切片实测，瓶局系原点 = 瓶底中心）：瓶身段中心 z=0.070 m、
    半长 0.032/0.032/0.070 m；瓶颈段中心 z=0.151 m、半长 0.016/0.016/0.013 m。
    对应 162 mm 高的 ``fanta_bottle.STL``，专供「持物 vs 障碍」碰撞检查。

    Args:
        name: CDPrim 节点名前缀（str），wrs 回调会以 ``name=...`` 传入。
        radius: 三轴外扩半径（米），逐段叠加到两个盒子的半长上；默认 0。

    Returns:
        CollisionNode: 含 2 个 ``CollisionBox`` 的 Panda3D 碰撞节点。
    """
    pdcnd = CollisionNode(name + "_cnode")
    pdcnd.addSolid(CollisionBox(
        Point3(0, 0, 0.070),
        x=.032 + radius, y=.032 + radius, z=.070 + radius))
    pdcnd.addSolid(CollisionBox(
        Point3(0, 0, 0.151),
        x=.016 + radius, y=.016 + radius, z=.013 + radius))
    return pdcnd


def make_fanta_bottle_obstacle():
    """构造芬达瓶子的碰撞模型（持物校验用，与 tea 版结构相同、几何不同）。

    外观用 ``cfg.CONFIG["path"]["object_stl_path"]``（随 active_object 切到 fanta STL），
    碰撞用 ``get_fanta_bottle_sample_cdprim()`` 的两段式 USER_DEFINED CDPrim。

    Returns:
        CollisionModel: 位姿为原点的瓶子碰撞模型，``set_pos((0,0,0))``、
        ``rgba=(1,0,0,0.3)``。调用方需自行摆到持物位姿。
    """
    bottle = mcm.CollisionModel(
        initor=str(cfg.CONFIG["path"]["object_stl_path"]),
        cdprimit_type="user_defined",
        expand_radius=0.0,
        userdefined_cdprimitive_fn=get_fanta_bottle_sample_cdprim,
    )
    bottle.set_pos(np.array([0.0, 0.0, 0.0]))
    bottle.set_rgba([1.0, 0.0, 0.0, 0.3])
    return bottle


def get_object_sample_cdprim(name="bottle_cdprim", radius=0):
    """按 ``active_object`` 分派到当前抓取对象的瓶子 CDPrim（规划层统一入口）。

    Args:
        name: CDPrim 节点名前缀（str），wrs 回调以 ``name=...`` 传入。
        radius: 三轴外扩半径（米）。

    Returns:
        CollisionNode: 当前对象（tea/fanta）的两段式 ``CollisionBox`` 节点。
    """
    fn = (get_fanta_bottle_sample_cdprim if cfg.CONFIG["active_object"] == "fanta"
          else get_tea_bottle_sample_cdprim)
    return fn(name=name, radius=radius)


def make_object_obstacle():
    """按 ``active_object`` 分派到当前抓取对象的持物碰撞模型（规划层统一入口）。

    Returns:
        CollisionModel: 当前对象（tea/fanta）的瓶子碰撞模型，位姿在世界原点。
    """
    fn = (make_fanta_bottle_obstacle if cfg.CONFIG["active_object"] == "fanta"
          else make_tea_bottle_obstacle)
    return fn()


def show_bottle_collision_body(object_pose=None, *, base=None,
                               rgb=(1.0, 0.35, 0.15), alpha=0.5):
    """在 wrs 仿真窗口**单独展示当前对象的 STL 外观 + 两段碰撞盒**，核对几何对不对。

    STL 外观（浅蓝半透明）与碰撞盒（橙色半透明）共用同一 ``object_pose``，
    叠加显示以便肉眼判断盒是否贴合/包住瓶子；不挂箱体/桌面/相机、不挂机器人。
    STL 走 ``make_object_obstacle()``、盒几何读 ``get_object_sample_cdprim()``。

    Args:
        object_pose: 可选 (4,4) 瓶子位姿（sim 世界系，原点 = 瓶底中心）。
            ``None`` = 世界原点、单位旋转。
        base: Panda3D 场景根节点（``wd.World``）。``None`` = 新建窗口并阻塞在
            ``base.run()`` 直到关窗。
        rgb: 长度 3 的颜色（0~1），默认 (1.0, 0.35, 0.15) 橙色。
        alpha: 不透明度（0~1），默认 0.5。

    Returns:
        list: 挂到 base 的碰撞盒 ``CollisionModel`` 列表（每段一个，应=2）；
        STL 外观也已 attach 但不计入本列表（保持既有接口）。
    """
    from gomp.obstacles.attached_object import read_boxes_from_cdprim

    owns_base = base is None
    if owns_base:
        base = wd.World(cam_pos=[0.5, -0.5, 0.45], lookat_pos=[0, 0, 0.1])
        mgm.gen_frame(length=0.2).attach_to(base)
    H = (np.eye(4) if object_pose is None
         else np.asarray(object_pose, dtype=float).reshape(4, 4))
    rgb_arr = np.asarray(rgb, dtype=float).reshape(3)
    rgba = [float(rgb_arr[0]), float(rgb_arr[1]), float(rgb_arr[2]), float(alpha)]
    # STL 外观与碰撞盒同口径：origin = 瓶底中心，故 homomat 直接用 H。
    stl_model = make_object_obstacle()
    stl_model.set_rgba([0.45, 0.65, 0.85, 0.30])
    stl_model.set_homomat(H.copy())
    stl_model.attach_to(base)
    boxes = read_boxes_from_cdprim(get_object_sample_cdprim())
    models = []
    for center, half in boxes:
        sgm = mgm.gen_box(extent=2.0 * half, rgba=rgba)
        model = mcm.CollisionModel(
            sgm, name="bottle_cd", cdprimit_type="aabb",
            expand_radius=0.0, rgba=rgba)
        model.set_homomat(_homomat(H[:3, :3] @ center + H[:3, 3], H[:3, :3]))
        model.attach_to(base)
        models.append(model)
    print("[show_bottle_collision_body] active={} stl={} 盒数={} 尺寸={}".format(
        cfg.CONFIG["active_object"],
        cfg.CONFIG["path"]["object_stl_path"].name, len(models),
        [np.round(2.0 * half, 4).tolist() for _, half in boxes]))
    if owns_base:
        print("窗口已打开；关窗即退出。")
        base.run()
    return models


def attach_obstacles_to_base(base, *, box_homomat: "np.ndarray | None" = None) -> dict:
    """把**全部障碍物**无条件挂到 wrs 仿真场景 ``base`` 上。

    箱体（5 块有厚度的板 = 碰撞 CDPrim 的几何）+ 桌面 + 相机碰撞体，
    三项一律展示，不设开关、不设画法选择：关掉任一项都会让「看见的」与「碰撞
    检测用的」不一致，无法凭画面判断路径是否穿箱。

    ⛔ 只影响画面：碰撞检测请照旧用 ``build_obstacles()``，两者不可互替。

    Args:
        base: Panda3D 场景根节点（``wd.World``），所有节点都 ``attach_to`` 它。
        box_homomat: (4,4) 箱位姿（sim 世界系）。``None`` = ``cfg.load_box_homomat()``
            读固定文件；将来接 ``detect_box_pose.py`` 的检测结果时从这里传入。

    Returns:
        dict: ``nodes``（本次 attach 的全部节点）、``box`` / ``table`` / ``camera``
        （各自的节点列表，``box`` 含 5 块厚板）。恒包含全部三类，
        调用方可据此核对画面与碰撞体是否逐块对上。
    """
    homomat = (
        cfg.load_box_homomat()
        if box_homomat is None
        else np.asarray(box_homomat, dtype=float).reshape(4, 4)
    )
    result = {"nodes": [], "box": [], "table": [], "camera": []}

    for model in box_panel_models(homomat):
        model.attach_to(base)
        result["box"].append(model)
    table = _make_obstacle_model("table")
    table.attach_to(base)
    result["table"].append(table)
    camera = _make_obstacle_model("camera")
    camera.attach_to(base)
    result["camera"].append(camera)

    for key in ("box", "table", "camera"):
        result["nodes"].extend(result[key])
    return result


def _show_scene(object_pose: "np.ndarray | None" = None, base=None) -> None:
    """打开 Panda3D 窗口显示全部障碍物（箱体/桌面/相机）与机器人，用于人工核对几何。

    障碍物一律经 ``attach_obstacles_to_base`` 无条件挂载（箱体厚板 +
    桌面 + 相机碰撞体，不叠棱线），另外把碰撞 CDPrim 也画出来对照。

    Args:
        object_pose: 可选，shape (4,4) 的物体位姿（sim 世界系，原点 = 底面中心）。
            传了就在真实抓点位置画红球（``object_pose @ [0,0,抓取高度,1]``）；
            ``None`` = 退化成「箱底中心 + 抓取高度」的示意点（假设物体摆在箱心）。

    Returns:
        None: 阻塞直到窗口被关闭。

    Note:
        ⛔ 必须有真实 X 显示：``wrs.visualization.panda.world.World.__init__``
        直接读 ``base.win.getProperties()``，而 Panda3D 的 offscreen buffer 只提供
        ``getFbProperties()``、``window-type none`` 时 ``base.win`` 是 None，
        两种离屏方式都会抛 AttributeError（已实测）。远程无 DISPLAY 时不要加 ``--show``。
    """
    box_homomat = cfg.load_box_homomat()
    # 取景：拉远抬高到能同时看到 箱体（z≈0.25）/ 机械臂 / 相机碰撞体（z=1.3）
    # 全部障碍物无条件挂进场景（5 块厚板 / 桌面 / 相机占位盒，不叠棱线）
    attached = attach_obstacles_to_base(base)
    print("仿真障碍物：箱体 {} 个节点，桌面 {} 个，相机 {} 个".format(
        len(attached["box"]), len(attached["table"]), len(attached["camera"])))
    for obstacle in build_obstacles():
        try:
            obstacle.show_cdprimit()
        except Exception as exc:  # 可视化失败不影响主流程
            print("show_cdprimit 跳过：", type(exc).__name__, exc)

    robot = u76.UR7EDH76()
    robot.goto_given_conf(cfg.CONFIG["joints"]["start_conf"])
    robot.gen_meshmodel(toggle_jnt_frames=False, alpha=0.8).attach_to(base)

    # 抓点标记：有物体位姿就按真实位姿算（物体系 z=204mm 处），否则退化成箱心示意点。
    if object_pose is None:
        grasp_local = np.array(
            [0.0, 0.0, box_floor_top_local_z() + cfg.CONFIG["object"]["grasp_height_local_m"], 1.0])
        grasp_world = (box_homomat @ grasp_local)[:3]
    else:
        grasp_local = np.array([0.0, 0.0, cfg.CONFIG["object"]["grasp_height_local_m"], 1.0])
        grasp_world = (np.asarray(object_pose, dtype=float) @ grasp_local)[:3]
    mgm.gen_sphere(pos=grasp_world, radius=0.012,
                   rgba=[1.0, 0.2, 0.2, 1.0]).attach_to(base)
    print("抓点标记（红球）世界坐标 =", np.round(grasp_world, 4).tolist())
    print("窗口已打开；关窗即退出。")





if __name__ == "__main__":

    base = wd.World(cam_pos=[2.60, -2.80, 2.55], lookat_pos=[0.60, 0.0, 0.55])
    mgm.gen_frame(length=0.2).attach_to(base)

    _show_scene(base=base)

    base.run()
