"""生成 GOMP 用的离线深度图 ``depthmap/scene.npz``。

碰撞体的**唯一来源**是 ``obstacle.build_obstacles()``（箱体 5 块厚板 + 桌面 +
相机占位板），与 ``plan_place`` / ``check_held_collision_idx`` 用的是同一套几何。
本脚本只做一次「WRS 碰撞模型 -> 2.5D 高度场 + 世界 AABB 表」的离线转换，
运行时 ``GompRefiner`` 只读这个 npz（缺文件直接报错，不做现场兜底）。

⛔ ``max_bottom_z`` 必须过滤掉相机：2.5D 高度场的语义是「该 (x,y) 列 z_top 以下
全部禁止通行」，表达不了悬空物体；相机板在 z=1.30（底面 1.245），不滤掉会把
大半工作区封死。机器人与相机的碰撞由 ``GompRefiner.validate``（WRS 精确判据）
兜底，不进高度场。

用法（Panda3D 场景需要真实 X 显示）::

    DISPLAY=:0 PYTHONPATH=<项目目录> python make_depthmap.py

不读命令行参数：``main()`` 的五个参数由调用方按需传入，脚本入口用默认值。
``show=True``（默认）会在保存 npz 后打开 Panda3D 窗口叠加显示碰撞体供人工
核对；批处理/离线传 ``show=False`` 跳过窗口。
"""
import os

import numpy as np

import config as cfg
import obstacle
from gomp.obstacles.wrs_depthmap import (
    from_obstacle_list, save_depthmap, visualize_depthmap,
    _print_height_levels,
)


def main(out_dir=None, resolution=0.01, margin=0.05, max_bottom_z=1.2,
         show=True):
    """把当前场景的 WRS 碰撞模型转成深度图 npz。

    Args:
        out_dir: str | Path | None 输出目录；``None`` = 本文件同级的 ``depthmap/``。
            目录不存在会自动创建，文件名为 ``scene.npz``。
        resolution: float 高度场网格边长（米）。默认 0.01（1 cm）——箱壁厚
            43 mm 在 1 cm 网格下最坏外扩 < 1 格（≈1 cm，偏保守）。
        margin: float 网格在盒并集之外留的边（米）。
        max_bottom_z: float 悬空障碍过滤阈值（米）：盒**底面** z 大于它的一律
            丢弃。``None`` = 不过滤（本场景会把相机算进来，不要这么做）。
        show: bool 保存 npz 后是否打开 Panda3D 窗口叠加显示碰撞体（红=原始
            OBB、蓝=AABB 输入、彩色实心柱=高度场本体）供真机运动前人工核对。
            默认 ``True``；``False`` = 纯生成 npz、不弹窗（离线/批处理）。

    Returns:
        dict: ``{"path": str, "grid_shape": tuple, "boxes": int, "dropped": list,
        "wall_top": float}``。``path`` 是落盘 npz 的绝对路径；``boxes`` 是保留
        进高度场的盒数；``dropped`` 是被过滤的 (名称, 原因) 列表。
    """
    from pathlib import Path
    import visualization.panda.world as wd

    here = Path(__file__).resolve().parent
    out_dir = Path(out_dir) if out_dir is not None else here / "depthmap"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scene.npz"

    # 场景只用于提供世界变换（extract_world_boxes 走 cdnp.getMat(base)）
    base = wd.World(cam_pos=[1.6, -1.9, 1.6], lookat_pos=[0.5, 0.0, 0.3])
    base.setBackgroundColor(1.0, 1.0, 1.0, 1.0)

    models = obstacle.build_obstacles()
    for m in models:
        m.attach_to(base)

    loaded = from_obstacle_list(models, base, resolution=float(resolution),
                                margin=float(margin), max_bottom_z=max_bottom_z)

    print("\n保留的碰撞盒（世界系 AABB，米）")
    print("{:<18s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "name", "x0", "y0", "z0", "z_top"))
    for b in loaded["boxes"]:
        a = b["aabb"]
        print("{:<18s} {:8.3f} {:8.3f} {:8.3f} {:8.3f}".format(
            b["name"], a[0], a[1], a[2], a[5]))
    if loaded["dropped"]:
        print("\n已排除")
        for name, reason in loaded["dropped"]:
            print("  {:<18s} {}".format(name, reason))

    dm = loaded["depth_map"]
    print("\n网格 {} @ {} m, origin = {}, 高度场 max = {:.3f} m".format(
        dm.grid.shape, dm.resolution, np.round(dm.origin, 3), float(dm.grid.max())))

    save_depthmap(dm, loaded["boxes"], str(out_path), dropped=loaded["dropped"],
                  meta={"resolution": float(resolution), "margin": float(margin),
                        "max_bottom_z": max_bottom_z,
                        "wall_height": float(obstacle.BOX_CDPRIM_WALL_HEIGHT),
                        "source": "obstacle.build_obstacles()",
                        "note": "场景障碍变更（箱位姿/桌面/相机）需重新运行 "
                                "make_depthmap.py"})
    print("\n已保存:", out_path)

    if show:
        # 展示 GOMP 实际约束的碰撞体，口径与 gomp/obstacles/wrs_depthmap.py
        # 的参考实现一致；视觉参数取该模块默认值，不额外暴露。
        _print_height_levels(dm, z_base=0.0)
        visualize_depthmap(base, loaded,
                           rgba_obb=(1, 0, 0, 0.0),     # 原始 OBB
                           rgba_aabb=(0, 0, 1, 0.20),   # 光栅化输入 AABB
                           z_base=0.0, alpha=0.20, show_boxes=True)
        print("红=原始OBB 蓝=AABB输入 彩柱=高度场碰撞体"
              "(每列 z_base~z_top 实心禁止区，按高度分层着色)")
        base.run()

    return {"path": str(out_path), "grid_shape": tuple(dm.grid.shape),
            "boxes": len(loaded["boxes"]), "dropped": loaded["dropped"],
            "wall_top": float(dm.grid.max())}


if __name__ == "__main__":
    main()
