"""WRS CollisionModel 障碍列表 -> DepthMapObstacle(2.5D 高度场) 的转换/存取/可视化。

背景：GOMP 的 SQP 碰撞约束(ObstacleConstraint)只接受单个 DepthMapObstacle，
而 yanpu_ur8 场景障碍是 WRS CollisionModel 列表（rbt_s.get_obstacle_list）。
本模块把 CollisionModel 的世界系碰撞盒光栅化成一张世界系(yaw=0)高度场网格，
并保存为 npz 供正式运行离线加载（正式运行不依赖 Panda3D 场景）。

用法：
  离线转换（在仿真进程里跑一次，人工核对可视化后保存 npz）：
    python GOMP/gomp/obstacles/wrs_depthmap.py
  正式运行（yanpu_ur8/gomp_refine.py）：
    loaded = load_depthmap(path)   # 只读 npz，零场景依赖
    planner = GOMPPlanner(..., obstacles=loaded.depth_map)

设计约定：
  - 所有障碍合并进同一张世界系网格（ObstacleConstraint 只收单个 depth_map）
  - 每个碰撞盒按【世界 AABB】光栅化（格内取 max 顶面 z）；旋转盒用 AABB 是
    保守近似（本项目障碍旋转都是 90° 的倍数，AABB 精确）
  - 【取整规则】填充索引范围两端都用 floor（= "格心/格起点落在 AABB 内才填"）：
    最大外扩 < 1 格(res)，不会像 ceil 那样凭空多填一整格（旧实现上边界用 ceil，
    实测把 1cm 网格的墙面从 0.043 膨胀到 0.0596，多出 1.66cm 幻影墙）
  - 高度场语义 = "xy 列上 z_top 以下全部禁止"：竖直墙/料箱壁表达精确；
    【悬空障碍】（相机、上方护栏，底面高于工作高度）会错误封锁其正下方区域，
    必须用 max_bottom_z 过滤掉；漏检由 gomp_refine.validate() 的 WRS 逐点
    碰撞校验兜底
  - npz 同时保存原始 OBB(center/half/rotmat) 与世界 AABB，供可视化核对与
    meta 一致性检查（场景参数变了要重新生成）
"""
import sys
import os
# 直接以脚本运行（python .../wrs_depthmap.py）时，模块级 import gomp 之前
# sys.path 还没被调用方设置好 —— 这里自行补齐：GOMP 根、GOMP/wrs、项目根。
_HERE = os.path.dirname(os.path.abspath(__file__))
_GOMP_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_GOMP_ROOT, ".."))
for _p in (_PROJECT_ROOT, os.path.join(_GOMP_ROOT, "wrs"), _GOMP_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gomp.obstacles.depth_map import DepthMapObstacle
import json
import numpy as np
import argparse
import math
import visualization.panda.world as wd
import modeling.geometric_model as gm
import basis.robot_math as rm
import robot_sim.robots.ur7e.ur7ewithoutmachine_1 as cbt
# npz 格式版本（字段含义变化时 +1）
# v2: 光栅化取整由上边界 ceil 改为 floor（去掉最多一整格的幻影墙），
#     网格内容与 v1 不同，读到旧 npz 直接报错要求重新生成。
FORMAT_VERSION = 2

# ---------------------------------------------------------------------------
# 几何提取：WRS CollisionModel -> 世界系盒
# ---------------------------------------------------------------------------

def _pdmat4_to_np(m):
    """Panda3D Mat4 -> numpy 4x4（保留 panda 的行向量约定：v' = v @ M3 + t）。"""
    out = np.zeros((4, 4))
    for i in range(4):
        row = m.getRow(i)
        out[i, 0] = row[0]
        out[i, 1] = row[1]
        out[i, 2] = row[2]
        out[i, 3] = row[3]
    return out


def extract_world_boxes(obstacle_list, base):
    """从 WRS CollisionModel 列表提取世界系碰撞盒。

    每个 CollisionModel 的 cdnp（child-1）里是局部系的 CollisionBox/
    CollisionSphere solid（gen_box_cdnp 用 mesh tightbounds 生成 AABB），
    乘 cdnp 相对 base 的世界变换得到世界盒。

    Parameters
    ----------
    obstacle_list : list of cm.CollisionModel
        rbt_s.get_obstacle_list(base, False) 的返回值。
    base : NodePath
        场景根（障碍 attach 到它下面）。

    Returns
    -------
    boxes : list of dict
        每项 {name, rotmat(3,3), obb_center(3,), obb_half(3,),
        aabb(6,)=x0,y0,z0,x1,y1,z1}
    dropped : list of (name, reason)
        无法转成盒（非 box/sphere solid 等）而被跳过的障碍。
    """
    from panda3d.core import CollisionBox, CollisionSphere

    if hasattr(base, "render"):  # wd.World/ShowBase -> 用其 render 根（attach_to 的落点）
        base = base.render
    boxes, dropped = [], []
    name_count = {}
    for cm_obj in obstacle_list:
        # 命名：优先 STL 文件名（CollisionModel 默认 name 全是 "auto" 无法区分），
        # 同名多实例追加序号（如 6 面墙共用一个 stl 的情况不存在，但保险起见）
        objpath = getattr(cm_obj, "objpath", None)
        if objpath:
            base_name = os.path.splitext(os.path.basename(str(objpath)))[0]
        else:
            base_name = getattr(cm_obj, "name", None) or "obstacle"
        idx = name_count.get(base_name, 0)
        name_count[base_name] = idx + 1
        name = base_name if idx == 0 else f"{base_name}#{idx}"
        # 本项目 wrs 的 CollisionModel 没有 .cdnp（那是 yanpu 版属性）：
        # 碰撞节点暴露为 .cdprim（NodePath，含 CollisionNode 作为 child-0）
        cdnp = cm_obj.cdprim
        M = _pdmat4_to_np(cdnp.getMat(base))    # 行向量约定
        M3, t = M[:3, :3], M[3, :3]
        node = cdnp.node()
        ok = False
        for i in range(node.getNumSolids()):
            solid = node.getSolid(i)
            if isinstance(solid, CollisionBox):
                center = np.array(solid.getCenter(), dtype=float)
                half = np.array(solid.getDimensions(), dtype=float) / 2.0
            elif isinstance(solid, CollisionSphere):
                center = np.array(solid.getCenter(), dtype=float)
                r = float(solid.getRadius())
                half = np.array([r, r, r])
            else:
                dropped.append((name, f"不支持的 solid 类型 {type(solid).__name__}"))
                continue
            # 局部 8 角点(以 solid 局部中心为基准) -> 世界 -> AABB
            sx, sy, sz = half
            cx, cy, cz = center
            corners = np.array([[x, y, z] for x in (cx - sx, cx + sx)
                                for y in (cy - sy, cy + sy)
                                for z in (cz - sz, cz + sz)])
            world = corners @ M3 + t
            # 世界旋转部分（行向量约定 R_row = M3.T，列向量约定用）
            R = M3.T.copy()
            boxes.append({
                "name": name,
                "rotmat": R,
                "obb_center": center @ M3 + t,  # 精确 OBB 中心
                "obb_half": half.copy(),
                "aabb": np.array([world[:, 0].min(), world[:, 1].min(),
                                  world[:, 2].min(), world[:, 0].max(),
                                  world[:, 1].max(), world[:, 2].max()]),
            })
            ok = True
        if not ok and node.getNumSolids() == 0:
            dropped.append((name, "无碰撞 solid"))
    return boxes, dropped


# ---------------------------------------------------------------------------
# 光栅化：世界盒 -> 单张高度场网格
# ---------------------------------------------------------------------------

def fill_index_range(aabb, x0, y0, nx, ny, resolution):
    """盒世界 AABB -> 网格填充索引范围（闭区间，含端点）。

    唯一取整规则来源：光栅化(boxes_to_depthmap)、膨胀量打印、可视化三者共用，
    避免"画的格子和实际填的格子不一致"。

    两端均用 floor：索引 i 的格子覆盖 [x0+i*res, x0+(i+1)*res)，floor 表示
    "该格起点落在 AABB 内才填"，因此填充足迹 ⊆ AABB 向外不足 1 格，最坏外扩
    ≈ res(1cm)，而不再是最坏 ≈ 2 格（ceil 会把完全在 AABB 外的下一格也填上）。

    Returns
    -------
    (ix0, ix1, iy0, iy1) : 填充的网格索引闭区间。
    """
    a = np.asarray(aabb, dtype=float)
    ix0 = int(max(0, np.floor((a[0] - x0) / resolution)))
    ix1 = int(min(nx - 1, np.floor((a[3] - x0) / resolution)))
    iy0 = int(max(0, np.floor((a[1] - y0) / resolution)))
    iy1 = int(min(ny - 1, np.floor((a[4] - y0) / resolution)))
    # 极端情况（盒完全落在网格外的浮点误差）保证至少占一格
    ix1 = max(ix0, ix1)
    iy1 = max(iy0, iy1)
    return ix0, ix1, iy0, iy1


def boxes_to_depthmap(boxes, resolution=0.01, margin=0.05):
    """把世界 AABB 盒列表光栅化成单个 DepthMapObstacle（世界系, yaw=0）。

    每格高度 = 覆盖该格的所有盒顶面 z 的最大值；网格范围 = 盒并集 + margin。
    填充索引范围见 fill_index_range（两端 floor，最坏外扩 < 1 格）。
    """
    if len(boxes) == 0:
        raise ValueError("boxes 为空，无法生成深度图")
    aabbs = np.array([b["aabb"] for b in boxes], dtype=float)
    x0 = aabbs[:, 0].min() - margin
    y0 = aabbs[:, 1].min() - margin
    x1 = aabbs[:, 3].max() + margin
    y1 = aabbs[:, 4].max() + margin
    nx = int(np.ceil((x1 - x0) / resolution)) + 1
    ny = int(np.ceil((y1 - y0) / resolution)) + 1
    grid = np.zeros((nx, ny))
    for a in aabbs:
        ix0, ix1, iy0, iy1 = fill_index_range(a, x0, y0, nx, ny, resolution)
        grid[ix0:ix1 + 1, iy0:iy1 + 1] = np.maximum(
            grid[ix0:ix1 + 1, iy0:iy1 + 1], a[5])
    return DepthMapObstacle(grid=grid, resolution=resolution,
                            origin=np.array([x0, y0]), yaw=0.0,
                            center_xy=np.zeros(2))


def from_obstacle_list(obstacle_list, base, resolution=0.01, margin=0.05,
                       max_bottom_z=None, exclude_names=()):
    """WRS 场景障碍 -> 深度图（在线转换入口，仅供离线 dump 脚本使用）。

    Parameters
    ----------
    max_bottom_z : float or None
        悬空障碍过滤：盒底面 z > 该值的一律排除（高度场会错误封锁其正下方，
        如相机/上方护栏）。None = 不过滤。
    exclude_names : tuple of str
        按名称排除的障碍（子串匹配）。

    Returns
    -------
    dict: {depth_map, boxes(保留), dropped(名称+原因列表)}
    """
    boxes, dropped = extract_world_boxes(obstacle_list, base)
    kept = []
    for b in boxes:
        if max_bottom_z is not None and b["aabb"][2] > max_bottom_z:
            dropped.append((b["name"],
                            f"悬空过滤: bottom_z={b['aabb'][2]:.3f} "
                            f"> {max_bottom_z}"))
            continue
        if any(s in b["name"] for s in exclude_names):
            dropped.append((b["name"], "按名称排除"))
            continue
        kept.append(b)
    dm = boxes_to_depthmap(kept, resolution=resolution, margin=margin)
    return {"depth_map": dm, "boxes": kept, "dropped": dropped}


# ---------------------------------------------------------------------------
# npz 存取
# ---------------------------------------------------------------------------

def save_depthmap(dm, boxes, path, dropped=None, meta=None):
    """深度图 + 原始盒 -> npz（正式运行唯一输入来源）。"""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(
        path,
        version=FORMAT_VERSION,
        grid=dm.grid,
        resolution=dm.resolution,
        origin=dm.origin,
        obb_center=np.array([b["obb_center"] for b in boxes]),
        obb_half=np.array([b["obb_half"] for b in boxes]),
        obb_rotmat=np.array([b["rotmat"] for b in boxes]),
        aabb=np.array([b["aabb"] for b in boxes]),
        names=json.dumps([b["name"] for b in boxes]),
        dropped=json.dumps(dropped or []),
        meta=json.dumps(meta or {}),
    )
    return path


def load_depthmap(path):
    """npz -> SimpleNamespace(depth_map, boxes, dropped, meta)。

    缺文件/版本不符直接报错，不做现场兜底转换（方案约定）。
    """
    from types import SimpleNamespace
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"深度图 npz 不存在: {path}\n请先运行离线转换: "
            f"python GOMP/gomp/obstacles/wrs_depthmap.py")
    data = np.load(path, allow_pickle=False)
    if int(data["version"]) != FORMAT_VERSION:
        raise ValueError(
            f"深度图格式版本 {int(data['version'])} != 当前 {FORMAT_VERSION}，"
            f"请重新生成: {path}")
    dm = DepthMapObstacle(grid=data["grid"], resolution=float(data["resolution"]),
                          origin=data["origin"], yaw=0.0,
                          center_xy=np.zeros(2))
    boxes = []
    for i in range(len(data["aabb"])):
        boxes.append({
            "name": json.loads(str(data["names"]))[i],
            "obb_center": data["obb_center"][i],
            "obb_half": data["obb_half"][i],
            "rotmat": data["obb_rotmat"][i],
            "aabb": data["aabb"][i],
        })
    return SimpleNamespace(depth_map=dm, boxes=boxes,
                           dropped=json.loads(str(data["dropped"])),
                           meta=json.loads(str(data["meta"])))


# ---------------------------------------------------------------------------
# 可视化（离线核对用）
# ---------------------------------------------------------------------------

def heightfield_runs(dm):
    """高度场 -> 竖直柱体矩形块 [(x0,x1,y0,y1,z_top)]。

    **这是 GOMP 实际使用的碰撞体几何**：高度场语义为「该 (x,y) 列 z_top 以下
    全部禁止」，等价于每根非零格子立一根从地面到 z_top 的实心柱。
    逐格建柱有数千个 mesh，这里做两趟合并（几何上等价于逐格柱的并集）：
      1) 每行(y=j)沿 x 合并同高度连续段；
      2) 纵向合并 (i0,i1,z) 相同且相邻的段。
    """
    grid = np.asarray(dm.grid, dtype=float)
    org = np.asarray(dm.origin, dtype=float)
    res = float(dm.resolution)
    nx, ny = grid.shape
    # 1) 行内 x 方向 run-length
    segs = []  # (i0, i1, j, z)
    for j in range(ny):
        col = grid[:, j]
        i = 0
        while i < nx:
            z = col[i]
            if z <= 0.0:
                i += 1
                continue
            k = i + 1
            while k < nx and col[k] == z:
                k += 1
            segs.append((i, k - 1, j, float(z)))
            i = k
    # 2) 纵向合并同 (i0,i1,z) 的相邻行
    rects, opened = [], {}
    for (i0, i1, j, z) in segs:
        key = (i0, i1, z)
        cur = opened.get(key)
        if cur is not None and cur[1] == j - 1:
            opened[key] = (cur[0], j)
        else:
            if cur is not None:
                rects.append((i0, i1, cur[0], cur[1], z))
            opened[key] = (j, j)
    for (i0, i1, z), (j0, j1) in opened.items():
        rects.append((i0, i1, j0, j1, z))
    # 3) 索引 -> 世界矩形
    return [(org[0] + i0 * res, org[0] + (i1 + 1) * res,
             org[1] + j0 * res, org[1] + (j1 + 1) * res, z)
            for (i0, i1, j0, j1, z) in rects]


# 高度层配色（按 z 从低到高取色；层数多于色板时循环，alpha 统一由调用方给）
_LEVEL_COLORS = [(0.15, 0.80, 0.55), (0.20, 0.62, 0.95), (0.70, 0.45, 0.95),
                 (0.95, 0.62, 0.20), (0.95, 0.30, 0.35), (0.55, 0.85, 0.20)]


def level_colors(dm, alpha=0.20):
    """高度场里每个非零高度层 -> rgba。返回 {z: rgba}（z 已 round 4 位）。"""
    zs = sorted({round(float(v), 4) for v in np.unique(dm.grid) if v > 0.0})
    out = {}
    for k, z in enumerate(zs):
        r, g, b = _LEVEL_COLORS[k % len(_LEVEL_COLORS)]
        out[z] = np.array([r, g, b, alpha])
    return out


def visualize_depthmap(base, loaded, rgba_obb=(1, 0, 0, 0.0),
                       rgba_aabb=(0, 0, 1, 0.20),
                       z_base=0.0, alpha=0.20, show_boxes=True):
    """叠加显示 GOMP 实际使用的碰撞体：红=原始 OBB，蓝=光栅化输入 AABB，
    彩色实体柱 = 高度场本身（每列从 z_base 到 grid 高度的实心柱，按高度分层着色）。

    柱体才是 SQP 里被约束的形状：`z_点 ≥ grid_height(x,y) + margin`
    等价于"点不得落入这些柱子里"。柱顶面即之前那块"天花板"，单独看会误以为
    障碍只有一层薄板。
    返回生成的 mesh 列表（attach 在 base 上）。
    """
    import modeling.collision_model as cm
    dm = loaded["depth_map"]
    colors = level_colors(dm, alpha=alpha)
    meshes = []
    if show_boxes:
        for b in loaded["boxes"]:
            R = np.asarray(b["rotmat"], dtype=float)
            c = np.asarray(b["obb_center"], dtype=float)
            h = np.asarray(b["obb_half"], dtype=float)
            hm = np.eye(4)
            hm[:3, :3] = R
            hm[:3, 3] = c
            m = cm.gen_box(extent=2.0 * h, homomat=hm,
                          rgba=np.array(rgba_obb))
            m.attach_to(base)
            meshes.append(m)
            a = np.asarray(b["aabb"], dtype=float)
            hm2 = np.eye(4)
            hm2[:3, 3] = (a[:3] + a[3:]) / 2.0
            m2 = cm.gen_box(extent=a[3:] - a[:3], homomat=hm2,
                            rgba=np.array(rgba_aabb))
            m2.attach_to(base)
            meshes.append(m2)
    for (x0, x1, y0, y1, z) in heightfield_runs(dm):
        if z <= z_base:
            continue
        rgb = colors.get(round(z, 4), np.array([0.8, 0.8, 0.8, alpha]))
        hm3 = np.eye(4)
        hm3[:3, 3] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0,
                      (z_base + z) / 2.0)
        m3 = cm.gen_box(extent=np.array([x1 - x0, y1 - y0, z - z_base]),
                        homomat=hm3, rgba=rgb)
        m3.attach_to(base)
        meshes.append(m3)
    return meshes


def _print_height_levels(dm, z_base=0.0):
    """打印高度场分层统计 + 配色图例（与可视化柱体颜色一一对应）。"""
    colors = level_colors(dm)
    grid = np.asarray(dm.grid, dtype=float)
    zs = sorted(colors)
    print(f"\n高度场碰撞体分层（柱体从 z={z_base} 起）：")
    for z in zs:
        n = int((np.round(grid, 4) == z).sum())
        r, g, b, _ = colors[z]
        print(f"  z_top={z:.4f}  格子数={n:5d}  颜色≈"
              f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}")
    runs = heightfield_runs(dm)
    print(f"  合并后柱体块数={len(runs)}（逐格柱的并集，渲染用）")


def _print_fill_table(boxes, dm):
    """打印 AABB 与「实际被填格子」足迹的偏差（光栅化膨胀量，单位 mm）。

    dx0/dy0 为负 = 网格向左/向下多填；dx1/dy1 为正 = 向右/向上多填。
    理论上四者都在 (-res, 0] / [0, res) 内，即最大外扩 < 1 格。
    """
    org = np.asarray(dm.origin, dtype=float)
    res = float(dm.resolution)
    nx, ny = dm.grid.shape
    print(f"\n{'名称':<22s}{'AABB x0':>9s}{'填x0':>9s}{'AABB x1':>9s}{'填x1':>9s}"
          f"{'dx0mm':>8s}{'dx1mm':>8s}{'dy0mm':>8s}{'dy1mm':>8s}")
    worst = 0.0
    for b in boxes:
        a = np.asarray(b["aabb"], dtype=float)
        ix0, ix1, iy0, iy1 = fill_index_range(a, org[0], org[1], nx, ny, res)
        fx0, fx1 = org[0] + ix0 * res, org[0] + (ix1 + 1) * res
        fy0, fy1 = org[1] + iy0 * res, org[1] + (iy1 + 1) * res
        d = np.array([fx0 - a[0], fx1 - a[3], fy0 - a[1], fy1 - a[4]]) * 1000.0
        worst = max(worst, float(np.abs(d).max()))
        print(f"{b['name']:<22s}{a[0]:9.3f}{fx0:9.3f}{a[3]:9.3f}{fx1:9.3f}"
              f"{d[0]:8.1f}{d[1]:8.1f}{d[2]:8.1f}{d[3]:8.1f}")
    print(f"最大外扩 {worst:.1f}mm（应 < {res * 1000:.0f}mm，即不足 1 格）")


# ---------------------------------------------------------------------------
# 离线 dump 入口：建场景 -> 转换 -> 可视化 -> 保存 npz
# ---------------------------------------------------------------------------

def _print_box_table(boxes, dropped):
    print(f"\n{'名称':<28s} {'x0':>7s} {'y0':>7s} {'z0':>7s} "
          f"{'x1':>7s} {'y1':>7s} {'z1':>7s}")
    for b in boxes:
        a = b["aabb"]
        print(f"{b['name']:<28s} " + " ".join(f"{v:7.3f}" for v in a))
    if dropped:
        print("\n已排除：")
        for name, reason in dropped:
            print(f"  {name:<28s} {reason}")


def main(argv=None):

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        _PROJECT_ROOT, "yanpu_ur9", "depthmap", "scene.npz"))   # 输出路径
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--max-bottom-z", dest="max_bottom_z", type=float,
                        default=1.2, help="悬空障碍过滤高度(m)，None 用 -1 关闭")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="只转换保存，不开可视化窗口")
    parser.add_argument("--z-base", dest="z_base", type=float, default=0.0,
                        help="碰撞柱体底面 z（高度场语义为 z_top 以下全禁止，"
                             "默认从地面 0 起；只看近地部分可调高）")
    parser.add_argument("--alpha", type=float, default=0.20,
                        help="碰撞柱体透明度（0~1，默认 0.20 半透明可透视内部）")
    parser.add_argument("--no-boxes", dest="show_boxes", action="store_false",
                        help="不画 OBB/AABB 参考盒，只看高度场柱体")
    parser.set_defaults(show=True, show_boxes=True)
    args = parser.parse_args(argv)
    mbz = None if args.max_bottom_z is not None and args.max_bottom_z < 0 \
        else args.max_bottom_z

    # 与 yanpu_ur8/main.py / path_plan.py 完全一致的场景搭建

    base = wd.World(cam_pos=[1.5, 1.5, 1.5], lookat_pos=[0, 0.5, 0.8])
    gm.gen_frame().attach_to(base)
    rbt_s = cbt.UR7E(pos=np.array([0.7, 0.2, 0.7]),
                     rotmat=rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi),
                     enable_cc=True)
    rbt_s.gen_meshmodel().attach_to(base)
    obstacle_list = rbt_s.get_obstacle_list_for_gomp(base, True)

    loaded_map = from_obstacle_list(obstacle_list, base,
                                    resolution=args.resolution,
                                    margin=args.margin,
                                    max_bottom_z=mbz)
    _print_box_table(loaded_map["boxes"], loaded_map["dropped"])
    _print_fill_table(loaded_map["boxes"], loaded_map["depth_map"])
    _print_height_levels(loaded_map["depth_map"], z_base=args.z_base)
    print(f"\n网格: {loaded_map['depth_map'].grid.shape} @ "
          f"{loaded_map['depth_map'].resolution}m, "
          f"origin={loaded_map['depth_map'].origin}")

    save_depthmap(loaded_map["depth_map"], loaded_map["boxes"], args.out,
                  dropped=loaded_map["dropped"],
                  meta={"resolution": args.resolution, "margin": args.margin,
                        "max_bottom_z": mbz,
                        "robot_pos": [0.7, 0.2, 0.7],
                        "note": "由 wrs_depthmap.py __main__ 生成；"
                                "场景障碍变更需重新生成"})
    print(f"已保存: {args.out}")

    if args.show:
        visualize_depthmap(base, loaded_map,
        rgba_obb = (1, 0, 0, 0.),      # 原始 OBB（透明，可改 alpha 看线框）
        rgba_aabb = (0, 0, 1, 0.20),   # 光栅化输入：世界 AABB
        z_base = args.z_base,          # 碰撞柱体底面
        alpha = args.alpha,            # 碰撞柱体透明度
        show_boxes = args.show_boxes,
        )
        print("红=原始OBB 蓝=AABB输入 彩柱=高度场碰撞体"
              "(每列 z_base~z_top 实心禁止区，按高度分层着色)")
        base.run()


if __name__ == "__main__":

    main()
