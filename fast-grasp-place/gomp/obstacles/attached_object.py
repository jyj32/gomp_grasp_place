"""
持有物体（抓在夹爪上的物体）的碰撞体：刚体绑 TCP 的盒组，配 DepthMapObstacle(2.5D)。

几何由调用方传入的【自定义碰撞体模型函数】提供（适配任意物体）：函数返回
Panda3D NodePath（第一个子节点为 CollisionNode）或直接返回 (center, half) 盒列表。
demo 中传入的洗衣液瓶函数内嵌 12 个 CollisionBox（瓶局系原点=瓶底中心，z=瓶轴），
数据原样取自旧版 wrs/ur7e_withouttable_dh76.py::get_bottle_cdprim。

盒 vs 高度图的语义：盒底面 >= 落脚圆内最大障碍高 + safety_margin
（盒体全部在底面之上，底面过了全过；落脚圆半径 = footprint 外接圆半径，
 与瓶子的旋转无关）。

12 盒按「z 相邻 + footprint 半径差 <= footprint_tol」自动合并成 G 组
（本瓶 tol=0.12 -> 4 组），每组在 QP 里只产生 1 行约束：障碍行数 m -> m*(1+G)。
"""

import numpy as np


def read_boxes_from_cdprim(cdprim_nodepath):
    """读 Panda3D CollisionNode 里的全部 CollisionBox。

    兼容两种约定：
    - NodePath 且第一个子节点为 CollisionNode（旧版 gen_bottle_cdprim）
    - 裸 CollisionNode（CollisionModel userdefined_cdprimitive_fn 约定，
      如 gen_U1_cdprim）

    Parameters
    ----------
    cdprim_nodepath : NodePath | CollisionNode

    Returns
    -------
    boxes : list of (center(3,), half(3,))
    """
    from panda3d.core import CollisionNode
    node = cdprim_nodepath.node() if hasattr(cdprim_nodepath, "node") else cdprim_nodepath
    if not isinstance(node, CollisionNode):
        node = cdprim_nodepath.getChild(0).node()
    boxes = []
    for i in range(node.getNumSolids()):
        solid = node.getSolid(i)
        center = np.array(solid.getCenter(), dtype=float)
        half = np.array(solid.getDimensions(), dtype=float) / 2.0
        boxes.append((center, half))
    return boxes


def merge_box_groups(boxes, footprint_tol=0.12):
    """z 相邻且 footprint 半径相近的盒合并成组。

    Parameters
    ----------
    boxes : list of (center(3,), half(3,))
    footprint_tol : float
        相邻盒 footprint 半径允许的相对差（0.12 -> 本瓶 12 盒合并为 4 组）。

    Returns
    -------
    groups : list of (bottom, top, fp_radius)
        bottom/top 为瓶局系 z 坐标；fp_radius 为该组最大 footprint 外接圆半径。
    """
    groups = []
    for center, half in sorted(boxes, key=lambda b: b[0][2]):
        bottom, top = center[2] - half[2], center[2] + half[2]
        fp = float(np.linalg.norm(half[:2]))
        # 邻接容差 1e-6：Panda3D 的 getCenter/getDimensions 是 float32，
        # "z 相邻"的盒之间有 ~1.5e-8 的浮点缝隙，用 1e-9 会误判为不相邻
        # （实测：本瓶 12 盒被切成 6 组而非 4 组）。
        if groups and abs(groups[-1][1] - bottom) < 1e-6 \
                and abs(fp - groups[-1][2]) <= groups[-1][2] * footprint_tol:
            groups[-1] = (groups[-1][0], top, max(groups[-1][2], fp))
        else:
            groups.append((bottom, top, fp))
    return groups


class AttachedObject:
    """刚体绑在 TCP 上的持有物体（合并后的盒组）。

    Parameters
    ----------
    groups : list of (bottom_z, top_z, fp_radius[, hx, hy]) 或 (bottom_z, fp_radius)
        物体局系下每组的底面 z、顶面 z 与落脚圆半径（盒心都在局部
        x=y=0 的轴上，故组底/顶心局部坐标为 (0, 0, bottom) / (0, 0, top)）。
        只给二元组时 tops 缺省等于 bottoms（旧约定）。
        可选给出 footprint 半尺寸 (hx, hy)：提供后 SQP 约束用【底面 4 角 +
        中心】5 点采样高度场取 max（连续双线性查表），比单点+常量余量精确
        且保守；不提供时退回单点+落脚圆半径常量余量。
    tcp_pos, tcp_rotmat : np.ndarray
        物体局系原点在 TCP 系下的位姿，抓取瞬间测一次、之后刚体绑定：
        T_tcp_obj = inv(T_world_tcp) @ T_world_obj。
    name : str

    每组同时给出底面(z=bottom)与顶面(z=top)在 TCP 系下的偏移 offsets/tops：
    物体若相对世界倒置（局部"底"朝上），SQP 约束行取【世界中较低的一端】
    才保守，故 collision._attached_rows 会按路点比较 offsets/tops 的 z 取低者。
    """

    def __init__(self, groups, tcp_pos, tcp_rotmat, name="attached"):
        self.name = name
        # 物体局系在 TCP 系下的位姿（供真实几何 OBB 使用：世界轴 = TCP_rot @ R_tb）
        self.tcp_pos = np.asarray(tcp_pos, dtype=float)
        self.tcp_rotmat = np.asarray(tcp_rotmat, dtype=float)
        self.half_xy = None          # (G,2) 局系 footprint 半尺寸（仅 5 元组）
        bottoms, tops, fpr = [], [], []
        corners_bot, corners_top = [], []
        has_corners = True
        for g in groups:
            if len(g) == 5:
                b, t, fp, hx, hy = g
            elif len(g) == 3:
                b, t, fp = g
                has_corners = False
            else:                 # 旧约定 (bottom, fp)：顶面缺省为底面
                b, fp = g
                t = b
                has_corners = False
            bottoms.append(b)
            tops.append(t)
            fpr.append(fp)
            if len(g) == 5:
                # 底/顶面 4 角的局系坐标（面是水平的，4 角与面心同 z）
                corners_bot.append(np.array([[sx * hx, sy * hy, b]
                                             for sx in (-1.0, 1.0)
                                             for sy in (-1.0, 1.0)]))
                corners_top.append(np.array([[sx * hx, sy * hy, t]
                                             for sx in (-1.0, 1.0)
                                             for sy in (-1.0, 1.0)]))
        self.offsets = np.array([tcp_rotmat @ np.array([0.0, 0.0, b]) + tcp_pos
                                 for b in bottoms])       # (G,3) TCP 系, 组底面心
        self.tops = np.array([tcp_rotmat @ np.array([0.0, 0.0, t]) + tcp_pos
                              for t in tops])             # (G,3) TCP 系, 组顶面心
        self.fp_radii = np.array(fpr)                     # (G,)
        if has_corners and corners_bot:
            # (G,4,3) TCP 系下的 4 角偏移：R @ 局系角点 + t
            self.corners_bot = np.array([
                (tcp_rotmat @ c.T).T + tcp_pos for c in corners_bot])
            self.corners_top = np.array([
                (tcp_rotmat @ c.T).T + tcp_pos for c in corners_top])
            # (G,9,3) 净空采样点，顺序固定：
            #   [面心, c0, c1, c2, c3, m01, m13, m32, m20]
            # 角点生成顺序为 0=(-hx,-hy) 1=(-hx,+hy) 2=(+hx,-hy) 3=(+hx,+hy)，
            # 故 4 条边的中点为 (0,1)(1,3)(3,2)(2,0)。
            # 为何要 4 个边中点：5 点（面心+4 角）在物体【倾斜】时，最先碰障的
            # 是底边中段而非角点（实测 place 段物体擦箱沿 1.3cm，接触点
            # x=0.043，而 4 角全在 x>=0.055，5 点采样整体漏检）。
            _ei = np.array([0, 1, 3, 2])
            _ej = np.array([1, 3, 2, 0])
            self.samples_bot = np.array([
                np.concatenate([self.offsets[g][None, :], c,
                                0.5 * (c[_ei] + c[_ej])], axis=0)
                for g, c in enumerate(self.corners_bot)])       # (G,9,3)
            self.samples_top = np.array([
                np.concatenate([self.tops[g][None, :], c,
                                0.5 * (c[_ei] + c[_ej])], axis=0)
                for g, c in enumerate(self.corners_top)])       # (G,9,3)
            # 局系 footprint 半尺寸（真实几何 OBB 用；不能从 samples 反推，
            # 因为 samples 已乘过 R_tb，旋转后其 |x|max 是外接盒而非 hx）
            self.half_xy = np.array([[np.abs(c[:, 0]).max(), np.abs(c[:, 1]).max()]
                                     for c in corners_bot])     # (G,2)
        else:
            self.corners_bot = None
            self.corners_top = None
            self.samples_bot = None
            self.samples_top = None

    @classmethod
    def from_cdprim_fn(cls, cdprim_fn, tcp_pos, tcp_rotmat, footprint_tol=0.12,
                       name="attached"):
        """从调用方自定义的碰撞体模型函数构造持有物体（适配任意物体）。

        Parameters
        ----------
        cdprim_fn : callable
            无参、返回该物体碰撞盒数据的函数。返回值可以是：
              - Panda3D NodePath：第一个子节点为 CollisionNode（同旧版
                get_bottle_cdprim 的约定），内部用 read_boxes_from_cdprim 读取；
              - list of (center(3,), half(3,))：直接给出盒中心与半尺寸。
        tcp_pos, tcp_rotmat : np.ndarray
            物体局系原点在 TCP 系下的位姿，抓取瞬间测一次、之后刚体绑定：
            T_tcp_bottle = inv(T_world_tcp) @ T_world_bottle。
        footprint_tol : float
            相邻盒 footprint 半径允许的相对差（0.12 -> 本瓶 12 盒合并为 4 组）。
        name : str
        """
        boxes = cdprim_fn()
        if not isinstance(boxes, list):
            boxes = read_boxes_from_cdprim(boxes)
        groups = merge_box_groups(boxes, footprint_tol=footprint_tol)
        return cls([(b, t, fp) for b, t, fp in groups], tcp_pos, tcp_rotmat,
                   name=name)
