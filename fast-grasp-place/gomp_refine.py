"""
GOMP 轨迹细化：把 path_plan.py 规划的抓取/放置路径做时间最优再优化。

用法（供 path_plan.py / main.py 集成）::

    refiner = GompRefiner(rbt_s)                     # 加载离线深度图 npz
    traj = refiner.refine(app_path)                  # 6DOF 关节路径 -> Trajectory
    pts  = refiner.traj_to_servo_points(traj)        # dt=0.002 等间隔点列, 直接 servoj
    wps  = refiner.traj_to_path(traj)                # 路点列表, 走 move_jntspace_path
    report = refiner.validate(traj, obstacle_list)   # WRS 逐点碰撞兜底校验

约定（方案确认）：
- 碰撞体唯一来源是离线 npz（由 GOMP/gomp/obstacles/wrs_depthmap.py 生成，
  默认 yanpu_ur8/depthmap/scene.npz）；运行时不做 obstacle_list 现场转换，
  npz 缺失直接 FileNotFoundError。
- 端点钉扎：首末路点位姿由 FK 反算成 theta 范围 [0,0] 的 Grasp，并用首末
  关节角预填 IK 缓存，SQP 不会移动抓取/放置点。
- 端点预检：`endpoint_collision()` 用与 validate 同判据查首/末路点（含持物
  模型）；端点自身已碰撞则不进入 SQP，直接回退原路径（端点钉死，优化改不了）。
- SQP 内含深度图碰撞约束（TCP 作用中心高于障碍高度场 + safety_margin）。
- 默认速度/加速度上限对齐实机 fast 参数（main.py: fast_v=3.14, fast_a=4.3）。
"""

import os
import sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
# gomp 包已复制进本项目目录（fast-grasp-place/gomp），只需把本目录塞进 sys.path：
# 不再引用 worktree 外的 GOMP/，也不插 _GOMP_ROOT/wrs（那是 yanpu 的顶层包布局）。
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gomp.planner.gomp_planner import GOMPPlanner
from gomp.planner.trajectory import Trajectory
from gomp.grasp.grasp import Grasp
from gomp.ur7e_robot_adapter import RobotAdapter as _UR7EAdapterBase
from gomp.obstacles.wrs_depthmap import load_depthmap
from gomp.obstacles.collision import ObstacleConstraint
import modeling.geometric_model as gm
from modeling._panda_cdhelper import is_collided as _cd_hit


# ``_cd_hit`` = ``modeling._panda_cdhelper.is_collided`` 直接别名，返回纯 bool。
# ⚠ 不要传 ``toggle_contact_points=True``：那条分支引用了未定义的 ``base``
#   （共享 wrs 的坑），会 NameError。需要接触点请自己从 CollisionHandlerQueue 取。
# 默认深度图 npz（离线转换产物）
DEFAULT_DEPTHMAP_PATH = os.path.join(_HERE, "depthmap", "scene.npz")

# 实机参数（yanpu_ur8/main.py L280-283）
DEFAULT_V_MAX = np.pi   # rad/s
DEFAULT_A_MAX = 7.0   # rad/s^2 ≈ 6.283

# GOMPPlanner.plan 内部把 ObstacleConstraint 的 safety_margin 硬编码为 0.01；
# 用户传别的值时，把差值烘焙进网格（grid += margin - 0.01），效果等价。
_PLAN_INTERNAL_MARGIN = 0.01

# collision.ObstacleConstraint._attached_rows 在走【底面 9 点（面心 + 4 角 +
# 4 边中点）逐点配对】采样分支时附加的「采样点间盲洞」余量（collision.py 内
# 硬编码 0.005）。端点豁免判据 / 预抬升目标必须与它逐项一致：否则用于「判定
# 几何不可满足」的界会比 SQP 实际约束宽得多，把本来可满足的路点误判成无解
# （实测 place 段 inflate=落脚圆半径 版本判 83/251 点违例，inflate=0 版本 0 点）。
_PLAN_CORNER_EXTRA = 0.005

# validate 的限位判定留的相对容差：OSQP eps=1e-3 下边界解会有 1e-4 量级溢出，
# 不吸收会把求解噪声报成「超限」（实测 6.2847 vs 2π=6.2832，0.02%）。
_LIMIT_REL_TOL = 1e-3


class _TunedUR7EAdapter(_UR7EAdapterBase):
    """覆盖 v_max/a_max：基类是硬编码 property，这里改为实例属性返回。"""

    @property
    def v_max(self) -> np.ndarray:
        return self._v_max

    @property
    def a_max(self) -> np.ndarray:
        return self._a_max


class GompRefiner:
    """
    GOMP 轨迹细化器。

    Parameters
    ----------
    rbt_s : UR7E (RobotInterface)
        仿真机器人实例（与 path_plan.py/main.py 同一实例，底座位姿须与
        离线 npz 生成时一致：pos=[0.7,0.2,0.7], rotmat=绕z π）。
    depthmap_path : str or None
        离线 npz 路径；None 用 DEFAULT_DEPTHMAP_PATH。缺文件直接报错。
    t_step : float
        SQP 离散时间步长（秒），默认 0.008（125Hz）。
    v_max, a_max : array-like(6,) or None
        关节速度/加速度上限（rad/s, rad/s^2）；None 用实机 fast 参数。
    min_H, max_H : int
        时间视界下限 / warm-start 估计的上限（步数）。
    safety_margin : float
        TCP 距障碍高度场的安全余量（米）。
    sqp_kwargs : dict or None
        透传 SQPSolver；默认 initial_trust_region=0.05（warm-start 局部打磨）。
    use_real_merit : bool
        L2 开关（默认 True）：线搜索的 merit 取
        `max(高度场 9 点代理违例, 真实几何持物 15 轴 SAT 违例)`，
        并把绝对帽的锚点固定成本段 warm start 的真实违例（全 H 共用，
        防止跨 H 棘轮）。只改验收，不改 QP 约束行 → 不会造出新的
        OSQP INFEASIBLE。关掉即回到纯代理口径（实测 place 会多吃 7mm
        真实穿透）。注意：真实几何路径要求 AttachedObject 带角点/半尺寸
        （5 元组 groups），旧的 from_cdprim_fn 3 元组路径会退化为纯代理。
    verbose : bool
        打印规划过程。
    hand_inflate : float
        【TCP 行的手爪足迹半径】>0 时，TCP 的障碍高度不再只查 TCP 这一个点，
        而是取「以 TCP 为中心 2*hand_inflate 窗口内的最大障碍高度」。
        高度场是 2.5D 点查表，看不见夹爪相对 TCP 轴的横向延伸（Dh50 指尖
        外缘 ≈27mm）：TCP 刚越过箱壁上沿时高度场读到 0（已进箱口），夹爪
        后侧却仍压在壁顶 —— 实测 4 个上边缘抓取点的 app 细化都在 wp46~59
        擦壁 0.4mm 被 validate 打回，加 25mm 足迹后 4 条路径全部 0 碰撞。
        0 = 关闭（旧行为，等价于把夹爪当质点）。
    """

    def __init__(self, rbt_s, depthmap_path=None, *,
                 t_step=0.008,
                 v_max=None, a_max=None,
                 min_H=50, max_H=500,
                 safety_margin=0.01,
                 sqp_kwargs=None,
                 use_real_merit=True,
                 hand_inflate=0.0,
                 verbose=False  # 打印规划过程
                 ):
        if depthmap_path is None:
            depthmap_path = DEFAULT_DEPTHMAP_PATH
        loaded = load_depthmap(depthmap_path)  # 缺文件/版本不符在此抛错
        self.depth_map = loaded.depth_map
        self.boxes = loaded.boxes
        # 真实障碍 AABB (K,6)：只供 SQP merit 的 φ_real 诊断（持物 15 轴 SAT），
        # 不进 QP。npz 里 obb->aabb 已核验一致（差 1e-17），故用 aabb 即可。
        self._real_aabb = (np.array([b["aabb"] for b in self.boxes], dtype=float)
                           if self.boxes else None)
        self._depthmap_path = depthmap_path

        # safety_margin 烘焙（plan() 内部硬编码 _PLAN_INTERNAL_MARGIN）
        extra = float(safety_margin) - _PLAN_INTERNAL_MARGIN
        if abs(extra) > 1e-9:
            self.depth_map.grid = self.depth_map.grid + extra

        # 机器人适配器：复用场景中的 rbt_s，速度/加速度用可配置子类
        self.rbt_s = rbt_s
        self.adapter = _TunedUR7EAdapter.from_wrs_robot(rbt_s)
        self.n_dof = self.adapter.n_dof
        self.adapter._v_max = (np.full(self.n_dof, DEFAULT_V_MAX)
                               if v_max is None else np.asarray(v_max, float))
        self.adapter._a_max = (np.full(self.n_dof, DEFAULT_A_MAX)
                               if a_max is None else np.asarray(a_max, float))

        self.t_step = float(t_step)
        self.min_H = int(min_H)
        self.max_H = int(max_H)
        self.safety_margin = float(safety_margin)
        self.verbose = verbose
        # L2：merit 取 max(高度场代理, 真实几何持物 SAT)。只改线搜索验收，不改
        # QP 约束行（不会引入新的 OSQP INFEASIBLE）。关掉即回到纯代理口径，
        # 供 A/B 对照（关掉时 optimize 会拿真实安全换代理可行，validate 会打回）。
        self.use_real_merit = bool(use_real_merit)
        # TCP 行的手爪足迹半径（0 = 把夹爪当质点，旧行为）。见类文档 hand_inflate。
        self.hand_inflate = max(0.0, float(hand_inflate))

        self._last_exempt_frac = (0.0, 0.0)

        self._last_held = None

        kw = {"initial_trust_region": 0.05}
        if sqp_kwargs:
            kw.update(sqp_kwargs)
        self.planner = GOMPPlanner(self.adapter,    # type:ignore
                                   t_step=self.t_step,
                                   initial_H=self.max_H,
                                   min_H=self.min_H,
                                   H_reduction="geometric",
                                   H_reduction_factor=0.8,
                                   sqp_kwargs=kw,
                                   verbose=verbose)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def refine(self, path, held_object=None, hand_inflate=None):
        """
        细化一条 6DOF 关节路径。

        Parameters
        ----------
        path : list of array-like(6,) | array-like (N, 6)
            初始路径（如 conf_dict['app'] / conf_dict['place']）。
        held_object : AttachedObject or None
            手爪刚性握持的物体（如抓着 U1 时的放置路径）。非 None 时在 SQP
            障碍约束里每路点额外加该物体的净空约束行（首末路点自动豁免，
            因为抓取/释放瞬间物体贴面必然违反底面净空）；端点物体仍在箱/墙
            内起步时，持物行按检出的连续违例段豁免（held_exempt_frac），
            TCP 行不豁免 —— 物体可瞬态穿障，手臂仍须避障。
        hand_inflate : float or None
            TCP 行的手爪足迹半径；None = 用构造时 self.hand_inflate。

        Returns
        -------
        Trajectory or None
            时间最优轨迹（路点含端点钉扎）；SQP 失败返回 None。
        """
        if hand_inflate is None:
            hand_inflate = self.hand_inflate
        hand_inflate = max(0.0, float(hand_inflate))
        q_in = np.asarray(path, dtype=float)
        if q_in.ndim != 2 or q_in.shape[1] != self.n_dof or len(q_in) < 2:
            raise ValueError("path 须为 (N>=2, %d) 关节路径, got %s"
                             % (self.n_dof, q_in.shape))

        Q, H = self._build_warm_start(q_in)
        self._last_warm_H = H
        Q = np.clip(Q, self.adapter.q_min, self.adapter.q_max)

        if H < self.max_H:
            _t_old = np.linspace(0.0, 1.0, H + 1)
            _t_new = np.linspace(0.0, 1.0, self.max_H + 1)
            Q = np.column_stack([
                np.interp(_t_new, _t_old, Q[:, j]) for j in range(Q.shape[1])
            ])
            if self.verbose:
                print(f"GompRefiner: 暖启动 H={H}(动力学最小) 拉伸至 max_H={self.max_H}"
                      f" (est {self.max_H * self.t_step:.3f}s)，给避障留时长裕度")
            H = self.max_H

        # [调试] 持物净空违例统计：与 _attached_rows 同口径，区分「要抬多少」与
        # 「inflate=落脚圆半径 是否把要求抬得过高」
        if held_object is not None and self.verbose:
            self.adapter.backup_state()
            try:
                _pos, _rot = [], []
                for _q in Q:
                    _p, _r = self.adapter.forward_kinematics(_q)
                    _pos.append(_p)
                    _rot.append(_r)
            finally:
                self.adapter.restore_state()
            _pos = np.asarray(_pos)
            _rot = np.asarray(_rot)
            for _g in range(len(held_object.offsets)):
                _ob = np.einsum("mij,j->mi", _rot, held_object.offsets[_g]) + _pos
                _ot = np.einsum("mij,j->mi", _rot, held_object.tops[_g]) + _pos
                _low = np.where((_ob[:, 2] <= _ot[:, 2])[:, None], _ob, _ot)
                for _tag, _inf in (("旧·单点+方窗", float(held_object.fp_radii[_g])),
                                   ("单点·inflate=0", 0.0)):
                    _zo = self.depth_map.get_obstacle_height_batch(
                        _low[:, :2], inflate=_inf)
                    _vio = _zo + _PLAN_INTERNAL_MARGIN - _low[:, 2]
                    _i = int(np.argmax(_vio))
                    print(f"[held-diag] 组{_g} {_tag} inflate={_inf:.4f} "
                          f"违例={int(np.sum(_vio > 0))}/{len(Q)} "
                          f"max={_vio[_i]:.4f}m @wp{_i} "
                          f"(障碍高={_zo[_i]:.4f}, 物体低端z={_low[_i, 2]:.4f})")
                # 第三行 = 豁免/预抬升实际使用的判据（与 _attached_rows 同口径）
                _cb, _ct = held_object.corners_bot, held_object.corners_top
                _ps, _zo, _ex, _ = self._held_clearance(
                    _pos, _rot,
                    np.asarray(held_object.offsets[_g], dtype=float),
                    np.asarray(held_object.tops[_g], dtype=float),
                    corners=None if _cb is None else (_cb[_g], _ct[_g]),
                    fp_radius=float(held_object.fp_radii[_g]),
                    samples=None if getattr(held_object, "samples_bot", None) is None
                    else (held_object.samples_bot[_g], held_object.samples_top[_g]))
                _vio = _zo + _PLAN_INTERNAL_MARGIN + _ex - _ps[:, 2]
                _i = int(np.argmax(_vio))
                _tag = ("SQP同口径·9点" if _cb is not None
                        else "SQP同口径·单点")
                print(f"[held-diag] 组{_g} {_tag} extra={_ex:.4f} "
                      f"违例={int(np.sum(_vio > 0))}/{len(Q)} "
                      f"max={_vio[_i]:.4f}m @wp{_i} "
                      f"(障碍高={_zo[_i]:.4f}, 物体低端z={_ps[_i, 2]:.4f})"
                      f"  <- 豁免判据用这条")

        ex = self._endpoint_exempt_fractions(Q, held_object)
        he_frac, el_frac, hef, hel = ex["held"]   # 持物段（真正豁免的段）
        te_frac, tel_frac, tef, tel = ex["tcp"]   # TCP 段（仅诊断）
        # 记录持物豁免比例供 validate 同步（无条件记录：ef=0 时记为 (0,0)）
        self._last_exempt_frac = (float(he_frac), float(el_frac))
        self._last_held = held_object
        if self.verbose and (hef > 0 or hel > 0 or tef > 0 or tel > 0):
            print(f"GompRefiner: 端点豁免 持物段 首{hef}点/末{hel}点, "
                  f"TCP段 首{tef}点/末{tel}点（共{len(Q)}点；仅持物行豁免，"
                  f"TCP 行全段有效；该段持物碰撞由 validate 单独统计，不阻塞）")
        if self.verbose and (tef > 1 or tel > 1):
            print(f"GompRefiner: [注意] TCP 自身在首{tef}点/末{tel}点也违反净空"
                  f"（首/末路点的 TCP 行在 SQP 内本就禁用；若 iter0 报 INFEASIBLE "
                  f"且 top 行是偶数(TCP)行，需评估 TCP 豁免或重规划抓取位姿）")

        # 持物暖启动预抬升：暖启动若违反持物净空，SQP 会因「步长触顶→信任域收缩」
        # 使每轮抬升量跟不上剩余违例，最终塌缩到真不可行（实测 H=1200 iter5
        # INFEASIBLE）。此处先用雅可比最小范数步把违例路点抬到满足（含小缓冲），
        # 让 SQP 从「约束不活跃」起步。豁免段（持物 ef/el）内持物组不抬；TCP 全段抬
        # （TCP 行在 SQP 全段有效，带违例进场会 iter0 直接不可行）。
        # 无条件调用：TCP 自身的净空违例与是否持物无关，held 为 None 时仍需抬 TCP。
        Q = self._presatisfy_held(Q, held_object, hef, hel)

        ws = Trajectory(waypoints=Q, velocities=np.zeros_like(Q),
                        t_step=self.t_step, H=H)

        # L2 的绝对帽锚点：取【本次 warm start】的真实几何违例，整段（所有 H）
        # 共用同一个锚。不固定的话，每个 H 都拿上一个 H 的结果重新起锚，违例会
        # 跨 H 累积（实测 place 4.8→9.7→14.6mm，物理间隙被吃到 0.3mm）。
        merit_anchor = None
        if (self.use_real_merit and self._real_aabb is not None
                and held_object is not None):
            _oc = ObstacleConstraint(
                self.adapter, self.depth_map,
                safety_margin=_PLAN_INTERNAL_MARGIN,
                fast_kin=self.planner.fast_kin, attached=held_object,
                held_exempt_frac=(he_frac, el_frac),
                real_aabb=self._real_aabb, use_real_merit=True,
                hand_inflate=hand_inflate)
            merit_anchor = _oc.max_violation(Q)
            if self.verbose:
                _cap = max(merit_anchor, 0.0) + 0.005
                print(f"GompRefiner: [L2] 锚点 φ_start={merit_anchor * 1000:+.2f}mm "
                      f"→ 全段帽 {_cap * 1000:.1f}mm "
                      f"(物理间隙 ≥ {(0.015 - _cap) * 1000:.1f}mm)")

        # 端点钉扎：位姿由 FK 反算，theta 范围 [0,0]，IK 缓存预填首末关节角
        self.adapter.backup_state()
        try:
            pos0, rot0 = self.adapter.forward_kinematics(Q[0])
            pos1, rot1 = self.adapter.forward_kinematics(Q[-1])
        finally:
            self.adapter.restore_state()
        axis = np.array([0.0, 0.0, 1.0])
        grasp_start = Grasp(pos=pos0, rotmat=rot0, axis=axis,
                            theta_min=0.0, theta_max=0.0)
        grasp_goal = Grasp(pos=pos1, rotmat=rot1, axis=axis,
                           theta_min=0.0, theta_max=0.0)

        if self.verbose:
            print(f"GompRefiner: warm start H={H} "
                  f"(est {H * self.t_step:.3f}s), npz={self._depthmap_path}")

        return self.planner.plan(grasp_start, grasp_goal,
                                 obstacles=self.depth_map,
                                 held_object=held_object,
                                 ik_seed_start=Q[0].copy(),
                                 ik_seed_goal=Q[-1].copy(),
                                 warm_start=ws,
                                 held_exempt_frac=(float(he_frac),
                                                   float(el_frac)),
                                 real_aabb=self._real_aabb,
                                 use_real_merit=self.use_real_merit,
                                 merit_anchor=merit_anchor,
                                 hand_inflate=hand_inflate)

    def refine_to_servo_points(self, path, dt=0.002, held_object=None):
        """细化并输出等间隔 dt（默认 500Hz，对齐 rtde）关节点列，供 servoj。"""
        traj = self.refine(path, held_object=held_object)
        if traj is None:
            return None
        return self.traj_to_servo_points(traj, dt=dt)

    def refine_to_path(self, path, held_object=None):
        """细化并输出一维关节路点列表，供 move_jntspace_path 执行。"""
        traj = self.refine(path, held_object=held_object)
        if traj is None:
            return None
        return self.traj_to_path(traj)

    # ------------------------------------------------------------------
    # 轨迹 -> 执行格式（可对已有 Trajectory 单独调用）
    # ------------------------------------------------------------------

    @staticmethod
    def traj_to_servo_points(traj, dt=0.002):
        """线性插值重采样为 (M, 6) 等间隔点列（首末点精确对齐）。"""
        n = int(np.floor(traj.duration / dt)) + 1
        times = np.linspace(0.0, traj.duration, n)
        pts = np.empty((n, traj.n_dof))
        for i, t in enumerate(times):
            pts[i] = traj.evaluate(float(t))[0]
        pts[-1] = traj.waypoints[-1]
        return pts

    @staticmethod
    def traj_to_path(traj):
        """轨迹路点 -> list[np.ndarray(6,)]（move_jntspace_path 输入约定）。"""
        return [q.copy() for q in traj.waypoints]

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------

    def validate(self, traj, obstacle_list=None, held=None, stride=1,
                 exempt_frac=None):
        """
        WRS 逐点碰撞兜底校验（深度图是 2.5D 保守近似，最终安全性以此为准）。

        会临时改变 rbt_s 关节状态，结束后恢复。

        Parameters
        ----------
        traj : Trajectory
        obstacle_list : list or None
            WRS 障碍物 CollisionModel 列表；None 时只校验机器人自身。
        held : tuple or None
            (obj_cm, t_tb, R_tb)：手爪刚性握持物体的碰撞模型 + 物体在 TCP 系下
            的相对位姿。非 None 时逐点用 T_world_obj = T_world_tcp @ T_tb 摆好
            该模型，判【物体 vs obstacle_list】是否碰撞，索引记入 collisions_held。
        stride : int
            抽样步长（1 = 每个路点都查）。
        exempt_frac : (first, last) or None
            端点豁免比例（占路点数），默认取最近一次 refine 的 _last_exempt_frac。
            豁免段跳过【持物】碰撞检查（机器人自身仍全量查），该段持物碰撞记入
            collisions_held_exempt 且不影响 is_valid —— 与 refine 内部
            held_exempt_frac 只禁持物约束行的区段保持一致，否则 QP 可解也会被
            全量打回。

        Returns
        -------
        report : dict
            is_valid / collisions(机器人路点) / collisions_held(物体路点) /
            collisions_held_exempt(豁免段物体路点) / n_checked / n_waypoints / limits
        """
        wps = traj.waypoints
        n = wps.shape[0]
        idxs = range(0, n, max(1, int(stride)))
        obstacles = obstacle_list if obstacle_list is not None else []
        if exempt_frac is None:
            exempt_frac = getattr(self, "_last_exempt_frac", (0.0, 0.0))
        _ef_frac, _el_frac = float(exempt_frac[0]), float(exempt_frac[1])
        _ef, _el = int(round(_ef_frac * n)), int(round(_el_frac * n))
        _exempt = (set(range(0, _ef)) | set(range(n - _el, n))
                   if (_ef > 0 or _el > 0) else set())
        collisions = []
        collisions_held = []
        collisions_held_exempt = []
        coll_detail = []   # 机器人碰撞明细 (路点, tcp_z, 当地障碍高, 是否豁免段)
        held_detail = []   # 持物碰撞明细 (路点, tcp, 净空(高度场5点), 接触点数)
        self.adapter.backup_state()
        try:
            for i in idxs:
                self.adapter.goto_given_conf(wps[i])
                if self.adapter.is_collided(obstacle_list=obstacle_list):
                    collisions.append(int(i))
                    if self.verbose:
                        _p, _ = self.adapter.forward_kinematics(wps[i])
                        _zo = float(self.depth_map.get_obstacle_height(
                            _p[0], _p[1]))
                        coll_detail.append((int(i), float(_p[2]), _zo,
                                            i in _exempt))
                if held is not None:
                    obj_cm, t_tb, R_tb = held
                    pos, rotmat = self.adapter.forward_kinematics(wps[i])
                    obj_cm.set_pos(pos + rotmat @ t_tb)
                    obj_cm.set_rotmat(rotmat @ R_tb)
                    _res = _cd_hit([obj_cm], obstacles)
                    if _res:
                        _in_ex = i in _exempt
                        if _in_ex:
                            collisions_held_exempt.append(int(i))
                        else:
                            collisions_held.append(int(i))
                        # 明细：同一路点的【5 点高度场违例量】（与 SQP 完全同口径）。
                        #   违例 > 0 → 2.5D 代理也看得见，是真穿透
                        #   违例 <=0 → 高度场判定安全，碰撞来自 5 点采样看不见的
                        #              侧面/侧棱，或深度图未烘焙的障碍
                        _vio = float("nan")
                        if self._last_held is not None:
                            _h = self._last_held
                            _ps, _zo, _ex, _ = self._held_clearance(
                                pos[None], rotmat[None],
                                np.asarray(_h.offsets[0], dtype=float),
                                np.asarray(_h.tops[0], dtype=float),
                                corners=(None if _h.corners_bot is None
                                         else (_h.corners_bot[0],
                                               _h.corners_top[0])),
                                fp_radius=float(_h.fp_radii[0]))
                            _v = _zo + _PLAN_INTERNAL_MARGIN + _ex - _ps[:, 2]
                            _k = int(np.argmax(_v))
                            _vio = float(_v[_k])
                            _lowz = float(_ps[_k, 2])
                            _obsz = float(_zo[_k] + _PLAN_INTERNAL_MARGIN + _ex)
                        else:
                            _lowz = _obsz = float("nan")
                        # 逐个障碍复测，定位到底碰到 obstacle_list 里的哪一个
                        _which = [k for k, _ob in enumerate(obstacles)
                                  if _cd_hit([obj_cm], [_ob])]
                        # 本机 is_collided 只返回 bool，不提供接触点
                        _cps = np.zeros((0, 3))
                        held_detail.append(
                            (int(i), pos.copy(),
                             np.array(obj_cm.get_pos(), dtype=float).copy(),
                             _vio, _which, _cps, _in_ex, _lowz, _obsz))
        finally:
            self.adapter.restore_state()
        # 加速度按 GOMP 自己的口径 2π 判（见 DEFAULT_A_MAX），并留 1‰ 相对容差：
        # OSQP 容差是 eps=1e-3，边界解会有 1e-4 量级溢出（实测 6.2847 vs 6.2832，
        # 0.02%），不吸收的话每段都会报 acceleration=False 的假告警。
        limits = traj.is_within_limits(
            self.adapter.q_min, self.adapter.q_max,
            self.adapter._v_max, self.adapter._a_max * (1.0 + _LIMIT_REL_TOL))
        # 加速度超限明细：OSQP 容差放宽后（eps=1e-3）会出现 1e-6 容差下的轻微越界，
        # 需要看幅度才能判断是「真超限」还是「求解精度噪声」。
        acc_over = None
        if not limits['acceleration']:
            _acc = np.abs(traj.accelerations)
            _a = np.asarray(self.adapter._a_max, dtype=float)
            _ratio = _acc / np.maximum(_a[None, :], 1e-9)
            _i, _j = np.unravel_index(int(np.argmax(_ratio)), _ratio.shape)
            acc_over = (int(_i), int(_j), float(_ratio[_i, _j]),
                        float(_acc[_i, _j]), float(_a[_j]))
        report = {"is_valid": len(collisions) == 0 and len(collisions_held) == 0,
                  "collisions": collisions,
                  "collisions_held": collisions_held,
                  "collisions_held_exempt": collisions_held_exempt,
                  "held_detail": held_detail,
                  "accel_over": acc_over,
                  "n_checked": len(list(idxs)),
                  "n_waypoints": int(wps.shape[0]),
                  "limits": limits}
        if self.verbose:
            _ex = (f", 豁免段持物碰撞 {len(collisions_held_exempt)} 处"
                   f"(首{_ef}/末{_el}点 抓取/放置瞬态穿障，不阻塞)"
                   if _exempt else "")
            print(f"GompRefiner.validate: {report['n_checked']} 点, "
                  f"碰撞 {len(collisions)} 处, 持物碰撞 {len(collisions_held)} 处{_ex}, "
                  f"limits={limits}")
            if coll_detail:
                # 明细：净空<0 = TCP 低于当地障碍高度场（2.5D 代理），链接也可能
                # 在 TCP 合格时侧碰箱壁；[豁免段] 标记该点落在持物豁免区内。
                _parts = [f"wp{i}:tcp_z={pz:.3f}/obs={zo:.3f}"
                          f"(净空{pz - zo:+.3f}){'[豁免段]' if exm else ''}"
                          for i, pz, zo, exm in coll_detail[:15]]
                print(f"  机器人碰撞明细({len(coll_detail)}处): "
                      f"{' | '.join(_parts)}"
                      + (" ..." if len(coll_detail) > 15 else ""))
            if held_detail:
                if held is not None and self._last_held is not None:
                    print(f"  [held-model] t_tb={np.round(np.asarray(held[1], float), 4)} "
                          f"offsets[0]={np.round(self._last_held.offsets[0], 4)} "
                          f"(offsets 应为 R_tb@[0,0,b]+t_tb)")
                _parts = []
                for dd in held_detail[:8]:
                    i, t, o, v, wh, cps, e, lz, oz = dd
                    _s = (f"wp{i}:违例{v:+.4f}(负=安全)"
                          f"{'[豁免]' if e else ''} 障碍{wh} "
                          f"物体低端={lz:.3f}/障碍顶={oz:.3f} "
                          f"obj=({o[0]:.3f},{o[1]:.3f},{o[2]:.3f})")
                    if len(cps):
                        _c = cps[0]
                        _s += f" 接触=({_c[0]:.3f},{_c[1]:.3f},{_c[2]:.3f})"
                    _parts.append(_s)
                print(f"  持物碰撞明细({len(held_detail)}处): "
                      f"{' | '.join(_parts)}"
                      + (" ..." if len(held_detail) > 10 else ""))
            if acc_over is not None:
                print(f"  加速度超限明细: {acc_over[2]:.4f}× a_max "
                      f"@wp{acc_over[0]}/joint{acc_over[1]} "
                      f"({acc_over[3]:.4f} vs {acc_over[4]:.4f} rad/s², "
                      f"超出 {(acc_over[2] - 1) * 100:.2f}%)")
        return report

    def endpoint_collision(self, path, obstacle_list=None, held=None):
        """【优化前】端点预检：首/末路点（含持物模型）是否与障碍真碰撞。

        端点被 SQP 钉扎（Grasp theta=[0,0] + IK 缓存），GOMP 改不了端点。端点
        本身若已碰撞，细化结果必然在 validate 的首/末点被打回 —— 白烧十几秒
        SQP。本函数用与 validate 完全相同的判据（同一 obstacle_list、同一持物
        碰撞模型与 T_tb、同样 stride=1 查首末点），因此 hit=True 等价于
        「这条路径怎么优化都过不了 validate」。

        典型触发场景：物体贴在箱壁边缘，抓取低点（手爪探入箱壁）或放置点
        （持物压在箱沿）已经相交 —— 该抓取/放置位姿本身不可用，应重选位姿，
        优化毫无意义。

        会临时改变 rbt_s 关节状态，结束后恢复。

        Parameters
        ----------
        path : array-like (N,6) | list[array-like(6,)]
        obstacle_list : list or None
            WRS 障碍物 CollisionModel 列表；None 时只查机器人自身。
        held : (obj_cm, t_tb, R_tb) or None
            同 validate：非 None 时把物体碰撞模型按端点 TCP 位姿摆好，判
            【物体 vs obstacle_list】。

        Returns
        -------
        dict
            hit    : bool  任一检查命中（起点或终点、机器人或持物）
            robot  : list  机器人碰撞的端点下标（0=起点, -1=终点）
            held   : list  持物碰撞的端点下标
            detail : list[(端点名, 'robot'|'held', tcp_pos(3,), 命中的障碍下标)]
        """
        wps = np.asarray(path, dtype=float)
        if wps.ndim != 2 or wps.shape[1] != self.n_dof or len(wps) < 1:
            raise ValueError("path 须为 (N>=1, %d) 关节路径, got %s"
                             % (self.n_dof, wps.shape))
        obstacles = obstacle_list if obstacle_list is not None else []
        ends = [(0, "起点")]
        if len(wps) > 1:
            ends.append((-1, "终点"))
        robot_hits, held_hits, detail = [], [], []
        self.adapter.backup_state()
        try:
            for i, tag in ends:
                self.adapter.goto_given_conf(wps[i])
                pos, rotmat = self.adapter.forward_kinematics(wps[i])
                if self.adapter.is_collided(obstacle_list=obstacle_list):
                    robot_hits.append(int(i))
                    detail.append((tag, "robot", np.asarray(pos, float).copy(),
                                   None))
                if held is not None:
                    obj_cm, t_tb, R_tb = held
                    obj_cm.set_pos(pos + rotmat @ np.asarray(t_tb, float))
                    obj_cm.set_rotmat(rotmat @ np.asarray(R_tb, float))
                    if _cd_hit([obj_cm], obstacles):
                        held_hits.append(int(i))
                        which = [k for k, _ob in enumerate(obstacles)
                                 if _cd_hit([obj_cm], [_ob])]
                        detail.append((tag, "held", np.asarray(pos, float).copy(),
                                       which))
        finally:
            self.adapter.restore_state()
        rep = {"hit": bool(robot_hits or held_hits),
               "robot": robot_hits, "held": held_hits, "detail": detail}
        if self.verbose and rep["hit"]:
            _parts = []
            for _tag, _sub, _p, _wh in detail:
                _s = ("%s%s碰撞 tcp=(%.3f,%.3f,%.3f)"
                      % (_tag, "机器人" if _sub == "robot" else "持物",
                         _p[0], _p[1], _p[2]))
                if _wh is not None:
                    _s += " 障碍%s" % _wh
                _parts.append(_s)
            print("GompRefiner.endpoint_collision: " + " | ".join(_parts))
        return rep

    # ------------------------------------------------------------------
    # 内部：端点违例检测 / warm start 构造
    # ------------------------------------------------------------------

    def _held_clearance(self, positions, rotmats, off, top,
                        corners=None, fp_radius=0.0, samples=None):
        """单个持物组的净空控制点与【SQP 同口径】高度场读数。

        逐项对齐 collision.ObstacleConstraint._attached_rows：
        - 控制点取底面心/顶面心中【世界 z 较低】的那个（倒置物体局部底朝上）；
        - 组带 (hx, hy) 时（samples 非空）：在较低那一面取【面心 + 4 角 + 4 边
          中点】9 点，inflate=0，且【逐点配对】—— 每个采样点用自己的 z 比
          自己脚下的 z_obs，取最坏点。倾斜物体最先碰障的是底边中段，5 点
          （面心+4 角）会整体漏检（实测 place 段擦箱沿 1.3cm 未报）。
        - 只有 corners（4 角）时退回旧 5 点口径（兼容路径）；
        - 无角点信息时退回单点查表 + inflate=落脚圆半径（守恒旧行为）。

        Returns
        -------
        p_s : (m,3) 世界系控制点（带采样时为【最坏采样点】坐标）
        z_obs : (m,) 各控制点脚下的最高障碍
        extra : float 附加余量（配合 _PLAN_INTERNAL_MARGIN 使用）
        use_bot : (m,) True = 用了底面心，False = 用了顶面心（供取 TCP 系偏移）
        """
        p_off_bot = np.einsum("mij,j->mi", rotmats, off)
        p_off_top = np.einsum("mij,j->mi", rotmats, top)
        use_bot = (positions + p_off_bot)[:, 2] <= (positions + p_off_top)[:, 2]
        p_s = positions + np.where(use_bot[:, None], p_off_bot, p_off_top)
        z_obs = self.depth_map.get_obstacle_height_batch(p_s[:, :2], inflate=0.0)
        extra = float(fp_radius)
        if samples is not None:
            s_local = np.where(use_bot[:, None, None], samples[0], samples[1])
            s_world = (positions[:, None, :]
                       + np.einsum("mij,mkj->mki", rotmats, s_local))
            m_s = s_world.shape[1]
            z_k = self.depth_map.get_obstacle_height_batch(
                s_world[..., :2].reshape(-1, 2), inflate=0.0
            ).reshape(s_world.shape[0], m_s)
            # margin/extra 对 argmax 是常数，可先只比 z_obs - z
            k_best = np.argmax(z_k - s_world[..., 2], axis=1)
            rows = np.arange(s_world.shape[0])
            p_s = s_world[rows, k_best]
            z_obs = z_k[rows, k_best]
            extra = _PLAN_CORNER_EXTRA
        elif corners is not None:
            c_bot, c_top = corners
            c_local = np.where(use_bot[:, None, None], c_bot, c_top)
            c_world = (positions[:, None, :]
                       + np.einsum("mij,mkj->mki", rotmats, c_local))
            for k in range(4):
                z_obs = np.maximum(
                    z_obs,
                    self.depth_map.get_obstacle_height_batch(
                        c_world[:, k, :2], inflate=0.0))
            extra = _PLAN_CORNER_EXTRA
        return p_s, z_obs, extra, use_bot

    def _held_sqp_rows(self, positions, rotmats, held_object):
        """持物各组按 SQP 同口径复算 [(p_s, z_obs, extra), ...]。"""
        cb, ct = held_object.corners_bot, held_object.corners_top
        sb = getattr(held_object, "samples_bot", None)
        st = getattr(held_object, "samples_top", None)
        rows = []
        for g in range(len(held_object.offsets)):
            rows.append(self._held_clearance(
                positions, rotmats,
                np.asarray(held_object.offsets[g], dtype=float),
                np.asarray(held_object.tops[g], dtype=float),
                corners=None if cb is None else (cb[g], ct[g]),
                fp_radius=float(held_object.fp_radii[g]),
                samples=None if sb is None else (sb[g], st[g]))[:3])
        return rows

    def _endpoint_exempt_fractions(self, Q, held_object=None):
        """统计 warm start 首/尾连续障碍违例段，按【TCP】与【持物】分组返回。

        违例判据与 SQP 障碍约束完全一致：z_obs(已烘焙 margin) + 内部 0.01
        余量 - p_z > 0。持物时净空取【TCP 与物体各组较低端】的较小值 —— 抓取
        /放置端点被钉死（且 v=0 使近端点几乎不动）时，若物体仍在箱/墙内，
        该约束几何不可满足。

        注意「完全一致」的要求：豁免的动作是【整行禁用安全约束】，属于主动
        放弃安全性，因此判据必须是【必要性】方向的 —— 只有确证无解才豁免。
        早期这里用的是深度图的 inflate=落脚圆半径 查询（实际展开成
        max-filter 的 ±7.5cm 轴对齐方窗，覆盖面积 = 物体足迹的 2.66 倍），
        那是「通过⇒一定不碰」的充分条件，证不出安全 ≠ 真碰：place 目标离箱壁
        5.3cm 就被判违例，首 83/251 点的持物行被整行拔掉，SQP 于是横穿箱体。
        现改为逐项复算 _attached_rows 的判据（5 点 / inflate=0 / extra=5mm）。

        返回 {"tcp": (ef_frac, el_frac, ef, el), "held": (同结构)}：
        - held 段：握持物体在箱/墙内起步（放置路径常见）→ 该段只有持物行需
          豁免（物体瞬态穿障），经 planner.held_exempt_frac 只禁持物行，
          TCP 行全段有效（手臂仍须避开障碍）。
        - tcp 段：TCP 自身 born-in-collision（抓取位姿擦障）。首/末路点的
          TCP 行在 ObstacleConstraint 内本就禁用（rhs=-inf），故段长 >1 时
          需警惕 SQP iter0 不可行；_presatisfy_held 会对 TCP 全段预抬升吸收。

        中段违例不豁免（由优化器抬升穿越），最终安全性由 validate() 的 WRS
        真碰撞兜底。
        """
        positions, rotmats, _ = self.planner.fast_kin.batch_fk_and_jacobian(Q)
        z_obs = self.depth_map.get_obstacle_height_batch(positions[:, :2])
        vio_tcp = ((positions[:, 2] - z_obs)
                   < (_PLAN_INTERNAL_MARGIN - 1e-9))
        vio_held = np.zeros(len(Q), dtype=bool)
        if held_object is not None:
            for p_s, z_obj, extra in self._held_sqp_rows(positions, rotmats,
                                                         held_object):
                vio_held |= ((p_s[:, 2] - z_obj)
                             < (_PLAN_INTERNAL_MARGIN + extra - 1e-9))

        def _runs(vio):
            m = len(vio)
            ef = 0
            while ef < m and vio[ef]:
                ef += 1
            el = 0
            while el < m - ef and vio[m - 1 - el]:
                el += 1
            return ef / m, el / m, ef, el

        return {"tcp": _runs(vio_tcp), "held": _runs(vio_held)}

    @staticmethod
    def _skew(v):
        """3 向量 -> 反对称矩阵（叉乘矩阵），用于刚体点雅可比。"""
        return np.array([[0.0, -v[2], v[1]],
                         [v[2], 0.0, -v[0]],
                         [-v[1], v[0], 0.0]])

    def _presatisfy_held(self, Q, held_object, ef=0, el=0,
                         buffer=0.005, max_iter=6, smooth=8):
        """暖启动预抬升：把违反净空约束的路点用最小范数关节步抬到满足。

        控制点同时包含【TCP 自身】与【持物各组的低端】。只抬持物是不够的：
        持物违例清零后 TCP 的净空违例会浮出水面（实测 H=1200 iter3 INFEASIBLE，
        top 行 29412/29414… 是 obstacle 段的偶数行 = 路点 294 的 TCP 行，
        违例 0.0235m；而此前持物未清零时 top 行是奇数行 = 持物组，把 TCP 挤出了前五）。

        暖启动若带违例进场，SQP 会因「步长触顶 → 信任域收缩」使每轮抬升量
        跟不上剩余违例，最终塌缩到真不可行。这里在 SQP 之前先用雅可比最小
        范数步 dq = J_pz·dz/‖J_pz‖² 一次性抬到位（含 buffer），让 SQP 从
        「约束不活跃」起步 —— 预抬升不受信任域限制，这正是它有效的关键。

        两个实现要点：
        - 用刚体点雅可比 J_p = J_lin − skew(R·r)·J_ang，而非直接取 TCP 的
          J_z，否则忽略旋转带来的 z 分量、抬升量会偏（TCP 的 r=0，退化为 J_lin）。
        - 需求先滑动平均平滑再施加：只在个别路点抬会形成尖峰，量级远大于
          相邻步位移（需抬 ~0.04rad vs 每步仅 0.0025rad），会直接破坏静态
          v/a 约束，反而让 QP 以另一种方式不可行。

        端点强制保持不动（由 grasp 钉扎）。豁免段（首 ef / 末 el）只跳过
        【持物组】的抬升 —— 物体在箱内起步、端点钉死，抬不动也不该抬；TCP
        全段照抬：TCP 行在 SQP 全段有效（held_exempt_frac 只禁持物行），
        带违例进场会 iter0 直接不可行。
        违例为 0 时本方法是 no-op。
        """
        m = len(Q)
        Q = np.array(Q, dtype=float, copy=True)
        q0, q1 = Q[0].copy(), Q[-1].copy()
        for _ in range(int(max_iter)):
            positions, rotmats, J = self.planner.fast_kin.batch_fk_and_jacobian(Q)
            # 控制点：(名称, 世界系点(m,3), TCP 系局部偏移(m,3), z_obs(m,),
            #         总余量 float, 豁免段是否跳过)
            # 净空一律按 SQP 同口径取（_held_clearance / _attached_rows）：
            # 用 inflate=落脚圆半径 的方窗会让预抬升把整段白抬 ~12cm。
            _zt = self.depth_map.get_obstacle_height_batch(positions[:, :2],
                                                           inflate=0.0)
            ctrl = [("TCP", positions, np.zeros((m, 3)), _zt,
                     _PLAN_INTERNAL_MARGIN, False)]
            if held_object is not None:
                _cb, _ct = held_object.corners_bot, held_object.corners_top
                for g in range(len(held_object.offsets)):
                    off = np.asarray(held_object.offsets[g], dtype=float)
                    top = np.asarray(held_object.tops[g], dtype=float)
                    p_s, z_obs_g, extra_g, use_bot = self._held_clearance(
                        positions, rotmats, off, top,
                        corners=None if _cb is None else (_cb[g], _ct[g]),
                        fp_radius=float(held_object.fp_radii[g]))
                    r_loc = np.where(use_bot[:, None], off[None, :], top[None, :])
                    ctrl.append((f"组{g}", p_s, r_loc, z_obs_g,
                                 _PLAN_INTERNAL_MARGIN + extra_g, True))
            any_vio = False
            for name, pts, r_loc, z_obs, margin, exemptible in ctrl:
                v = np.maximum(z_obs + margin + buffer - pts[:, 2], 0.0)
                if exemptible:
                    if ef > 0:
                        v[:ef] = 0.0
                    if el > 0:
                        v[m - el:] = 0.0
                n_bad = int(np.sum(v > 0))
                if n_bad == 0:
                    continue
                any_vio = True
                d = v.copy()
                for _ in range(int(smooth)):
                    d[1:-1] = (d[:-2] + 2.0 * d[1:-1] + d[2:]) / 4.0
                d[0] = d[-1] = 0.0
                r_world = np.einsum("mij,mj->mi", rotmats, r_loc)
                for i in np.where(d > 1e-6)[0]:
                    J_p = J[i, :3, :] - self._skew(r_world[i]) @ J[i, 3:, :]
                    jz = J_p[2, :]
                    s = float(jz @ jz)
                    if s < 1e-12:
                        continue
                    Q[i] = Q[i] + jz * (d[i] / s)
                if self.verbose:
                    print(f"[pre-satisfy] {name} 违例 {n_bad}/{m} 点, "
                          f"max={float(v.max()):.4f}m, "
                          f"平滑后抬升 {int(np.sum(d > 1e-6))} 点 "
                          f"(峰值 {float(d.max()):.4f}m)")
            Q[0] = q0
            Q[-1] = q1
            Q = np.clip(Q, self.adapter.q_min, self.adapter.q_max)
            if not any_vio:
                break
        return Q

    def _build_warm_start(self, q_in):
        """
        构造满足 SQP 静态约束的 warm start。
        SQP 静态约束要求 v_0=v_H=0 且 |Δv| ≤ a_max*t_step（每步仅
        a_max*t_step 的速度变化预算）。真实 RRT 路径是折线，折角处会产生
        离散加速度脉冲，直接重采样必超 a_max 被判 primal infeasible。
        流程：弧长均匀重采样 -> 滑动平均切角平滑（端点保持）-> 梯形速度
        剖面 -> 按离散速度/加速度超限比例迭代降速重定时，直到静态约束满足。
        """
        n = self.n_dof
        seg = np.linalg.norm(np.diff(q_in, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        S = float(s[-1])
        if S < 1e-9:
            return np.repeat(q_in[:1], self.min_H + 1, axis=0), self.min_H

        # 1) 弧长均匀重采样
        M = int(np.clip(4 * (len(q_in) - 1), 120, 400))
        ts = np.linspace(0.0, S, M + 1)
        P = np.column_stack([np.interp(ts, s, q_in[:, j]) for j in range(n)])
        # 2) 滑动平均切角（仅内部点；端点由抓取/放置位姿钉死）
        for _ in range(8):
            P[1:-1] = (P[:-2] + 2.0 * P[1:-1] + P[2:]) / 4.0
        P[0] = q_in[0]
        P[-1] = q_in[-1]

        # 3) 平滑后重新参数化：弧长、斜率、梯形剖面
        seg2 = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s2 = np.concatenate([[0.0], np.cumsum(seg2)])
        S2 = float(s2[-1])
        slope = np.maximum(
            np.max(np.abs(np.diff(P, axis=0))
                   / np.maximum(seg2, 1e-12)[:, None], axis=0), 1e-6)
        s_v = float(np.min(self.adapter._v_max / slope))
        s_a = float(np.min(self.adapter._a_max / slope))

        # 4) 折角离散加速度超限 -> 把剖面速度/加速度上限同除以放慢因子 g，
        #    H 由放慢后的剖面时长决定；g 受 max_H 封顶（t3(g)<=H*t_step），
        #    否则终点钉扎会产生跳变，测出的 rv/ra 会失真爆炸。
        def _t3(g_):
            vv, aa = s_v / g_, s_a / g_
            if S2 >= vv * vv / aa:
                return S2 / vv + vv / aa
            return 2.0 * np.sqrt(S2 / aa)

        # [调试] 打印剖面参数，判断"所需时间远超几何所需"的来源
        if self.verbose:
            print(f"[ws] S2={S2:.4f}rad slope_max={float(np.max(slope)):.4f} "
                  f"s_v={s_v:.4f} s_a={s_a:.4f} "
                  f"t3(g=1)={_t3(1.0):.4f}s H(g=1)={int(np.ceil(_t3(1.0) * 1.08 / self.t_step))} "
                  f"max_H={self.max_H}")

        Q = None
        g = 1.0
        rv = ra = 0.0
        for _ in range(8):
            t3 = _t3(g)
            H = int(np.ceil(t3 * 1.08 / self.t_step))
            if H > self.max_H:
                H = self.max_H
                while g > 1e-3 and _t3(g) * 1.08 > H * self.t_step:
                    g *= 0.9  # 封顶：放慢到 max_H 装得下的程度
                t3 = _t3(g)
            t = np.arange(H + 1) * self.t_step
            s_prof = self._trapezoid_profile(t, S2, s_v / g, s_a / g)
            Q = np.column_stack([np.interp(s_prof, s2, P[:, j])
                                 for j in range(n)])
            Q[0] = q_in[0]
            Q[-1] = q_in[-1]
            v = np.diff(Q, axis=0) / self.t_step
            a = np.diff(v, axis=0) / self.t_step
            rv = float(np.max(np.abs(v) / self.adapter._v_max))
            ra = float(np.max(np.abs(a) / self.adapter._a_max))
            # [调试] 定位超限来源：单步最大 Δq 及其位置（单步跳变会让 rv、ra 同倍率放大）
            if self.verbose:
                dQ = np.diff(Q, axis=0)
                _ij = np.unravel_index(int(np.argmax(np.abs(dQ))), dQ.shape)
                print(f"[ws] iter g={g:.4f} H={H} t3={t3:.4f}s "
                      f"max|dQ|={np.abs(dQ).max():.4f}@step{_ij[0]}/joint{_ij[1]} "
                      f"(=v_max*dt*{np.abs(dQ).max() / (float(np.max(self.adapter._v_max)) * self.t_step):.2f}) "
                      f"rv={rv:.2f} ra={ra:.2f}")
            if rv <= 0.95 and ra <= 0.95:
                break
            grow = max(rv / 0.9, ra / 0.9, 1.2)
            g_old = g
            g = g * grow
            if g <= g_old * 1.001:  # 已被 max_H 封顶，无法再放慢
                break
        if self.verbose and (rv > 1.0 or ra > 1.0):
            print(f"GompRefiner: warm start 仍超限 rv={rv:.2f} ra={ra:.2f} "
                  f"(H={H}, max_H={self.max_H})")
        return Q, H

    @staticmethod
    def _trapezoid_profile(t, S, v, a):
        """距离 S、速度上限 v、加速度上限 a 的时间最优剖面 s(t)（起停静止）。"""
        if S >= v * v / a:
            v_peak = v
        else:
            v_peak = np.sqrt(S * a)
        t1 = v_peak / a                      # 加速段
        t_cruise = (S - v_peak * v_peak / a) / v_peak
        t2 = t1 + t_cruise                   # 匀速段结束
        t3 = t2 + t1                         # 减速段结束
        tt = np.clip(t, 0.0, t3)
        s = np.where(tt <= t1, 0.5 * a * tt ** 2,
                     np.where(tt <= t2,
                              0.5 * a * t1 * t1 + v_peak * (tt - t1),
                              S - 0.5 * a * (t3 - tt) ** 2))
        return s


# ----------------------------------------------------------------------
# 可视化：在 base 场景中展示优化后的轨迹
# ----------------------------------------------------------------------

def show_trajectory(base, rbt_s, traj, n_ghosts=12, n_tcp=60,
                    ghost_rgba=(0, 1, 0, 0.3), tcp_rgba=(1, 0, 0, 1)):
    """在 base 里展示优化轨迹：绿色半透明幽灵臂（采样位形）+ 红色 TCP 折线。

    traj.waypoints 是 SQP 解出的 (H+1, 6) 关节路点；n_ghosts/n_tcp 控制采样密度
    （153 步全画网格太重，默认只画 12 个幽灵臂）。返回生成的模型列表。
    注意：会改变 rbt_s 当前关节状态（fk 到最后一次采样位形），与 path_plan.py
    的展示惯例一致。
    """
    wps = np.asarray(traj.waypoints, dtype=float)
    m = len(wps)
    ghost_idx = set(np.unique(np.linspace(0, m - 1, min(n_ghosts, m)).astype(int)))
    tcp_idx = set(np.unique(np.linspace(0, m - 1, min(n_tcp, m)).astype(int)))
    meshes, tcp_pts = [], []
    for i in sorted(ghost_idx | tcp_idx):
        rbt_s.fk(jnt_values=wps[i])
        if i in tcp_idx:
            tcp_pts.append(np.asarray(rbt_s.arm.get_gl_tcp()[0], dtype=float))
        if i in ghost_idx:
            mesh = rbt_s.gen_meshmodel(rgba=list(ghost_rgba))
            mesh.attach_to(base)
            meshes.append(mesh)
    for a, b in zip(tcp_pts[:-1], tcp_pts[1:]):
        seg = gm.gen_stick(spos=a, epos=b, thickness=0.004, rgba=list(tcp_rgba))
        seg.attach_to(base)
        meshes.append(seg)
    return meshes


# ----------------------------------------------------------------------
# 冒烟自测：建场景 -> 合成一条高处路径 -> 细化 -> 校验
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import math
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import basis.robot_math as rm
    import robot_sim.robots.ur7e.ur7ewithoutmachine as cbt

    base = wd.World(cam_pos=[1.5, 1.5, 1.5], lookat_pos=[0, 0.5, 0.8])
    gm.gen_frame().attach_to(base)
    rbt_s = cbt.UR7E(pos=np.array([0.7, 0.2, 0.7]),
                     rotmat=rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi),
                     enable_cc=True)
    rbt_s.gen_meshmodel().attach_to(base)
    obstacle_list = rbt_s.get_obstacle_list_for_gomp(base, True)
    # 优化器
    refiner = GompRefiner(
        rbt_s,
        depthmap_path=os.path.join(_HERE, "depthmap", "scene.npz")    # 障碍物路径
    )

    # 两个高处 TCP 目标（桌面/箱墙之上），IK 得到首末关节角，关节空间直线插值
    seed = np.array([0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2])
    rbt_s.fk(jnt_values=seed)
    rot_ref = rbt_s.arm.get_gl_tcp()[1]  # 用种子位形的 TCP 姿态，保证可解
    tgt0 = np.array([0.7, -0.15, 1.0])
    tgt1 = np.array([0.3, 0.3, 0.85])
    q0 = rbt_s.ik(tgt_pos=tgt0, tgt_rotmat=rot_ref, seed_jnt_values=seed)
    q1 = rbt_s.ik(tgt_pos=tgt1, tgt_rotmat=rot_ref, seed_jnt_values=seed)
    assert q0 is not None and q1 is not None, "IK 失败，请检查目标点可达性"
    init_path = [q0 + (q1 - q0) * f for f in np.linspace(0, 1, 30)]
    # 优化轨迹
    traj = refiner.refine(init_path)
    if traj is None:
        print("细化失败"); sys.exit(1)
    print(f"\n初始 warm start {refiner._last_warm_H * refiner.t_step:.3f}s"
          f" -> 优化后 duration={traj.duration:.3f}s, H={traj.H}")
    pts = refiner.traj_to_servo_points(traj)
    print(f"servoj 点列: {pts.shape} @ 500Hz")
    rep = refiner.validate(traj, obstacle_list)
    print("端到端冒烟通过" if rep["is_valid"] else f"存在碰撞路点: {rep['collisions']}")

    show_trajectory(base, rbt_s, traj)
    print("展示：绿色半透明=优化轨迹采样位形, 红色折线=TCP 路径；关闭窗口后流程结束")
    base.run()
