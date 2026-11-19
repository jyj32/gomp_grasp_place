"""
Linearized obstacle avoidance constraints for the SQP.

From GOMP Section IV-C: The obstacle avoidance constraint is linearized
using the robot's current configuration and Jacobian:

    z_obstacle - p(q_i^{(k)}) + J_z^{(k)} * q_i^{(k)} ≤ J_z^{(k)} * q_i^{(k+1)}

where J_z is the z-translation row of the Jacobian.

Performance: Uses FastKinematics for batched FK+Jacobian computation,
replacing the serial per-waypoint Python loop that was the #1 bottleneck
(87% of total planning time in profiling).
"""

import numpy as np
from typing import Tuple

from gomp.robot_adapter import RobotAdapter
from gomp.obstacles.depth_map import DepthMapObstacle
from gomp.obstacles.attached_object import AttachedObject


class ObstacleConstraint:
    """
    Computes linearized obstacle avoidance constraints for each waypoint.

    For each waypoint, the constraint ensures the end-effector stays above
    the depth map surface. The constraint is linearized around the current
    configuration and updated at each SQP iteration.

    Parameters
    ----------
    robot : RobotAdapter
        Robot model for FK and Jacobian.
    depth_map : DepthMapObstacle
        The obstacle depth map.
    safety_margin : float
        Additional clearance above the depth map surface (meters).
    fast_kin : FastKinematics or None
        Pre-compiled fast kinematics engine. If None, falls back to
        robot_adapter FK/Jacobian (slower).
    attached : AttachedObject or None
        Object rigidly held by the gripper. Each merged box group adds one
        linearized constraint row per waypoint (box bottom above the max
        obstacle height inside its footprint circle). Rows of the first/last
        waypoint are disabled (-inf): at grasp/release the object rests on a
        surface and would violate the bottom-clearance constraint by design.
    held_exempt_frac : (first, last)
        持物行单独豁免比例（占路点数）：首/末连续段的【持物物体行】禁用
        （rhs=-inf, J=0），TCP 行不受影响。用于放置路径起点握持物体仍在
        箱/墙内（从箱内起步）的场景 —— 该瞬态只豁免物体，不豁免手臂，
        否则 SQP 在豁免段失去 TCP 净空约束，会把手臂伸进障碍（实测 place
        段 7 处机器人碰撞全部来自该段）。豁免路点同时不参与 dup 合并
        （控制点在 TCP 正下方时把持物要求并入 TCP 行），避免 TCP 行被
        持物的必然违例污染。
    """

    def __init__(self, robot: RobotAdapter, depth_map: DepthMapObstacle,
                 safety_margin: float = 0.01, fast_kin=None,
                 attached: AttachedObject = None,
                 held_exempt_frac=(0.0, 0.0),
                 real_aabb=None, use_real_merit: bool = False,
                 hand_inflate: float = 0.0):
        self.robot = robot
        self.depth_map = depth_map
        self.safety_margin = safety_margin
        self.fast_kin = fast_kin
        self.attached = attached
        # 【TCP 行的手爪足迹半径】高度场只按 TCP 一个点查表时，夹爪本体相对
        # TCP 轴有 ±27mm 的横向延伸（Dh50：指尖外缘 0.013+0.014）。TCP 刚越过
        # 箱壁上沿时，高度场在 TCP 处读到 0（已进箱口），但夹爪后侧仍压在壁上
        # —— 实测 app 细化在 wp48-51 以 0.4mm 侵入量擦壁（validate 打回）。
        # inflate>0 时改为「以 TCP 为中心 2*inflate 窗口内的最大障碍高度」，
        # 把夹爪的横向足迹纳入约束（heightfield 侧有 max-filter 缓存，几乎零成本）。
        self.hand_inflate = max(0.0, float(hand_inflate))
        self.held_exempt_frac = (float(held_exempt_frac[0]),
                                 float(held_exempt_frac[1]))
        # 真实几何障碍盒 (K,6)=x0,y0,z0,x1,y1,z1（世界轴对齐）。
        # 非 None 时可用 max_violation_real：持物 OBB vs 真盒 15 轴 SAT。
        self.real_aabb = (None if real_aabb is None else
                          np.asarray(real_aabb, dtype=float).reshape(-1, 6))
        # L2：merit 里对「高度场代理违例」与「真实几何违例」取 max 后再验收。
        # 只影响线搜索的 φ（QP 的 A/l/u 一个字节不变），因此不会产生新的
        # OSQP INFEASIBLE，只会拒绝「代理看不出来但真实在退化」的步长。
        self.use_real_merit = bool(use_real_merit) and self.real_aabb is not None

    def compute_for_waypoint(self, q: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the linearized obstacle constraint for a single waypoint.

        The constraint is:
            z_obstacle + margin - p_z(q^{(k)}) + J_z * q^{(k)} ≤ J_z * q^{(k+1)}

        Rearranged for the QP: J_z * q ≥ z_obstacle + margin - p_z(q^{(k)}) + J_z * q^{(k)}

        Which becomes:  -J_z * q ≤ -(z_obstacle + margin - p_z(q^{(k)}) + J_z * q^{(k)})

        Parameters
        ----------
        q : np.ndarray, shape (n_dof,)
            Current joint configuration at this waypoint.

        Returns
        -------
        J_z : np.ndarray, shape (n_dof,)
            The z-row of the Jacobian (linear velocity z-component).
        rhs : float
            The right-hand side of the linearized constraint:
            z_obstacle + margin - p_z(q^{(k)}) + J_z * q^{(k)}
        """
        if self.fast_kin is not None:
            pos, _, J = self.fast_kin.fk_and_jacobian(q)
        else:
            # Forward kinematics at current config
            pos, _ = self.robot.forward_kinematics(q)
            # Jacobian
            J = self.robot.jacobian(q)

        p_z = pos[2]  # z-position of end-effector
        p_x, p_y = pos[0], pos[1]

        # Get obstacle height at the (x, y) position
        z_obstacle = self.depth_map.get_obstacle_height(
            p_x, p_y, inflate=self.hand_inflate)

        # Jacobian z-row (translation z)
        J_z = J[2, :]  # z-translation row

        # Right-hand side: z_obstacle + margin - p_z + J_z @ q
        rhs = z_obstacle + self.safety_margin - p_z + J_z @ q

        return J_z, rhs

    def compute_all_waypoints(self, waypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute linearized obstacle constraints for all waypoints.

        Uses batch FK+Jacobian computation for significant speedup.

        Parameters
        ----------
        waypoints : np.ndarray, shape (H+1, n_dof)
            Joint configurations at each waypoint.

        Returns
        -------
        J_z_all : np.ndarray, shape (H+1, n_dof)
            J_z row for each waypoint.
        rhs_all : np.ndarray, shape (H+1,)
            Right-hand side for each waypoint.
        """
        H_plus_1, n = waypoints.shape

        if self.fast_kin is not None:
            # Batched computation using fast kinematics
            positions, rotmats, jacobians = self.fast_kin.batch_fk_and_jacobian(waypoints)

            # Extract z-rows of Jacobians: shape (H+1, n)
            J_z_all = jacobians[:, 2, :]

            # Get obstacle heights for all xy positions (vectorized batch)
            z_obstacles = self.depth_map.get_obstacle_height_batch(
                positions[:, :2], inflate=self.hand_inflate)

            # Vectorized RHS computation
            p_z_all = positions[:, 2]
            # rhs = z_obstacle + margin - p_z + J_z @ q  (per waypoint)
            Jz_dot_q = np.sum(J_z_all * waypoints, axis=1)
            rhs_all = z_obstacles + self.safety_margin - p_z_all + Jz_dot_q

            if self.attached is not None:
                J_z_all, rhs_all = self._attached_rows(
                    waypoints, positions, rotmats, jacobians, J_z_all, rhs_all)

            return J_z_all, rhs_all
        else:
            # Fallback: serial computation
            J_z_all = np.zeros((H_plus_1, n))
            rhs_all = np.zeros(H_plus_1)
            for i in range(H_plus_1):
                J_z_all[i], rhs_all[i] = self.compute_for_waypoint(waypoints[i])
            if self.attached is not None:
                positions = np.zeros((H_plus_1, 3))
                rotmats = np.zeros((H_plus_1, 3, 3))
                jacobians = np.zeros((H_plus_1, 6, n))
                for i in range(H_plus_1):
                    positions[i], rotmats[i] = self.robot.forward_kinematics(waypoints[i])
                    jacobians[i] = self.robot.jacobian(waypoints[i])
                J_z_all, rhs_all = self._attached_rows(
                    waypoints, positions, rotmats, jacobians, J_z_all, rhs_all)
            return J_z_all, rhs_all

    def check_violation(self, q: np.ndarray) -> float:
        """
        Check how much a waypoint violates the obstacle constraint.

        Returns
        -------
        violation : float
            Positive if the constraint is violated (EE below obstacle).
            Negative if satisfied (EE above obstacle).
        """
        if self.fast_kin is not None:
            pos, _ = self.fast_kin.fk(q)
        else:
            pos, _ = self.robot.forward_kinematics(q)
        z_obstacle = self.depth_map.get_obstacle_height(
            pos[0], pos[1], inflate=self.hand_inflate)
        return z_obstacle + self.safety_margin - pos[2]

    def _attached_rows(self, waypoints, tcp_positions, tcp_rotmats, jacobians,
                       J_z_all, rhs_all):
        """持有物体的约束行：每个合并盒组 1 行/路点，交错插在 TCP 行之后。

        组控制点 p_s（刚体挂点，偏移 d 在 TCP 系固定）：
            p_s   = p_tcp + R_tcp @ d
            J_s,z = J_tcp,z + (R_tcp d)_y * J_tcp[3] - (R_tcp d)_x * J_tcp[4]
                  （v_s = v_tcp + w x p_off，(w x p_off)_z = w_x*p_off_y - w_y*p_off_x，
                   免重算 FK；已用中心有限差分校验，误差 ~1e-11）
        约束：p_s,z >= z_obs(落脚圆内最大) + margin
              -> J_s,z * q >= rhs,  rhs = z_obs + margin - p_s,z + J_s,z @ q_k

        控制点按路点取【组底面心 / 组顶面心中世界里较低的一端】：物体相对世界
        倒置（局部"底"朝上，如 U1 倒放）时，局部底未必是世界最低点，取低者才
        保守。行序：(m, 1+G, n) 交错 ravel —— 每路点 [TCP 行, 组0..G-1]，
        与 constraints._obstacle_constraints 的 rows_per_wp 映射一致。
        首/末路点禁用（l = -inf）：抓取态/释放态物体贴桌，必然违反底面约束。
        held_exempt_frac 首/末段内该组行同样禁用（物体在箱内起步的持物路径），
        且豁免路点不参与 dup 合并，避免持物违例经 max() 污染 TCP 行。
        """
        m, n = waypoints.shape
        G = len(self.attached.offsets)
        # 持物行单独豁免段（首/末连续路点）：该段只禁持有物体的行，TCP 行
        # 保持有效 —— 放置起点物体在箱内是抓取瞬态，手臂仍须避开障碍。
        exempt = np.zeros(m, dtype=bool)
        _ef = int(round(self.held_exempt_frac[0] * m))
        _el = int(round(self.held_exempt_frac[1] * m))
        if _ef > 0:
            exempt[:_ef] = True
        if _el > 0:
            exempt[m - _el:] = True
        J_obj = np.zeros((m, G, n))
        rhs_obj = np.zeros((m, G))
        for g in range(G):
            p_off_bot = tcp_rotmats @ self.attached.offsets[g]          # (m,3)
            p_off_top = tcp_rotmats @ self.attached.tops[g]            # (m,3)
            z_bot = (tcp_positions + p_off_bot)[:, 2]
            z_top = (tcp_positions + p_off_top)[:, 2]
            # 取世界中较低的一端作为净空控制点（倒置物体局部底朝上）
            lower_is_bot = z_bot <= z_top
            p_off = np.where(lower_is_bot[:, None], p_off_bot,
                             p_off_top)                                # (m,3)
            p_s = tcp_positions + p_off                                # (m,3) 控制点
            # (w x p_off)_z = w_x*p_off_y - w_y*p_off_x
            J_z = (jacobians[:, 2, :]
                   + p_off[:, 1:2] * jacobians[:, 3, :]
                   - p_off[:, 0:1] * jacobians[:, 4, :])                # (m,n)
            # 平滑采样版：不在高度场上做「膨胀求 max」（膨胀网格在网格边界是
            # 阶跃的，SQP 每步 xy 移动跨格会让 z_obs 跳变 ~墙高，上一迭代解对
            # 新线性化巨额违反 → 误判不可行）。改为在【较低那一面】上取
            # 9 点（面心 + 4 角 + 4 边中点）连续双线性查表。
            #
            # 关键：逐点配对 —— 每个采样点用【自己的 z】比【自己脚下的 z_obs】，
            # 再取最坏点决定本行约束。早期版本是「9/5 点脚下取 max(z_obs)，
            # 但减去的是【面心】z」，物体倾斜时最低点比面心低 1~4cm，等于把
            # 物体想高了；同时只有 4 角时最先碰障的底边中段完全不在采样集里。
            # 两者叠加导致 place 段真擦箱沿 1.3cm 而 SQP 仍以为有 8~26cm 余量。
            if self.attached.samples_bot is not None:
                s_local = np.where(lower_is_bot[:, None, None],
                                   self.attached.samples_bot[g],
                                   self.attached.samples_top[g])    # (m,9,3)
                s_world = (tcp_positions[:, None, :]
                           + np.einsum("mij,mkj->mki", tcp_rotmats, s_local))
                m_s = s_world.shape[1]
                z_k = self.depth_map.get_obstacle_height_batch(
                    s_world[..., :2].reshape(-1, 2), inflate=0.0
                ).reshape(m, m_s)                                    # (m,9)
                extra = 0.005        # 采样点间盲洞的小余量（<< 落脚圆常量余量）
                vio_k = (z_k + self.safety_margin + extra
                         - s_world[..., 2])                          # (m,9)
                k_best = np.argmax(vio_k, axis=1)                    # (m,)
                rows = np.arange(m)
                # 线性化也用最坏采样点的力臂：J_s,z 的修正项由 p_off_xy 决定
                p_off_best = s_world[rows, k_best] - tcp_positions    # (m,3)
                p_off_dup = p_off_best
                J_z = (jacobians[:, 2, :]
                       + p_off_best[:, 1:2] * jacobians[:, 3, :]
                       - p_off_best[:, 0:1] * jacobians[:, 4, :])    # (m,n)
                rhs_g = (vio_k[rows, k_best]
                         + np.sum(J_z * waypoints, axis=1))          # (m,)
            elif self.attached.corners_bot is not None:
                c_local = np.where(lower_is_bot[:, None, None],
                                   self.attached.corners_bot[g],
                                   self.attached.corners_top[g])      # (m,4,3)
                c_world = (tcp_positions[:, None, :]
                           + np.einsum("mij,mkj->mki", tcp_rotmats, c_local))
                z_obs = self.depth_map.get_obstacle_height_batch(
                    p_s[:, :2], inflate=0.0)
                for k in range(4):
                    zk = self.depth_map.get_obstacle_height_batch(
                        c_world[:, k, :2], inflate=0.0)
                    z_obs = np.maximum(z_obs, zk)
                extra = 0.005
                rhs_g = (z_obs + self.safety_margin + extra
                         - p_s[:, 2] + np.sum(J_z * waypoints, axis=1))
                p_off_dup = p_off
            else:
                z_obs = self.depth_map.get_obstacle_height_batch(
                    p_s[:, :2], inflate=0.0)
                extra = self.attached.fp_radii[g]
                rhs_g = (z_obs + self.safety_margin + extra
                         - p_s[:, 2] + np.sum(J_z * waypoints, axis=1))
                p_off_dup = p_off
            # 控制点与 TCP 同列（竖直抓取 p_off_xy≈0）时，本组行与 TCP 行雅可比
            # 几乎相同而 rhs 不同：两行同时活跃会使 QP 对偶解不唯一，OSQP ADMM
            # 收敛极慢（maximum iterations）→ SQP 信任域塌缩 → 假性不可行。
            # 这类路点不保留独立行，把要求并入 TCP 行（rhs 取较大者），约束不放松。
            # 判据必须用【本行线性化实际用的力臂】p_off_dup：9 点采样时它是最坏
            # 采样点（角/边中点），xy 通常不为 0，不该合并 —— 合并会把力臂项丢掉，
            # 使 rhs 与 TCP 行的 J 不同口径（实测 1.6cm 误差）。
            dup = ((np.abs(p_off_dup[:, 0]) < 2e-3)
                   & (np.abs(p_off_dup[:, 1]) < 2e-3)
                   & ~exempt)
            rhs_all[dup] = np.maximum(rhs_all[dup], rhs_g[dup])
            rhs_obj[:, g] = np.where(dup, -np.inf, rhs_g)
            J_obj[:, g, :] = np.where(dup[:, None], 0.0, J_z)
            # 豁免段该组行整行禁用（rhs=-inf, J=0）
            rhs_obj[exempt, g] = -np.inf
            J_obj[exempt, g, :] = 0.0
        J_all = np.concatenate([J_z_all[:, None, :], J_obj], axis=1).reshape(-1, n)
        rhs_all = np.concatenate([rhs_all[:, None], rhs_obj], axis=1).reshape(-1)
        rhs_all[:(1 + G)] = -np.inf
        rhs_all[-(1 + G):] = -np.inf
        return J_all, rhs_all

    # ------------------------------------------------------------------
    # 真实（非线性）净空评估 —— 供 SQP 线搜索做步长验收
    # ------------------------------------------------------------------
    def _fk_pos_rot(self, waypoints: np.ndarray):
        """批量 FK → (positions (m,3), rotmats (m,3,3))。

        batch_fk 是 python for 循环（慢 3~4 倍），这里统一走向量化版并丢掉 J。
        """
        if self.fast_kin is not None:
            positions, rotmats, _ = self.fast_kin.batch_fk_and_jacobian(waypoints)
        else:
            m = waypoints.shape[0]
            positions = np.zeros((m, 3))
            rotmats = np.zeros((m, 3, 3))
            for i in range(m):
                positions[i], rotmats[i] = self.robot.forward_kinematics(waypoints[i])
        return positions, rotmats

    def _violation_rows(self, waypoints: np.ndarray, fk=None) -> np.ndarray:
        """逐行真实净空违例（>0 = 违反，<0 = 还剩多少净空），不做线性化。

        行序与约束行完全一致：每路点 [TCP 行, 组0..G-1]（同 _attached_rows）。
        返回 (m, 1+G)，其中【被禁用的行】置 -inf，取 max 时自然跳过：
          - 首/末路点的所有行（_attached_rows 末尾把 rhs 整体设为 -inf；
            TCP 行同样被禁 —— SQP 侧本就不约束端点净空）
          - 持物豁免段（held_exempt_frac 首/末段）的各组行
          - dup 合并掉的组行（其要求已并入 TCP 行，违例也一并并入）

        与 _attached_rows 的口径逐项对齐（这是关键 —— 判据不同口径会
        「证不出安全就放弃安全」）：9 点采样、inflate=0、extra=5mm、
        逐点配对（每个采样点拿自己的 z_obs 比自己的 z）。

        fk : (positions, rotmats) 或 None
            调用方已算过 FK 时传入，避免同一路点重复求运动学（merit 每轮
            要算 1+5 次，重复 FK 是主要浪费）。
        """
        m = waypoints.shape[0]
        G = 0 if self.attached is None else len(self.attached.offsets)

        positions, rotmats = (self._fk_pos_rot(waypoints) if fk is None else fk)
        out = np.full((m, 1 + G), -np.inf)

        # --- TCP 行（与 compute_all_waypoints 同口径：普通查表，无 5 点采样）---
        z_obs = self.depth_map.get_obstacle_height_batch(
            positions[:, :2], inflate=self.hand_inflate)
        out[:, 0] = z_obs + self.safety_margin - positions[:, 2]

        if self.attached is not None:
            exempt = np.zeros(m, dtype=bool)
            _ef = int(round(self.held_exempt_frac[0] * m))
            _el = int(round(self.held_exempt_frac[1] * m))
            if _ef > 0:
                exempt[:_ef] = True
            if _el > 0:
                exempt[m - _el:] = True

            for g in range(G):
                p_off_bot = rotmats @ self.attached.offsets[g]
                p_off_top = rotmats @ self.attached.tops[g]
                z_bot = (positions + p_off_bot)[:, 2]
                z_top = (positions + p_off_top)[:, 2]
                lower_is_bot = z_bot <= z_top
                p_off = np.where(lower_is_bot[:, None], p_off_bot, p_off_top)
                p_s = positions + p_off

                if self.attached.samples_bot is not None:
                    # 9 点（面心 + 4 角 + 4 边中点）逐点配对，与 _attached_rows 同口径
                    s_local = np.where(lower_is_bot[:, None, None],
                                       self.attached.samples_bot[g],
                                       self.attached.samples_top[g])
                    s_world = (positions[:, None, :]
                               + np.einsum("mij,mkj->mki", rotmats, s_local))
                    m_s = s_world.shape[1]
                    z_k = self.depth_map.get_obstacle_height_batch(
                        s_world[..., :2].reshape(-1, 2), inflate=0.0
                    ).reshape(m, m_s)
                    extra = 0.005
                    v_k = (z_k + self.safety_margin + extra - s_world[..., 2])
                    viol_g = np.max(v_k, axis=1)
                    # dup 判据要用与 _attached_rows 相同的力臂（最坏采样点）
                    _kb = np.argmax(v_k, axis=1)
                    _rows = np.arange(m)
                    p_dup = s_world[_rows, _kb] - positions
                elif self.attached.corners_bot is not None:
                    c_local = np.where(lower_is_bot[:, None, None],
                                       self.attached.corners_bot[g],
                                       self.attached.corners_top[g])
                    c_world = (positions[:, None, :]
                               + np.einsum("mij,mkj->mki", rotmats, c_local))
                    z_g = self.depth_map.get_obstacle_height_batch(
                        p_s[:, :2], inflate=0.0)
                    for k in range(4):
                        zk = self.depth_map.get_obstacle_height_batch(
                            c_world[:, k, :2], inflate=0.0)
                        z_g = np.maximum(z_g, zk)
                    extra = 0.005
                    viol_g = z_g + self.safety_margin + extra - p_s[:, 2]
                    p_dup = p_off
                else:
                    # 旧路径（3 元组 / from_cdprim_fn，无角点）：与 _attached_rows
                    # 的 else 分支同口径 —— 那里是 inflate=0 + extra=fp_radius，
                    # 这里不能再传 inflate（max-filter 窗口 ≈15 格，会把 6cm 外
                    # 的箱墙顶也吞进来，把 φ_start 抬松，绝对帽随之失效）。
                    z_g = self.depth_map.get_obstacle_height_batch(
                        p_s[:, :2], inflate=0.0)
                    extra = self.attached.fp_radii[g]
                    viol_g = z_g + self.safety_margin + extra - p_s[:, 2]
                    p_dup = p_off

                # dup 路点：该组行在 SQP 里被并入 TCP 行（rhs 取较大者），
                # 真实违例也应并入 TCP 行，否则这一路的穿障会被漏掉。
                dup = ((np.abs(p_dup[:, 0]) < 2e-3)
                       & (np.abs(p_dup[:, 1]) < 2e-3)
                       & ~exempt)
                out[dup, 0] = np.maximum(out[dup, 0], viol_g[dup])
                out[:, 1 + g] = np.where(dup | exempt, -np.inf, viol_g)

        # 首/末路点：SQP 侧该路点所有行 rhs=-inf（无约束），这里同样跳过
        out[0, :] = -np.inf
        out[-1, :] = -np.inf
        return out

    def max_violation(self, waypoints: np.ndarray) -> float:
        """merit 用的最大违例（米），语义由 use_real_merit 决定。

        use_real_merit=False → 只有高度场代理（原行为，零额外成本）；
        use_real_merit=True  → max(代理, 真实几何持物 SAT)。

        不做线性化：直接在世界坐标查高度场 / 做 OBB-AABB 分离量。全部行都被
        禁用时返回 0.0（= 无活跃约束，视为无违例），避免 -inf 参与比较产生 nan。
        """
        return self.max_violation_split(waypoints)[2]

    def max_violation_split(self, waypoints: np.ndarray, want_real=None):
        """一次 FK 同时算三联：返回 (proxy, real, used)。

        proxy : 高度场 9 点代理违例（原 max_violation 的口径）
        real  : 真实几何持物违例（15 轴 SAT），不可用时为 nan
        used  : merit 实际采用的量 = max(proxy, real) if use_real_merit else proxy

        want_real=None 时跟随 self.use_real_merit；诊断想无条件看 real 就传 True。
        共用同一次 FK —— 两者分别求运动学会让每次 merit 多付一遍 FK。
        """
        fk = self._fk_pos_rot(waypoints)
        proxy = self._max_of(self._violation_rows(waypoints, fk=fk))
        if want_real is None:
            want_real = self.use_real_merit
        if not want_real or self.real_aabb is None or self.attached is None:
            return proxy, float("nan"), proxy
        vr, _ = self._real_viol_rows(waypoints, fk=fk)
        if not np.isfinite(vr).any():
            return proxy, float("nan"), proxy
        real = float(vr.max())
        used = max(proxy, real) if self.use_real_merit else proxy
        return proxy, real, used

    def max_violation_detail(self, waypoints: np.ndarray):
        """同 max_violation，另返回最差位置 (viol, wp_idx, row_kind)。"""
        v = self._violation_rows(waypoints)
        val = self._max_of(v)
        if not np.isfinite(v).any():
            return val, -1, -1
        flat = int(np.argmax(v))
        wp, rp = np.unravel_index(flat, v.shape)
        return val, int(wp), int(rp)

    @staticmethod
    def _max_of(v: np.ndarray) -> float:
        return float(v.max()) if np.isfinite(v).any() else 0.0

    # ------------------------------------------------------------------
    # 真实几何持物违例（诊断 / merit 兜底，不进 QP）
    # ------------------------------------------------------------------
    def _real_viol_rows(self, waypoints: np.ndarray, fk=None):
        """持物 OBB vs real_aabb 的逐路点违例 (m, G) 与最近障碍索引 (m, G)。

        clearance = 最近障碍的 15 轴 SAT 分离量（>0 分离，<0 = -最小平移深度）
        viol      = safety_margin + extra - clearance，>0 判违例。
        与 _attached_rows 的「9 点 + extra 5mm」同 margin/extra 口径，可比。
        """
        m = waypoints.shape[0]
        G = 0 if self.attached is None else len(self.attached.offsets)
        out = np.full((m, G), -np.inf)
        outk = np.full((m, G), -1, dtype=int)
        if (G == 0 or self.real_aabb is None
                or getattr(self.attached, "samples_bot", None) is None):
            return out, outk

        pos, rotmats = (self._fk_pos_rot(waypoints) if fk is None else fk)

        A = self.real_aabb
        Ca = (A[:, :3] + A[:, 3:]) / 2.0                      # (K,3)
        Ha = (A[:, 3:] - A[:, :3]) / 2.0                       # (K,3)

        exempt = np.zeros(m, dtype=bool)
        _ef = int(round(self.held_exempt_frac[0] * m))
        _el = int(round(self.held_exempt_frac[1] * m))
        if _ef > 0:
            exempt[:_ef] = True
        if _el > 0:
            exempt[m - _el:] = True

        extra = 0.005
        for g in range(G):
            if getattr(self.attached, "half_xy", None) is None:
                out[0, :] = -np.inf
                out[-1, :] = -np.inf
                return out, outk
            hx, hy = (float(self.attached.half_xy[g][0]),
                      float(self.attached.half_xy[g][1]))
            p_b = np.asarray(self.attached.offsets[g], dtype=float)
            p_t = np.asarray(self.attached.tops[g], dtype=float)
            # 局系半高 = |顶-底|/2（旋转不改变该长度，R_tb 有倾角时也成立）
            half = np.array([hx, hy, 0.5 * float(np.linalg.norm(p_t - p_b))])
            c_tcp = 0.5 * (p_b + p_t)                          # 组心（TCP 系）
            cw = pos + np.einsum("mij,j->mi", rotmats, c_tcp)   # (m,3)
            # 物体局系轴在世界系的朝向 = TCP_rot @ R_tb（R_tb 已存在 attached 里）
            Rob = np.einsum("mij,jk->mik", rotmats,
                            np.asarray(self.attached.tcp_rotmat, dtype=float))
            clear, kbest = self._obb_aabb_clearance(cw, Rob, half, Ca, Ha)
            out[:, g] = np.where(exempt, -np.inf,
                                 self.safety_margin + extra - clear)
            outk[:, g] = kbest
        out[0, :] = -np.inf
        out[-1, :] = -np.inf
        return out, outk

    @staticmethod
    def _obb_aabb_clearance(cw, R, half, Ca, Ha):
        """OBB(中心 cw / 轴= R 的列 / 半尺寸 half) vs K 个 AABB 的最小分离量。

        15 轴 SAT：3 世界轴 + 3 持物轴 + 9 叉积轴。返回 (m,)：
          >0 = 分离（值为最宽松轴的间隙）；<0 = -最小平移深度。
        """
        m = cw.shape[0]
        K = Ca.shape[0]
        d = cw[:, None, :] - Ca[None, :, :]                    # (m,K,3)
        ra = Ha[None, :, :]
        rb = half
        best = np.full((m, K), -np.inf)
        # 3 根世界轴：r_B = sum_j half_j |R[i,j]|
        for i in range(3):
            r_b = np.abs(R[:, i, :]) @ rb
            best = np.maximum(best, np.abs(d[..., i]) - ra[..., i]
                              - r_b[:, None])
        # 3 根持物轴：r_A = sum_c Ha_c |R[c,j]|
        for j in range(3):
            ax = R[:, None, :, j]
            r_a = np.sum(ra * np.abs(ax), axis=-1)
            best = np.maximum(best, np.abs(np.sum(d * ax, axis=-1))
                              - rb[j] - r_a)
        # 9 根叉积轴
        for i in range(3):
            e = np.zeros(3)
            e[i] = 1.0
            for j in range(3):
                ax = np.cross(e, R[:, :, j])
                nrm = np.linalg.norm(ax, axis=1)
                good = nrm > 1e-8
                a = np.where(good[:, None],
                             ax / np.maximum(nrm, 1e-12)[:, None], 0.0)
                r_a = np.abs(a) @ np.array([Ha[:, 0], Ha[:, 1], Ha[:, 2]])
                au = np.abs(np.einsum("mc,mcj->mj", a, R))
                gap = (np.abs(np.sum(d * a[:, None, :], axis=-1))
                       - r_a - (au @ rb)[:, None])
                best = np.maximum(best, np.where(good[:, None], gap, -np.inf))
        kbest = np.argmin(best, axis=1)
        return best[np.arange(m), kbest], kbest

    def max_violation_real(self, waypoints: np.ndarray):
        """真实几何持物违例 (viol, wp_idx, box_idx)；不可用时 (0.0, -1, -1)。"""
        v, k = self._real_viol_rows(waypoints)
        if not np.isfinite(v).any():
            return 0.0, -1, -1
        flat = int(np.argmax(v))
        wp, _g = np.unravel_index(flat, v.shape)
        return float(v.max()), int(wp), int(k[wp, _g])

    def check_attached_violation(self, q: np.ndarray) -> np.ndarray:
        """持有物体每组的违反量（>0 = 该组较低端低于当地障碍 + margin）。"""
        if self.attached is None:
            return np.zeros(0)
        if self.fast_kin is not None:
            pos, rotmat, _ = self.fast_kin.fk_and_jacobian(q)
        else:
            pos, rotmat = self.robot.forward_kinematics(q)
        out = np.zeros(len(self.attached.offsets))
        for g in range(len(self.attached.offsets)):
            p_s = pos + rotmat @ self.attached.offsets[g]
            p_t = pos + rotmat @ self.attached.tops[g]
            if p_t[2] < p_s[2]:                 # 顶面心世界更低 → 用它做控制点
                p_s = p_t
            z_obs = self.depth_map.get_obstacle_height(
                p_s[0], p_s[1], inflate=self.attached.fp_radii[g])
            out[g] = z_obs + self.safety_margin - p_s[2]
        return out
