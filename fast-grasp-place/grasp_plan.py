# 抓取规划
import contextlib
import math
import pickle
import time
from pathlib import Path

import numpy as np

import config as cfg
import obstacle
import basis.robot_math as rm
import motion.probabilistic.rrt_connect as rrt_connect
import robot_sim.robots.ur7e.ur7e_withouttable_dh76 as ur7e
import gomp_refine
from gomp.obstacles.attached_object import (AttachedObject, read_boxes_from_cdprim)
# ============================================================================
# 抓取 / 放置规划的输出：纯 dict（加字段 = 加一个 key，不用改类定义）
# ============================================================================

class Grasp_planner():

    def __init__(self, rbt_s: ur7e.UR7EDH76, obstacles=None, seed_conf=None, base0=None):
        """构造抓取规划器。

        Args:
            rbt_s: ``UR7EDH76`` 实例（sim，默认 ``enable_cc=True``）。本类会调用它的
                ``fk`` / ``goto_given_conf`` / ``is_collided`` / ``arm.jlc``。
            obstacles: ``obstacle.build_obstacles()`` 的结果（CollisionModel 列表）。
                ``None`` = 空列表，此时碰撞检查只覆盖关节限位 + 自碰撞。
            seed_conf: (6,) 默认 IK 种子兼「行程基准」构型，弧度。``None`` = 用
                ``config.START_CONF``（= 抓取起点，从它出发算 q6 行程最短）。
        """
        self.rbt_s = rbt_s
        self.obstacle_list = [] if obstacles is None else list(obstacles)
        self.seed_conf = np.asarray(
            cfg.CONFIG["joints"]["start_conf"] if seed_conf is None else seed_conf, dtype=float
        ).reshape(6).copy()
        # 单臂 RRT：UR7EDH76 属于 UR7EBase（不是双臂接口），RRTConnect 只需 robot
        self.rrt_planner = rrt_connect.RRTConnect(rbt_s)
        #: 持物碰撞检查的模型对（瓶子 + 箱四壁）实例级缓存；首次调用时构造，
        #: 实测 2.7 ms（planner 本身构造是 407 ms）。
        self._held_models = None
        #: GOMP 精修器实例级缓存；首次 plan_place 时构造（会读 depthmap/scene.npz，
        #: 缺文件时按约定硬报错）。构造一次约 0.1 s，精修一次约 7 s。
        self._gomp = None


    def plan_path(self, current_conf, target_conf, step=0.07):  # 从当前位置移动到目标位置
        """
        :param current_conf: 起始关节角
        :param target_conf: 目标关节角
        """
        path_list = self._rrt_result_to_path(self.rrt_planner.plan(
            start_conf=current_conf,
            goal_conf=target_conf,
            obstacle_list=self.obstacle_list,
            other_robot_list=[],
            ext_dist=step,  # 扩展步长
            max_n_iter=300,  # 最大迭代次数
            max_time=15.0,  # 最大规划时间（秒）
            smoothing_n_iter=50,  # 平滑迭代次数
            animation=False))
        if path_list is None:
            # 检查首末点是否碰撞
            if self._conf_in_collision(current_conf):
                print(f"RRT开始点 {list(current_conf)} 碰撞!")
                return None
            if self._conf_in_collision(target_conf):
                print(f"RRT结束点 {list(target_conf)} 碰撞!")
                return None
            print("第一次RRT路径规划失败")
            for i in range(5):
                path_list = self._rrt_result_to_path(self.rrt_planner.plan(
                    start_conf=current_conf,
                    goal_conf=target_conf,
                    obstacle_list=self.obstacle_list,
                    other_robot_list=[],
                    ext_dist=0.05,  # 扩展步长,最小0.05
                    max_n_iter=300,  # 最大迭代次数
                    max_time=15.0,  # 最大规划时间（秒）
                    smoothing_n_iter=50,  # 平滑迭代次数
                    animation=False
                ))
                if path_list is not None:
                    break
                else:
                    print(f"第{i + 2}次rrt规划失败")
        return path_list


    def get_line_path(self, seed=None, start=None, goal=None, rot=None):  # 直线路径规划,返回关节点路径
        jnts = []
        cur_seed = seed.copy()  # 更新种子，提高连续性
        # 计算起点到终点的距离
        distance = np.linalg.norm(np.array(goal) - np.array(start))
        # 根据距离和步长确定点数（至少 15 点）
        num = max(15, int(distance * 30))  # type: ignore
        for pos in np.linspace(start, goal, num):
            conf = self._safe_ik(tgt_pos=pos, tgt_rotmat=rot, seed=cur_seed)
            if conf is None:  # ik求解失败
                print(f"机器人tracik求解位置{pos}旋转{rot}关节seed{cur_seed}碰撞，直线路径求解失败")
                return None
            else:
                jnts.append(conf)
                cur_seed = conf  # 更新种子，提高连续性
        return [np.array(p) for p in jnts] if jnts else None


    def get_line_change_rot(self, seed=None, start_pos=None, goal_pos=None, start_rot=None, angle: float = 0):
        '''
        从起始到终点的直线路径，但姿态也会绕z轴旋转angle角度
        输入:
            seed:初始关节角种子
            start_pos:开始位置
            goal_pos:终点位置
            start_rot:初始姿态
            angle:绕世界坐标系z轴旋转的角度
        输出:
            jnts:关节路径点
        '''
        if angle == 0:  # 规划直线路径
            print("改为规划角度不变的直线路径")
            return self.get_line_path(seed=seed, start=start_pos, goal=goal_pos, rot=start_rot)
        # 规划直线+旋转路径
        jnts = []
        cur_seed = seed.copy()  # 更新种子，提高连续性
        # 计算起点到终点的距离
        distance = np.linalg.norm(np.array(goal_pos) - np.array(start_pos))
        # 根据距离和步长确定点数（至少 15 点，最长 cm一个点）
        num = max(15, int(distance * 50))  # type:ignore
        pos_change = (goal_pos - start_pos) / num
        angle_change = angle / num
        cur_pos = start_pos.copy()
        cur_angle = 0
        cur_conf = self._safe_ik(tgt_pos=cur_pos, tgt_rotmat=start_rot, seed=cur_seed)  # 起点关节点
        # 起始点
        if cur_conf is not None:
            # print(f"起点关节点pos:{cur_pos},start_rot:{start_rot},cur_conf:`{cur_conf}`")
            jnts.append(cur_conf)
            cur_seed = cur_conf.copy()  # 更新关节点种子
        else:
            print(f"直线旋转路径起点关节点计算失败，start_pos:{start_pos},start_rot:{start_rot},seed:{cur_seed}")
            return None
        for i in range(num):  # 总共num次循环
            cur_pos += pos_change
            cur_angle += angle_change
            cur_rot = rm.rotmat_from_axangle([0, 0, 1], cur_angle) @ start_rot
            cur_conf = self._safe_ik(cur_pos, cur_rot, seed=cur_seed)
            if cur_conf is not None:
                jnts.append(cur_conf)
                cur_seed = cur_conf.copy()  # 更新关节点种子
            else:
                print(f"直线旋转路径关节点计算失败，pos:{cur_pos},cur_rot:{cur_rot},cur_seed:{cur_seed}")
                return None
        return [np.array(p) for p in jnts] if jnts else None

    def _copy_path(self,path):
        """把路径复制成独立的一维 float 数组列表, 避免后续修改污染原数据。"""
        return [np.asarray(p, dtype=float).reshape(-1).copy() for p in path]

    def _concat_paths(self, path_a, path_b):
        """把两段原始路径首尾相接（去掉重复的衔接点）。

        用作 smooth_two_path_2 平滑失败时的回退：返回的就是"平滑前"的几何，
        不含任何插值，因此不会引入新的碰撞。
        """
        merged = [np.asarray(p, dtype=float).reshape(-1).copy() for p in path_a]
        for p in path_b:
            q = np.asarray(p, dtype=float).reshape(-1).copy()
            if merged and np.allclose(q, merged[-1], atol=1e-9, rtol=0.0):
                continue  # 衔接点重复（path_b 起点 == path_a 终点），跳过避免零步长
            merged.append(q)
        return merged


    def _safe_ik(self, tgt_pos, tgt_rotmat, seed=None, check_collision=True):
        """解 TCP 位姿的关节角（pyikfast 全解里取第一个无碰撞的），并查碰撞。

        对「工具轴竖直」的腕部奇异位姿，pyikfast 直接 0 解；第一遍 ``_ikfast_all``
        只做一次 1e-3 rad 微扰+抛光，精确竖直时仍可能
        全空。此时本函数用递增微扰角（1.5e-3 → 2e-3）重新做「微扰 → ikfast →
        DLS 抛光回真值 → 碰撞检查」，直到拿到可用解。微扰角序列是恒定实现细节，
        不对外暴露为参数；非奇异位姿完全走不到兜底，零额外开销。

        Args:
            tgt_pos: (3,) TCP（= 手爪中心 jaw center）目标位置，sim 世界系，单位 m。
            tgt_rotmat: (3,3) TCP 目标姿态，sim 世界系。
            seed: (6,) 参照构型（弧度）；``None`` = ``self.seed_conf``。pyikfast 用它
                在等价角（±2π）里挑最近的，也决定返回顺序，直线路径里应传上一个路点；
                兜底抛光时也用它做初值/限位折回的参照。
            check_collision: 是否用 ``self.obstacle_list`` 查碰撞；直线路径逐点调用时
                保持 True（否则会生成穿模路径）。

        Returns:
            np.ndarray | None: (6,) 关节角（已折回关节限位）；None = 常规解与奇异兜底
                都无解，或所有解都碰撞。
        """
        for conf in self._ikfast_all(tgt_pos, tgt_rotmat, seed=seed,
                                     max_solutions=3):
            if check_collision and self._conf_in_collision(conf):
                continue
            return conf

        # 奇异兜底：1e-3 微扰仍无解时加大微扰重新求解（恒定量，非开关/参数）
        seed_conf = (self.seed_conf if seed is None
                     else np.asarray(seed, dtype=float).reshape(6))
        pos = np.asarray(tgt_pos, dtype=float).reshape(3)
        rot = np.asarray(tgt_rotmat, dtype=float).reshape(3, 3)
        for tilt_rad in (1.5e-3, 2e-3):
            tilt = rm.rotmat_from_axangle(np.array([1.0, 0.0, 0.0]), tilt_rad)
            for conf in self._pyikfast_raw(pos, tilt @ rot, seed_conf):
                q, _, _ = self._dls_ik(pos, rot, seed=conf, max_n_iter=5)
                if q is None:
                    continue
                if check_collision and self._conf_in_collision(q):
                    continue
                return q
        return None

    def _ikfast_all(self, tgt_pos, tgt_rotmat, seed=None, max_solutions=None):
        """pyikfast 解析 IK 的解，按与 seed 的关节距离升序。

        Args:
            tgt_pos: (3,) TCP 目标位置（= jaw center），sim 世界系，单位 m。
            tgt_rotmat: (3,3) TCP 目标姿态，sim 世界系。
            seed: (6,) 参照构型（弧度）；``None`` = ``self.seed_conf``。pyikfast 用它
                把每个关节折到最近的等价角，并按距离排序。
            max_solutions: int | None 最多返回几个解，**只作用于竖直奇异的微扰路径**
                （按 seed 距离取前 N 支抛光）。None = 全部。抛光 + 碰撞检查是主要
                成本，实测每支 ~2 ms；非奇异位姿直接返回 pyikfast 全解，不抛光。

        Returns:
            list[np.ndarray]: 每个元素 (6,) 关节角；无解返回 ``[]``。

        Note:
            ⛔「工具轴竖直」是 ikfast 模型的奇异位形：实测该姿态直接 0 解
            （微扰 <1e-3 rad 仍 0，=1e-3 rad 才有 8 解）。本项目全部抓取候选都是
            竖直下抓，因此空解时绕工具 x 轴加 1e-3 rad
            再解一次，然后用 ``_dls_ik`` 抛光回真值位姿（残差 <1e-9）。
        """
        seed_conf = (self.seed_conf if seed is None
                     else np.asarray(seed, dtype=float).reshape(6))
        pos = np.asarray(tgt_pos, dtype=float).reshape(3)
        rot = np.asarray(tgt_rotmat, dtype=float).reshape(3, 3)

        sols = self._pyikfast_raw(pos, rot, seed_conf)
        if sols:
            return sols

        tilt = rm.rotmat_from_axangle(np.array([1.0, 0.0, 0.0]), 1e-3)
        limit = None if max_solutions is None else max(1, int(max_solutions))
        polished = []
        # ``pyikfast`` 已按 seed 距离排好序：从最近的一支开始抛光，够数就停
        for conf in self._pyikfast_raw(pos, tilt @ rot, seed_conf):
            q, _, _ = self._dls_ik(pos, rot, seed=conf, max_n_iter=5)
            if q is not None:
                polished.append(q)
                if limit is not None and len(polished) >= limit:
                    break
        polished.sort(key=lambda c: float(np.linalg.norm(c - seed_conf)))
        return polished

    def _pyikfast_raw(self, tgt_pos, tgt_rotmat, seed):
        """直接调 ``rbt_s.pyikfast``（TCP 位姿入、全解出），并把 ImportError 变可读错误。

        Args:
            tgt_pos: (3,) TCP 目标位置，sim 世界系，单位 m。
            tgt_rotmat: (3,3) TCP 目标姿态，sim 世界系。
            seed: (6,) 参照构型（弧度），非 None。

        Returns:
            list[np.ndarray]: (6,) 关节角列表；无解为 ``[]``。

        Raises:
            RuntimeError: 环境里没有 pyikfast（wrs 的 import 守卫已把它降级成 None）。
        """
        try:
            return list(self.rbt_s.pyikfast(tgt_pos, tgt_rotmat, seed_jnt_values=seed))
        except ImportError as exc:
            raise RuntimeError(
                "pyikfast 不可用（{}）；本项目 IK 已改为解析解，需先装好 pyikfast 扩展"
                .format(exc)) from exc

    def _dls_ik(self, tgt_pos, tgt_rotmat, seed=None, max_n_iter=5):
        """阻尼最小二乘（DLS）数值 IK —— 现仅用于 pyikfast 微扰解的抛光。

        求解器已改为 ``_ikfast_all``（解析解）；只有 ikfast 在竖直工具轴零解时，
        才用它把 1e-3 rad 微扰解收敛回真值位姿。

        Args:
            tgt_pos: (3,) TCP 目标位置，sim 世界系，单位 m。
            tgt_rotmat: (3,3) TCP 目标姿态，sim 世界系。
            seed: (6,) 初值（弧度）；``None`` = ``self.seed_conf``。
            max_n_iter: 最大迭代次数，默认 5（本项目仅用于微扰解抛光，2 次即收敛）。

        Returns:
            tuple: ``(conf, pos_err_m, rot_err_rad)``。``conf`` 为 (6,) 关节角或 None
            （None 时两个误差是最后一次迭代的残余量，用于诊断）。
        """
        loc_p = np.asarray(self.rbt_s.arm.loc_tcp_pos, dtype=float)
        loc_r = np.asarray(self.rbt_s.arm.loc_tcp_rotmat, dtype=float)
        jlc = self.rbt_s.arm.jlc
        tgt_r = np.asarray(tgt_rotmat, dtype=float).reshape(3, 3)
        # TCP -> 法兰：jlc.fk 报的是法兰位姿
        tgt_p_flange = np.asarray(tgt_pos, dtype=float).reshape(3) - tgt_r @ loc_p
        tgt_r_flange = tgt_r @ loc_r.T

        q = np.asarray(self.seed_conf if seed is None else seed, dtype=float).reshape(6).copy()
        n_iter = int(max_n_iter)
        pos_tol = 1e-4
        rot_tol = 1e-3

        pos_err = rot_err = float("inf")
        for _ in range(n_iter):
            f_pos, f_rot, j_mat = jlc.fk(jnt_values=q, toggle_jacobian=True, update=False)
            pos_err, rot_err, err_vec = rm.diff_between_poses(
                src_pos=f_pos, src_rotmat=f_rot, tgt_pos=tgt_p_flange, tgt_rotmat=tgt_r_flange)
            if pos_err < pos_tol and rot_err < rot_tol:
                wrapped = self._wrap_conf_to_limits(q, seed=seed)
                return (wrapped, pos_err, rot_err) if wrapped is not None else (None, pos_err, rot_err)
            step = np.copy(err_vec)
            if pos_err >= 0.1:      # 与 wrs 一致的误差钳位
                step[:3] = 0.1 * err_vec[:3] / pos_err
            if rot_err >= np.pi / 10:
                step[3:6] = (np.pi / 10) * err_vec[3:6] / rot_err
            delta = np.linalg.lstsq(j_mat, step, rcond=1e-4)[0]
            if np.linalg.norm(delta) < 1e-12:   # 真正停滞（不是 sum≈0 的误判）
                return None, pos_err, rot_err
            q = q + delta
        return None, pos_err, rot_err


    def _wrap_conf_to_limits(self, conf, seed=None):
        """把 6 个关节角各自按 2π 平移折回关节限位。

        Args:
            conf: (6,) 关节角（弧度），可能因数值 IK 多转整数圈而超限。
            seed: (6,) 参照构型；给了就在等价解里选离它最近的那个（省行程）。

        Returns:
            np.ndarray | None: (6,) 合法关节角；某关节无论 ±2π 都落不进限位时返回 None。
        """
        q = np.asarray(conf, dtype=float).reshape(6).copy()
        ranges = np.asarray(self.rbt_s.arm.jlc.jnt_ranges, dtype=float)
        ref = None if seed is None else np.asarray(seed, dtype=float).reshape(6)
        two_pi = 2.0 * np.pi
        for j in range(6):
            lo, hi = float(ranges[j, 0]), float(ranges[j, 1])
            k0 = int(math.ceil((lo - q[j]) / two_pi))
            options = [q[j] + two_pi * k for k in range(k0 - 1, k0 + 3)]
            options = [v for v in options if lo - 1e-9 <= v <= hi + 1e-9]
            if not options:
                return None
            if ref is None:
                q[j] = min(options, key=abs)
            else:
                q[j] = min(options, key=lambda v: abs(v - ref[j]))
        return q

    def _conf_in_collision(self, conf):
        """把机器人摆到 conf 并查碰撞（关节限位 + 自碰撞 + ``self.obstacle_list``）。

        Args:
            conf: (6,) 关节角（弧度）。

        Returns:
            bool: True = 碰撞。

        Note:
            会改变 ``rbt_s`` 的当前关节状态，调用方需要自己复位（不影响 IK，
            因为 ``_dls_ik`` 抛光走的是 ``jlc.fk(update=False)``）。
        """
        self.rbt_s.goto_given_conf(np.asarray(conf, dtype=float).reshape(6))
        hit = self.rbt_s.is_collided(obstacle_list=self.obstacle_list, toggle_contacts=False)
        return bool(hit[0] if isinstance(hit, tuple) else hit)

    @staticmethod
    def _rrt_result_to_path(result):
        """把 ``RRTConnect.plan()`` 的返回值统一成关节路径列表。

        Args:
            result: ``RRTConnect.plan()`` 的返回 —— 实测是
                ``wrs.motion.motion_data.MotionData`` 对象（不是 list！
                它的 ``__len__`` 会误导：``len(mot_data) == 26`` 看着像 26 个路点，
                但 ``mot_data[0]`` 抛 TypeError），也可能是 None（规划失败）。

        Returns:
            list[np.ndarray] | None: 每个元素 (6,) 关节角（弧度）；失败时 None。

        Note:
            ``MotionData.jv_list`` 才是路点列表。UR9 那套 ``rrt_planner.plan``
            返回的是裸 list，直接搬过来会踩这个坑。
        """
        if result is None:
            return None
        jv_list = getattr(result, "jv_list", None)
        if jv_list is None:
            jv_list = result
        path = [np.asarray(q, dtype=float).reshape(6) for q in jv_list]
        return path or None

    # 平滑路径处理
    def get_smooth_path(self, path, step=0.05):
        """用 RRT 的 shortcut 平滑器缩短关节路径。

        Args:
            path: list[(6,)] 关节路径（弧度），至少 3 点。
            step: 捷径插值步长（rad），越小越保守。

        Returns:
            list[np.ndarray]: 平滑后的路径；输入为空时返回空列表。
        """
        if not path:
            return []
        smoothed = self.rrt_planner._smooth_path(
            path,
            obstacle_list=self.obstacle_list,
            other_robot_list=[],
            granularity=step,
        )
        return [np.array(p) for p in smoothed] if smoothed else []


    def _blend_joint_window(self, path, left_idx, half_window=3, max_step=0.5,
                            max_move=0.30, max_turn_deg=120.0, stats=None):
        """把 ``left_idx`` 与 ``left_idx+1`` 之间的拼接折角用三次 Hermite 局部抹平。

        窗口为 ``[left_idx-half_window, left_idx+half_window]``（共 ``2*hw+1`` 点），
        只改写窗口**内部**点：首尾锚点与整条路径端点严格不动，点数不变。
        切线口径与 ``blend_shortcut_seam`` 一致（算法出处，本项目无跨项目依赖）：
        左端取**左窗外入射切线**（不能用窗内首步——那正是要修的折角），右端取
        窗口内末步方向（右锚点右侧已被平滑器改向，窗外方向不再是原方向）。

        Args:
            path: list[(6,)] 关节路径（弧度）。函数内不改动它。
            left_idx: 折角顶点下标，折角位于 ``path[left_idx]`` 与
                ``path[left_idx+1]`` 之间。需满足 ``0 < left_idx < len(path)-1``。
            half_window: int 窗口半宽（点）。默认 3（7 点窗口），与
                ``blend_shortcut_seam`` 默认一致；给 2 时折角抹不干净，
                实测 90° 折角会残留约 20° 并被搬到窗口两端。
            max_step: float 窗口内相邻点的单关节步长上限（rad）。默认 0.5，
                真实路径步长约 0.02，只作安全阀，不是调参旋钮。
            max_move: float 窗口内任一点相对**原位**的最大位移上限
                （6 轴欧氏范数，rad）。默认 0.30（约 17°）。这是防「整条路径
                被拉直」的护栏：逐点碰撞检查只能证明点不撞，证明不了几何没
                走远（150° 折返实测位移 0.189 rad、长度缩短 27.5%）。
            max_turn_deg: float 折角大于该角度（度）时直接放弃。默认 120。
                接近 180° 的折返（典型：approach 下去、lift 原路升回来，实测
                176°）是路径语义本身，抹平等于把「下去再上来」改成「绕过旁边」，
                而且三次 Hermite 在 7 点窗口内根本抹不平（实测 176° → 176.9°）。
            stats: dict | None 诊断字典。无论成功失败都会写 ``blend_reason``
                （``ok`` / ``no_left_tangent`` / ``no_right_room`` /
                ``window_too_small`` / ``turn_too_sharp`` / ``non_finite`` /
                ``max_step`` / ``max_move`` / ``collision``）、``blend_lo`` /
                ``blend_hi`` / ``blend_max_move_deg`` / ``blend_turn_deg_before`` /
                ``blend_turn_deg_after``。后两个转折角都取**窗口内最大转折角**，
                ``after`` 还额外把曲线与窗外的两个衔接点算进去（否则「折角被
                搬到窗口边界」这种失败模式看不见）。不传则不收集。

        Returns:
            tuple[int, int, list] | None: ``(lo, hi, curve)``，``curve`` 长度恒
            等于 ``hi-lo``，调用方 ``path[lo:hi] = curve`` 即等长替换；
            任一校验不过返回 ``None``，调用方必须保留原窗口（绝不降级）。

        Note:
            与 ``blend_shortcut_seam`` 的差别只有两点：窗口以折角为中心（而非
            左锚点向右）、失败时不打 print（由调用方按 ``stats`` 决定日志）。
        """
        def _turn(seq, i):
            """算 ``seq[i]`` 处的转折角（度）；任一侧零步长按无折角 0.0。"""
            v1 = np.asarray(seq[i]) - np.asarray(seq[i - 1])
            v2 = np.asarray(seq[i + 1]) - np.asarray(seq[i])
            a, b = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            if a <= 1e-12 or b <= 1e-12:
                return 0.0
            cos = float(np.clip(np.dot(v1, v2) / (a * b), -1.0, 1.0))
            return float(math.degrees(math.acos(cos)))

        lo = max(0, left_idx - int(half_window))
        hi = min(len(path), left_idx + int(half_window) + 1)
        if lo <= 0:
            reason = "no_left_tangent"
        elif hi >= len(path):
            reason = "no_right_room"
        elif hi - lo < 4:
            reason = "window_too_small"
        else:
            reason = None
        before = 0.0
        if reason is None:
            # 取**窗口内**最大转折角，而不是 left_idx 处那一个：合并窗口后中心
            # 未必是折角顶点（实测 lift/transit 的真实折角在窗口内 index 10，
            # 而合并后的中心是 11，_turn(path, 11) = 0，诊断会完全失真）。
            before = max(_turn(path, j) for j in range(lo + 1, hi))
            if before > float(max_turn_deg):
                reason = "turn_too_sharp"
        if stats is not None:
            stats.update(blend_reason=reason, blend_lo=lo, blend_hi=hi,
                         blend_max_move_deg=0.0,
                         blend_turn_deg_before=float(before),
                         blend_turn_deg_after=float(before))
        if reason is not None:
            return None

        win = [np.asarray(p, dtype=float).copy() for p in path[lo:hi]]
        n = len(win) - 1
        p0, p1 = win[0], win[-1]
        d0 = (win[0] - np.asarray(path[lo - 1], dtype=float)) * n   # ★ 左窗外入射
        d1 = (win[-1] - win[-2]) * n                                # 右锚点取窗内末步
        curve = []
        for u in np.linspace(0.0, 1.0, n + 1):
            u2, u3 = u * u, u ** 3
            curve.append((2 * u3 - 3 * u2 + 1) * p0
                         + (u3 - 2 * u2 + u) * d0
                         + (-2 * u3 + 3 * u2) * p1
                         + (u3 - u2) * d1)
        curve[0] = p0.copy()
        curve[-1] = p1.copy()
        if any(not np.all(np.isfinite(p)) for p in curve):
            if stats is not None:
                stats["blend_reason"] = "non_finite"
            return None
        if max(float(np.max(np.abs(curve[j + 1] - curve[j]))) for j in range(n)) > max_step:
            if stats is not None:
                stats["blend_reason"] = "max_step"
            return None
        move = max(float(np.linalg.norm(curve[j] - win[j])) for j in range(n + 1))
        if stats is not None:
            stats["blend_max_move_deg"] = float(math.degrees(move))
        if move > max_move:
            if stats is not None:
                stats["blend_reason"] = "max_move"
            return None
        if self._path_collision_idx(curve) >= 0:
            print("smooth_two_path: 拼接点平滑后碰撞，保留原窗口")
            if stats is not None:
                stats["blend_reason"] = "collision"
            return None
        if stats is not None:
            stats["blend_reason"] = "ok"
            # 把曲线与窗外的两个衔接点一起算：折角被「搬到窗口边界」是三次
            # Hermite 最常见的失败模式，只量窗口内部会看不见。
            probe = ([np.asarray(path[lo - 1], dtype=float)] + curve
                     + [np.asarray(path[hi], dtype=float)])
            stats["blend_turn_deg_after"] = float(
                max(_turn(probe, j) for j in range(2, len(probe) - 1)))
        return lo, hi, curve


    def smooth_two_path(self, path1, path2, step=0.2,
                        overlap_a=1.0 / 3.0, overlap_b=1.0 / 3.0, blend_joints=False):
        if path1 is None and path2 is None:
            print("平滑路径1和2都为None")
            return None
        elif path1 is None:
            print("平滑路径1为None")
            return self.get_smooth_path(path2,step)
        elif path2 is None:
            print("平滑路径2为None")
            return self.get_smooth_path(path1,step)
        # 中间1/3部分进行平滑
        head = self._copy_path(path1)
        tail = self._copy_path(path2)
        k = int(len(head) * (1-overlap_a))
        m = int(len(tail) * overlap_b)
        smooth_path = self.get_smooth_path(head[k:] + tail[:m], step)
        # 合并最终路径
        final_path = head[:k] + self._copy_path(smooth_path) + tail[m:]
        if not final_path:
            return None
        if blend_joints and len(final_path) >= 4:
            # 两个拼接界面各做一次局部 Hermite 补接：k-1（head 末点 / 平滑段首点）
            # 与 k+len(smooth)-1（平滑段末点 / tail 首点）。索引从右往左处理只为
            # 日志顺序稳定：替换是等长的，索引不会失效。
            joints = [k - 1]
            if smooth_path:
                joints.append(k + len(smooth_path) - 1)
            joints = sorted({i for i in joints if 0 < i < len(final_path) - 1})
            half_window = 3
            # 两窗口重叠时合并成一个大窗口，但**不放大 half_window**：放大后
            # 窗口可能覆盖整条路径（实测 4+4 点短段覆盖 86%，等于把整条路径
            # 换成一条 Hermite 曲线），碰撞检查验不出这种几何走远。
            if len(joints) == 2 and joints[1] - joints[0] <= 2 * half_window:
                joints = [(joints[0] + joints[1]) // 2]
            for i in sorted(joints, reverse=True):
                res = self._blend_joint_window(final_path, i, half_window=half_window)
                if res is not None:
                    lo, hi, curve = res
                    final_path[lo:hi] = curve
        return final_path


    def _path_collision_idx(self, path, jaw_width=None):
        """沿关节路径逐点查碰撞，返回首个碰撞路点的下标；全程无碰撞返回 -1。

        判据与规划/校验一致：``rbt_s.goto_given_conf(conf)`` 后用
        ``rbt_s.is_collided(self.obstacle_list)``（机器人本体 cdprim vs 障碍物，
        含自碰撞 + 关节限位）。

        Args:
            path: list[(6,)] 关节路径（弧度）；``None`` / 空列表直接返回 -1。
            jaw_width: 查碰撞时用的夹爪开口（指尖全距，米）；``None`` = 不改变当前
                开口。不同段开口不同（接近 0.041 / 夹持 0.031 / 松爪 0.092），
                想把「手指张开会不会撞」算进去就必须传。

        Returns:
            int: 首个碰撞路点下标；无碰撞返回 -1。

        Note:
            会临时改变 rbt_s 的关节状态与夹爪开口，返回前都会恢复，不影响调用方
            后续计算。机器人未开碰撞模型（``cc is None``）时跳过检查返回 -1。
        """
        if jaw_width is not None:
            with self._jaw(jaw_width):
                return self._path_collision_idx(path)
        if path is None or len(path) == 0:
            return -1
        if getattr(self.rbt_s, "cc", None) is None:
            return -1
        try:
            saved = self.rbt_s.get_jnt_values()
        except Exception:
            saved = None
        hit = -1
        try:
            for idx, conf in enumerate(path):
                self.rbt_s.goto_given_conf(np.asarray(conf, dtype=float).reshape(6))
                if self.rbt_s.is_collided(obstacle_list=self.obstacle_list,
                                          toggle_contacts=False):
                    hit = idx
                    break
        finally:
            if saved is not None:
                self.rbt_s.goto_given_conf(np.asarray(saved, dtype=float).reshape(6))
        return hit

    def simplify_path(self, path, tol=0.01, check_collision=True, sample_step=0.05,
                      max_step=0.5):
        """关节空间 Douglas–Peucker 简化：删掉冗余路点，首末点严格保留。
        参数
        ----
        path : list[np.ndarray(6)]
        tol : float
            允许的最大关节偏差（rad，6 轴等权、欧氏范数）。判据是「原始点到
            简化折线的最近距离」，所以它就是简化后的整体误差上界。
        check_collision : bool
            简化把折线拉直，可能蹭到原来绕开的东西 —— 简化后沿新折线（含段内
            插值）再查一遍碰撞，有碰撞就放弃简化、原样返回 path。
        sample_step : float
            碰撞复查时相邻路点间的插值步长（rad），只在 check_collision 为真时用。
        max_step : float or None
            相邻路点的最大关节步长（rad），None 或 <=0 关闭。
        返回
        ----
        list[np.ndarray(6)]；path 为 None 或不足 3 点时原样返回。
        """
        if path is None or len(path) < 3:
            return path
        Q = np.asarray(path, dtype=float)
        n = len(Q)

        # ---- Douglas–Peucker（迭代栈，避免深递归） ----
        keep = np.zeros(n, dtype=bool)
        keep[0] = keep[-1] = True
        stack = [(0, n - 1)]
        while stack:
            i, j = stack.pop()
            if j <= i + 1:      # 相邻点之间没有可删的点
                continue
            a, ab = Q[i], Q[j] - Q[i]
            L2 = float(ab @ ab)
            mid = Q[i + 1:j]
            if L2 < 1e-18:      # 首末重合：退化成到点的距离
                dev = np.linalg.norm(mid - a, axis=1)
            else:               # 点到线段距离，投影截断在 [0,1]
                t = np.clip((mid - a) @ ab / L2, 0.0, 1.0)
                dev = np.linalg.norm(mid - (a + t[:, None] * ab), axis=1)
            k = int(np.argmax(dev))
            if dev[k] > tol:    # 该点必须留，左右两段继续分治
                keep[i + 1 + k] = True
                stack.append((i, i + 1 + k))
                stack.append((i + 1 + k, j))

        # ---- 步长上限：沿原始点补回中间点，使相邻点 max|dq| <= max_step ----
        if max_step is not None and max_step > 0.0:
            add = []
            idx = np.flatnonzero(keep)
            for i, j in zip(idx[:-1], idx[1:]):
                c = i
                while c < j:
                    seg = np.max(np.abs(Q[c + 1:j + 1] - Q[c]), axis=1)
                    ok = np.flatnonzero(seg <= max_step)
                    # 满足约束的最远点；单步就超限时只能硬留相邻点
                    nxt = c + 1 + int(ok[-1]) if len(ok) else c + 1
                    if nxt < j:
                        add.append(nxt)
                    c = nxt
            if add:
                keep[np.asarray(add, dtype=int)] = True
        out = Q[keep]

        # ---- 实际最大偏差（tol 是否放太松，看这个数） ----
        A, AB = out[:-1], out[1:] - out[:-1]
        L2s = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-18)
        dev_max = 0.0
        for p in Q:
            rel = p - A
            t = np.clip(np.einsum("ij,ij->i", rel, AB) / L2s, 0.0, 1.0)
            dev_max = max(dev_max, float(np.linalg.norm(rel - t[:, None] * AB, axis=1).min()))

        # ---- 碰撞复查：新折线 + 段内插值 ----
        if check_collision:
            probe = [out[0]]
            for q0, q1 in zip(out[:-1], out[1:]):
                seg = q1 - q0
                k = max(1, int(math.ceil(float(np.max(np.abs(seg))) / sample_step)))
                probe.extend(q0 + seg * (s / k) for s in range(1, k + 1))
            hit = self._path_collision_idx(probe)
            if hit >= 0:
                print(f"[simplify] 简化到 {len(out)} 点后第 {hit}/{len(probe)} 个插值点碰撞"
                      f"（tol={tol}），放弃简化，返回原路径")
                return [np.array(q) for q in Q]

        step_max = float(np.max(np.abs(np.diff(out, axis=0)))) if len(out) > 1 else 0.0
        print(f"[simplify] {n} -> {len(out)} 点 (tol={tol} rad, max_step={max_step}, "
              f"实际最大偏差 {dev_max:.4f} rad, 实际最大步长 {step_max:.3f} rad)")
        return [np.array(q) for q in out]

    def smooth_two_path_2(
        self,
        path_a,
        path_b,
        *,
        overlap_a=1.0 / 3.0,
        overlap_b=1.0 / 3.0,
        max_joint_step_rad=0.5,
        check_collision=False,
    ):
        """用五次厄米特(Hermite)插值把两段路径平滑拼接成一条连续路径。
        参数:
            path_a: 第一段路径 (list of array_like, 每点 6 个关节角)
            path_b: 第二段路径, 其起点应与 path_a 终点(或附近)衔接
            overlap_a: path_a 尾部参与平滑的比例 (0,1], 默认 1/3; 实际点数 = round(比例 * len(path_a))
            overlap_b: path_b 前面参与平滑的比例 (0,1], 默认 1/3; 实际点数 = round(比例 * len(path_b))
            max_joint_step_rad: 拼接后相邻点的最大关节步长 (弧度)
            check_collision: 是否检查平滑新生成的那段曲线 (默认开)。
        返回:
            list[np.ndarray] —— 平滑拼接后的路径(每段 6 个关节角);
            以下三种情况会打印原因并回退:
              1) 五次厄米特拼接不可行 -> smooth_two_path(path_a, path_b, step=max_joint_step_rad);
              2) check_collision 为真且平滑曲线存在碰撞路点 -> 平滑前的路径
                 (path_a + path_b 首尾相接、衔接点去重, 不含任何插值)。
        """
        # 与 smooth_two_path 一致的空输入处理
        if path_a is None and path_b is None:
            print("平滑路径1和2都为None")
            return None
        elif path_a is None:
            print("平滑路径1为None")
            return [np.array(p) for p in path_b]
        elif path_b is None:
            print("平滑路径2为None")
            return [np.array(p) for p in path_a]

        original_a = self._copy_path(path_a)
        original_b = self._copy_path(path_b)

        # 基本合法性检查 (两段都需 >=3 点, 以便各自取出头/尾局部导数)
        if len(original_a) < 3 or len(original_b) < 3:
            print("smooth_two_path_2: both paths need at least three points")
            return self.smooth_two_path(path_a, path_b,
                                        overlap_a= overlap_a, overlap_b=overlap_b)
        if not np.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0.0:
            print("smooth_two_path_2: max_joint_step_rad must be a positive finite value")
            return self.smooth_two_path(path_a, path_b,
                                        overlap_a= overlap_a, overlap_b=overlap_b)
        n_a = len(original_a)
        n_b = len(original_b)
        # path_a 尾部重叠窗口大小(比例, 0 < ratio <= 1, 默认 1/3) 与曲线起点
        if overlap_a is None:
            overlap_a = 1.0 / 3.0
        if not (np.isfinite(overlap_a) and 0.0 < overlap_a <= 1.0):
            print("smooth_two_path_2: overlap_a must be a ratio in (0, 1]")
            return self.smooth_two_path(path_a, path_b,
                                        overlap_a= overlap_a, overlap_b=overlap_b)
        overlap_a_pts = int(round(overlap_a * n_a))
        overlap_a_pts = int(min(max(overlap_a_pts, 2), n_a - 1))
        start_a = n_a - overlap_a_pts        # 曲线起点取自 path_a[start_a]
        a_intervals = overlap_a_pts - 1      # path_a 尾部窗口的间隔数

        # path_b 前面重叠窗口大小(比例, 0 < ratio <= 1, 默认 1/3) 与曲线终点
        if overlap_b is None:
            overlap_b = 1.0 / 3.0
        if not (math.isfinite(overlap_b) and 0.0 < overlap_b <= 1.0):
            print("smooth_two_path_2: overlap_b must be a ratio in (0, 1]")
            return self.smooth_two_path(path_a, path_b,
                                        overlap_a= overlap_a, overlap_b=overlap_b)
        overlap_b_pts = int(round(overlap_b * n_b))
        overlap_b_pts = int(min(max(overlap_b_pts, 2), n_b - 1))
        end_b = overlap_b_pts                # 曲线终点 = path_b[overlap_b_pts]; path_b[:overlap_b_pts] 被替换
        b_intervals = overlap_b_pts          # path_b 头部窗口的间隔数

        p0 = original_a[start_a].copy()
        p1 = original_b[end_b].copy()        # 曲线终点 = path_b 第 overlap_b 个点

        # 起点处(path_a)的位置/速度/加速度
        d0 = (original_a[start_a + 1] - original_a[start_a]) * a_intervals
        if start_a + 2 < n_a:
            dd0 = (original_a[start_a + 2] - 2.0 * original_a[start_a + 1]
                   + original_a[start_a]) * a_intervals ** 2
        else:
            dd0 = np.zeros(6, dtype=float)

        # 终点加速度不约束
        if end_b + 1 < n_b:
            downstream_step = original_b[end_b + 1] - original_b[end_b]
        else:
            downstream_step = original_b[end_b] - original_b[end_b - 1]
        if float(np.max(np.abs(downstream_step))) > max_joint_step_rad + 1e-12:
            print("smooth_two_path_2: path_b's downstream step at the blend end exceeds max_joint_step_rad; cannot blend")
            return self.smooth_two_path(path_a, path_b,
                                        overlap_a= overlap_a, overlap_b=overlap_b)
        dd1 = np.zeros(6, dtype=float)

        # 五次 Hermite 基函数
        def _build_curve(intervals):
            scale = float(intervals)
            curve = []
            for sid in range(intervals + 1):
                u = sid / scale
                u2, u3, u4, u5 = u * u, u ** 3, u ** 4, u ** 5
                h00 = 1.0 - 10.0 * u3 + 15.0 * u4 - 6.0 * u5
                h10 = u - 6.0 * u3 + 8.0 * u4 - 3.0 * u5
                h20 = 0.5 * (u2 - 3.0 * u3 + 3.0 * u4 - u5)
                h01 = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
                h11 = -4.0 * u3 + 7.0 * u4 - 3.0 * u5
                h21 = 0.5 * (u3 - 2.0 * u4 + u5)
                curve.append(
                    h00 * p0 + h10 * d0 + h20 * dd0
                    + h01 * p1 + h11 * d1 + h21 * dd1
                )
            curve[0] = p0.copy()
            curve[-1] = p1.copy()
            return curve

        # 估计初始细分数量, 使起点处步长不超限制(末端步长由 downstream_step 自身的限位保证)
        est_start = math.ceil(
            float(np.max(np.abs(original_a[start_a + 1] - original_a[start_a])))
            * a_intervals / max_joint_step_rad
        )
        curve_intervals = max(2, a_intervals, b_intervals, est_start)
        curve_parameter_scale = float(curve_intervals)
        d1 = downstream_step * curve_parameter_scale
        curve = None
        # 迭代细化, 保证曲线内部相邻关节步长满足限制
        max_refine = 12
        for _ in range(max_refine):
            curve = _build_curve(curve_intervals)
            max_step = 0.0
            for prev, curr in zip(curve[:-1], curve[1:]):
                s_ = float(np.max(np.abs(curr - prev)))
                if s_ > max_step:
                    max_step = s_
            if max_step <= max_joint_step_rad + 1e-12:
                break
            curve_intervals *= 2
            curve_parameter_scale = float(curve_intervals)
            d1 = downstream_step * curve_parameter_scale

        # 组装最终路径: 保留 a 的前段 + 曲线 + b 在 end_b 之后的部分
        # (曲线末点已是 b[end_b], 故 b[end_b:] 从 end_b+1 起接, 避免重复)
        blended = ([p.copy() for p in original_a[:start_a]] + curve
                   + [p.copy() for p in original_b[end_b + 1:]])

        # 合法性复核
        if any(not np.all(np.isfinite(p)) for p in blended):
            print("smooth_two_path_2: quintic blend produced invalid samples")
            return self.smooth_two_path(path_a, path_b,
                                        overlap_a= overlap_a, overlap_b=overlap_b)
        if not np.allclose(blended[0], original_a[0], atol=1e-12, rtol=0.0):
            print("smooth_two_path_2: 拼接段起点改了")   # 拼接段起点改了
            return self.smooth_two_path(path_a, path_b,
                                        overlap_a= overlap_a, overlap_b=overlap_b)
        if not np.allclose(blended[-1], original_b[-1], atol=1e-12, rtol=0.0):
            print("smooth_two_path_2: 拼接段终点改了") # 拼接段终点改了
            return self.smooth_two_path(path_a, path_b,
                                        overlap_a= overlap_a, overlap_b=overlap_b)
        if any(np.any(p < -2.0 * np.pi) or np.any(p > 2.0 * np.pi) for p in blended):
            print("smooth_two_path_2: 拼接段关节角超出范围") # 拼接段关节角超出范围
            return self.smooth_two_path(path_a, path_b,
                                        overlap_a= overlap_a, overlap_b=overlap_b)
        max_step = 0.0
        for prev, curr in zip(curve[:-1], curve[1:]):
            s_ = float(np.max(np.abs(curr - prev)))
            if s_ > max_step:
                max_step = s_   # 整条曲线里最大的单关节步长
        if max_step > max_joint_step_rad + 1e-12:   # 确认平滑曲线里相邻两点的关节步长没超过max_joint_step_rad
            print(f"smooth_two_path_2: blend joint step {np.degrees(max_step):.3f}deg exceeds limit")
            return self.smooth_two_path(path_a, path_b,
                                        overlap_a= overlap_a, overlap_b=overlap_b)

        # 平滑后碰撞检测
        if check_collision:
            hit = self._path_collision_idx(curve)
            if hit >= 0:    # 有碰撞
                print(f"smooth_two_path_2: 平滑后碰撞，改用smooth_two_path平滑")
                return self.smooth_two_path(path_a, path_b,
                                            overlap_a= overlap_a, overlap_b=overlap_b)
        return [np.array(p) for p in blended]

    # ============================== P0-2：抓取规划 ==============================
    def load_candidates(self, object_pose, pickle_path=None, jaw_width=None):
        """读 pickle，把物体系抓取位姿换算成世界系候选。

        Args:
            object_pose: shape (4,4) 物体位姿（sim 世界系，原点 = 底面中心），
                即 ``main.py`` 用 ``config.make_object_pose(位置, 旋转矩阵)`` 组装的结果。
            pickle_path: pickle 路径；``None`` = ``config.GRASP_PICKLE_PATH``
                （13 项，每项 ``(jaw_width, jaw_pos_local, jaw_rot_local, hnd_pos, hnd_rot)``，
                后两项是未填充的占位）。
            jaw_width: 覆盖 pickle 里的指尖全距（米）；``None`` = 沿用 pickle 的 0.031。

        Returns:
            list[dict]: 世界系候选（键 ``index`` / ``jaw_width`` / ``pos`` / ``rotmat``），
                顺序与 pickle 一致（13 个）。

        Raises:
            ValueError: object_pose 不是 4x4 / pickle 每项不足 3 元素 /
                jaw_rot 不是 3x3。

        Note:
            换算公式：``pos_world = R_obj @ pos_local + t_obj``、
            ``rot_world = R_obj @ rot_local``。pickle 里的 ``jaw_pos_local = [0, 0, 0.204]``
            是物体系里的抓取高度，不是世界系坐标。
        """
        pose = np.asarray(object_pose, dtype=float)
        if pose.shape != (4, 4):
            raise ValueError(f"object_pose 必须是 4x4，实际 shape={pose.shape}")
        rot_obj = pose[:3, :3]
        pos_obj = pose[:3, 3]

        path = Path(cfg.CONFIG["path"]["grasp_pickle_path"] if pickle_path is None else pickle_path)
        with open(path, "rb") as fp:
            raw = pickle.load(fp)
        if not isinstance(raw, (list, tuple)):
            raise ValueError(f"pickle 内容应是 list，实际 {type(raw).__name__}：{path}")

        candidates = []
        for idx, item in enumerate(raw):
            if len(item) < 3:
                raise ValueError(f"pickle 第 {idx} 项应至少 3 个元素"
                                 f"（jaw_width, jaw_pos, jaw_rot），实际 {len(item)}")
            width = float(item[0]) if jaw_width is None else float(jaw_width)
            pos_local = np.asarray(item[1], dtype=float).reshape(-1)
            rot_local = np.asarray(item[2], dtype=float)
            if pos_local.shape != (3,):
                raise ValueError(f"pickle 第 {idx} 项 jaw_pos 必须是 3 个数，"
                                 f"实际 shape={pos_local.shape}")
            if rot_local.shape != (3, 3):
                raise ValueError(f"pickle 第 {idx} 项 jaw_rot 必须是 3x3，"
                                 f"实际 shape={rot_local.shape}")
            candidates.append({
                "index": int(idx),
                "jaw_width": float(width),
                "pos": np.asarray(pos_obj + rot_obj @ pos_local,
                                  dtype=float).reshape(3),
                "rotmat": np.asarray(rot_obj @ rot_local, dtype=float).reshape(3, 3),
            })
        return candidates

    def solve_ik(self, candidate, seed_conf=None, check_collision=True):
        """解单个候选的抓取关节角（pyikfast 全解里挑第一个无碰撞的）。

        Args:
            candidate: dict（键 ``pos`` / ``rotmat``，世界系 TCP 位姿）。
            seed_conf: (6,) 参照构型（弧度）；``None`` = ``self.seed_conf``。
            check_collision: 是否用 ``self.obstacle_list`` 查碰撞（默认 True）。

        Returns:
            np.ndarray | None: (6,) 关节角；无解 / 全部解碰撞时返回 None。
        """
        seed = self.seed_conf if seed_conf is None else np.asarray(seed_conf, dtype=float)
        for conf in self._ikfast_all(candidate["pos"], candidate["rotmat"], seed=seed,
                                     max_solutions=3):
            if check_collision and self._conf_in_collision(conf):
                continue
            return conf
        return None


    def select_candidate(self, candidates, seed_conf=None, check_collision=True):
        """对全部候选解 IK + 查碰撞，按「q6 行程最小」排序取最优。

        Args:
            candidates: ``load_candidates`` 的输出。
            seed_conf: (6,) 行程基准构型（弧度）；``None`` = ``self.seed_conf``
                （默认 = ``config.START_CONF``，即机器人抓取前所在构型）。
            check_collision: 是否逐候选查碰撞（默认 True；关掉只用于离线测 IK）。

        Returns:
            tuple | None: ``(candidate dict, conf(6,), meta(dict), others(list))``；
            全部不可解时 None。``others`` 是 ``[(candidate, conf, q6_travel_deg), ...]``，
            按行程升序、最多 3 个，供 P1 抓取失败重试。
            ``meta`` 含 ``selected_index`` / ``conf_norm_rad``（离 seed 的总关节距离）
            / ``q6_travel_deg`` / ``pos_err_m`` /
            ``rot_z_err_deg`` / ``n_tried`` / ``n_usable`` /
            ``rejected``（[(index, reason)]）/ ``conf_deg``。

        """
        seed = np.asarray(self.seed_conf if seed_conf is None else seed_conf,
                          dtype=float).reshape(6)
        usable = []
        rejected = []

        for cand in candidates:
            sols = self._ikfast_all(cand["pos"], cand["rotmat"], seed=seed,
                                    max_solutions=3)
            if not sols:
                rejected.append((cand["index"], "ik"))
                continue
            best = None
            for conf in sols:
                if check_collision and self._conf_in_collision(conf):
                    continue
                # 实到 TCP 位姿 vs 目标：工具 z 轴夹角（夹爪接近轴，错 1° 就可能蹭壁）
                self.rbt_s.goto_given_conf(conf)
                tcp_pos, tcp_rot = self.rbt_s.fk(jnt_values=conf)
                tcp_rot = np.asarray(tcp_rot, dtype=float)
                z_dot = float(np.clip(tcp_rot[:, 2] @ cand["rotmat"][:, 2], -1.0, 1.0))
                z_err_deg = float(np.degrees(np.arccos(z_dot)))
                tcp_err_m = float(np.linalg.norm(np.asarray(tcp_pos, dtype=float) - cand["pos"]))
                rot_err = float(np.arccos(np.clip(
                    (np.trace(tcp_rot @ cand["rotmat"].T) - 1.0) / 2.0, -1.0, 1.0)))
                q6_travel = abs(float(self._wrap_pi(conf[5] - seed[5])))
                conf_norm = float(np.linalg.norm(conf - seed))
                # 同一候选可能有多支解：取**离 seed 总关节距离最短**的那支。
                # ⛔ 不能只看 q6 行程：实测远端支 q6 行程同样可以很小（0.47°），
                #    但整条路径要多转 ~3.9 rad，转移段 37 点变 158 点。
                best = (conf_norm, q6_travel, cand, conf,
                        tcp_err_m, rot_err, z_err_deg, tcp_err_m)
                # sols 已按「离 seed 的关节距离」升序：最近的一支无碰撞就定了，
                # 不必再对更远的支做碰撞检查（每次 ~5 ms，13 候选能省 100+ ms）
                break
            if best is None:
                rejected.append((cand["index"], "collision"))
                continue
            usable.append(best)

        if not usable:
            print("[select_candidate] 13 个候选全部不可用：{}".format(rejected))
            return None

        usable.sort(key=lambda row: (row[0], row[1]))
        conf_norm, q6_travel, cand, conf, pos_err, rot_err, z_err_deg, tcp_err_m = usable[0]
        meta = {
            "selected_index": cand["index"],
            "jaw_width": cand["jaw_width"],
            "q6_travel_deg": float(np.degrees(q6_travel)),
            "conf_norm_rad": conf_norm,
            "pos_err_m": float(pos_err),
            "rot_err_rad": float(rot_err),
            "rot_z_err_deg": z_err_deg,
            "tcp_err_m": tcp_err_m,
            "n_tried": len(candidates),
            "n_usable": len(usable),
            "rejected": rejected,
            "conf_deg": np.round(np.degrees(conf), 2).tolist(),
        }
        # 循环里用 goto_given_conf 查碰撞，机器人会停在「最后一个候选」上；
        # 复位到 seed，保证调用方拿到的 sim 状态与进入本函数前一致。
        self.rbt_s.goto_given_conf(seed)

        others = [
            (row[2], row[3], float(np.degrees(row[1])))
            for row in usable[1:1 + 3]
        ]
        return cand, conf, meta, others

    @staticmethod
    def _wrap_pi(angle):
        """把角度折到 (-π, π]，用于算两构型在同一关节上的最短行程。

        Args:
            angle: 角度（弧度），可为任意实数。

        Returns:
            float: 等价值，范围 (-π, π]。
        """
        return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi

    # ------------------------------------------------------------------ P0-3
    @contextlib.contextmanager
    def _jaw(self, width):
        """临时把 sim 夹爪开口设为 ``width``（米），退出时恢复原值。

        为什么需要：``is_collided`` 用的是**当前**手指几何，而 sim 构造时把开口设成
        ``jaw_range[0]``=0.016（全闭）。不同段实际开口不同（接近 0.041 / 夹持 0.031 /
        松爪 0.092），不显式设置就会拿最窄的手指去查碰撞 ⇒ 结论偏乐观。

        Args:
            width: 指尖全距（米），必须落在 ``jaw_range``=[0.016, 0.092] 内；
                越界时 ``change_jaw_width`` 抛 ValueError。

        Yields:
            None: 上下文内 ``rbt_s.end_effector`` 的开口 = ``width``。

        Note:
            机器人没有夹爪（``end_effector`` 为 None）时什么也不做，直接 yield。
        """
        hnd = getattr(self.rbt_s, "end_effector", None)
        if hnd is None:
            yield
            return
        prev = float(hnd.get_jaw_width())
        hnd.change_jaw_width(float(width))
        try:
            yield
        finally:
            hnd.change_jaw_width(prev)

    def _plan_approach_segment(self, candidate, conf, seed_conf):
        """为一个已解出抓取构型的候选生成接近段。

        Args:
            candidate: dict（键 ``pos`` / ``rotmat``，世界系 TCP 位姿）。
            conf: (6,) 该候选的抓取关节角（弧度，已折回限位）。
            seed_conf: (6,) 接近点的 IK 种子，一般是当前构型（如 ``cfg.CONFIG["joints"]["start_conf"]``）；
                种子决定落到哪一支解，所以必须跟执行时的来路一致。

        Returns:
            dict | None: 键 ``q_approach``(6,) / ``approach_path``(list) /
            ``path_ms``(float)；接近点无 IK 解或接近段有路点碰撞时返回 None
            （调用方换下一个候选）。

        Note:
            抬升段不在这里：抬升高度要按「物体最低点过壁」算，而物体位姿是
            ``plan_place`` 的入参；抬起 + 搬运 + 下放在 ``plan_place`` 里拼成一条。
            段是**笛卡尔直线 + 逐点 pyikfast**，姿态恒等于抓取候选姿态，所以手爪沿
            ``-rotmat[:, 2]``（远离物体方向）进出，不会一边转一边扎。
        """
        t0 = time.perf_counter()
        pos = np.asarray(candidate["pos"], dtype=float).reshape(3)
        rot = np.asarray(candidate["rotmat"], dtype=float).reshape(3, 3)
        retreat = -rot[:, 2]  # 单位向量：从物体指向夹爪外侧

        appr_pos = pos + retreat * float(cfg.CONFIG["motion"]["approach_distance_m"])

        # 接近时手指开到 APPROACH_WIDTH_FULL（比抓取宽 1 cm），碰撞也按这个开口算
        with self._jaw(cfg.approach_width_full()):
            q_appr = self._safe_ik(appr_pos, rot, seed=seed_conf)
            if q_appr is None:
                print("[segments] 候选 #{} 接近点无 IK 解（pos={}）".format(
                    candidate["index"], np.round(appr_pos, 4).tolist()))
                return None
            approach_path = self.get_line_path(seed=q_appr, start=appr_pos, goal=pos, rot=rot)
        if approach_path is None:
            return None

        approach_path = [np.asarray(p, dtype=float).reshape(6) for p in approach_path]
        conf = np.asarray(conf, dtype=float).reshape(6)
        # 接缝吸附：段首末点是同一位姿的两次 IK，实测差 2e-4 rad；不吸附执行层会
        # 在段边界跳一下（也是「点数/索引对不上」这类坑的源头）。
        approach_path[0] = np.array(q_appr, dtype=float).reshape(6)
        approach_path[-1] = conf.copy()

        return {
            "q_approach": approach_path[0].copy(),
            "approach_path": approach_path,
            "path_ms": (time.perf_counter() - t0) * 1000.0,
        }

    def _direct_path(self, start, goal, step=None):
        """关节空间直线插值（含首末点，步长 <= step）。

        Args:
            start: (6,) 起点关节角（弧度）。
            goal: (6,) 终点关节角（弧度）。
            step: 相邻点最大关节差（弧度）；``None`` = ``cfg.CONFIG["motion"]["collision_step_rad"]``。

        Returns:
            list[(6,)] 路径；首点严格等于 ``start``、末点严格等于 ``goal``
            （``i=0`` / ``i=n`` 直接取端点，不做浮点累加）。
        """
        start = _as_conf(start, "direct.start")
        goal = _as_conf(goal, "direct.goal")
        step = float(cfg.CONFIG["motion"]["collision_step_rad"] if step is None else step)
        span = float(np.max(np.abs(goal - start)))
        n = max(int(np.ceil(span / step)), 1)
        return [start + (goal - start) * (i / n) for i in range(n + 1)]


    def _transfer(self, start, goal, name, prefer_direct=True, jaw_width=None):
        """两点之间的转移段：优先关节直连，碰撞才退 RRT。碰撞**只查这一次**。

        Args:
            start: (6,) 起点关节角（弧度）。
            goal: (6,) 终点关节角（弧度）。
            name: 段名（``SEGMENT_ORDER`` 之一，决定默认查碰撞开口与日志前缀）。
            prefer_direct: True = 先试直连（实测 26 ms vs RRT 2050 ms）。
            jaw_width: float | None 查碰撞用的指尖全距（米）；``None`` =
                ``segment_jaw_width(name)``。

        Returns:
            tuple[list, str, int]: ``(path, mode, collision_idx)``。``mode`` 属于
            ``direct`` / ``rrt`` / ``direct-collided``（RRT 也失败，这里仍返回带碰撞的
            直连并把下标带出去 —— ⛔ 执行层不再有门禁，调用方需自己看这个下标）。
        """
        jaw = segment_jaw_width(name) if jaw_width is None else float(jaw_width)
        direct = self._direct_path(start, goal)
        hit = -1
        if prefer_direct:
            hit = self._path_collision_idx(direct, jaw_width=jaw)
            if hit < 0:
                return direct, "direct", -1
            print("[plan] {} 直连在路点 {} 处碰撞，改走 RRT".format(name, hit))
        path = self.plan_path(start, goal, step=0.07)
        if path is None:
            if hit < 0:                      # prefer_direct=False 时也得算一次
                hit = self._path_collision_idx(direct, jaw_width=jaw)
            print("[plan] {} 的 RRT 也失败，返回带碰撞的直连（⚠️ 无门禁，会照跑）".format(name))
            return direct, "direct-collided", hit
        path = [_as_conf(p, name) for p in path]
        return path, "rrt", self._path_collision_idx(path, jaw_width=jaw)

    def _return_transfer(self, return_path, q_start, q_approach):
        """把「放置点 → 抓取起点」固定路径与「起点 → 接近点」转移段拼成一条。

        第 2 轮起机器人停在放置构型，``to_pick_above`` 必须先沿 config 的
        ``place_to_start`` 回抓取起点，再走转移段到接近点。两段用
        ``smooth_two_path_2(check_collision=True)`` 平滑拼接；该函数在曲线
        碰撞时退到 ``smooth_two_path``（该回退**不查碰撞**），所以这里对整条
        结果再显式跑一次 ``_path_collision_idx``。

        Args:
            return_path: list[(6,)] 固定回程路径（每点弧度），首点 = 上一轮
                放置构型，末点 = 抓取起点 ``start_conf``。
            q_start: (6,) 抓取起点关节角（弧度），用来核对 ``return_path``
                末点是否漂移（偏差 > 1e-6 直接拒绝，防 config 改了没同步）。
            q_approach: (6,) 接近点关节角（弧度），拼接路径的终点。

        Returns:
            tuple[list, int] | None: ``(路径, -1)`` —— 路径首点强制 =
            ``return_path[0]``、末点强制 = ``q_approach``；``-1`` = 全程无
            碰撞。回程末点不符 / 转移段碰撞 / 平滑与回退后仍碰撞时返回
            ``None``（调用方换候选，绝不把带碰撞的路径交给执行层）。
        """
        prev_q = _as_conf(return_path[0], "return_path 首点")
        end_q = _as_conf(return_path[-1], "return_path 末点")
        if not np.allclose(end_q, q_start, atol=1e-6):
            print("[plan_grasp] 回程路径末点 != 抓取起点，拒绝拼接")
            return None
        q_appr = _as_conf(q_approach, "q_approach")
        seg, _, seg_hit = self._transfer(q_start, q_appr, "to_pick_above")
        if seg_hit >= 0:
            print("[plan_grasp] 回程后的转移段在第 {} 点碰撞，换候选".format(seg_hit))
            return None
        jaw = segment_jaw_width("to_pick_above")
        with self._jaw(jaw):
            merged = self.smooth_two_path_2(return_path, seg, overlap_a=0.5,overlap_b=1.0, check_collision=True)
            if merged is None:
                merged = self._concat_paths(return_path, seg)
            hit = self._path_collision_idx(merged, jaw_width=jaw)
            if hit >= 0:
                merged = self._concat_paths(return_path, seg)
                hit = self._path_collision_idx(merged, jaw_width=jaw)
        if hit >= 0:
            print("[plan_grasp] 回程拼接后仍碰撞（下标 {}），换候选".format(hit))
            return None
        merged = [np.asarray(p, dtype=float).reshape(6) for p in merged]
        merged[0] = prev_q
        merged[-1] = q_appr
        return merged, hit


    def check_object_clearance(self, q_grasp, object_pose, segments, margin=0.0):
        """检查被夹物体在给定段上不会撞箱壁（机器人的碰撞模型里没有物体）。

        ⛔ 只算**壁体正上方**那圈（``外廓投影 & ~内腔投影``）：抓取点和抬升段整段都在
        箱内腔里，拿「外廓投影内就比壁顶」会把箱内的正常姿态全判成穿壁（实测 lift
        12 处误报、最小 −0.2450 m）。内腔尺寸见 ``obstacle.box_inner_bounds()``。
        不判底板：抓取点角点最低正好等于底板顶（0.006 m），贴底是正常工况，且底板
        已被机器人自身碰撞体覆盖。

        Args:
            q_grasp: (6,) 抓取关节角（弧度）。用它 ``fk`` 出来的 TCP 位姿把物体 bbox
                的 8 个角点定到 TCP 系（刚性随动，整段不变）。
            object_pose: (4,4) 抓取时的物体位姿（sim 世界系，原点 = 底面中心）。
            segments: list[dict] 要查的段，一般是 ``[place_plan["place"]]``。
            margin: 要求物体最低点高出壁顶的额外余量（米），默认 0 = 只要不穿壁。

        Returns:
            tuple[list, float]: ``(violations, min_clearance)``。
            ``violations`` 为 ``[(段内下标, 箱局坐标(3,)), ...]``；
            ``min_clearance`` = 全程物体角点落在「壁体正上方那圈」时相对壁顶的最小
            高度差（米，负数=穿壁）；没有任何角点落在那一圈时为 ``inf``。
        """
        corners_world = obstacle.object_corners_world(np.asarray(object_pose).reshape(4, 4))
        tcp_pos, tcp_rot = self.rbt_s.fk(jnt_values=_as_conf(q_grasp, "clearance"))
        tcp_pos = np.asarray(tcp_pos, dtype=float).reshape(3)
        tcp_rot = np.asarray(tcp_rot, dtype=float).reshape(3, 3)
        corners_tcp = (tcp_rot.T @ (corners_world - tcp_pos).T).T  # 8x3，刚性随动

        box_inv = np.linalg.inv(cfg.load_box_homomat())
        outer = obstacle.box_outer_bounds()
        inner = obstacle.box_inner_bounds()
        wall_top = obstacle.box_wall_top_local_z()
        violations = []
        min_clear = float("inf")
        for seg in segments:
            for i, q in enumerate(seg["path"]):
                pos, rot = self.rbt_s.fk(jnt_values=_as_conf(q, "clearance"))
                pts = (np.asarray(rot, float) @ corners_tcp.T).T + np.asarray(pos, float)
                local = (box_inv[:3, :3] @ pts.T).T + box_inv[:3, 3]
                over = ((local[:, 0] >= outer[0, 0]) & (local[:, 0] <= outer[1, 0])
                        & (local[:, 1] >= outer[0, 1]) & (local[:, 1] <= outer[1, 1]))
                inside = ((local[:, 0] >= inner[0, 0]) & (local[:, 0] <= inner[1, 0])
                          & (local[:, 1] >= inner[0, 1]) & (local[:, 1] <= inner[1, 1]))
                ring = over & ~inside          # 只看正好压在壁体上方的那圈
                if not np.any(ring):
                    continue
                low = float(local[ring, 2].min())
                min_clear = min(min_clear, low - wall_top)
                if low < wall_top + float(margin):
                    violations.append(
                        (i, local[ring][int(np.argmin(local[ring, 2]))]))
        return violations, min_clear


    def _held_collision_models(self, box_homomat=None):
        """取持物碰撞检查用的模型对：瓶子 + 箱四壁（实例级惰性缓存）。

        Args:
            box_homomat: (4,4) 箱位姿（sim 世界系）。``None`` = ``cfg.load_box_homomat()``
                并复用 ``self._held_models``；传非 None 时**强制重建**（换箱子位姿的场景）。

        Returns:
            dict: ``{"bottle": CollisionModel, "walls": CollisionModel}``。瓶子位姿由
            调用方逐点写 ``homomat``；四壁位姿在构造时就已固定。
        """
        if self._held_models is None or box_homomat is not None:
            self._held_models = {
                "bottle": obstacle.make_object_obstacle(),
                "walls": obstacle.make_box_wall_collision_obstacle(
                    cfg.load_box_homomat() if box_homomat is None
                    else np.asarray(box_homomat, dtype=float).reshape(4, 4)),
            }
        return self._held_models


    def check_held_collision_idx(self, q_grasp, object_pose, path):
        """沿路径查「被夹瓶子会不会碰箱壁」，返回首个碰撞路点下标。

        为什么必须单独查：``rbt_s.is_collided`` 只查机器人自己的 cdprim，手里那瓶
        瓶子不在里面；而放置全程都夹着它，瓶底离 TCP 204 mm，蹭壁最先发生在瓶底。

        Args:
            q_grasp: (6,) 抓取关节角（弧度）。用它 ``fk`` 出来的 TCP 位姿定义
                「瓶子相对 TCP」的刚性变换。
            object_pose: (4,4) 抓取时的物体位姿（sim 世界系，原点 = 底面中心）。
            path: list[(6,)] 待查关节路径（弧度）。

        Returns:
            int: 首个碰撞路点下标；全程无碰撞返回 -1。

        Note:
            检查对象只有**箱四壁**（``obstacle.make_box_wall_collision_obstacle``）：
            底板不查是因为抓取点瓶底就贴在上面（``box_inner_bounds()`` 的 ``z_min``
            = 0.006 m），查了恒报；桌面不查是因为瓶子最后就落在桌面上，相机不查
            是因为它在箱外远端。
            ``CollisionModel.is_pcdwith`` 的返回值 bool / tuple 不一致，这里统一归一化。
            实测 0.183 ms/点（含 ``fk`` 与 ``homomat`` 赋值；只算 ``is_pcdwith`` 是
            0.04 ms/点），比 8 角点法的 0.24 ms/点还快。
        """
        if path is None or len(path) == 0:
            return -1
        models = self._held_collision_models()
        bottle = models["bottle"]
        walls = models["walls"]
        R_tb, t_tb = object_to_tcp(np.asarray(object_pose, dtype=float).reshape(4, 4),
                                   *self.rbt_s.fk(jnt_values=_as_conf(q_grasp, "held")))
        for idx, q in enumerate(path):
            tcp_pos, tcp_rot = self.rbt_s.fk(jnt_values=_as_conf(q, "held"))
            tcp_pos = np.asarray(tcp_pos, dtype=float).reshape(3)
            tcp_rot = np.asarray(tcp_rot, dtype=float).reshape(3, 3)
            bottle.set_homomat(_homomat(tcp_pos + tcp_rot @ t_tb, tcp_rot @ R_tb))
            if bottle.is_pcdwith(walls):
                return idx
        return -1

    def _refine_place_with_gomp(self, path, q_grasp, object_pose):
        """对已无碰撞的 place 路径做 GOMP 精修（带瓶身持物约束），失败则原样返回。

        Args:
            path: list[(6,)] 已平滑且已通过本体/持物检查的关节路径（弧度）。
                本函数**不修改**它；回退时返回它的逐元素副本。
            q_grasp: (6,) 抓取关节角（弧度）。用它 ``fk`` 出的 TCP 位姿算
                「瓶子相对 TCP」的刚性变换（抓取那一刻测一次，全程刚体）。
            object_pose: (4,4) 抓取那一刻的物体位姿（sim 世界系，原点 = 底心）。

        Returns:
            list[(6,)]: 精修后的路径（GOMP 轨迹的 waypoints，通常 201 点）。
            以下任一情况回退为入参 path 的副本：SQP 返回 None、GOMP 的 validate
            判不过、项目自有的瓶身/本体判据查出碰撞。

        Note:
            端点**不回写**：SQP 的端点位姿残差实测 Δpos ≈ 1.0 mm / Δrot ≈ 1.0e-2，
            相对 6 cm 过壁余量可忽略；强行回写关节角会破坏末端速度连续性。
            持物几何必须是 5 元组 groups ``(bottom, top, fp, hx, hy)``：只给 3 元组
            会退化成纯 2.5D 单点代理，看不见侧向蹭壁（实测漏检 → 瓶身碰壁下标 39）。
        """
        # 延迟 import：只有真的精修时才加载 gomp，模块 import 保持轻量。
        import gomp_refine
        from gomp.obstacles.attached_object import (AttachedObject,
                                                    read_boxes_from_cdprim)

        if self._gomp is None:
            self._gomp = gomp_refine.GompRefiner(self.rbt_s, verbose=False)

        grasp_pos, grasp_rot = self.rbt_s.fk(jnt_values=q_grasp)
        R_tb, t_tb = object_to_tcp(object_pose, grasp_pos, grasp_rot)

        # 瓶身 12 盒 → 5 元组组（bottom/top/fp/hx/hy），合并规则与
        # merge_box_groups 一致（z 相邻 + footprint 半径相对差 ≤ 0.12），
        # 但额外保留 hx/hy 供 SQP 用「底面 4 角 + 中心」5 点采样高度场。
        boxes = read_boxes_from_cdprim(obstacle.get_object_sample_cdprim())
        groups = []
        for center, half in sorted(boxes, key=lambda b: b[0][2]):
            bottom, top = center[2] - half[2], center[2] + half[2]
            fp = float(np.linalg.norm(half[:2]))
            if groups and abs(groups[-1][1] - bottom) < 1e-6 \
                    and abs(fp - groups[-1][2]) <= groups[-1][2] * 0.12:
                g = groups[-1]
                groups[-1] = (g[0], top, max(g[2], fp),
                              max(g[3], float(half[0])), max(g[4], float(half[1])))
            else:
                groups.append((bottom, top, fp, float(half[0]), float(half[1])))
        attached = AttachedObject(groups, t_tb, R_tb)

        raw_path = np.asarray(path, dtype=float).reshape(-1, 6)
        traj = self._gomp.refine(raw_path, held_object=attached)
        if traj is None:
            print("[plan_place] GOMP 精修失败，保留平滑路径")
            return [p.copy() for p in path]

        report = self._gomp.validate(
            traj, self.obstacle_list,
            held=(obstacle.make_object_obstacle(), t_tb, R_tb))
        new_path = self._gomp.traj_to_path(traj)
        new_path = [np.asarray(p, dtype=float).reshape(6) for p in new_path]
        if not report["is_valid"]:
            print("[plan_place] GOMP 轨迹校验不过（本体 {} / 持物 {}），保留平滑路径".format(
                len(report["collisions"]), len(report["collisions_held"])))
            return [p.copy() for p in path]
        hit = self._path_collision_idx(
            new_path, jaw_width=cfg.CONFIG["gripper"]["grasp_width_full"])
        held_hit = self.check_held_collision_idx(q_grasp, object_pose, new_path)
        if hit >= 0 or held_hit >= 0:
            print("[plan_place] GOMP 轨迹项目判据不过（本体 {} / 瓶身 {}），保留平滑路径".format(
                hit, held_hit))
            return [p.copy() for p in path]

        print("[plan_place] 放置路径GOMP 精修：{} → {} 点，{:.3f} s，本体/持物校验通过".format(
            len(path), len(new_path), float(traj.duration)))
        return new_path

    def _refine_grasp_with_gomp(self, path, q_first, q_last):
        """对第 2 轮起的 `to_pick_above`（回程+转移平滑拼接）做 GOMP 精修。

        与 ``_refine_place_with_gomp`` 的区别：**不夹持瓶子**，但把张开 0.041
        的手爪外接盒作为 attached 盒传给 SQP（优化阶段即按此保守包络避障，
        消掉 validate 阶段才暴露的指尖蹭箱壁）；端点强制回写（首=上一轮
        place_conf / 末=q_approach）。validate 仍用真实指尖 cdelement 校验。

        Args:
            path: list[(6,)] 已平滑拼接、已通过本体碰撞检查的关节路径（弧度）。
                本函数**不修改**入参；回退时返回它的逐元素副本。
            q_first: (6,) 回写后的首点（= ``return_path[0]`` = place_conf，弧度）。
            q_last: (6,) 回写后的末点（= q_approach 接近点，弧度）。

        Returns:
            list[(6,)]: 精修后的路径（GOMP waypoints，通常 201 点，首末点已
            强制回写）。SQP 返回 None / GOMP validate 判不过 / 项目自有本体
            判据查出碰撞时，回退为入参 path 的逐元素副本。
        """
        if self._gomp is None:
            self._gomp = gomp_refine.GompRefiner(self.rbt_s, verbose=False)
        # 抓取段手爪外接盒：运行时读 dh76 新增的 _gomp_finger_cdprim 几何，
        # SQP 优化阶段即按"张开 0.041 的外接盒"避障（比真实指尖大），消掉
        # validate 阶段才暴露的指尖蹭箱壁。盒心相对 TCP 偏移 -0.0255 m 是
        # probe 实测（盒心 gripper 系 z=0.178、TCP z=0.2035；二者同受
        # coupling 变换，差值恒定）。不夹持瓶子，validate 仍用真实指尖。
        # 本机扁平 wrs 的 dh76 没有这个外接盒，几何复刻在 obstacle 里（逐项同上游）。
        _, _box_half = read_boxes_from_cdprim(obstacle.get_gomp_finger_cdprim())[0]
        _dz = -0.0255
        _b = _dz - float(_box_half[2])
        _t = _dz + float(_box_half[2])
        _fp = float(np.linalg.norm(_box_half[:2]))
        gripper_box = AttachedObject(
            [(_b, _t, _fp, float(_box_half[0]), float(_box_half[1]))],
            tcp_pos=np.zeros(3), tcp_rotmat=np.eye(3),
            name="gripper_open_041")
        jaw = segment_jaw_width("to_pick_above")
        raw_path = np.asarray(path, dtype=float).reshape(-1, 6)
        traj = self._gomp.refine(raw_path, held_object=gripper_box)
        if traj is None:
            print("[plan_grasp] GOMP 精修失败，保留平滑拼接路径")
            return [p.copy() for p in path]
        report = self._gomp.validate(
            traj, self.obstacle_list, held=None, exempt_frac=(0.0, 0.0))
        new_path = [np.asarray(p, dtype=float).reshape(6)
                    for p in self._gomp.traj_to_path(traj)]
        if not report["is_valid"]:
            print("[plan_grasp] GOMP 轨迹本体校验不过（{}），保留平滑拼接路径".format(
                len(report["collisions"])))
            return [p.copy() for p in path]
        # 端点回写（先回写再判据，判的是最终要下发的那条路径）。
        new_path[0] = np.asarray(q_first, dtype=float).reshape(6).copy()
        new_path[-1] = np.asarray(q_last, dtype=float).reshape(6).copy()
        hit = self._path_collision_idx(new_path, jaw_width=jaw)
        if hit >= 0:
            print("[plan_grasp] GOMP 轨迹项目判据不过（本体 {}），保留平滑拼接路径".format(hit))
            return [p.copy() for p in path]
        print("[plan_grasp] 抓取路径GOMP 精修：{} → {} 点，{:.3f} s，本体校验通过（端点回写）".format(
            len(path), len(new_path), float(traj.duration)))
        return new_path

    def _blend_grasp_paths(self, pre_path, approach_path):
        """把转移段与接近段小窗拼接成一条 grasp 路径（抹平接缝折角）。

        两段各自已逐点查过碰撞。GOMP 精修只回写端点、不约束末端切线，末端
        可能与 approach 竖直流下扎形成近折返角（实测 raw 最大 124.2°）。本函数
        用 ``_concat_paths`` 去掉重复衔接点，再用 ``_blend_joint_window`` 在
        接缝处做局部 Hermite 抹平（护栏 130°，覆盖实测最大 124.2° 的 raw 折角，
        同时保留对 176° 原路折返的拒绝），最后按 **全开 0.092** 对整条路径复核
        碰撞（并行调爪期间爪宽从上一状态过渡到 0.041，首瓶甚至从全开起步，
        故取最保守口径）。

        历史：曾用 ``_align_grasp_tail`` 先把末端切线扭到 approach 方向再抹平，
        但 Hermite 重采样会在 pre 末段制造随窗口长度增长的 TCP 笛卡尔上移
        （tail_n=10 时 8 mm、30 时 26 mm）。改为直接 concat + 抬护栏后，上移
        归零，9 瓶实测接缝角抹到 13–54°、碰撞全部 -1。

        Args:
            pre_path: list[(6,)] 转移段（首轮 direct / 第 2 轮起回程+平滑+GOMP），
                弧度；末点应等于 ``approach_path`` 首点。
            approach_path: list[(6,)] 竖直接近段（弧度），来自
                ``_plan_approach_segment`` 返回的 ``approach_path``。

        Returns:
            list[(6,)] | None: 拼接（并尽可能抹平接缝）后的路径，首末点严格
            等于 ``pre_path[0]`` / ``approach_path[-1]``；抹平护栏不过或抹平后
            碰撞时回退 raw concat（仍无碰撞）；raw concat 本身碰撞才返回 None
            （该候选不可用，由 ``plan_grasp`` 换下一个候选）。
        """
        merged = self._concat_paths(pre_path, approach_path)
        seam = len(pre_path) - 1
        stats = {}
        jaw = float(cfg.CONFIG["gripper"]["open_width_full"])
        with self._jaw(jaw):
            res = self._blend_joint_window(
                merged, seam, half_window=2, max_move=0.30,
                max_turn_deg=130.0, stats=stats)
            if res is not None:
                lo, hi, curve = res
                trial = (merged[:lo]
                         + [np.asarray(p, dtype=float).reshape(6) for p in curve]
                         + merged[hi:])
                hit = self._path_collision_idx(trial, jaw_width=jaw)
                if hit < 0:
                    print("[plan_grasp] grasp 拼接抹平 OK：{} → {} 点，"
                          "接缝折角 {:.2f}° → {:.2f}°，最大位移 {:.2f}°".format(
                              len(merged), len(trial),
                              stats["blend_turn_deg_before"],
                              stats["blend_turn_deg_after"],
                              stats["blend_max_move_deg"]))
                    return trial
                print("[plan_grasp] grasp 抹平后碰撞（{}），回退 raw concat".format(hit))
            else:
                print("[plan_grasp] grasp 抹平护栏不过（{}），回退 raw concat".format(
                    stats.get("blend_reason", "unknown")))
        # raw concat 两段原本各自无碰撞，但拼接点相邻关系是新引入的，复核一次。
        hit = self._path_collision_idx(merged, jaw_width=jaw)
        if hit >= 0:
            print("[plan_grasp] grasp raw concat 在 {} 点碰撞，该候选不可用".format(hit))
            return None
        return merged

    def plan_grasp(self, object_pose, candidates=None, seed_conf=None,
                   check_collision=True, return_path=None):
        """抓取规划门面：物体位姿 -> 抓取构型 + 接近段。

        Args:
            object_pose: (4,4) 物体位姿（sim 世界系，原点 = 底面中心）。
            candidates: 已算好的候选列表；``None`` = 内部调 ``load_candidates``。
            seed_conf: (6,) IK 种子 / q6 行程基准（弧度）；``None`` = ``self.seed_conf``。
            check_collision: 是否逐候选查碰撞（默认 True）。
            return_path: list[(6,)] | None 上一轮「放置点 → 抓取起点」的固定
                回程路径（= ``cfg.CONFIG["move_path"]["place_to_start"]``，
                首点 place_conf、末点 start_conf，弧度）。``None`` = 首轮，
                转移段从 ``seed_conf`` 直达接近点；非 None = 第 2 轮起，
                回程与转移段平滑拼成一条 ``to_pick_above`` 并查碰撞，
                失败换候选。

        Returns:
            dict | None: 抓取规划结果，键见函数末尾的 dict 字面量——
            ``candidate`` / ``conf`` / ``object_pose``，外加一条以段名为键的段
            ``grasp``（= ``to_pick_above`` + ``approach`` 小窗拼接并抹平接缝，
            dict，字段见模块顶部说明）。
            主选及前 3 个备选的接近段或转移段都不可行时
            返回 None（带碰撞的路径不会返回给调用方）。

        Note:
            抬升段归 ``plan_place``：抬升高度要按「物体最低点过壁」算，还要与搬运、
            下放拼成一条。原因见 ``cfg.CONFIG["motion"]["lift_distance_m"]`` 的注释 —— 瓶子握在瓶口，
            整只瓶身垂在夹爪下方 204 mm，按「手爪本体」算的 0.15 m 会让瓶底蹭着
            壁顶拖过去。
        """
        seed = np.asarray(self.seed_conf if seed_conf is None else seed_conf, dtype=float)
        if candidates is None:
            candidates = self.load_candidates(object_pose)
        picked = self.select_candidate(candidates, seed_conf=seed,
                                       check_collision=check_collision)
        if picked is None:
            return None
        cand0, conf0, meta0, others = picked

        trials = [(cand0, np.asarray(conf0, dtype=float), meta0)]
        trials += [(c, np.asarray(q, dtype=float), {"q6_travel_deg": float(travel)})
                   for c, q, travel in others]
        rejected = []
        for cand, conf, meta in trials:
            segs = self._plan_approach_segment(cand, conf, seed)
            if segs is None:
                rejected.append((cand["index"], "approach 不可行"))
                continue
            meta = dict(meta)
            meta["approach_points"] = len(segs["approach_path"])
            print("[plan_grasp] 选中候选 #{}：q6 行程 {:.2f}°，位置误差 {:.2e} m，"
                  "工具 z 轴误差 {:.4f}°（{} 个候选 / {} 个可用），"
                  "接近段 {} 点".format(
                      cand["index"], meta.get("q6_travel_deg", float("nan")),
                      meta.get("pos_err_m", float("nan")),
                      meta.get("rot_z_err_deg", float("nan")),
                      meta.get("n_tried", len(trials)), meta.get("n_usable", len(trials)),
                      meta["approach_points"]))

            # 起始构型 → 接近点：首轮直接从抓取起点转移；第 2 轮起先走 config
            # 的 place_to_start 回程再接转移段，平滑拼成一条 to_pick_above。
            q_appr = np.asarray(segs["q_approach"], dtype=float).reshape(6)
            if return_path is None:
                pre_path, pre_mode, pre_hit = self._transfer(seed, q_appr,
                                                             "to_pick_above")
            else:
                blended = self._return_transfer(return_path, seed, q_appr)
                if blended is None:
                    rejected.append((cand["index"], "回程拼接不可行"))
                    print("[plan_grasp] 候选 #{} 的回程拼接不可行，换下一个".format(
                        cand["index"]))
                    continue
                pre_path, pre_hit = blended
                pre_mode = "return+smooth"
            print("[plan_grasp] to_pick_above {} 点 [{}]，碰撞 {}".format(
                len(pre_path), pre_mode, pre_hit))
            if pre_hit >= 0:
                # 直连撞、RRT 也撞 ⇒ 该候选的转移段不可行，换下一个候选重规划。
                rejected.append((cand["index"],
                                 "to_pick_above 在 {} 点碰撞".format(pre_hit)))
                print("[plan_grasp] 候选 #{} 的转移段不可行，换下一个".format(
                    cand["index"]))
                continue
            # 第 2 轮起：回程+转移的平滑拼接路径做无持物 GOMP 精修，端点回写。
            # 首轮 direct 不精修（用户要求）；失败在函数内回退，不影响候选可用性。
            if pre_mode == "return+smooth":
                pre_path = self._refine_grasp_with_gomp(
                    pre_path, pre_path[0], q_appr)
            grasp_path = self._blend_grasp_paths(pre_path, segs["approach_path"])
            if grasp_path is None:
                rejected.append((cand["index"], "grasp 拼接碰撞"))
                print("[plan_grasp] 候选 #{} 的 grasp 拼接碰撞，换下一个".format(
                    cand["index"]))
                continue
            grasp_dict = {
                "candidate": cand,
                "conf": conf,
                "object_pose": np.asarray(object_pose, dtype=float).reshape(4, 4),
                # 转移段 + 接近段小窗拼接成一条 grasp：段间不停车，速度更快。
                # approach 在 _plan_approach_segment 里已由 _safe_ik 逐点查碰撞；
                # 拼接后 _blend_grasp_paths 又按全开 0.092 全路径复核（并行调爪
                # 期间爪宽过渡，取最保守口径）。段以段名为键直挂 plan。
                "grasp": {
                    "path": grasp_path,
                    "action_after": _SEGMENT_ACTION["grasp"],
                    "jaw_full": _SEGMENT_JAW_WIDTH["grasp"],
                },
            }
            return grasp_dict
        print("[plan_grasp] {} 个候选的接近段都不可行：{}".format(len(trials), rejected))
        return None


    def plan_place(self, q_grasp, object_pose, place_conf=None, approach_distance=None,
                   check_object=True):
        """放置规划：抬起 → 搬运 → 下放，三段**拼成一条**平滑路径 `place`。

        Args:
            q_grasp: (6,) 抓取关节角（弧度）。抬升段起点就是它；它 ``fk`` 出来的
                TCP 位姿与 ``plan_grasp`` 的 ``candidate["pos"] / ["rotmat"]``
                实测只差 1.1e-07 m（IK 残差），所以这里不必再收 candidate。
            object_pose: (4,4) **抓取那一刻**的物体位姿（sim 世界系，原点 = 底面
                中心）。两个用途：算「物体最低点过壁」所需的最小抬升；算「瓶子相对
                TCP」的刚性位姿，供持物碰撞检查逐点还原瓶身。
            place_conf: (6,) 放置关节角（弧度）；``None`` = ``cfg.CONFIG["joints"]["place_conf"]``。
                放置目标是**固定关节角**（用户已确认），所以这一层不需要 IK 选解，
                只需要把「竖直进出」补出来。
            approach_distance: 垂直下放距离（米）；``None`` = ``cfg.CONFIG["motion"]["place_approach_distance_m"]``。
            check_object: bool 是否做持物检查（瓶身 vs 箱四壁 + 8 角点过壁诊断），
                默认 True。只有调参/画图这类不执行的场景才该设 False。

        Returns:
            dict | None: 键见 ``place_plan`` 的模块级说明——只有 ``place``（**单段**
            dict）与 ``object_violations``（8 角点过壁诊断）。抬升点或上方点无 IK 解、
            放置构型折不回限位、平滑后与回退的未平滑拼接都碰壁（瓶身或机器人本体）
            时返回 None（带碰撞的路径不会返回给调用方）。

        Note:
            三段在规划期拼成一条理由：执行层每调一次 ``move_path`` 就做一次 TOPPRA
            参数化，每段都从 0 速起、0 速止 ⇒ 接缝处 stop-and-go。拼成一条后只有
            一次参数化，接缝只是折角，不停顿。
            下放方向取**世界 +z**，不是工具轴：``cfg.CONFIG["joints"]["place_conf"]`` 的工具 z 轴实测
            [-0.018, 0.020, -1.0]，沿工具轴抬升等于又往下扎。
        """
        q_grasp = _as_conf(q_grasp, "plan_place.q_grasp")
        obj = np.asarray(object_pose, dtype=float).reshape(4, 4)
        q_place = np.asarray(cfg.CONFIG["joints"]["place_conf"] if place_conf is None else place_conf,
                             dtype=float).reshape(6)
        # 本地扁平 wrs 的 JLChain 只有 is_jnt_values_in_ranges（新版叫 are_jnts_in_ranges）
        if not self.rbt_s.arm.jlc.is_jnt_values_in_ranges(q_place):
            wrapped = self._wrap_conf_to_limits(q_place)
            if wrapped is None:
                print("[plan_place] 放置构型 {} 超关节限位且无法 ±2π 折回".format(
                    np.round(np.degrees(q_place), 2).tolist()))
                return None
            q_place = wrapped
        dist = float(cfg.CONFIG["motion"]["place_approach_distance_m"]
                     if approach_distance is None else approach_distance)

        # ① 抬起：沿工具轴（= 抓取姿态的 -z），与旧的 _plan_grasp_segments 同口径。
        #    高度不是常量：max(最低抬升, 物体最低点过壁所需)。
        grasp_pos, grasp_rot = self.rbt_s.fk(jnt_values=q_grasp)
        grasp_pos = np.asarray(grasp_pos, dtype=float).reshape(3)
        grasp_rot = np.asarray(grasp_rot, dtype=float).reshape(3, 3)
        lift_distance = max(float(cfg.CONFIG["motion"]["lift_distance_m"]),
                            float(obstacle.min_lift_to_clear_wall(obj)))
        lift_pos = grasp_pos - grasp_rot[:, 2] * lift_distance
        with self._jaw(cfg.CONFIG["gripper"]["grasp_width_full"]):
            q_lift = self._safe_ik(lift_pos, grasp_rot, seed=q_grasp)
            if q_lift is None:
                print("[plan_place] 抬升点无 IK 解（抬升 {:.4f} m，pos={}）".format(
                    lift_distance, np.round(lift_pos, 4).tolist()))
                return None
            lift_path = self.get_line_path(seed=q_grasp, start=grasp_pos,
                                           goal=lift_pos, rot=grasp_rot)
            if lift_path is None:
                print("[plan_place] 抬升段有路点不可解（IK 或碰撞）")
                return None
            lift_path = [np.asarray(p, dtype=float).reshape(6) for p in lift_path]
            lift_path[0] = q_grasp.copy()
            lift_path[-1] = np.asarray(q_lift, dtype=float).reshape(6)

        # ② 放置点正上方 + 竖直下放段（世界 +z）
        pos, rot = self.rbt_s.fk(jnt_values=q_place)
        pos = np.asarray(pos, dtype=float).reshape(3)
        rot = np.asarray(rot, dtype=float).reshape(3, 3)
        above = pos + np.array([0.0, 0.0, dist])
        with self._jaw(cfg.CONFIG["gripper"]["grasp_width_full"]):
            q_above = self._safe_ik(above, rot, seed=q_place)
            if q_above is None:
                print("[plan_place] 放置点上方 {:.3f} m 处无 IK 解（pos={}）".format(
                    dist, np.round(above, 4).tolist()))
                return None
            descend = self.get_line_path(seed=q_above, start=above, goal=pos, rot=rot)
            if descend is None:
                print("[plan_place] 下放段有路点不可解（IK 或碰撞）")
                return None
            descend = [np.asarray(p, dtype=float).reshape(6) for p in descend]
            descend[0] = np.asarray(q_above, dtype=float).reshape(6)
            descend[-1] = q_place.copy()

        # ③ 搬运：抬升点 → 放置点上方。name 用 "place"（= 该段最终归属），这样
        #    self._transfer 查碰撞用的是搬运行的真实开口（已闭爪 0.031）。
        transit_path, transit_mode, transit_hit = self._transfer(
            lift_path[-1], q_above, "place")
        transit_path = [np.asarray(p, dtype=float).reshape(6) for p in transit_path]
        transit_path[0] = lift_path[-1].copy()
        transit_path[-1] = np.asarray(q_above, dtype=float).reshape(6)

        # ④ 拼成一条并平滑。⛔ check_collision 的签名默认是 False（docstring 却写
        #    「默认开」），必须显式传 True；开口由 _jaw 管，_path_collision_idx 在
        #    里面不传 jaw_width ⇒ 用当前开口 = GRASP_WIDTH_FULL。
        raw = lift_path + transit_path[1:] + descend[1:]
        with self._jaw(cfg.CONFIG["gripper"]["grasp_width_full"]):
            merged = self.smooth_two_path_2(lift_path, transit_path, check_collision=True)
            merged = self.smooth_two_path_2(merged, descend, check_collision=True)
        if merged is None:
            print("[plan_place] 平滑拼接失败，退回未平滑拼接（接缝折角大但可行）")
            merged = [p.copy() for p in raw]
        merged = [np.asarray(p, dtype=float).reshape(6) for p in merged]
        merged[0] = q_grasp.copy()
        merged[-1] = q_place.copy()

        # ⑤ 瓶身复核：机器人碰撞模型里没有瓶子，平滑抄近路可能把瓶底带进箱壁。
        #    碰了就退回未平滑拼接（那是设计路径，几何上还没被曲线改写）。
        held_hit = -1
        if check_object:
            held_hit = self.check_held_collision_idx(q_grasp, obj, merged)
            if held_hit >= 0:
                print("[plan_place] 平滑后瓶身在第 {} 点碰箱壁，退回未平滑拼接".format(
                    held_hit))
                merged = [p.copy() for p in raw]
                held_hit = self.check_held_collision_idx(q_grasp, obj, merged)
                if held_hit >= 0:
                    print("[plan_place] ⛔ 未平滑拼接也在第 {} 点碰箱壁".format(held_hit))
                    return None



        # ⑥ 段级碰撞按**最终**路径重算（平滑会生成新路点）。平滑可能抄近路撞上
        #    箱外的东西 ⇒ 先回退未平滑拼接；仍碰撞说明设计路径本身不可行，直接
        #    返回 None —— **绝不把带碰撞的路径交给执行层**。
        hit = self._path_collision_idx(merged, jaw_width=cfg.CONFIG["gripper"]["grasp_width_full"])
        if hit >= 0:
            print("[plan_place] 平滑后本体在第 {} 点碰撞，退回未平滑拼接".format(hit))
            merged = [p.copy() for p in raw]
            hit = self._path_collision_idx(merged, jaw_width=cfg.CONFIG["gripper"]["grasp_width_full"])
            if check_object:
                held_hit = self.check_held_collision_idx(q_grasp, obj, merged)
            if hit >= 0 or held_hit >= 0:
                print("[plan_place] ⛔ 未平滑拼接仍不可行（本体 {} / 瓶身 {}）".format(hit, held_hit))
                return None
        # ⑥.5 GOMP 精修：对已无碰撞的平滑路径做轨迹优化（全程持瓶，带瓶身约束）。
        #     端点不回写、不改目标函数；SQP 失败或任一校验不过 ⇒ 回退到上面
        #     已验证无碰撞的 merged（绝不把带碰撞的路径交给执行层）。
        merged = self._refine_place_with_gomp(merged, q_grasp, obj)

        place_seg = {
            "path": merged,
            "action_after": _SEGMENT_ACTION["place"],
            "jaw_full": _SEGMENT_JAW_WIDTH["place"],
        }

        # ⑦ 8 角点过壁诊断（只用来给「离壁顶多少米」这个数值，判定权在 ⑤）
        violations, min_clear = ([], float("inf"))
        if check_object:
            violations, min_clear = self.check_object_clearance(q_grasp, obj, [place_seg])

        print("[plan_place] 抬起 {:.4f} m / {} 点 → 搬运 {} 点 [{}] → 下放 {:.3f} m / "
              "{} 点 ⇒ place {} 点，本体碰撞 {}".format(
                  lift_distance, len(lift_path), len(transit_path), transit_mode,
                  dist, len(descend), len(merged), hit))
        print("[plan_place] 放置 TCP = {}，物体离壁顶 {:+.4f} m（过壁违规 {} 处），"
              "瓶身碰箱壁下标 {}".format(
                  np.round(pos, 4).tolist(),
                  min_clear if np.isfinite(min_clear) else float("nan"),
                  len(violations), held_hit))

        place_dict = {
            "place": place_seg,
            "object_violations": violations,
        }

        return place_dict

# ============================================================================
# wrs 仿真环境（Panda3D）展示：把 pick&place 路径摆进真实场景里看
# ============================================================================

#: 每段在仿真里的轨迹线颜色（RGBA，0~1）。刻意选深色：场景是白背景 + 浅灰箱子，
#: 亮色（黄/浅绿）会和背景糊在一起。与 matplotlib 的 ``_SEGMENT_COLOR`` 同色系。
_SIM_SEGMENT_RGBA = {
    "grasp": (0.00, 0.52, 0.58, 1.0),
    "place": (0.90, 0.45, 0.00, 1.0),
}

#: 「瓶子已经夹住」的段：这些段的 ghost 上要同步画随动瓶身。
_HELD_SEGMENTS = ("place",)

#: 仿真场景默认取景（世界系，米）。箱心与放置点的中间，拉高到能看见相机碰撞体。
_SIM_CAM_POS = (2.05, -2.45, 2.20)
_SIM_LOOKAT_POS = (0.55, -0.05, 0.35)

#: 模块级场景 / 机器人单例：`check_grasp_and_place_path` 与 `show_path` 共用，
#: 避免"每画一段就新建一个 World"（新建会让之前 attach 的 mesh 全丢）。
_SIM_BASE = None
_SIM_ROBOT = None


def _sim_base():
    """取模块级仿真场景（首次调用时创建：白底 + 障碍物 + 世界系）。

    Returns:
        wd.World: 全局唯一的 Panda3D 场景。重复调用返回同一实例；已 attach
        的 mesh 不会丢。
    """
    global _SIM_BASE
    if _SIM_BASE is None:
        import modeling.geometric_model as _gm
        import visualization.panda.world as wd

        _SIM_BASE = wd.World(cam_pos=list(_SIM_CAM_POS),
                             lookat_pos=list(_SIM_LOOKAT_POS))
        _sim_setup_window(_SIM_BASE)
        _set_camera(_SIM_BASE, _SIM_CAM_POS, _SIM_LOOKAT_POS)
        _gm.gen_frame(length=0.15).attach_to(_SIM_BASE)
        obstacle.attach_obstacles_to_base(_SIM_BASE)
    return _SIM_BASE


def _sim_robot(rbt=None):
    """取模块级 sim 机器人；传入实例时替换单例。

    Args:
        rbt: ``UR7EDH76`` 实例或 ``None``。规划侧建议传 ``planner.rbt_s``
            复用（新建一个约 0.5~1 s，且多一份 mesh）。

    Returns:
        UR7EDH76: 全局唯一的机器人实例。
    """
    global _SIM_ROBOT
    if rbt is not None:
        _SIM_ROBOT = rbt
    if _SIM_ROBOT is None:
        _SIM_ROBOT = ur7e.UR7EDH76()
    return _SIM_ROBOT

# ============================================================================
# 可执行段（原 path_plan.py，11-13 并入本模块；11-14 去掉拼接层）
#
# 只做「段常量 + 转移段 + 段工具」，不做 IK：IK 在 Grasp_planner 里完成。
# 段序 = 执行顺序，夹爪动作挂在段末（close 在 grasp 后，release 在 place 后）。
# ⛔ 没有「拼完统一重查碰撞」的层：谁生成段谁查这一次，结果写进段里。
# ============================================================================
#: 段名顺序（= 执行顺序）。改顺序等于改动作时序，改前先看模块 docstring。
SEGMENT_ORDER = ("grasp", "place")

#: 段末夹爪动作的**打印标签**。闭爪/松爪由执行层按时间顺序直接调用，不读这些值。
ACTION_CLOSE = "close"
ACTION_RELEASE = "release"
_VALID_ACTIONS = (None, ACTION_CLOSE, ACTION_RELEASE)

#: 各段的夹爪开口（指尖全距，米）。查碰撞必须用**该段真实开口**，否则偏乐观。
_SEGMENT_JAW_WIDTH = {
    # ⛔ to_pick_above 不在 SEGMENT_ORDER 里（已被 grasp 吸收），但 _transfer /
    #    _return_transfer / _refine_grasp_with_gomp 内部查碰撞仍按全开取它。
    "to_pick_above": cfg.CONFIG["gripper"]["open_width_full"],
    # grasp 段全程按全开 0.092 查碰撞（并行调爪期间爪宽过渡，取最保守口径）；
    # 字段值 0.041 是执行终值（打印/记录用），不是查碰撞口径。
    "grasp": cfg.approach_width_full(),
    "place": cfg.CONFIG["gripper"]["grasp_width_full"],
}

#: 段末动作（⛔ 仅用于 ``describe_segments`` 打印，不是控制信号；
#: 真正的动作顺序写死在 ``grasp_execute.execute_pick`` / ``execute_place``）
_SEGMENT_ACTION = {
    "grasp": ACTION_CLOSE,
    "place": ACTION_RELEASE,
}


def segment_jaw_width(name: str) -> float:
    """该段执行时的夹爪开口（指尖全距，米）。

    Args:
        name: 段名，必须是 ``SEGMENT_ORDER`` 之一。

    Returns:
        float: 指尖全距（米）。``grasp`` = 接近宽度（pickle 派生；
        查碰撞另按全开 0.092，见 ``_blend_grasp_paths``），``place``
        （抬起 + 搬运 + 下放）= 抓取宽度 0.031（已闭爪，正夹着物体）。
        内部转移段 ``to_pick_above`` 保留 = 全开 0.092。
    """
    return float(_SEGMENT_JAW_WIDTH[name])


def _as_conf(q, label=""):
    """把任意 array_like 规整成 (6,) float 关节角。

    Args:
        q: array_like，长度 6。
        label: 出错时打印的段名/用途，便于定位。

    Returns:
        np.ndarray: shape (6,)、dtype float64 的新数组。
    """
    arr = np.asarray(q, dtype=float).reshape(6)
    if not np.all(np.isfinite(arr)):
        raise ValueError("{}: 关节角含 NaN/inf: {}".format(label, arr.tolist()))
    return arr.copy()


def seg_n_points(seg):
    """段的路点数。

    Args:
        seg: dict，一段（至少含 ``path``）。

    Returns:
        int: ``len(seg["path"])``，含首末两端点。
    """
    return len(seg["path"])


def plan_segments(plan):
    """按 ``SEGMENT_ORDER`` 取出 plan 里实际存在的段，连带段名（**只为遍历**）。

    ⛔ 执行层取单段一律直接下标（``pick_plan["grasp"]``），不要绕本函数——
    它只服务「按执行顺序遍历」的场景：打印段表、逐段画图。

    Args:
        plan: dict ``plan_grasp`` / ``plan_place`` 的返回（段以段名为键）。

    Returns:
        list[tuple[str, dict]]: ``(段名, 段)``，顺序 = ``SEGMENT_ORDER``。
    """
    return [(name, plan[name]) for name in SEGMENT_ORDER if name in plan]


def describe_segments(segments):
    """把各段的点数 / 开口 / 动作整理成可打印的行。

    Args:
        segments: list[tuple[str, dict]] ``plan_segments`` 的返回。

    Returns:
        list[str]: 每段一行，外加一行总计。
    """
    lines = ["  {:<14s} {:>4s} {:>8s} {:>8s}".format(
        "段", "点数", "开口mm", "动作")]
    for name, seg in segments:
        lines.append("  {:<14s} {:3d} 点  {:8.1f} {:<8s}".format(
            name, seg_n_points(seg), float(seg["jaw_full"]) * 1000.0,
            seg["action_after"] or "-"))
    lines.append("  {} 段合计 {} 点".format(
        len(segments), sum(seg_n_points(s) for _, s in segments)))
    return lines


def _homomat(pos, rot):
    """把位置 + 旋转矩阵拼成 4x4 齐次变换矩阵。

    Args:
        pos: array-like，长度 3，平移量（米）。
        rot: array-like，(3,3) 旋转矩阵（无量纲）。

    Returns:
        np.ndarray: (4,4) 齐次矩阵，``[:3,:3] = rot``、``[:3,3] = pos``、
        末行 ``[0, 0, 0, 1]``。
    """
    hom = np.eye(4, dtype=float)
    hom[:3, :3] = np.asarray(rot, dtype=float).reshape(3, 3)
    hom[:3, 3] = np.asarray(pos, dtype=float).reshape(3)
    return hom


def object_to_tcp(object_pose, tcp_pos, tcp_rot):
    """算「物体相对 TCP」的刚性位姿，即 ``show_path(held=...)`` 要的那两个量。

    Args:
        object_pose: (4,4) 物体位姿（sim 世界系，原点 = 底面中心），必须取
            **抓取那一刻**的值，否则抬升后瓶身会错位。
        tcp_pos: (3,) 抓取构型下 TCP 的世界位置（米）。
        tcp_rot: (3,3) 抓取构型下 TCP 的世界旋转。

    Returns:
        tuple: ``(R_tb, t_tb)``。``R_tb`` 是 (3,3) 物体在 TCP 系下的姿态；
        ``t_tb`` 是 (3,3)->(3,) 物体原点在 TCP 系下的位置（米）。两者只对这一次抓取有效。
    """
    T_world_tcp = _homomat(tcp_pos, tcp_rot)
    T_tcp_obj = np.linalg.inv(T_world_tcp) @ np.asarray(object_pose, dtype=float).reshape(4, 4)
    return T_tcp_obj[:3, :3].copy(), T_tcp_obj[:3, 3].copy()


def show_path(path, base=None, rbt=None, n_ghosts=10, n_tcp=10,
              ghost_rgba=(0, 1, 0, 0.3), tcp_rgba=(1, 0, 0, 1),
              held=None, held_cm=None, only_tcp=False, all_conf=False,
              line_radius=0.007):
    """在 wrs/Panda3D 场景里画一条关节路径：TCP 轨迹线（+ 可选机器人 ghost）。

    与 ``yanpu_ur8/path_plan.py`` 的 ``Path_plan.show_path`` 同一形态：**只传路径**
    就画，场景与机器人走模块级单例。TCP 一律取 ``fk`` 的返回值
    （= ``arm.get_gl_tcp()``，jaw center），轨迹线一律用 ``mgm.gen_stick``。

    Args:
        path: list[(6,)] 关节角（弧度）。
        base: Panda3D 场景根节点。``None`` = ``_sim_base()`` 单例（首次调用
            自动建窗口 + 障碍物 + 世界系）。
        rbt: sim 机器人（``UR7EDH76``）。``None`` = ``_sim_robot()`` 单例。
            会被 ``fk`` 依次改到路径上的采样构型，函数返回时停在**最后被采样到
            的**构型，调用方要复位就自己 ``goto_given_conf``。
        n_ghosts: int 画多少个机器人 ghost 姿态（沿路径等间隔采样）。签名默认
            ``10``；``0`` = 不画、只留轨迹线——多段各画 ghost 时 alpha 0.2 叠起来
            会把箱内路径重新挡住，所以 ``check_grasp_and_place_path`` 想只看线时
            应显式传 ``n_ghosts=0``。
        n_tcp: int TCP 轨迹线的采样点数（沿路径等间隔采样）。签名默认 ``10``；
            想看顺滑曲线传 ``150``。
        ghost_rgba: 长度 4 的序列，机器人 ghost 的 RGBA（0~1）。alpha 建议 0.2~0.35，
            太实会把后面的路径挡住。
        tcp_rgba: 长度 4 的序列，TCP 轨迹线的 RGBA。
        held: tuple | None ``(R_tb, t_tb)``，手持物体相对 TCP 的刚体位姿，
            一般由 ``object_to_tcp`` 返回。``None`` = 不画手持物体。
        held_cm: CollisionModel | None 手持物体的模型（``tea_bottle.STL``）。
            ``held`` 非 None 时必须给；内部每帧 ``copy()`` 一份再摆位，不会移动
            传进来的原模型。
        only_tcp: bool True 时只画 TCP 线，跳过机器人 ghosts 与手持物体
            （段数多、要看清轨迹时用）。
        all_conf: bool True 时把 ``path`` 里**每一个**关节点都画成 ghost，用于逐帧
            核对姿态。此时忽略 ``n_ghosts`` 与 ``held``，且点数多会很卡。
        line_radius: float TCP 轨迹线半径（米）。默认 0.007，比 ``gen_stick`` 的
            默认 0.0025 粗近三倍——细线在 1600x900 的整场视角下几乎看不见。

    Returns:
        list: 本次 attach 到 ``base`` 的节点（机器人 mesh、瓶身 mesh、线段），
        顺序与创建顺序一致，调用方可自行清理。
    """
    meshes = []
    base = _sim_base() if base is None else base
    rbt = _sim_robot() if rbt is None else rbt
    m = len(path)
    if m == 0:
        return meshes

    if all_conf:
        for q in path:
            rbt.fk(jnt_values=np.asarray(q, dtype=float).reshape(6))
            mesh = rbt.gen_meshmodel(rgba=list(ghost_rgba))
            mesh.attach_to(base)
            meshes.append(mesh)
        return meshes

    tcp_idx = set(int(v) for v in np.unique(
        np.linspace(0, m - 1, min(int(n_tcp), m)).astype(int)))
    ghost_idx = set()
    if n_ghosts and int(n_ghosts) > 0 and not only_tcp:
        ghost_idx = set(int(v) for v in np.unique(
            np.linspace(0, m - 1, min(int(n_ghosts), m)).astype(int)))

    tcp_pts = []
    for i in sorted(tcp_idx | ghost_idx):
        tcp_pos, tcp_rot = rbt.fk(
            jnt_values=np.asarray(path[i], dtype=float).reshape(6))
        tcp_pos = np.asarray(tcp_pos, dtype=float).reshape(3)
        tcp_rot = np.asarray(tcp_rot, dtype=float).reshape(3, 3)
        if i in tcp_idx:
            tcp_pts.append(tcp_pos)
        if i in ghost_idx:
            mesh = rbt.gen_meshmodel(rgba=list(ghost_rgba))
            mesh.attach_to(base)
            meshes.append(mesh)
            if held is not None and held_cm is not None:
                R_tb, t_tb = held
                held_obj = held_cm.copy()
                held_obj.set_homomat(_homomat(
                    tcp_pos + tcp_rot @ np.asarray(t_tb, dtype=float).reshape(3),
                    tcp_rot @ np.asarray(R_tb, dtype=float).reshape(3, 3)))
                held_obj.attach_to(base)
                meshes.append(held_obj)

    import modeling.geometric_model as gm

    for a, b in zip(tcp_pts[:-1], tcp_pts[1:]):
        # 本机 gen_stick 的签名是 (thickness, rgba)，没有 radius/rgb/alpha
        seg = gm.gen_stick(spos=a, epos=b, thickness=float(line_radius),
                           rgba=list(tcp_rgba))
        seg.attach_to(base)
        meshes.append(seg)
    return meshes


def _set_camera(base, cam_pos, lookat_pos, up=(0.0, 0.0, 1.0)):
    """显式摆一次相机（位置 + lookAt + up）。

    ``wd.World.__init__`` 已经摆过一次，但这里仍要再摆：``taskMgr.step()`` 期间
    只要有一次 lens/render 重算，初始视角就可能漂掉（实测截图会变成贴脸视角）。

    Args:
        base: ``wd.World`` 场景。
        cam_pos: 长度 3 的相机世界位置（米）。
        lookat_pos: 长度 3 的注视点（米）。
        up: 长度 3 的相机 up 向量，默认 +z（世界系竖直）。

    Returns:
        None
    """
    from panda3d.core import Point3, Vec3

    px, py, pz = [float(v) for v in np.asarray(cam_pos, float).reshape(3)]
    lx, ly, lz = [float(v) for v in np.asarray(lookat_pos, float).reshape(3)]
    ux, uy, uz = [float(v) for v in np.asarray(up, float).reshape(3)]
    base.cam.setPos(px, py, pz)
    base.cam.lookAt(Point3(lx, ly, lz), Vec3(ux, uy, uz))


def _sim_setup_window(base, width=1600, height=900, bg=(1.0, 1.0, 1.0)):
    """把 Panda3D 窗口改成指定尺寸并强制白底清屏。

    ⛔ 两个已踩过的坑：
        1. ``wd.World`` 建的窗口实际是 **800x600**（不是它签名里的 1920x1080），
           截图细节全靠这一步放大。
        2. ``World.__init__`` 的 ``setBackgroundColor(1, 1, 1)`` 把 alpha 设成了 0，
           ``saveScreenshot`` 存出的 PNG 天空区域 alpha=0 → 看图软件里显示成**黑底**。
           必须显式 ``setClearColor(Vec4(1,1,1,1))``。

    Args:
        base: ``wd.World`` 场景。
        width: int 窗口宽度（像素）。
        height: int 窗口高度（像素）。
        bg: 长度 3 的 RGB 背景色（0~1）。

    Returns:
        None
    """
    from panda3d.core import Vec4, WindowProperties

    try:
        props = WindowProperties()
        props.setSize(int(width), int(height))
        base.win.requestProperties(props)
    except Exception:  # 窗口尺寸改不动不算致命，继续用原尺寸截图
        pass
    # ⛔ 必须给 alpha=1：World.__init__ 只调了 3 参数的 setBackgroundColor，
    #    framebuffer alpha 为 0，saveScreenshot 存出的 PNG 天空区会变成黑色。
    base.setBackgroundColor(float(bg[0]), float(bg[1]), float(bg[2]), 1.0)
    base.win.setClearColorActive(True)
    base.win.setClearColor(Vec4(float(bg[0]), float(bg[1]), float(bg[2]), 1.0))


def _sim_snapshot(base, out_path, frames=6):
    """把当前 Panda3D 画面推进若干帧后截图存成 PNG。

    ⛔ 必须有真实 X 显示：``world-type none``（离屏）时 ``base.win`` 是 None，
    Panda3D 的 offscreen buffer 又不提供 ``win.getProperties()``，两条路都走不通
    （已在 ``obstacle._show_scene`` 实测记录）。

    Args:
        base: 已建好窗口的 ``wd.World``。
        out_path: str | Path 输出 PNG 路径；父目录不存在会自动创建。
        frames: int 截图前推进的帧数。太少会拍到首帧（相机还没 rig 好、模型未刷）。

    Returns:
        Path: 落盘 PNG 的绝对路径。

    Raises:
        RuntimeError: 没有可用窗口，或 ``saveScreenshot`` 返回 False。
    """
    from panda3d.core import Filename

    if getattr(base, "win", None) is None:
        raise RuntimeError("没有可用的 Panda3D 窗口（需要真实 DISPLAY，不能用离屏）")
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(max(1, int(frames))):
        base.taskMgr.step()
    base.graphicsEngine.renderFrame()
    if not bool(base.win.saveScreenshot(Filename(str(out)))):
        raise RuntimeError("Panda3D 截图失败：{}".format(out))
    return out


def visualize_plan_sim(pick_plan, place_plan, robot, object_pose, *,
                       base=None, shot_path=None,
                       n_ghosts=0, n_tcp=150,
                       show_placed_object=True, cam_pos=(2.05, -2.45, 2.20),
                       lookat_pos=(0.55, -0.05, 0.35), line_radius=0.007,
                       robot_alpha=0.22, ghost_alpha=0.20,
                       win_size=(1600, 900), reset_robot=True, shot_frames=8,
                       run_window=True):
    """在 wrs 仿真环境（Panda3D 窗口）里展示整条 pick&place 路径。

    场景内容（全部在 **sim 世界系**，与 ``cfg`` / ``obstacle`` 同一套坐标）：

    - 箱体（默认 ``"panel"``：5 块**有厚度**的板 = 碰撞 CDPrim 的几何，壁厚 43 mm）
      + 桌面实体（``wholetable.STL``）+ 相机碰撞体（1.30x1.3x0.1 m 的板）
      —— 由 ``obstacle.attach_obstacles_to_base`` 统一无条件挂载（不设展示开关）
    - 物体初始位姿（蓝，按 ``object_pose`` 摆）与放置后位姿（绿，按 descend 末帧摆）
    - 机器人本体（起点构型，默认 alpha 0.22，只留轮廓）+ 每段各 ``n_ghosts``
      个 ghost（按段配色，默认 alpha 0.22）
    - 三段 TCP 轨迹线（``gen_stick``，按段配色，粗 7 mm）+ 抓点红球 / 放点绿球
    - 持物段（``place``）的 ghost 上同步画随动瓶身

    ⛔ 机器人本体与 ghost 都刻意做成半透明：实体渲染时它们会挡掉 70% 以上的路径
    （实测），这个视图的唯一目的就是「看路径有没有穿箱 / 绕远」。

    Args:
        pick_plan: dict ``plan_grasp`` 的返回（有碰撞也能画，正好肉眼找问题段）。
            段以段名为键，本函数用 ``plan_segments`` 按 ``SEGMENT_ORDER`` 遍历。
        place_plan: dict ``plan_place`` 的返回。⛔ 两批段**不相加**：
            函数内部按「先 pick 后 place」的时间顺序**分批遍历**，只为画一整张图。
        robot: sim 机器人（``UR7EDH76``）。会被反复 ``fk`` / ``goto_given_conf``；
            ``reset_robot=True`` 时返回前复位到路径起点。
        object_pose: (4,4) 物体位姿（sim 世界系，原点 = 底面中心）。
        base: ``wd.World`` | None 复用已有场景；``None`` = 新建一个。
        shot_path: str | Path | None 非 None 时把画面截图存到该 PNG。
        n_ghosts: int 每段画几个 ghost 姿态。默认 0 = 只画轨迹线（最清楚，
            能一眼看出路径是否绕远）；给 1~2 可额外看到机器人姿态，但 12 个
            alpha 0.2 的 ghost 叠加后接近不透明，会把箱内路径重新挡住。
        n_tcp: int 每段 TCP 线的采样点数。转移段点多，150 足够顺滑。
        show_placed_object: bool 是否在 place 段末帧画一个半透明绿「放好之后」的瓶子。
        cam_pos: 长度 3 的相机位置（世界系，米）。默认 (2.05, -2.45, 2.20)：
            比旧值 (1.60, -1.95, 1.75) 拉高拉远，为的是把 z≈1.3 m 的相机碰撞体
            也纳入画面；箱体与路径仍占画面主体。
        lookat_pos: 长度 3 的相机注视点（米）。默认 (0.55, -0.05, 0.35)：
            箱心与放置点的中间。
        line_radius: float TCP 轨迹线半径（米），逐段转发给 ``show_path``。
        robot_alpha: float 机器人本体不透明度。默认 0.22（只留轮廓）。
        ghost_alpha: float 每个 ghost 姿态的不透明度。
        win_size: tuple[int, int] 窗口像素尺寸，默认 1600x900。
        reset_robot: bool 结束时是否把机器人复位到路径起点。
        shot_frames: int 截图前推进的帧数（沿用 ``_sim_snapshot``）。
        run_window: bool True 时最后 ``base.run()`` 阻塞窗口；截图脚本要设 False。

    Returns:
        dict: ``{"base": base, "shot": Path|None, "held": (R_tb, t_tb),
        "meshes": list, "obstacles": dict}``。``held`` 是本次抓取的物体-相对-TCP 位姿，
        可复用；``meshes`` 是本次新建的路径节点（不含机器人本体与障碍物）；
        ``obstacles`` 是 ``obstacle.attach_obstacles_to_base`` 的返回（含
        ``box`` / ``table`` / ``camera`` 节点列表，便于核对画面与碰撞体）。
    """
    import modeling.collision_model as mcm
    import modeling.geometric_model as mgm
    import visualization.panda.world as wd

    if base is None:
        base = wd.World(cam_pos=list(cam_pos), lookat_pos=list(lookat_pos))
    _sim_setup_window(base, width=win_size[0], height=win_size[1])
    _set_camera(base, cam_pos, lookat_pos)
    mgm.gen_frame(length=0.15).attach_to(base)

    # 1) 障碍物：全部交给 obstacle.attach_obstacles_to_base（箱体 5 块厚板 +
    #    桌面 / 相机，不叠外廓棱线），一律无条件展示，不设开关。
    #    厚板几何与碰撞 CDPrim 逐块重合（底板 5 mm + 四壁 43 mm）。
    obstacles = obstacle.attach_obstacles_to_base(base)

    # 2) 瓶子：start（初始位姿）/ held（随动模板）两个独立实例，避免 copy() 纠缠
    bottle_kwargs = dict(initor=str(cfg.CONFIG["path"]["object_stl_path"]),
                         cdprimit_type="box",
                         # 显式 0：本机 expand_radius 默认 0.002，会把瓶身 AABB 撑大 2 mm
                         expand_radius=0.0)
    bottle_start = mcm.CollisionModel(name="tea_bottle_start", **bottle_kwargs)
    bottle_start.set_rgba([0.30, 0.50, 0.95, 0.95])
    bottle_start.set_homomat(np.asarray(object_pose, dtype=float).reshape(4, 4))
    bottle_start.attach_to(base)
    bottle_held = mcm.CollisionModel(name="tea_bottle_held", **bottle_kwargs)
    bottle_held.set_rgba([0.30, 0.50, 0.95, 0.55])

    # 3) 抓取那一刻的 物体-相对-TCP 位姿（后续所有 ghost 都用它还原瓶身）
    grasp_q = np.asarray(pick_plan["grasp"]["path"][-1], dtype=float).reshape(6)
    p_grasp, r_grasp = robot.fk(jnt_values=grasp_q)
    held = object_to_tcp(object_pose, p_grasp, r_grasp)

    # 4) 机器人本体摆在起点，各段各自画 ghost + TCP 线
    start_q = np.asarray(pick_plan["grasp"]["path"][0],
                         dtype=float).reshape(6)
    robot.goto_given_conf(start_q)
    robot.gen_meshmodel(toggle_jnt_frames=False, alpha=float(robot_alpha)).attach_to(base)

    meshes = []
    for plan in (pick_plan, place_plan):
        for name, seg in plan_segments(plan):
            rgba = _SIM_SEGMENT_RGBA.get(name, (0.2, 0.2, 0.2, 1.0))
            meshes.extend(show_path(
                seg["path"], base=base, rbt=robot,
                n_ghosts=n_ghosts, n_tcp=n_tcp,
                ghost_rgba=(rgba[0], rgba[1], rgba[2], float(ghost_alpha)), tcp_rgba=rgba,
                held=held if name in _HELD_SEGMENTS else None,
                held_cm=bottle_held if name in _HELD_SEGMENTS else None,
                line_radius=line_radius,
            ))

    # 5) 抓点 / 放点标记
    p_place, r_place = robot.fk(
        jnt_values=np.asarray(place_plan["place"]["path"][-1], dtype=float).reshape(6))
    mgm.gen_sphere(pos=np.asarray(p_grasp, float).reshape(3), radius=0.012,
                   rgba=[1.0, 0.15, 0.15, 1.0]).attach_to(base)
    mgm.gen_sphere(pos=np.asarray(p_place, float).reshape(3), radius=0.012,
                   rgba=[0.10, 0.70, 0.25, 1.0]).attach_to(base)

    # 6) 放好之后的瓶子（半透明绿），用来核对落点是否在箱内/是否压壁
    if show_placed_object:
        R_tb, t_tb = held
        placed = bottle_held.copy()
        placed.set_homomat(_homomat(
            np.asarray(p_place, float).reshape(3)
            + np.asarray(r_place, float).reshape(3, 3) @ t_tb,
            np.asarray(r_place, float).reshape(3, 3) @ R_tb))
        placed.set_rgba([0.20, 0.75, 0.35, 0.55])
        placed.attach_to(base)

    if reset_robot:
        robot.goto_given_conf(start_q)

    # 7) 截图（可选）+ 开窗
    shot = None
    if shot_path is not None:
        _set_camera(base, cam_pos, lookat_pos)
        shot = _sim_snapshot(base, shot_path, frames=shot_frames)
    if run_window:
        base.run()
    return {"base": base, "shot": shot, "held": held, "meshes": meshes,
            "obstacles": obstacles}



# ============================================================================
# 直接运行本文件 = 在 wrs 仿真环境里展示一条完整的 pick&place 路径
# ============================================================================
def check_grasp_and_place_path(object_pose=None):
    """规划一次 pick&place，把各段路径逐段画进仿真场景并开窗。

    形态对齐 ``yanpu_ur8/path_plan.py`` 的 ``check_U1_grasp_and_place_path``：
    规划 → 逐段 ``show_path`` → ``base.run()``。画 TCP 轨迹线（按段配色）+ 机器人
    本体（α0.22）+ ``show_path`` 默认的 10 个 ghost，并在 ghost 的**夹爪上挂随动
    瓶身**——只有 ``place`` 段持瓶（见 ``_HELD_SEGMENTS``）；不另摆静态「起始瓶」/
    「放置后瓶」，也不存截图、不贴图例：那几样用 ``visualize_plan_sim``。

    Args:
        object_pose: (4,4) 物体位姿（sim 世界系，原点 = 底面中心，米）；
            ``None`` = 读 ``main.py`` 的 ``OBJECT_POSITION`` / ``OBJECT_ROTMAT``。

    Returns:
        dict: ``{"base", "pick", "place", "object_pose", "meshes"}``。两阶段的
        plan 原样分开返回，**不合并**；``meshes`` 是各段 attach 的轨迹线 / ghost /
        夹爪上的随动瓶身（不含机器人本体与障碍物）。``base.run()`` 阻塞到用户关窗，
        之后才返回。

    Raises:
        AssertionError: ``plan_grasp`` / ``plan_place`` 返回 None。
    """
    if object_pose is None:
        import main
        object_pose = cfg.make_object_pose(main.OBJECT_POSITION_4, main.OBJECT_ROTMAT)
    planner = Grasp_planner(ur7e.UR7EDH76(), obstacle.build_obstacles(),
                            seed_conf=cfg.CONFIG["joints"]["start_conf"])
    pick = planner.plan_grasp(object_pose)
    assert pick is not None, "plan_grasp 返回 None"
    place = planner.plan_place(pick["conf"], pick["object_pose"])
    assert place is not None, "plan_place 返回 None"
    # 一张完整图要按时间顺序画两阶段的段，但两批段本身**不相加**
    base = _sim_base()
    robot = _sim_robot(planner.rbt_s)     # 复用规划用机器人，不再建第二个
    start_q = np.asarray(pick["grasp"]["path"][0],
                         dtype=float).reshape(6)
    robot.goto_given_conf(start_q)
    robot.gen_meshmodel(toggle_jnt_frames=False, alpha=0.22).attach_to(base)
    # 瓶子只做「随动模板」挂到 ghost 的夹爪上，不 attach 任何静态瓶：place 段
    # 全程持瓶，所以只有该段传 held。物体-相对-TCP 位姿必须取**抓取那一刻**，
    # 否则抬升之后瓶身会整体错位（口径与 visualize_plan_sim 完全相同）。
    import modeling.collision_model as mcm
    bottle_held = mcm.CollisionModel(
        name="tea_bottle_held", initor=str(cfg.CONFIG["path"]["object_stl_path"]),
        cdprimit_type="box",
        # 同 visualize_plan_sim：本机默认 0.002 会撑大瓶身，显式 0
        expand_radius=0.0)
    bottle_held.set_rgba([0.50, 0.50, 0.0, 0.55])
    p_grasp, r_grasp = robot.fk(
        jnt_values=np.asarray(pick["grasp"]["path"][-1], dtype=float).reshape(6))
    held = object_to_tcp(object_pose, p_grasp, r_grasp)
    meshes = []
    for plan in (pick, place):
        for name, seg in plan_segments(plan):
            rgba = _SIM_SEGMENT_RGBA.get(name, (0.2, 0.2, 0.2, 1.0))
            # 走 show_path 的签名默认（n_ghosts=10 / n_tcp=10）：ghost 叠多了
            # 会把箱内路径糊掉（实测），要纯轨迹线就传 n_ghosts=0, n_tcp=150。
            held_name = name in _HELD_SEGMENTS
            meshes.extend(show_path(
                seg["path"], tcp_rgba=rgba,
                held=held if held_name else None,
                held_cm=bottle_held if held_name else None))
    # 关窗前把机器人摆回起点，否则它会停在最后一段的最后采样构型
    robot.goto_given_conf(start_q)
    base.run()
    return {"base": base, "pick": pick, "place": place,
            "object_pose": object_pose, "meshes": meshes}


if __name__ == "__main__":
    # 无参数：规划 → 在仿真场景里逐段展示 → 开窗（关窗后返回）。
    check_grasp_and_place_path()
