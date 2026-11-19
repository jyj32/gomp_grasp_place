"""
GOMP Planner: Grasp-Optimized Motion Planning (Algorithm 1).

This implements the full GOMP algorithm from the paper:
1. Compute IK for initial top-down grasps
2. Initialize trajectory via spline interpolation
3. For H from initial_H down to min_H:
   a. Run SQP solver
   b. If infeasible → return previous best trajectory
   c. Warm-start shorter trajectory from current solution
   d. Enforce v_H = 0
4. Return shortest feasible trajectory

For multiple grasp candidates, run GOMP on each and select the fastest.
"""

import numpy as np
from typing import List, Optional
import time as time_module

from gomp.ur7e_robot_adapter import RobotAdapter
from gomp.grasp.grasp import Grasp
from gomp.grasp.grasp_set import GraspSet
from gomp.obstacles.depth_map import DepthMapObstacle
from gomp.obstacles.collision import ObstacleConstraint
from gomp.obstacles.attached_object import AttachedObject
from gomp.optimization.qp_builder import QPBuilder
from gomp.optimization.sqp_solver import SQPSolver
from gomp.optimization.warm_start import spline_warm_start, interpolate_to_shorter
from gomp.optimization.fast_kinematics import FastKinematics
from gomp.planner.trajectory import Trajectory


class GOMPPlanner:
    """
    Grasp-Optimized Motion Planner.

    Computes minimum-time trajectories that avoid obstacles while
    constraining start/end configurations to grasp sets with
    rotational degrees of freedom.

    Parameters
    ----------
    robot : RobotAdapter
        Robot model.
    t_step : float
        Time step between waypoints (seconds).
        Default 0.008 matches UR5e 125Hz control loop.
    initial_H : int
        Initial time horizon (number of steps).
    min_H : int
        Minimum time horizon to try.
    H_reduction : str
        Strategy for reducing H: 'linear' (subtract 1) or
        'geometric' (multiply by factor).
    H_reduction_factor : float
        Factor for geometric reduction. E.g., 0.8 means H_new = int(H * 0.8).
    sqp_kwargs : dict
        Additional arguments for the SQP solver.
    verbose : bool
        Print progress information.
    """

    def __init__(self,
                 robot: RobotAdapter,
                 t_step: float = 0.008,
                 initial_H: int = 60,
                 min_H: int = 50,
                 H_reduction: str = 'geometric',
                 H_reduction_factor: float = 0.7,
                 sqp_kwargs: dict = None,  # 透传给 SQPSolver 的额外参数。
                 verbose: bool = True):
        self.robot = robot
        self.t_step = t_step  # 路点时间步长
        self.initial_H = initial_H  # 初始时间视界（步数）
        self.min_H = min_H  # 最短视界下限（步数）
        self.H_reduction = H_reduction  # 缩短 H 的策略：'linear' 每轮减 1 步；'geometric' 每轮乘因子
        self.H_reduction_factor = H_reduction_factor  # 几何缩短因子。H_new = int(H × factor)
        self.verbose = verbose  # 	是否打印每轮 SQP 进度

        # SQP solver configuration
        self.sqp_kwargs = sqp_kwargs or {}
        if 'verbose' not in self.sqp_kwargs:
            self.sqp_kwargs['verbose'] = verbose
        self.sqp_solver = SQPSolver(**self.sqp_kwargs)

        # Pre-compile fast kinematics engine
        self.fast_kin = FastKinematics(robot)

    def _shorter_H(self, H: int) -> int:
        """按 H_reduction 策略返回下一个（更短的）时间视界步数。

        'geometric'：H_new = clip(int(H*factor), H-40, H-20)（每轮至少减 20 步，
        避免因子接近 1 时退化成逐 1 步递减）；'linear'：H-1。
        调用方负责判断 H_new < min_H 时退出。
        """
        if self.H_reduction == 'geometric':
            H_min, H_max = max(H - 50, 20), max(H - 20, 20)
            return min(max(int(H * self.H_reduction_factor), H_min), H_max)
        return H - 1

    def plan(self,
             grasp_start: Grasp,
             grasp_goal: Grasp,
             obstacles: Optional[DepthMapObstacle] = None,
             held_object: Optional[AttachedObject] = None,
             ik_seed_start: Optional[np.ndarray] = None,
             ik_seed_goal: Optional[np.ndarray] = None,
             warm_start: Optional[Trajectory] = None,
             held_exempt_frac=(0.0, 0.0),
             real_aabb=None,
             use_real_merit: bool = False,
             merit_anchor: Optional[float] = None,
             hand_inflate: float = 0.0) -> Optional[Trajectory]:
        """
        Plan a minimum-time trajectory from grasp_start to grasp_goal.

        This is the main GOMP algorithm (Algorithm 1):
        1. Compute IK for top-down grasps
        2. Spline warm start
        3. Iteratively reduce H until infeasible
        4. Return shortest feasible trajectory

        Parameters
        ----------
        grasp_start : Grasp
            Start grasp with rotational DOF.
        grasp_goal : Grasp
            Goal grasp with rotational DOF.
        obstacles : DepthMapObstacle or None
            Obstacle depth map.
        held_object : AttachedObject or None
            Object rigidly held by the gripper (e.g. the grasped bottle).
            Adds one linearized clearance constraint per merged box group and
            waypoint; the first/last waypoints are exempt (grasp/release
            contact).
        held_exempt_frac : (first, last)
            持物行单独豁免比例（占路点数），见 ObstacleConstraint。只禁持物
            物体的约束行，TCP 行全段有效 —— 放置起点物体在箱内时豁免物体
            瞬态穿障，但手臂仍受净空约束（否则 SQP 会把手臂伸进障碍）。
        ik_seed_start : np.ndarray or None
            Known IK solution for start grasp (bypasses IK solver).
        ik_seed_goal : np.ndarray or None
            Known IK solution for goal grasp (bypasses IK solver).
        warm_start : Trajectory or None
            If given, use this trajectory as the initial guess instead of the
            spline warm start, and start the time-minimization loop at
            warm_start.H. It is expected to be (approximately) feasible, e.g.
            the hand-stitched result of two shorter GOMP plans. Pair it with a
            SMALL initial_trust_region (<= 0.1): the SQP then only polishes the
            trajectory locally. With a large trust region the linearization is
            driven out of its valid neighbourhood and the solve dies.

        Returns
        -------
        trajectory : Trajectory or None
            Optimized trajectory, or None if no feasible solution found.
        """
        n = self.robot.n_dof
        t0 = time_module.time()

        if self.verbose:
            print(f"GOMP: Planning with t_step={self.t_step}s, "
                  f"initial_H={self.initial_H}, n_dof={n}")

        # --- Step 1: Compute initial IK solutions ---
        start_grasp_set = GraspSet(grasp_start, self.robot)
        goal_grasp_set = GraspSet(grasp_goal, self.robot)

        # Use provided IK seeds or solve IK
        if ik_seed_start is not None:
            # Pre-seed the cache so get_topdown_ik returns the known config
            start_grasp_set._ik_cache[0.0] = ik_seed_start
            q_start = ik_seed_start
        else:
            q_start = start_grasp_set.get_topdown_ik()

        if q_start is None:
            if self.verbose:
                print("GOMP: Failed to find IK for start grasp")
            return None

        if ik_seed_goal is not None:
            goal_grasp_set._ik_cache[0.0] = ik_seed_goal
            q_goal = ik_seed_goal
        else:
            q_goal = goal_grasp_set.get_topdown_ik(seed_jnt_values=q_start)

        if q_goal is None:
            if self.verbose:
                print("GOMP: Failed to find IK for goal grasp")
            return None

        if self.verbose:
            print(f"GOMP: IK solutions found")
            print(f"  q_start = {np.round(q_start, 3)}")
            print(f"  q_goal  = {np.round(q_goal, 3)}")

        # --- Step 2: Create obstacle constraint ---
        obs_constraint = None
        if obstacles is not None:
            obs_constraint = ObstacleConstraint(
                self.robot, obstacles, safety_margin=0.01,
                fast_kin=self.fast_kin, attached=held_object,
                held_exempt_frac=held_exempt_frac,
                real_aabb=real_aabb,
                use_real_merit=use_real_merit,
                hand_inflate=hand_inflate
            )

        # --- Step 3: Initialize trajectory ---
        if warm_start is not None:
            # 外部给的【已可行】轨迹当初值（例如两段规划的手工拼接结果）：
            # GOMP 的目标函数本身就是 1/2*sum||v_{i+1}-v_i||^2，配小信任域即可在原地抹平加速度。
            H = int(warm_start.H)
            ws_q = np.asarray(warm_start.waypoints, dtype=float)
            ws_v = np.asarray(warm_start.velocities, dtype=float).copy()
            if ws_q.shape != (H + 1, n) or ws_v.shape != (H + 1, n):
                raise ValueError(
                    "warm_start shapes %s != %s" % ((ws_q.shape, ws_v.shape), (H + 1, n)))
            # 对齐动力学 q_{i+1} = q_i + v_i*t_step，让等式残差为 0
            # （拼接轨迹的速度是中心差分，不满足该等式；不修的话 SQP 要先花一轮修它）
            ws_v[:-1] = (ws_q[1:] - ws_q[:-1]) / self.t_step
            ws_v[-1] = 0.0
            x = QPBuilder(H, n).pack_x(ws_q, ws_v)
            if self.verbose:
                print(f"GOMP: warm start from supplied trajectory (H={H})")
        else:
            H = self.initial_H
            x = spline_warm_start(q_start, q_goal, H, n, self.t_step)
            if self.verbose:
                print(f"GOMP: Spline warm start initialized (H={H})")

        # --- Step 4: Time minimization loop ---
        best_trajectory = None

        while H >= self.min_H:
            if self.verbose:
                print(f"\nGOMP: Solving for H={H} "
                      f"(duration={H * self.t_step:.3f}s)...")

            # Run SQP
            result = self.sqp_solver.solve(
                x_init=x,
                H=H,
                n=n,
                t_step=self.t_step,
                robot=self.robot,
                obstacle_constraint=obs_constraint,
                start_grasp_set=start_grasp_set,
                goal_grasp_set=goal_grasp_set,
                fast_kin=self.fast_kin,
                merit_anchor=merit_anchor
            )

            if not result.feasible:
                if self.verbose:
                    print(f"GOMP: H={H} is INFEASIBLE after "
                          f"{result.n_iterations} iterations")
                # H 递减的语义是「最短时间搜索」，不是「求解失败后的退路」：
                # 可行域对 H 单调（时间越长、速度越低、约束越容易满足），
                # 即 Feasible(H_short) ⊆ Feasible(H_long)。所以当前 H 不可行时，
                # 任何更短的 H 只会在同一处几何违例上再叠加"时间不够→超速"
                # （实测 H=130→10 的 maxviol 0.18→37.5），必然更不可行。
                # 因此这里直接收工：有 best 返回 best，没有就返回 None。
                break

            # Build trajectory from solution
            qp = QPBuilder(H, n)
            waypoints = qp.extract_waypoints(result.x)
            velocities = qp.extract_velocities(result.x)

            # Velocities are already in rad/s from the SQP
            best_trajectory = Trajectory(
                waypoints=waypoints,
                velocities=velocities,
                t_step=self.t_step,
                H=H
            )

            if self.verbose:
                print(f"GOMP: H={H} FEASIBLE (cost={result.final_cost:.4f}, "
                      f"iters={result.n_iterations}, "
                      f"duration={best_trajectory.duration:.3f}s)")
                if result.n_accepted == 0:
                    # SQP 一步都没被接受 ⇒ 返回的其实是 warm start 本身
                    # （位置块未经精修）。如实标出来，别把兜底当收敛。
                    # 同时报【位置差分】的真实加速度比：速度块已被 accel clamp
                    # 投影到带内，报告值不再反映这条路本身需要多少加速度。
                    _vfd = np.diff(waypoints, axis=0) / self.t_step
                    _ac = self.robot.a_max * self.t_step
                    _ar = float(np.max(np.abs(np.diff(_vfd, axis=0)) / _ac))
                    print(f"GOMP: [warn] H={H} 无任何步被接受（LS 全拒），"
                          f"返回 warm start 兜底；位置差分真实加速度 "
                          f"{_ar:.3f}× a_max")

            # --- Step 5: Reduce H and warm start ---
            H_prev = H
            H = self._shorter_H(H)

            if H < self.min_H:
                break

            # Warm start with interpolated previous solution
            #
            # ⚠️ 时间压缩会把加速度按 (H_prev/H_new)² 放大：上一段是时间最优解，
            # 加速几乎必然贴着 a_max，压缩后即超限。若不处理，这段超限的 warm
            # start 一旦「SQP 全线被线搜索拒绝、原样返回」，就会得到
            # 「FEASIBLE 但 accel=False」的矛盾结果（实测 place H=200：
            # wp0/joint5 6.6833 > 6.2832，超出 6.37%）。
            # 这里把速度块投影回加速度带（位置块不动 ⇒ 几何路径与碰撞安全不变，
            # 动力学等式残差交给 QP 修，warm start 本就不必满足等式）。
            _stats = {}
            x = interpolate_to_shorter(
                result.x, H_prev, H, n, self.t_step,
                a_max=self.robot.a_max, v_max=self.robot.v_max,
                start_velocity=getattr(self.sqp_solver, "start_velocity", None),
                stats=_stats)
            if self.verbose:
                _ab = _stats.get("a_ratio_before", float("nan"))
                _aa = _stats.get("a_ratio_after", float("nan"))
                _v0 = _stats.get("v0_max", 0.0)
                print(f"GOMP: warm start H {H_prev}->{H} accel clamp: "
                      f"max|Δv|/(a_max·dt) {_ab:.3f} -> {_aa:.3f}"
                      f"{'' if _v0 < 1e-9 else f', |v_0|={_v0:.4f} (QP 会修回 0)'}")

        elapsed = time_module.time() - t0

        if best_trajectory is not None:
            print(f"GOMP: Done in {elapsed:.3f}s. "
                  f"H={best_trajectory.H}, "
                  f"duration={best_trajectory.duration:.3f}s")
        else:
            print(f"GOMP: Failed to find feasible trajectory "
                  f"({elapsed:.3f}s)")

        return best_trajectory

    def plan_multi_grasp(self,
                         grasp_candidates: List[Grasp],
                         grasp_goal: Grasp,
                         obstacles: Optional[DepthMapObstacle] = None,
                         held_object: Optional[AttachedObject] = None,
                         held_exempt_frac=(0.0, 0.0)) -> Optional[Trajectory]:
        """
        Run GOMP for multiple candidate grasps and return the fastest trajectory.

        From Section V: "For each object, we compute trajectories to all of its
        candidate grasp points in parallel, and then select the grasp that results
        in the shortest execution time."

        Parameters
        ----------
        grasp_candidates : list of Grasp
            Multiple candidate start grasps.
        grasp_goal : Grasp
            Goal grasp.
        obstacles : DepthMapObstacle or None
            Obstacle depth map.
        held_object : AttachedObject or None
            Object rigidly held by the gripper (see plan()).

        Returns
        -------
        trajectory : Trajectory or None
            Fastest trajectory among all candidates.
        """
        if self.verbose:
            print(f"GOMP Multi-Grasp: {len(grasp_candidates)} candidates")

        best_trajectory = None
        best_duration = np.inf

        for i, grasp_start in enumerate(grasp_candidates):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Candidate {i+1}/{len(grasp_candidates)} "
                      f"(quality={grasp_start.quality:.3f})")
                print(f"{'='*60}")

            traj = self.plan(grasp_start, grasp_goal, obstacles,
                             held_object=held_object,
                             held_exempt_frac=held_exempt_frac)

            if traj is not None and traj.duration < best_duration:
                best_trajectory = traj
                best_duration = traj.duration
                if self.verbose:
                    print(f"  → New best! duration={best_duration:.3f}s")

        if self.verbose:
            if best_trajectory is not None:
                print(f"\nGOMP Multi-Grasp: Best duration = "
                      f"{best_trajectory.duration:.3f}s")
            else:
                print(f"\nGOMP Multi-Grasp: No feasible trajectory found")

        return best_trajectory
