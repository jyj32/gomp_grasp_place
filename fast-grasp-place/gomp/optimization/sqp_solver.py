"""
Sequential Quadratic Programming (SQP) solver for GOMP.

From GOMP Section IV: The SQP method solves a sequence of QP subproblems,
iteratively updating trust regions and linearized constraints until
convergence or infeasibility.

Uses OSQP as the underlying QP solver for its warm-starting capabilities
and infeasibility detection.

Performance optimizations:
- OSQP solver reuse via in-place update when sparsity pattern is unchanged
- Pre-computed static constraints (reused across iterations)
- Timing instrumentation for profiling
"""

import numpy as np
import osqp
import scipy.sparse as sp
from dataclasses import dataclass
from typing import Optional
import time as time_module

from gomp.optimization.qp_builder import QPBuilder
from gomp.optimization.constraints import ConstraintBuilder
from gomp.ur7e_robot_adapter import RobotAdapter
from gomp.grasp.grasp_set import GraspSet
from gomp.obstacles.collision import ObstacleConstraint


@dataclass
class SQPResult:
    """
    Result of the SQP optimization.

    Attributes
    ----------
    x : np.ndarray or None
        Solution decision variable, or None if infeasible.
    feasible : bool
        Whether a feasible solution was found.
    n_iterations : int
        Number of SQP iterations performed.
    final_cost : float
        Final objective value.
    constraint_violation : float
        Maximum constraint violation at convergence.
    n_accepted : int
        Number of accepted steps (线搜索通过并更新 x_current 的次数)。
        `feasible=True` 且 `n_accepted==0` 表示返回的其实是 warm start：
        SQP 一步都没走成功（全被线搜索拒绝），此时 x 的位置块就是初值。
    """
    x: Optional[np.ndarray]
    feasible: bool
    n_iterations: int
    final_cost: float
    constraint_violation: float
    n_accepted: int = 0


class SQPSolver:
    """
    SQP solver for the GOMP trajectory optimization.

    Iteratively solves QP subproblems with updated linearizations
    of the obstacle and grasp constraints, and adaptively sized
    trust regions.

    Parameters
    ----------
    max_iterations : int
        Maximum number of SQP iterations.
    constraint_tol : float
        Tolerance for constraint satisfaction.
    trust_region_tol : float
        Tolerance for trust region convergence.
    initial_trust_region : float
        Initial trust region radius.
    trust_region_expand : float
        Factor to expand trust region on improvement.
    trust_region_shrink : float
        Factor to shrink trust region on violation.
    endpoint_jnt_tol : float
        Joint-space tolerance for endpoint pinning (radians).
        Controls how tightly the endpoint joints are constrained
        near the IK solution to prevent FK drift.
    stall_window : int
        Stagnation-stopping window (iterations). 0 disables it (default).
        When > 0 and the trust region is already pinned at TRUST_REGION_MIN
        (i.e. the step size is budget-limited rather than the solve being
        unconverged), the solve stops as soon as the relative cost
        improvement over the last `stall_window` iterations drops below
        `stall_rtol`. Useful for "polish an already feasible warm start"
        runs, where the tail iterations only crawl along a flat valley.
    stall_rtol : float
        Relative cost-improvement threshold for `stall_window`.
    osqp_eps : float
        OSQP primal/dual tolerance (eps_abs = eps_rel). Loosening it from
        1e-5 to ~1e-4 cuts per-iteration solve time noticeably; GOMP only
        needs the descent direction from each QP subproblem.
    osqp_max_iter : int
        OSQP maximum iterations per subproblem.
    osqp_polish : bool
        OSQP polish step (solution refinement after convergence).
    line_search : bool
        回溯线搜索开关（默认开）。QP 子问题用的是【一阶代理模型】：障碍
        高度 z_obs 被当作常数、J 只含 z 平移行，因此对双线性高度场里
        「6mm 内 z_obs 涨 0.889m」这种陡坡完全失明 —— 一步大跨格能让真实
        穿透达 0.1m。开这个开关后，每步先用真实（非线性）高度场验收
        max_violation，恶化则缩步长。这本质上是把标准 SQP 应有的 merit
        function 验收补回来。
    line_search_max_steps : int
        步长减半的最多次数（α = 1, 1/2, ... 1/2^k）。
    line_search_shrink : float
        每次回溯的步长系数（默认 0.5）。
    line_search_abs_tol : float
        允许的真实违例恶化量（米）。判据是 `φ_try <= φ_cur + abs_tol`，
        以【当前点的违例水平】为基准 —— 不能用绝对 0：place 段起点握持
        物体仍在箱内时 φ_cur 本身就 > 0，绝对门槛会让 α 缩到底也不通过
        （死锁）。⚠️ 基准也不能写成 `max(φ_cur, 0)`：φ_cur<0（有净空）
        时它退化成「允许恶化到 +abs_tol」，净空每轮溜走 5mm、多轮累积
        成棘轮（实测 place 段 -0.0339 → -0.0023 → +0.0049 → +0.0094 →
        下一轮不可行）。
    verbose : bool
        Print iteration info.
    """

    # 信任域硬边界（值与原 clip 硬编码一致）。⚠️ 看 TRUST_REGION_MIN 时必须连带看
    # trust_region_tol：MIN(0.01) > tol(1e-3)，所以当每步解都顶在信任域边界
    # （delta_x ≈ trust_r）时，「收敛」在数学上不可达 —— 只能靠 max_iterations
    # 或 stall_window 停滞判据兜底。精修段 60 轮里有 30 轮就是这样白跑的。
    TRUST_REGION_MIN = 0.01
    TRUST_REGION_MAX = 5.0

    def __init__(self,
                 max_iterations: int = 50,
                 constraint_tol: float = 1e-4,
                 trust_region_tol: float = 1e-3,
                 initial_trust_region: float = 1.0,
                 trust_region_expand: float = 1.5,
                 trust_region_shrink: float = 0.5,
                 endpoint_jnt_tol: float = 0.02,
                 stall_window: int = 0,
                 stall_rtol: float = 0.08,
                 osqp_eps: float = 1e-5,
                 osqp_max_iter: int = 4000,
                 osqp_polish: bool = True,
                 osqp_eps_prim_inf: float = 1e-4,
                 start_velocity: Optional[np.ndarray] = None,
                 infeas_retry_limit: int = 0,
                 line_search: bool = True,
                 line_search_max_steps: int = 5,
                 line_search_shrink: float = 0.5,
                 line_search_abs_tol: float = 0.005,
                 verbose: bool = False):
        self.max_iterations = max_iterations
        self.constraint_tol = constraint_tol
        self.trust_region_tol = trust_region_tol
        self.initial_trust_region = initial_trust_region
        self.trust_region_expand = trust_region_expand
        self.trust_region_shrink = trust_region_shrink
        self.endpoint_jnt_tol = endpoint_jnt_tol
        # 停滞判据（stall_window=0 关闭）：见类属性 TRUST_REGION_MIN 的注释。
        self.stall_window = stall_window
        self.stall_rtol = stall_rtol
        # OSQP 精度（默认 = 原硬编码值，行为不变）
        self.osqp_eps = osqp_eps
        self.osqp_max_iter = osqp_max_iter
        self.osqp_polish = osqp_polish
        # OSQP 判「原问题不可行」的容差（默认 1e-4 = 原 OSQP 硬编码值）。
        # H 递减扫描的最后一档必然不可行，该档要花两次 max_iter 才能证明；
        # 放宽到 1e-2 后同一档迭代数约减半，耗时 148ms→37ms，而可行档的解
        # 逐点不变（eps_prim_inf 只影响不可行判定，不参与可行问题的迭代）。
        # 边界：2e-2 起会误判可行档不可行（H 从 140 退化到 176）；5e-2 直接
        # 秒判不可行返回 None。1e-2 是实测安全上限。
        self.osqp_eps_prim_inf = float(osqp_eps_prim_inf)
        # 起点速度（None ⇒ v_0 = 0）；透传给 ConstraintBuilder。
        self.start_velocity = start_velocity
        # OSQP 判不可行时的扩域重试次数上限（0 = 关闭，恢复旧的硬失败行为）。
        # 见 solve() 内 infeasible 分支的说明。
        self.infeas_retry_limit = int(infeas_retry_limit)
        # 回溯线搜索（默认开）：对真实非线性净空验收 QP 步长，见类 docstring。
        self.line_search = bool(line_search)
        self.line_search_max_steps = int(line_search_max_steps)
        self.line_search_shrink = float(line_search_shrink)
        self.line_search_abs_tol = float(line_search_abs_tol)
        self.verbose = verbose

    def solve(self, x_init: np.ndarray,
              H: int, n: int, t_step: float,
              robot: RobotAdapter,
              obstacle_constraint: Optional[ObstacleConstraint] = None,
              start_grasp_set: Optional[GraspSet] = None,
              goal_grasp_set: Optional[GraspSet] = None,
              fast_kin=None,
              merit_anchor: Optional[float] = None) -> SQPResult:
        """
        Run the SQP loop for a fixed horizon H.

        Parameters
        ----------
        x_init : np.ndarray
            Initial guess for the decision variable.
        H : int
            Time horizon (number of steps).
        n : int
            Number of DOFs.
        t_step : float
            Time step between waypoints.
        robot : RobotAdapter
            Robot model.
        obstacle_constraint : ObstacleConstraint or None
            Obstacle model.
        start_grasp_set : GraspSet or None
            Start grasp set with DOF.
        goal_grasp_set : GraspSet or None
            Goal grasp set with DOF.
        fast_kin : FastKinematics or None
            Pre-compiled fast kinematics for the grasp constraint
            Jacobian computation.
        merit_anchor : float or None
            本段【整段不变】的违例锚点（米）。给定时所有 H 的线搜索绝对帽都
            锚在它上面，避免「每个 H 都以前一个 H 的结果重新起锚」造成跨 H
            棘轮（实测 place 4.8→9.7→14.6mm，物理间隙被吃到 0.3mm）。
            None = 沿用旧行为（每个 H 各以自己的 warm start 起锚）。

        Returns
        -------
        SQPResult
            The result of the optimization.
        """
        qp = QPBuilder(H, n)
        cb = ConstraintBuilder(qp, t_step, robot, fast_kin=fast_kin,
                               start_velocity=self.start_velocity)

        # Build objective (doesn't change)
        P = qp.build_P()
        p = qp.build_p()
        P_upper = sp.triu(P, format='csc')

        # Build static constraints (don't change between SQP iterations)
        A_static, l_static, u_static = cb.build_static_constraints()

        # Initialize
        x_current = x_init.copy()
        trust_radius = self.initial_trust_region
        prev_nnz = None  # Track sparsity pattern for OSQP reuse
        osqp_solver = None
        cost_hist = []  # 每轮目标值，供停滞判据按窗口比较
        infeas_retries = 0  # OSQP 报不可行后已扩域重试的次数
        phi_start = merit_anchor  # 线搜索绝对帽的锚点（warm start 的真实违例）
        _merit_printed = False
        n_accepted = 0  # 被接受的步数（0 ⇒ 返回的是 warm start 兜底）

        for iteration in range(self.max_iterations):
            # Extract current waypoints for linearization
            waypoints = qp.extract_waypoints(x_current)

            # Build dynamic constraints (updated each iteration)
            t0 = time_module.perf_counter() if self.verbose else 0
            A_dynamic, l_dynamic, u_dynamic = cb.build_dynamic_constraints(
                waypoints=waypoints,
                obstacle_constraint=obstacle_constraint,
                start_grasp_set=start_grasp_set,
                goal_grasp_set=goal_grasp_set,
                trust_region=trust_radius,
                endpoint_jnt_tol=self.endpoint_jnt_tol
            )
            t_constraints = time_module.perf_counter() - t0 if self.verbose else 0

            # Stack all constraints
            A = sp.vstack([A_static, A_dynamic], format='csc')
            l = np.concatenate([l_static, l_dynamic])
            u = np.concatenate([u_static, u_dynamic])

            # Try to reuse OSQP solver if sparsity pattern is unchanged
            t0 = time_module.perf_counter() if self.verbose else 0
            current_nnz = A.nnz

            if osqp_solver is not None and current_nnz == prev_nnz:
                # Attempt in-place update (same sparsity pattern)
                try:
                    osqp_solver.update(Ax=A.data, l=l, u=u)
                    osqp_solver.warm_start(x=x_current)
                except Exception:
                    # Sparsity pattern changed despite same nnz — recreate
                    osqp_solver = self._create_solver(
                        P_upper, p, A, l, u, x_current,
                        eps=self.osqp_eps, max_iter=self.osqp_max_iter,
                        polish=self.osqp_polish,
                        eps_prim_inf=self.osqp_eps_prim_inf)
            else:
                osqp_solver = self._create_solver(
                    P_upper, p, A, l, u, x_current,
                    eps=self.osqp_eps, max_iter=self.osqp_max_iter,
                    polish=self.osqp_polish,
                    eps_prim_inf=self.osqp_eps_prim_inf)
            prev_nnz = current_nnz

            result = osqp_solver.solve()
            t_solve = time_module.perf_counter() - t0 if self.verbose else 0

            # Check feasibility
            # OSQP 实际返回的是 'primal infeasible inaccurate'（无括号），
            # 只判带括号的写法会漏匹配：不可行轮次会误走下方「缩域 continue」
            # 分支，白跑一轮并把信任域继续磨向 TRUST_REGION_MIN。三种写法全收。
            if result.info.status in ('primal infeasible',
                                      'primal infeasible inaccurate',
                                      'primal infeasible (inaccurate)'):
                # OSQP 判不可行时是否扩信任域重试。默认关闭（limit=0）。
                # 曾试过开启：实测在"上一步行长跨过 z_obs 阶跃（高度场格边界）"
                # 导致的不可行里，扩域只会让一阶线性化更失真 —— 扩到
                # trust_r=1.0 仍不可行，白花 0.2~0.5s 后才硬失败。
                # 真正对症的是【回退步长】（线搜索 / merit 比较），不是放大步长。
                # 保留此路径仅供实验，默认不启用。
                if (infeas_retries < self.infeas_retry_limit
                        and trust_radius < self.TRUST_REGION_MAX - 1e-12):
                    infeas_retries += 1
                    trust_radius = min(trust_radius / self.trust_region_shrink,
                                       self.TRUST_REGION_MAX)
                    if self.verbose:
                        print(f"  SQP iter {iteration}: OSQP infeasible "
                              f"({result.info.status}) -> retry "
                              f"{infeas_retries}/{self.infeas_retry_limit} "
                              f"with trust_r={trust_radius:.4f}")
                    continue
                if self.verbose:
                    # [调试] x_current 是否满足本轮线性化后的 (A,l,u)：若满足则
                    # OSQP 报不可行属数值误判；若违反则给出最差行（static/dynamic）
                    try:
                        Ax = A @ x_current
                        viol = np.maximum(l - Ax, Ax - u)
                        n_stat = A_static.shape[0]
                        order = np.argsort(-viol)[:5]
                        _rows = [(int(r), 'static' if int(r) < n_stat else 'dynamic',
                                  float(viol[r])) for r in order]
                        print(f"  SQP iter {iteration}: INFEASIBLE "
                              f"status={result.info.status} "
                              f"x_current_maxviol={viol.max():.3e} "
                              f"(static {n_stat} rows / dynamic {A_dynamic.shape[0]}) "
                              f"top={_rows}")
                    except Exception as _e:
                        print(f"  SQP iter {iteration}: INFEASIBLE "
                              f"(diag failed: {_e})")
                return SQPResult(
                    x=None,
                    feasible=False,
                    n_iterations=iteration + 1,
                    final_cost=np.inf,
                    constraint_violation=np.inf
                )

            if result.info.status not in ('solved', 'solved inaccurate'):
                if self.verbose:
                    print(f"  SQP iter {iteration}: status={result.info.status}")
                # Try shrinking trust region
                trust_radius *= self.trust_region_shrink
                if trust_radius < 1e-6:
                    return SQPResult(
                        x=None, feasible=False,
                        n_iterations=iteration + 1,
                        final_cost=np.inf,
                        constraint_violation=np.inf
                    )
                continue

            x_new = result.x

            # ---- 回溯线搜索：对真实（非线性）净空验收 QP 步长 ----
            # 锚点 x_current 对本轮线性化约束恰好取等号（rhs 定义的必然结果：
            # 代入 q_{k+1}=q_k 得 p_z(q_k) >= z_obs(q_k)+m+extra，即 φ(x_k)<=0）。
            # 速度/信任域/端点钉扎全是凸线性约束，凸组合 x(α)=x_k+αΔ 对它们
            # 全部安全 ⇒ 唯一会变差的量是真实 φ，而 φ(α→0)=φ(x_k) 已知有限。
            alpha = 1.0
            phi_cur = None
            phi_proxy = None
            phi_real = None
            ls_rejected = False
            if (self.line_search and self.line_search_max_steps > 0
                    and obstacle_constraint is not None):
                # 基准取【当前点】的真实违例，只接受不恶化的步；phi_start 是
                # warm start 的违例水平，作为绝对帽的锚点（见 _line_search）。
                # use_real_merit=True 时 phi_cur = max(proxy, real)，phi_start 由
                # 同一函数取，锚点自动同口径（不同口径会重新引入棘轮）。
                _wps = qp.extract_waypoints(x_current)
                _sp = getattr(obstacle_constraint, "max_violation_split", None)
                if _sp is not None:
                    # verbose 时无条件算 real（诊断要看），否则跟随 use_real_merit
                    phi_proxy, phi_real, phi_cur = _sp(
                        _wps, want_real=True if self.verbose else None)
                else:
                    phi_cur = obstacle_constraint.max_violation(_wps)
                if not _merit_printed:
                    _merit_printed = True
                    if phi_start is None:
                        phi_start = phi_cur
                    # [diag] 锚点的 proxy/real 拆分（真实几何 15 轴 SAT，带
                    # wp/box 定位）。每段只跑一次，成本可忽略。
                    _rf = getattr(obstacle_constraint, "max_violation_real", None)
                    if _rf is not None and self.verbose:
                        _vr, _wr, _kr = _rf(_wps)
                        print(f"  [merit] phi_start "
                              f"{'(anchor) ' if merit_anchor is not None else ''}"
                              f"proxy={phi_proxy:+.4f} "
                              f"({phi_proxy * 1000:+.1f}mm)  "
                              f"real={_vr:+.4f} ({_vr * 1000:+.1f}mm) "
                              f"@wp{_wr}/box{_kr}")
                alpha, _, ok_ls = self._line_search(
                    obstacle_constraint, qp, x_current, x_new,
                    phi_cur, phi_start)
                ls_rejected = not ok_ls

            if ls_rejected:
                # 不更新 x_current（它本身是 φ=0 的安全点），缩信任域重解
                trust_radius = max(trust_radius * self.trust_region_shrink,
                                   self.TRUST_REGION_MIN)
                if trust_radius <= self.TRUST_REGION_MIN + 1e-12:
                    if self.verbose:
                        print(f"  SQP iter {iteration}: LS-FAIL "
                              f"(α_min={self.line_search_shrink ** self.line_search_max_steps:.4f}, "
                              f"phi_cur={phi_cur:.4f}) — 无下降步，停在当前点")
                    break
                if self.verbose:
                    print(f"  SQP iter {iteration}: LS-FAIL "
                          f"(phi_cur={phi_cur:.4f}) -> retry with "
                          f"trust_r={trust_radius:.4f}")
                continue

            x_acc = (x_new if alpha == 1.0
                     else x_current + alpha * (x_new - x_current))

            # 步长/收敛/代价一律用【实际接受点】算。若线搜索砍了步长却仍按
            # 未缩放的解判断「顶到信任域边界」，信任域会被误缩到下限后卡死。
            #
            # 且必须只取 q 块：x = [q; v]（qp_builder.py:42，n_vars = 2*m*n），
            # 而 _trust_region_constraints 只作用于 x[:m*n]。若取全 x 的最大值，
            # delta_x 会被速度块主导（首轮 v 从初值跳到合理值可差 0.5 rad/s），
            # 于是每轮都满足 delta_x > 0.75*trust_r → 一路 shrink 到下限，
            # expand 分支永远进不去（实测 grasp 段 trust_r 卡死 0.01、35 轮爬坡 4.47s）。
            _nq = qp.n_waypoints * qp.n
            delta_x = float(np.max(np.abs(x_acc[:_nq] - x_current[:_nq])))
            delta_v = float(np.max(np.abs(x_acc[_nq:] - x_current[_nq:])))

            # Check convergence
            # 线搜索介入（alpha<1）时 delta_x 被人为缩小，不能凭它判「收敛」——
            # 那是模型不可信的信号，不是已到最优点。要求 alpha==1 才算真收敛。
            converged = (delta_x < self.trust_region_tol) and (alpha == 1.0)

            # 目标值：停滞判据要用，verbose 打印也要用
            cost = 0.5 * x_acc @ P @ x_acc
            cost_hist.append(cost)

            if self.verbose:
                _ls_msg = "" if alpha == 1.0 else f", LS α={alpha:.4f}"
                _phi_msg = "" if phi_cur is None else f", phi={phi_cur:.4f}"
                # [diag] proxy / real 拆分（与 phi_cur 同一次 FK，不额外开销）
                if phi_real is not None and np.isfinite(phi_real):
                    _phi_msg += (f"(proxy={phi_proxy:+.4f},"
                                 f"real={phi_real * 1000:+.1f}mm)")
                print(f"  SQP iter {iteration}: cost={cost:.6f}, "
                      f"delta_x={delta_x:.6f} (q/v={delta_x:.4f}/{delta_v:.4f}), "
                      f"trust_r={trust_radius:.4f}, "
                      f"constraints={t_constraints*1000:.1f}ms, "
                      f"solve={t_solve*1000:.1f}ms"
                      f"({getattr(result.info, 'iter', -1)}it)"
                      f"{_ls_msg}{_phi_msg}")

            # Update solution
            x_current = x_acc
            n_accepted += 1
            infeas_retries = 0  # 成功一轮即重置扩域配额（只统计「连续」不可行）

            if converged:
                break

            # 停滞判据（可选，默认关闭）：信任域已顶在下限 ⇒ 步长是被预算卡住的，
            # 不是"还没收敛"；此时若窗口内相对 cost 改善已低于 stall_rtol，继续迭代
            # 只是以最小步长爬坡（实测精修段后 30 轮只买 cost -10.8%），直接收工。
            if self.stall_window > 0 and len(cost_hist) > self.stall_window:
                c_ref = cost_hist[-1 - self.stall_window]
                if (trust_radius <= self.TRUST_REGION_MIN + 1e-12 and c_ref > 0
                        and (c_ref - cost) <= self.stall_rtol * c_ref):
                    if self.verbose:
                        print(f"  SQP iter {iteration}: STALL "
                              f"(window={self.stall_window}, "
                              f"rel_improve={(c_ref - cost) / c_ref:.4f} "
                              f"<= {self.stall_rtol})")
                    break

            # Adaptive trust region：唯一判据是【线搜索有没有退步】
            #   α<1        ⇒ 真实净空逼我们缩步 ⇒ 一阶模型在 δ 尺度上不可信，缩域；
            #   α=1 且顶满 ⇒ 这一步被真实高度场逐点验收通过 ⇒ 模型可信，放大；
            #   α=1 未顶满 ⇒ QP 自己不需要走那么远（接近局部最优），保持不变。
            #
            # 原规则里的 `delta_x < 0.25δ → 放大` 和 `delta_x > 0.75δ → 缩小` 是
            # 「步长占满率」判据，对线性 QP + 二次目标不成立：QP 几乎总会走满
            # 边界，于是「顶满即缩」每轮必中 ⇒ trust_r 单调缩到 TRUST_REGION_MIN
            # 再放不回去（实测 grasp 段 38 轮 × 0.01rad 爬坡 4.38s，而全程 α=1，
            # 即模型每一步都是准的）。可信与否只由线搜索回答。
            if alpha < 1.0:
                trust_radius *= self.trust_region_shrink
            elif delta_x > trust_radius * 0.75:
                trust_radius *= self.trust_region_expand

            trust_radius = np.clip(trust_radius, self.TRUST_REGION_MIN,
                                   self.TRUST_REGION_MAX)

        # Compute final cost and constraint violation
        final_cost = 0.5 * x_current @ P @ x_current
        constraint_violation = self._compute_constraint_violation(
            A, l, u, x_current)

        return SQPResult(
            x=x_current,
            feasible=True,
            n_iterations=iteration + 1,
            final_cost=final_cost,
            constraint_violation=constraint_violation,
            n_accepted=n_accepted
        )

    def _line_search(self, obstacle_constraint, qp, x_current, x_new,
                     phi_cur, phi_start=None):
        """在 x(α)=x_current+α(x_new-x_current) 上做回溯，返回 (alpha, phi, ok)。

        双重判据，取更严者：
            phi_allow = min( max(φ_cur, 0) + tol ,  max(φ_start, 0) + tol )

        - 前项（相对）：φ_cur ≤ 0（有净空）时等于 `tol`，即允许净空一直缩到
          「刚好 5mm 穿透」为止 —— 这是 GOMP 时间最优的正常行为（实测 grasp
          段靠 8.78cm→2.34cm 换时间），不能禁。φ_cur > 0 时随当前值抬升。
        - 后项（绝对帽）：锚定在 warm start 的违例水平 φ_start 上。没有它，
          φ_cur > 0 之后每轮都能再涨 tol，一路累积成棘轮（实测 place 段
          -0.0339 → -0.0023 → +0.0049 → +0.0094 → +0.0142 不可行）；
          有了它，φ_start ≤ 0 时上限恒为 tol（最多 5mm 穿透），φ_start > 0
          （豁免段起点物体在箱内）时上限为 φ_start + tol（允许维持、不许爬）。
        α→0 时 φ=φ_cur 恒满足，所以任何情况都不会死锁。
        """
        if phi_start is None:
            phi_start = phi_cur
        alpha = 1.0
        phi_cap = max(phi_start, 0.0) + self.line_search_abs_tol
        phi_allow = min(max(phi_cur, 0.0) + self.line_search_abs_tol, phi_cap)
        phi_try = phi_cur
        for _ in range(self.line_search_max_steps):
            x_try = (x_new if alpha == 1.0
                     else x_current + alpha * (x_new - x_current))
            phi_try = obstacle_constraint.max_violation(qp.extract_waypoints(x_try))
            if phi_try <= phi_allow:
                return alpha, phi_try, True
            alpha *= self.line_search_shrink
        # 全部失败：回传最后一次实际尝试过的 alpha（调用方不使用）
        return (alpha / self.line_search_shrink if self.line_search_max_steps > 0
                else 1.0), phi_try, False

    @staticmethod
    def _create_solver(P_upper, p, A, l, u, x_warm,
                       eps=1e-5, max_iter=4000, polish=True,
                       eps_prim_inf=1e-4):
        """Create a fresh OSQP solver instance."""
        solver = osqp.OSQP()
        solver.setup(
            P=P_upper,
            q=p,
            A=A,
            l=l,
            u=u,
            eps_abs=eps,
            eps_rel=eps,
            eps_prim_inf=eps_prim_inf,
            max_iter=max_iter,
            warm_start=True,
            verbose=False,
            polish=polish
        )
        solver.warm_start(x=x_warm)
        return solver

    @staticmethod
    def _compute_constraint_violation(A, l, u, x) -> float:
        """Compute maximum constraint violation."""
        Ax = A @ x
        lower_viol = np.maximum(0, l - Ax)
        upper_viol = np.maximum(0, Ax - u)
        return max(np.max(lower_viol), np.max(upper_viol))
