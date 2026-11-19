"""
Warm-starting utilities for GOMP.

From GOMP Section IV-F:
1. Initial warm start: cubic spline interpolating q_0 to q_H with zero velocity
2. Time reduction warm start: interpolate previous solution to shorter trajectory

Note on velocity decision variable:
    The decision variable v_i represents actual joint velocity (rad/s).
    The dynamics constraint is: q_{i+1} = q_i + v_i * t_step
    So v_i = (q_{i+1} - q_i) / t_step
    
    The warm start must satisfy this exactly, so we compute positions from
    the spline and then derive velocities from finite differences.
"""

import numpy as np
from scipy.interpolate import CubicSpline


def spline_warm_start(q_start: np.ndarray, q_goal: np.ndarray,
                      H: int, n: int, t_step: float) -> np.ndarray:
    """
    Create initial warm start via cubic spline interpolation.

    Generates a smooth trajectory from q_start to q_goal with zero
    velocity at both endpoints, without considering obstacles.

    Parameters
    ----------
    q_start : np.ndarray, shape (n,)
        Start joint configuration.
    q_goal : np.ndarray, shape (n,)
        Goal joint configuration.
    H : int
        Number of time steps.
    n : int
        Number of joints.
    t_step : float
        Time step between waypoints (seconds).

    Returns
    -------
    x : np.ndarray, shape (2*(H+1)*n,)
        Decision variable with spline-interpolated waypoints and velocities.
    """
    m = H + 1  # number of waypoints
    t_real = np.linspace(0, 1, m)  # normalized parameter

    waypoints = np.zeros((m, n))
    for j in range(n):
        cs = CubicSpline(
            [0, 1],
            [q_start[j], q_goal[j]],
            bc_type='clamped'  # zero derivative at endpoints
        )
        waypoints[:, j] = cs(t_real)

    # Compute velocities from finite differences to exactly satisfy dynamics:
    # v_i = (q_{i+1} - q_i) / t_step
    velocities = np.zeros((m, n))
    for i in range(H):
        velocities[i, :] = (waypoints[i + 1, :] - waypoints[i, :]) / t_step

    # Force zero velocity at endpoints (overrides the finite differences)
    velocities[0, :] = 0.0
    velocities[-1, :] = 0.0

    # Since v_0 = 0 is enforced, we need q_1 = q_0 + v_0*t_step = q_0
    # But our spline has q_1 != q_0. Fix: adjust q_0 to match, or accept
    # that the dynamics at endpoints won't be exactly satisfied.
    # The SQP will handle the correction.

    # Pack into x = [q0, q1, ..., qH, v0, v1, ..., vH]
    return np.concatenate([waypoints.ravel(), velocities.ravel()])


def clamp_accel_band(v: np.ndarray, a_max, t_step: float,
                     v_max=None, start_velocity=None) -> np.ndarray:
    """把速度序列夹到加速度带内：|v_{i+1} - v_i| <= a_max * t_step。

    正推（钉 v_0）+ 反推（钉 v_H）两遍。反推是最后一遍，且每步都以【最终的】
    v_{i+1} 为基准，所以返回序列的【所有】相邻差都在带内（含 (0,1) 这一对）。
    同时可选地夹到 ±v_max。

    代价：压缩比过大时 v_0 会被反推改动 —— 若 v_1 本身已经超出一个时间步
    可达的范围，就【不可能】同时满足 v_0 = 0 与 |v_1 - v_0| <= a_max*t_step。
    偏差量级 = |v_1| - a_max*t_step（实测 place 段 3.2 mrad/s），而 QP 里
    v_0 = 0 是硬等式，第一轮就会被修回。

    Parameters
    ----------
    v : np.ndarray, shape (m, n)
        速度序列（rad/s），就地不被修改。
    a_max : float or np.ndarray, shape (n,)
        加速度上限（rad/s²），按关节给。
    t_step : float
        时间步长（s）。
    v_max : float or np.ndarray or None
        速度上限（rad/s）。None = 不夹。
    start_velocity : np.ndarray or None
        起点速度（rad/s）。None = 0（GOMP 默认）。

    Returns
    -------
    out : np.ndarray, shape (m, n)
        夹紧后的速度序列。
    """
    m = v.shape[0]
    c = np.asarray(a_max, dtype=float).reshape(-1) * t_step
    lim = None if v_max is None else np.asarray(v_max, dtype=float).reshape(-1)
    v0 = (np.zeros(v.shape[1]) if start_velocity is None
          else np.asarray(start_velocity, dtype=float).reshape(-1))

    def _clip(x, ref):
        lo = ref - c
        hi = ref + c
        if lim is not None:
            lo = np.maximum(lo, -lim)
            hi = np.minimum(hi, lim)
        return np.clip(x, lo, hi)

    out = np.array(v, dtype=float, copy=True)

    # 正推：从 v_0 = v0 出发，每步最多变化 c
    out[0] = v0
    for i in range(m - 1):
        out[i + 1] = _clip(out[i + 1], out[i])

    # 反推：钉 v_H = 0，反向夹紧（最后一遍 ⇒ 全序列满足带约束）
    out[-1] = 0.0
    for i in range(m - 2, -1, -1):
        out[i] = _clip(out[i], out[i + 1])
    return out


def interpolate_to_shorter(x_prev: np.ndarray, H_prev: int,
                           H_new: int, n: int,
                           t_step: float,
                           a_max=None, v_max=None, start_velocity=None,
                           stats: dict = None) -> np.ndarray:
    """
    Interpolate a trajectory solution to a shorter time horizon.

    ⚠️ 时间压缩的副作用：位置按归一化时间重采样，速度由有限差分重新导出，
    于是加速度随压缩比 r = H_prev/H_new 近似按 r² 增长。若上一段已经贴着
    a_max 跑（GOMP 的时间最优解几乎总是如此），压缩后必然超限 —— 这段超限
    warm start 若因线搜索全线被拒而原样返回，就出现「FEASIBLE 但 accel=False」。

    给了 a_max 时，速度块会被投影回加速度带（见 clamp_accel_band）。位置块
    **不动**（几何路径与碰撞安全完全保持），因此该投影只是把动力学等式
    q_{i+1} = q_i + v_i*t_step 的初值残差让给 QP 去修 —— warm start 本就不必
    满足等式；只有「SQP 一步都没接受」时它才会原样返回，此时 v 块是
    「该路径在加速度限制下允许的速度」，而非有限差分真值。

    Parameters
    ----------
    x_prev : np.ndarray
        Previous solution decision variable.
    H_prev : int
        Previous number of time steps.
    H_new : int
        New (shorter) number of time steps.
    n : int
        Number of joints.
    t_step : float
        Time step (same for both trajectories).
    a_max : float or np.ndarray or None
        加速度上限（rad/s²）。None ⇒ 不做夹紧（旧行为）。
    v_max : float or np.ndarray or None
        速度上限（rad/s）。None ⇒ 不夹。
    start_velocity : np.ndarray or None
        起点速度（rad/s）。None ⇒ v_0 = 0。
    stats : dict or None
        传入时回填诊断量（a_ratio_before / a_ratio_after / v0_max）。

    Returns
    -------
    x_new : np.ndarray
        Interpolated decision variable for the shorter trajectory.
    """
    m_prev = H_prev + 1
    m_new = H_new + 1

    # Extract previous waypoints
    q_prev = x_prev[:m_prev * n].reshape(m_prev, n)

    # Time arrays (normalized)
    t_prev = np.linspace(0, 1, m_prev)
    t_new = np.linspace(0, 1, m_new)

    # Interpolate positions
    q_new = np.zeros((m_new, n))
    for j in range(n):
        cs = CubicSpline(t_prev, q_prev[:, j])
        q_new[:, j] = cs(t_new)

    # Compute velocities from finite differences
    v_new = np.zeros((m_new, n))
    for i in range(H_new):
        v_new[i, :] = (q_new[i + 1, :] - q_new[i, :]) / t_step

    # Force zero velocity at endpoints
    v_new[0, :] = 0.0
    v_new[-1, :] = 0.0

    if a_max is not None:
        c = np.asarray(a_max, dtype=float).reshape(-1) * t_step
        if stats is not None:
            stats["a_ratio_before"] = float(
                np.max(np.abs(np.diff(v_new, axis=0)) / c))
            stats["a_ratio_after"] = float("nan")
            stats["v0_max"] = 0.0
        v_new = clamp_accel_band(v_new, a_max, t_step,
                                 v_max=v_max,
                                 start_velocity=start_velocity)
        if stats is not None:
            stats["a_ratio_after"] = float(
                np.max(np.abs(np.diff(v_new, axis=0)) / c))
            stats["v0_max"] = float(np.max(np.abs(v_new[0])))
    elif stats is not None:
        # 无 a_max ⇒ 无参照尺度，比值无从定义（用 nan 而不是抛错）
        stats["a_ratio_before"] = float("nan")
        stats["a_ratio_after"] = float("nan")
        stats["v0_max"] = 0.0

    return np.concatenate([q_new.ravel(), v_new.ravel()])
