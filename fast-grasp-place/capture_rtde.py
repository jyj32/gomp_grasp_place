"""live 拍照用的最小 RTDE 封装（从 yanjiuyuan/pick_place_rtde_utils.py 精简抄入）。

只保留 ``detect_box_pose.capture_world_pointcloud()`` 需要的两个动作：
获取共享 RTDE 会话、运动到拍照构型。原文件的 3000 余行规划/执行逻辑不搬。
⛔ live 模式会驱动真机；本模块不含任何自动执行入口。
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from robot_con.ur.ur7e_dh76_rtde import get_rtde_robot_singleton

_rtde_robot_shared = None
_FT_ZERO_MAX_ATTEMPTS = 3
_FT_ZERO_RETRY_DELAY_S = 0.2


def get_rtde_robot(args: SimpleNamespace):
    """获取（懒加载）全局共享的 RTDE 会话，避免重复建连。

    Args:
        args: 运行时配置，需含 ``robot_ip``（str）与 ``gp_port``（str）；两者传给
            ``get_rtde_robot_singleton``。其余字段忽略。

    Returns:
        object: ``ur7e_dh76_rtde`` 的模块级 RTDE 单例（同一进程内复用）。
    """
    global _rtde_robot_shared
    if _rtde_robot_shared is not None:
        return _rtde_robot_shared
    _rtde_robot_shared = get_rtde_robot_singleton(args.robot_ip, args.gp_port)
    return _rtde_robot_shared


def _zero_ft_sensor_best_effort(rtde_robot, *, context: str) -> bool:
    """尝试清零 FT 力传感器；控制器拒绝时重试，仍失败只告警不抛异常。

    Args:
        rtde_robot: RTDE 机器人对象，需有 ``rtde_c``（或 ``_rtde_c``）控制接口。
        context: 日志用的场景标签（str），例如 ``"capture_conf_rad"``。

    Returns:
        bool: 清零成功为 True；三次都被控制器拒绝为 False。

    Raises:
        RuntimeError: 拿不到 RTDE 控制接口。
    """
    ctl = getattr(rtde_robot, "rtde_c", getattr(rtde_robot, "_rtde_c", None))
    if ctl is None:
        raise RuntimeError("RTDE control interface is unavailable; cannot zero FT sensor")
    for attempt in range(1, _FT_ZERO_MAX_ATTEMPTS + 1):
        if ctl.zeroFtSensor() is not False:
            return True
        if attempt < _FT_ZERO_MAX_ATTEMPTS:
            print(f"[box_pose] FT 清零被控制器拒绝（{context}，第 {attempt}/"
                  f"{_FT_ZERO_MAX_ATTEMPTS} 次），{_FT_ZERO_RETRY_DELAY_S}s 后重试...")
            time.sleep(_FT_ZERO_RETRY_DELAY_S)
    return False


def move_to_capture_point(args: SimpleNamespace, *, rtde_robot,
                          slow_v: float = 3.14, slow_a: float = 3.14,
                          fast_v: float = 3.14, fast_a: float = 5.0) -> None:
    """把机器人移到拍照构型 ``capture_conf_rad``（弧度，直接下发，不做 deg 转换）。

    顺序：先 ``moveL`` 竖直上移到拍照点高度（仅改 z，避免从低位斜着爬升），
    再 ``move_jnts`` 到拍照关节构型，最后尽力清零 FT 传感器。竖直上移失败只告警
    不中断，仍继续 moveJ。

    Args:
        args: 运行时配置，需含 ``capture_conf_rad``（6 个弧度值，来自
            ``CONFIG["joints"]["capture_conf_rad"]``）与 ``capture_pose``（6 位姿值，
            只用 ``[2]`` 的 z 分量做预抬升）；``capture_conf_rad`` 为 ``None`` 直接返回。
            ``execute_dry_run`` 或 ``mock`` 为真时只打印、不运动。
        rtde_robot: ``get_rtde_robot`` 返回的 RTDE 会话对象。
        slow_v: 竖直预抬升 ``moveL`` 的线速度（m/s），默认 3.14。
        slow_a: 竖直预抬升 ``moveL`` 的线加速度（m/s²），默认 3.14。
        fast_v: 到拍照构型 ``move_jnts`` 的关节速度（rad/s），默认 3.14。
        fast_a: 到拍照构型 ``move_jnts`` 的关节加速度（rad/s²），默认 5.0。

    Returns:
        None: 无返回值；运动完成（或被 dry-run 跳过）后返回。
    """
    conf_rad = getattr(args, "capture_conf_rad", None)
    if conf_rad is None:
        return
    conf_rad = np.asarray(conf_rad, dtype=float).reshape(-1)
    dry_run = bool(getattr(args, "execute_dry_run", False) or getattr(args, "mock", False))
    if dry_run:
        print("[box_pose] Dry-run: 跳过拍照点运动。")
        return

    try:
        cur = np.asarray(rtde_robot.getActualTCPPose(), dtype=float).reshape(6)
        cap_z = float(args.capture_pose[2])
        prelift = cur.copy()
        prelift[2] = cap_z
        if float(prelift[2]) > float(cur[2]) + 1e-4:
            rtde_robot.moveL(prelift.tolist(), float(slow_v),
                             float(slow_a), wait=True)
        else:
            print("[box_pose] 跳过拍照点前的竖直上移（当前 z 已足够高）")
    except Exception as exc:  # noqa: BLE001
        print(f"[box_pose] 竖直上移失败，直接 moveJ 到拍照点：{exc}")

    rtde_robot.move_jnts(conf_rad, vel=float(fast_v),
                         acc=float(fast_a), wait=True)
    if _zero_ft_sensor_best_effort(rtde_robot, context="capture_conf_rad"):
        print("[box_pose] FT sensor zeroed at capture_conf_rad.")
    print("[box_pose] 已运动到拍照点。")
