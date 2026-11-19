"""
执行层：按**时间顺序**做完一次 pick&place（对齐 anygrasp_wrs/d435_grasp/robot_process.py）。
"""

from __future__ import annotations

import json
import queue
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

import config as cfg
import grasp_plan


class GraspFailed(RuntimeError):
    """执行期的业务失败（抓空 / 未到位），与「程序 bug」区分开，便于上层单独处理。"""


def _to_stroke(width_full):
    """指尖全距 → 夹爪行程 stroke。

    Args:
        width_full: float 目标指尖全距（米）。有效区间
            ``[cfg.CONFIG["gripper"]["zero_gap"], cfg.CONFIG["gripper"]["open_width_full"]]`` = [0.016, 0.092]。

    Returns:
        float: 行程（米）= ``width_full - 0.016``，并 clamp 到
        ``[0, cfg.CONFIG["gripper"]["max_stroke"]]`` = [0, 0.076]。
        ⚠️ 超出上界的输入会被**静默截断**成 0.076，不报错 —— 需要区分「夹到底」
        和「下发了非法值」的调用方请自行先校验。
    """
    stroke = float(width_full) - float(cfg.CONFIG["gripper"]["zero_gap"])
    return float(min(max(stroke, 0.0), float(cfg.CONFIG["gripper"]["max_stroke"])))


def _from_stroke(stroke):
    """夹爪行程 stroke → 指尖全距。

    Args:
        stroke: float | None 驱动回读的行程（米），0 = 全闭、0.076 = 全开。
            None 表示串口回读失败。

    Returns:
        float | None: 指尖全距（米）= ``stroke + 0.016``；``stroke`` 为 None 时返回 None
        （**不要**当成 0，否则会把「读不到」误判成「空夹到底」）。
    """
    if stroke is None:
        return None
    return float(stroke) + float(cfg.CONFIG["gripper"]["zero_gap"])


def make_gripper(port=None, max_jaw_width=None):
    """构造 AG145 串口夹爪驱动（真机夹爪的唯一构造入口）。

    ``ag145.Ag145driver.__init__`` 会打开串口、必要时做一次夹爪初始化（回零），
    最后 ``open_g()`` 张开到全开 —— 也就是说**调用本函数夹爪会动一下**。
    所以只在真机执行路径上调用。

    Args:
        port: str | None 串口设备路径；None 用 ``cfg.CONFIG["robot"]["gp_port"]``（当前
            ``/dev/ttyUSB0``）。设备不存在抛 ``FileNotFoundError``，无读写权限抛
            ``PermissionError``，端口被别的进程占住抛 ``ConnectionError``。
        max_jaw_width: float | None 驱动口径的**行程上限**（米，= 编码器 1000
            对应的宽度）；None 用 ``cfg.CONFIG["gripper"]["max_stroke"]``（0.076）。
            ⚠️ 必须与 ``UR7EDH76_RTDE.__init__`` 里的 ``Ag145driver(
            max_jaw_width=0.076)`` 一致：``conv2encoder`` 按这个值定比例，两处
            不一致会让同一个 stroke 下发到不同编码器位置（夹爪开合量整体缩放）。

    Returns:
        ag145.Ag145driver: 已打开串口、已 ``open_g()`` 的驱动实例。
        用完必须由**唯一持有方**调 ``close()`` 放掉串口：``dh_device.serialPort``
        是模块级单例，任何实例 close 都会把整个进程的串口关掉。
    """
    # 延迟 import：dry-run 全程不碰 pyserial / 串口驱动。
    import drivers.devices.dh.ag145 as ag145
    return ag145.Ag145driver(
        port=str(cfg.CONFIG["robot"]["gp_port"] if port is None else port),
    )


class GraspExecuter:
    """持有真机 + 夹爪，按时间顺序走完「抓取」与「放置」。

    积木（``set_jaw`` / ``_send_path`` / ``confirm_grasp`` / ``recover``）一次只做
    一件事；两段程序（``execute_pick`` / ``execute_place``）是**直线顺序代码**，
    一行一个动作，顺序直接读函数体就能看出来。**不读** 段里的 ``action_after``，
    也没有「按段名触发夹爪」的分支或参数。

    夹爪串口由单一 worker 线程独占（``_gripper_worker_loop``）：主线程通过
    ``_submit_gripper`` 投命令，``wait=True`` 阻塞取结果、``wait=False`` 立即返回
    句柄用于并行。所有 ``gripper.*`` 串口调用只发生在 worker 线程内。

    Attributes:
        rbt_r: ``UR7EDH76_RTDE`` 实例。
        gripper: ``ag145.Ag145driver`` 实例（``make_gripper()`` 造）；只做夹爪
            串口动作，**仅**由本类 worker 线程访问。
        log_dir: Path 日志目录。
        events: list[dict] 最近一次动作记录（也给日志用）。
        _gripper_q: queue.Queue 手爪命令队列；元素为带 ``done`` Event 的 dict。
        _gripper_thread: threading.Thread daemon worker；``shutdown()`` 停它。
    """

    def __init__(self, rbt_r, gripper, *, log_dir=None):
        """构造执行器。

        Args:
            rbt_r: ``UR7EDH76_RTDE`` 实例（真机）。真机执行必须非 None。
            gripper: ``ag145.Ag145driver`` 实例（推荐用 ``make_gripper()`` 造），
                负责夹爪串口动作。真机执行必须非 None。
                ⚠️ 执行器**不接管串口生命周期**：``close()`` 仍由调用方 main.py
                唯一持有（单例，close 一次全进程失效）；本类只调 ``jaw_to`` /
                ``get_current_width`` / ``set_force`` / ``set_speed``，且**全部**
                只在内部 worker 线程里调。
            log_dir: str | Path | None 日志目录；None 用 ``<本文件目录>/logs``。

        Raises:
            ValueError: ``rbt_r`` / ``gripper`` 为 None（本类是纯真机执行器，
                没有 dry-run 模式；dry-run 由 ``main.py`` 直接打印段表完成）。
        """
        if rbt_r is None:
            raise ValueError("rbt_r 为 None：本执行器只跑真机，dry-run 请走 main.py 打印")
        if gripper is None:
            raise ValueError("gripper 为 None：真机执行请先用 make_gripper() 造一个")
        self.rbt_r = rbt_r
        self.gripper = gripper
        self._vels = np.asarray(cfg.CONFIG["motion"]["toppra_vels"], dtype=float)
        self._accs = np.asarray(cfg.CONFIG["motion"]["toppra_accs"], dtype=float)
        self.log_dir = (Path(log_dir) if log_dir is not None
                        else Path(__file__).resolve().parent / "logs")
        #: 最近一次回读到的指尖全距（米）；None = 还没读过。用于「回读相等就跳过下发」。
        #: 由 worker 线程写、主线程读（单写者，无需加锁）。
        self._width_full = None
        #: 本次 run 的动作序号（打印用）
        self._step = 0
        #: 逐动作执行记录
        self.events = []
        #: 手爪命令队列（主线程投 → worker 线程取）。
        self._gripper_q = queue.Queue()
        #: 手爪 worker 线程（daemon）；串口单写者。收尾用 shutdown() 停。
        self._gripper_thread = threading.Thread(
            target=self._gripper_worker_loop, name="gripper-worker", daemon=True)
        self._gripper_thread.start()

    # -------------------------------------------------------------- 手爪线程
    def _gripper_worker_loop(self):
        """手爪命令 worker：独占串口，从 ``self._gripper_q`` 取命令逐条执行。

        每条命令是 dict：``op`` = ``"set_jaw"`` / ``"read"`` / ``"stop"``；
        必带 ``done`` (threading.Event)；执行结果写 ``result``、异常写 ``error``，
        最后无条件 ``done.set()`` 唤醒等待方。``"stop"`` 令线程退出循环。

        Args:
            无（线程入口，读实例属性 ``self._gripper_q`` / ``self.gripper``）。

        Returns:
            None: 无返回值；副作用是驱动夹爪串口并把结果写回命令 dict。
        """
        while True:
            cmd = self._gripper_q.get()
            try:
                op = cmd["op"]
                if op == "stop":
                    return
                if op == "read":
                    cmd["result"] = _from_stroke(self.gripper.get_current_width())
                elif op == "set_jaw":
                    cmd["result"] = self._set_jaw_body(
                        cmd["width_full"], label=cmd["label"],
                        force=cmd["force"], speed=cmd["speed"],
                        skip_if_same=cmd["skip_if_same"])
                else:
                    cmd["error"] = ValueError("未知手爪命令 op: {}".format(op))
            except Exception as exc:  # 单条命令异常不杀死线程
                cmd["error"] = exc
            finally:
                cmd["done"].set()

    def _submit_gripper(self, op, *, wait=True, **kwargs):
        """向手爪 worker 投一条命令，可选阻塞到完成并取回结果。

        Args:
            op: str 命令名，取 ``"set_jaw"`` / ``"read"`` / ``"stop"``。
            wait: bool True = 阻塞到命令完成，返回结果或抛回异常；
                False = 立即返回命令 dict（句柄，调用方可稍后自己等 ``done``）。
            **kwargs: 随命令携带的参数（``set_jaw`` 用 width_full/label/force/
                speed/skip_if_same），会被原样写进命令 dict。

        Returns:
            dict | Any: ``wait=False`` 返回命令 dict（含 ``done`` Event）；
            ``wait=True`` 返回命令的 ``result`` 字段（``set_jaw`` 为 result dict，
            ``read`` 为 float | None）。

        Raises:
            Exception: ``wait=True`` 且 worker 执行该命令抛异常时，原样重抛。
        """
        cmd = dict(kwargs)
        cmd["op"] = op
        cmd["done"] = threading.Event()
        self._gripper_q.put(cmd)
        if not wait:
            return cmd
        cmd["done"].wait()
        if "error" in cmd:
            raise cmd["error"]
        return cmd.get("result")

    def shutdown(self):
        """
        停掉手爪 worker 线程（必须在 ``gripper.close()`` 之前调用）。

        Args:
            无（投 ``"stop"`` 命令并 join 线程）。

        Returns:
            None: 幂等 —— 线程已不在则直接返回。join 有 5 s 上限，避免卡死。
                刻意**不**等 stop 命令的 ``done``（worker 若卡在串口读，等 done
                会无限挂起），只投递哨兵后有界 join。
        """
        th = self._gripper_thread
        if th is None or not th.is_alive():
            return
        self._gripper_q.put({"op": "stop", "done": threading.Event()})
        th.join(timeout=5.0)

    # ------------------------------------------------------------------ 积木
    def read_width_full(self):
        """回读当前指尖全距（经 worker 线程读串口，主线程阻塞等待结果）。

        Args:
            无。

        Returns:
            float | None: 指尖全距（米） = ``gripper.get_current_width() + 0.016``；
            串口没响应时返回 None（**不是** 0）。
        """
        return self._submit_gripper("read")

    def _set_jaw_body(self, width_full, *, label, force=None, speed=None,
                      skip_if_same=True):
        """set_jaw 的实际执行体：**只在 worker 线程内调用**，直接读写串口。

        与公开 ``set_jaw`` 逻辑一致，区别仅在于回读写 ``self.gripper`` 而非再投
        命令（避免 worker 内二次入队造成自我死锁）。

        Args:
            width_full: float 目标指尖全距（米），[0.016, 0.092]；越界被
                ``_to_stroke()`` 静默 clamp。
            label: str 动作说明（中文），只进打印和日志。
            force: int | None 夹持力百分比 0~100；None 用 cfg 的 force_percent。
            speed: int | None 速度百分比 0~100；None 用 cfg 的 speed_percent。
            skip_if_same: bool 为 True 且回读已在目标 ±jaw_move_eps_m 内时跳过下发。

        Returns:
            dict: ``{"label", "target_full", "stroke", "readback_full",
            "grip_state", "skipped", "ms"}``。``grip_state`` 是
            ``Ag145driver.jaw_to()`` 的返回（None = 超时未停稳）。
        """
        target = float(width_full)
        t0 = time.perf_counter()
        before = _from_stroke(self.gripper.get_current_width())
        if (skip_if_same and before is not None
                and abs(before - target) <= float(cfg.CONFIG["gripper"]["jaw_move_eps_m"])):
            self._width_full = before
            return {
                "label": label, "target_full": target, "stroke": _to_stroke(target),
                "readback_full": before, "grip_state": None, "skipped": True,
                "ms": (time.perf_counter() - t0) * 1000.0,
            }

        self.gripper.set_speed(int(cfg.CONFIG["gripper"]["speed_percent"] if speed is None else speed))
        self.gripper.set_force(int(cfg.CONFIG["gripper"]["force_percent"] if force is None else force))
        stroke = _to_stroke(target)
        state = self.gripper.jaw_to(stroke, timeout=float(cfg.CONFIG["gripper"]["wait_timeout_s"]))
        readback = _from_stroke(self.gripper.get_current_width())
        self._width_full = readback
        return {
            "label": label, "target_full": target, "stroke": stroke,
            "readback_full": readback, "grip_state": state, "skipped": False,
            "ms": (time.perf_counter() - t0) * 1000.0,
        }

    def set_jaw(self, width_full, *, label, force=None, speed=None,
                skip_if_same=True):
        """把夹爪开/闭到指定指尖全距，并等它停稳（投给 worker 线程执行）。

        Args:
            width_full: float 目标指尖全距（米），[0.016, 0.092]。越界会被
                ``_to_stroke()`` 静默 clamp，所以调用方要保证值合法。
            label: str 动作说明（中文），只进打印和日志。
            force: int | None 夹持力百分比 0~100；None 用 ``cfg.CONFIG["gripper"]["force_percent"]``。
            speed: int | None 速度百分比 0~100；None 用 ``cfg.CONFIG["gripper"]["speed_percent"]``。
            skip_if_same: bool 为 True 且回读已在目标 ±``cfg.CONFIG["gripper"]["jaw_move_eps_m"]`` 内时，
                不下发命令，直接返回（``skipped=True``）。

        Returns:
            dict: ``{"label", "target_full", "stroke", "readback_full",
            "grip_state", "skipped", "ms"}``。``grip_state`` 是
            ``Ag145driver.jaw_to()`` 的返回（驱动报的夹持状态码；None = 2 s 内没停稳）。
        """
        return self._submit_gripper(
            "set_jaw", width_full=width_full, label=label, force=force,
            speed=speed, skip_if_same=skip_if_same)

    def start_jaw_async(self, width_full, *, label, force=None, speed=None,
                        skip_if_same=False):
        """投一条调爪命令但不等待，用于与机器人路径并行（本轮只调一次）。

        命令排队后立刻返回句柄；worker 线程独占串口执行。并行窗口结束时，靠
        闭爪的 ``set_jaw`` 排队等它完成 —— 队列 FIFO 保证「异步调爪」一定
        先于「闭爪」执行（合并后 grasp 段内不再单独设爪）。

        Args:
            width_full: float 目标指尖全距（米），[0.016, 0.092]。
            label: str 动作说明（中文），进记录。
            force: int | None 夹持力百分比；None 用 cfg 的 force_percent。
            speed: int | None 速度百分比；None 用 cfg 的 speed_percent。
            skip_if_same: bool 默认 False —— 并行调爪要保证真正下发，不跳过。

        Returns:
            dict: 命令句柄（含 ``done`` Event、``result``/``error`` 槽），调用方
            可用 ``wait_async(handle)`` 阻塞取结果；不取也不影响后续 FIFO 串行。
        """
        return self._submit_gripper(
            "set_jaw", wait=False, width_full=width_full, label=label,
            force=force, speed=speed, skip_if_same=skip_if_same)

    def wait_async(self, handle):
        """等待一条异步手爪命令完成并取回结果（把句柄变回 result）。

        Args:
            handle: dict ``start_jaw_async`` 返回的命令句柄（含 ``done`` Event）。

        Returns:
            dict: 该命令的 ``result``（set_jaw 的 result dict）。

        Raises:
            Exception: worker 执行该命令抛出的异常，原样重抛。
        """
        handle["done"].wait()
        if "error" in handle:
            raise handle["error"]
        return handle.get("result")

    def _send_path(self, seg, move_to_start=True, move_to_end=True, *, name):
        """只下发这一段关节路径（TOPPRA + servoJ），不碰夹爪 —— 供持物/并行时序用。

        Args:
            seg: dict 一段（``path`` / ``jaw_full``）；只读 ``path``，
                ``jaw_full`` 由调用方在需要时自行处理（本函数不设爪、不打印开口）。
            move_to_start: bool 下发前是否先把机器人移到路径首点。默认 True
                （单点执行）；多瓶连续执行时传 False，省掉每段起步回位。
            move_to_end: bool 路径末尾是否收敛到末点。默认 True。
            name: str 段名，只用于日志与执行记录。

        Returns:
            dict: ``{"name", "n_points", "move_ms", "vel"}``。

        Raises:
            RuntimeError: 底层 ``move_jntspace_path`` 抛出的任何失败
                （半连接、servoJ 被拒、末端未收敛），原样向上抛，不包装。
        """
        vels = self._vels.tolist()
        accs = self._accs.tolist()
        n_points = grasp_plan.seg_n_points(seg)
        # servoJ 的 velocity 是「关节速度上限」，必须 >= 规划出的样本峰值，
        # 否则控制器会主动限速、计划时间失效。
        vel = max(float(cfg.CONFIG["motion"]["joint_move_vel"]), float(np.max(self._vels)))
        self._step += 1
        t0 = time.perf_counter()
        self.rbt_r.move_jntspace_path(
            seg["path"],
            vel=vel,
            acc=float(cfg.CONFIG["motion"]["joint_move_acc"]),
            toppra_vels=vels,
            toppra_accs=accs,
            move_to_start=move_to_start,
            move_to_end=move_to_end,
            use_toppra=True,
        )
        rec = {
            "name": name, "n_points": n_points,
            "move_ms": (time.perf_counter() - t0) * 1000.0, "vel": vel,
        }
        self.events.append(rec)
        return rec

    def confirm_grasp(self):
        """闭爪后回读，判断指间到底有没有夹到东西。

        判据（不含物理建模，只用宽度区间）：
        - 回读 None ⇒ 读不到，判**失败**（不能当成夹住）
        - 回读 <= ``GRASP_CONFIRM_MIN_WIDTH_M`` (0.020) ⇒ 一路闭到底 = **空夹**
        - 回读 > ``GRASP_CONFIRM_MAX_WIDTH_M`` (0.035) ⇒ 没到预定开口，可能夹在
          箱壁/瓶身斜面上 ⇒ ``too_wide=True``（告警，但仍算夹住）
        - 其余 ⇒ 夹住

        Args:
            无（用最近一次 ``set_jaw`` 的 ``self._width_full``；没有则现读一次）。

        Returns:
            dict: ``{"width_full", "grasped", "too_wide", "reason"}``。
            ``width_full`` 为 None 表示回读失败；``reason`` 是中文判据说明。
        """
        width = self._width_full
        if width is None:
            width = self.read_width_full()
        if width is None:
            return {"width_full": None, "grasped": False, "too_wide": False,
                    "reason": "夹爪回读失败（串口无响应），无法确认是否夹住"}
        min_w = float(cfg.CONFIG["gripper"]["grasp_confirm_min_width_m"])
        max_w = float(cfg.CONFIG["gripper"]["grasp_confirm_max_width_m"])
        if width <= min_w:
            return {"width_full": width, "grasped": False, "too_wide": False,
                    "reason": "回读 {:.1f} mm <= {:.1f} mm：一路闭到底 = 空夹".format(
                        width * 1000.0, min_w * 1000.0)}
        if width > max_w:
            return {"width_full": width, "grasped": True, "too_wide": True,
                    "reason": "回读 {:.1f} mm > {:.1f} mm：没到预定开口（可能夹在壁/斜面上）".format(
                        width * 1000.0, max_w * 1000.0)}
        return {"width_full": width, "grasped": True, "too_wide": False,
                "reason": "回读 {:.1f} mm ∈ ({:.1f}, {:.1f}] mm：夹住".format(
                    width * 1000.0, min_w * 1000.0, max_w * 1000.0)}

    def recover(self, reason, start_conf=None):
        """失败兜底：把机器人开回起始构型（best-effort，绝不掩盖原始异常）。

        Args:
            reason: str 失败原因，只用于打印。
            start_conf: (6,) 起始关节角（弧度）；None 表示跳过回位。

        Returns:
            dict: ``{"attempted", "ok", "detail", "jaw_width_full"}``。
            ``ok=False`` 且 ``attempted=True`` 表示回位本身也失败了 ——
            这时更危险（机器人可能停在箱内 / 手里还夹着瓶子），调用方必须把
            detail 打出来交人工处理。
        """
        out = {"attempted": False, "ok": False, "detail": "", "jaw_width_full": None}
        try:
            out["jaw_width_full"] = self.read_width_full()
        except Exception:
            pass
        if start_conf is None:
            out["detail"] = "未提供 start_conf，跳过回位"
            return out
        out["attempted"] = True
        print("[exec] ⚠️ {} ⇒ 回位到 START_CONF".format(reason))
        try:
            self.rbt_r.move_jnts(np.asarray(start_conf, dtype=float).reshape(6),
                                 vel=float(cfg.CONFIG["motion"]["joint_move_vel"]),
                                 acc=float(cfg.CONFIG["motion"]["joint_move_acc"]), wait=True)
            out["ok"] = True
            out["detail"] = "已回位"
        except Exception as exc:
            out["detail"] = "回位失败（{}: {}）—— 机器人可能仍在箱内，需人工介入".format(
                type(exc).__name__, exc)
            print("[exec] ⛔ " + out["detail"])
        return out

    # ------------------------------------------------------------------ 抓取
    def execute_pick(self, pick_plan, *, start_conf=None):
        """抓取程序：一条**直线写死**的时间顺序（对齐 robot_process.py 的 run()）。

            ① 异步调爪到接近宽度（与 ② 并行）→ ② grasp（转移+接近小窗拼接）
            → ③ 闭爪 + 确认

        ① 把爪从起点全开 (0.092) 调到接近宽度（抓取 pickle 派生 = jaw_width + 0.010，
        当前 0.041），
        由手爪 worker 线程执行；主线程不等它，立刻下发 ② 的 grasp 路径 ——
        调爪与走路径并行，省掉串行等待。规划按全开 0.092 查碰撞，执行期爪只会
        更窄 (0.041)，包络更小、更安全。② 是 to_pick_above + approach 经
        ``_blend_joint_window`` 小窗抹平接缝后的一条路径：段间不停车，速度更快；
        ② 里不再单独设爪，闭爪 ③ 入队后 FIFO 保证异步调爪先完成。

        段名只用于「从 pick_plan 里取路径」和打日志，**不参与任何动作分派**：
        闭爪就写在 grasp 之后，没有 ``if name == ...`` 这种判断。
        抬起不在这里：抬起 + 搬运 + 下放由 ``execute_place`` 的一条 ``place`` 段走完。

        Args:
            pick_plan: dict ``grasp_plan.plan_grasp`` 的返回；直接下标取
                ``pick_plan["grasp"]`` 一段（转移+接近拼接）。
                段里的 ``action_after`` **完全不读**。
            start_conf: (6,) 失败回位构型（弧度）；None = 失败不回位。
                正常传 ``cfg.CONFIG["joints"]["start_conf"]``。

        Returns:
            dict: ``{"phase", "ok", "error", "actions", "grasp", "elapsed_s",
            "log_path", "recover"}``。``grasp`` 为 dict，其中 ``grasped=None``
            表示**抓取确认已按用户要求停用**（不回读、不判断夹住与否）；``ok``
            仍为 True，正常返回后调用方照常执行放置（用户要求：抓没抓住都要
            走完放置），不抛异常、不回位。

        Raises:
            KeyError: plan 里缺 ``grasp`` / ``place`` 键。
            RuntimeError: 运动或夹爪命令失败（底层原样异常），已尝试回位。
        """
        grasp = pick_plan["grasp"]

        with self._phase("pick", start_conf) as result:
            # ① 异步调爪到接近宽度（pickle 派生，worker 线程执行，不等待）
            target_w = cfg.approach_width_full()
            self.start_jaw_async(target_w, label="调爪到接近宽度（并行）", skip_if_same=True)
            # ② 一条 grasp 路径走到抓取构型：转移 + 接近小窗拼接，段间不停车。
            self._send_path(grasp, move_to_start=False, move_to_end=True, name="grasp")
            # ③ 闭爪。不检查是否夹住，抓没抓住都继续走放置。
            closed = self.set_jaw(float(cfg.CONFIG["gripper"]["zero_gap"]),
                                  label="闭爪（approach 之后）", skip_if_same=False)
            self.events.append({"action": "close", "detail": closed})
            # grasp 仍填 dict 以维持返回/日志结构；grasped=None = 未检查。
            result["grasp"] = {
                "width_full": None, "grasped": None, "too_wide": False,
                "reason": "抓取确认按用户要求跳过",
            }
            print("[exec] ⚠️ " + result["grasp"]["reason"])
        return result

    # ------------------------------------------------------------------ 放置
    def execute_place(self, place_plan, *, start_conf=None):
        """放置程序：一条**直线写死**的时间顺序。

            ① place（抬起 + 搬运 + 下放，规划期已拼成一条）→ ② 松爪

        与 ``execute_pick`` 是两段独立程序：只吃 ``place_plan`` 自己的段列表，
        和抓取段**不合并、不拼接**。松爪写在 ``place`` 之后。
        松爪后机器人停在放置点，不回起点 —— 多物体时下一轮 ``execute_pick`` 的
        ``grasp`` 会从当前位置直接走过去。

        Args:
            place_plan: dict ``grasp_plan.plan_place`` 的返回；直接下标取
                ``place_plan["place"]``。段里的 ``action_after`` **完全不读**。
            start_conf: (6,) 失败回位构型（弧度）；None = 失败不回位。

        Returns:
            dict: ``{"phase", "ok", "error", "actions", "grasp", "elapsed_s",
            "log_path", "recover"}``；``grasp`` 恒为 None。

        Raises:
            KeyError: plan 里缺 ``grasp`` / ``place`` 键。
            RuntimeError: 运动或夹爪命令失败（底层原样异常），已尝试回位。
        """
        place = place_plan["place"]
        with self._phase("place", start_conf) as result:
            # ① 抬起 → 搬运 → 下放：一条平滑路径，只调一次 _send_path ⇒ 只有一次
            #    TOPPRA 参数化，接缝处不会 stop-and-go。
            #    ⛔ 持物段**不设爪**：若按段 jaw_full(0.031) 重新设爪，会把闭爪
            #    夹持状态往外调到 0.031 ⇒ 物体松脱。持物全程保持闭爪。
            #    move_to_start=False：上一瓶 place 已在放置点松爪停下，直接起步。
            self._send_path(place, move_to_start=False, move_to_end=True, name="place")
            # # ② 松爪（机器人停在放置点，不回起点）—— 保留本地每瓶松爪行为。
            # opened = self.set_jaw(float(cfg.CONFIG["gripper"]["grasp_width_full"]),
            #                       label="松爪（place 之后）", skip_if_same=False)
            # self.events.append({"action": "release", "detail": opened})
        return result

    # ------------------------------------------------------------------ 边界
    @contextmanager
    def _phase(self, phase, start_conf):
        """给一段顺序程序套上统一的「计时 / 异常回位 / 日志」边界。

        ⛔ 这里**不生成也不分派任何动作**：``yield`` 出去的就是调用方写的直线
        代码，先做什么后做什么完全由 ``execute_pick`` / ``execute_place`` 的函数体
        决定。对应 ``robot_process.py`` 的 ``run()`` 里那个 ``try/finally``。

        Args:
            phase: str ``"pick"`` / ``"place"``，只用于打印、日志与返回。
            start_conf: (6,) | None 失败回位构型（弧度）；None = 不回位。

        Yields:
            dict: ``{"phase", "ok", "error", "actions", "grasp", "elapsed_s",
            "log_path", "recover"}``。调用方就地写 ``grasp`` 等键；``actions``
            与 ``self.events`` 是同一个 list，``_send_path`` 会自动往里追加。

        Raises:
            GraspFailed / Exception: 主体里的异常一律**原样重抛**（已尝试回位），
            绝不吞掉、绝不包装。
        """
        t0 = time.perf_counter()
        self.events = []
        self._step = 0
        self._width_full = None
        result = {"phase": phase, "ok": False, "error": None, "actions": self.events,
                  "grasp": None, "elapsed_s": 0.0, "log_path": None, "recover": None}
        try:
            yield result
            result["ok"] = True
        except GraspFailed as exc:
            result["error"] = "{}".format(exc)
            result["recover"] = self.recover("抓取失败", start_conf)
            raise
        except Exception as exc:
            result["error"] = "{}: {}".format(type(exc).__name__, exc)
            result["recover"] = self.recover(
                "{} 阶段异常 {}".format(phase, type(exc).__name__), start_conf)
            raise
        finally:
            result["elapsed_s"] = time.perf_counter() - t0
            try:
                result["log_path"] = str(self._write_log(result))
            except Exception as exc:
                print("[exec] ⚠️ 日志落盘失败：{}: {}".format(type(exc).__name__, exc))
        print("[exec] ✅ {} 阶段完成，耗时 {:.2f} s".format(phase, result["elapsed_s"]))
    # ------------------------------------------------------------------ 日志
    def _write_log(self, result):
        """把一个阶段的执行结果写成 JSON。

        Args:
            result: dict ``_phase`` 产出的返回容器（就地读 ``phase`` / ``actions`` 等）。

        Returns:
            Path: 落盘的文件路径 ``<log_dir>/<phase>_YYYYmmdd-HHMMSS-ffffff.json``。
            用**微秒**而非毫秒：单段 dry-run 不到 0.5 ms，连跑会全落进同一毫秒；
            仍撞名则追加 ``-N`` 序号，保证「一次执行一个文件」，绝不静默覆盖。
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        stem = now.strftime("%Y%m%d-%H%M%S-%f")
        path = self.log_dir / "{}_{}.json".format(result["phase"], stem)
        seq = 1
        while path.exists():
            path = self.log_dir / "{}_{}-{}.json".format(result["phase"], stem, seq)
            seq += 1
        payload = {
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "phase": result.get("phase"),
            "ok": result.get("ok"),
            "error": result.get("error"),
            "toppra_vels": self._vels.tolist(),
            "toppra_accs": self._accs.tolist(),
            "elapsed_s": result.get("elapsed_s"),
            "recover": result.get("recover"),
            "grasp": result.get("grasp"),
            "actions": [_jsonable(rec) for rec in result.get("actions", [])],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                        encoding="utf-8")
        print("[exec] 日志 → {}".format(path))
        return path


def _jsonable(obj):
    """把结果字典里的 numpy 标量/数组转成可 JSON 序列化的 Python 对象。

    Args:
        obj: 任意对象（dict/list/ndarray/np 标量/基础类型）。

    Returns:
        dict | list | float | int | str | bool | None: 递归转换后的结构。
        ndarray 一律转 list（不做 round，保留原始精度备查）。
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj
