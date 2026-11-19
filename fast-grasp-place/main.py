"""fast-grasp-place 主入口：不拍照，物体位姿手动给定。

⛔ 坐标系（铁律 19）
    ``OBJECT_POSITION`` / ``OBJECT_ROTMAT`` 属 **sim 世界系**，即
    ``UR7EDH76().fk()`` 的输出系，与本项目 ``box_pose_detect/results/fixed_box_transform.txt``
    同源。它不是 ``getActualTCPPose()`` / ``moveL()`` 那套真实基系
    （两者实测差 2.4 m 且 x 反号），严禁相减或混用。
    ⇒ 规划全在 sim 世界系里做完，执行层只收关节角，不做任何跨框架换算。

⛔ 时间顺序（对齐 anygrasp_wrs/d435_grasp/robot_process.py）
    阶段 A：把 5 个瓶子的 plan_grasp → plan_place 全部算完（不连机器人），
    任一失败 ⇒ 一个动作都不执行。第 2 轮起 grasp 把 config 的固定回程
    place_to_start 与「起点 → 接近点」转移段平滑拼接、GOMP 精修，再与接近段
    小窗拼成一条并查碰撞（首轮 grasp 无回程，仅转移+接近拼接）。
    阶段 B：只连一次真机/串口，循环 5 次 execute_pick → execute_place。
    段末夹爪动作写死在执行层，不按 ``action_after`` 名字分派。

用法
    python main.py                         # 直接跑：规划 + 打印动作表（dry-run，不连机器人）
    import main; main()                    # 同上；被 import 调用不再解析宿主 sys.argv
    import main; main(execute=True)                     # 真机执行（首跑手放急停）
    import main; main(plot=True)           # 规划后在 wrs 仿真展示抓放路径（只截图）
"""

import sys
import time

import numpy as np

import config as cfg
import grasp_execute
import grasp_plan
import obstacle
import robot_sim.robots.ur7e.ur7e_withouttable_dh76 as u76
from robot_con.ur.ur7e_dh76_rtde import (
    disconnect_rtde_robot_singleton,
    get_rtde_robot_singleton,
)

# ========================= ① 物体位姿（手动给定，改这里） =========================
# 位置：物体原点 = CAD 原点 = **底面中心**（不是质心），单位 m，sim 世界系。
OBJECT_POSITION_1 = np.array([0.747529793, 0.011525319, 0.012])+np.array([-0.10, -0.10, 0])
OBJECT_POSITION_2 = OBJECT_POSITION_1+np.array([0, 0.1, 0])
OBJECT_POSITION_3 = OBJECT_POSITION_1+np.array([0, 0.2, 0])
OBJECT_POSITION_4 = OBJECT_POSITION_1+np.array([0.1, 0.2, 0])
OBJECT_POSITION_5 = OBJECT_POSITION_1+np.array([0.1, 0.1, 0])
OBJECT_POSITION_6 = OBJECT_POSITION_1+np.array([0.1, 0, 0])
OBJECT_POSITION_7 = OBJECT_POSITION_1+np.array([0.2, 0.2, 0])
OBJECT_POSITION_8 = OBJECT_POSITION_1+np.array([0.2, 0.1, 0])
OBJECT_POSITION_9 = OBJECT_POSITION_1+np.array([0.2, 0, 0])

# 旋转：R_world_obj，**列** = 物体系 x/y/z 轴在世界系下的方向余弦；
# 物体系 z 轴 = 瓶身高度方向（+z 从底面指向瓶口）。
# ⚠️ 下面这组值 = 瓶子直立、绕 z 轴只偏 0.225°（原 object_pose.example.txt 实测值），
#    换摆放就改数字，别改矩阵形状。想绕某个轴转 theta：
#    import wrs.basis.robot_math as rm; R = rotmat @ rm.rotmat_from_axangle([0,0,1], theta)
OBJECT_ROTMAT = np.array(
    [
        [0.999992294, 0.003871222, -0.000651635],
        [-0.003871094, 0.999992488, 0.000198379],
        [0.000652398, -0.000195855, 0.999999768],
    ]
)
#: 5 个瓶子的抓取初位（姿态全都用上面的 OBJECT_ROTMAT）。
OBJECT_POSITIONS = (
    OBJECT_POSITION_1, OBJECT_POSITION_2, OBJECT_POSITION_3,
    OBJECT_POSITION_4, OBJECT_POSITION_5, OBJECT_POSITION_6,
    OBJECT_POSITION_7, OBJECT_POSITION_8, OBJECT_POSITION_9,
)


def make_planner():
    """造一台 sim 机器人 + 障碍物，返回规划器（本脚本唯一构造入口）。

    ``UR7EDH76()`` 构造实测 458 ms（整个规划链里最贵的一步），所以只造一次，
    抓取与放置共用同一个 ``planner``。

    Args:
        无（障碍物走 ``obstacle.build_obstacles()``，种子构型走 ``cfg.CONFIG["joints"]["start_conf"]``）。

    Returns:
        grasp_plan.Grasp_planner: 已复位到 ``cfg.CONFIG["joints"]["start_conf"]`` 的规划器，``.rbt_s``
        是那台 sim 机器人（画图也复用它，别再新建 World）。
    """
    robot = u76.UR7EDH76()                      # sim；enable_cc 默认 True
    return grasp_plan.Grasp_planner(robot, obstacle.build_obstacles(),
                                    seed_conf=cfg.CONFIG["joints"]["start_conf"])


def show_phase(title, plan):
    """按执行顺序打印一个 plan 的各段（dry-run 的全部输出，不碰机器人）。

    Args:
        title: str 标题，例如 ``"② 放置（plan_place → execute_place）"``。
        plan: dict ``plan_grasp`` / ``plan_place`` 的返回；段由
            ``grasp_plan.plan_segments`` 按 ``SEGMENT_ORDER`` 取出。

    Returns:
        None。副作用：打印段表。
    """
    print("\n{}：".format(title))
    for line in grasp_plan.describe_segments(grasp_plan.plan_segments(plan)):
        print(line)


def check_phase(*, phase, object_violations=()):
    """校核一个 plan：被夹物体过壁就直接抛（**任何运动之前**）。

    Args:
        phase: str ``"抓取"`` / ``"放置"``，只进报错文字。
        object_violations: list 该阶段 ``object_violations``；非空 = 被夹物体过壁。

    Returns:
        None。

    Raises:
        RuntimeError: 被夹物体过壁（校核恒定开启，无开关）。
    """
    if object_violations:
        raise RuntimeError("{}阶段被夹物体过壁 {} 处：{}".format(
            phase, len(object_violations),
            [i for i, _ in object_violations]))


def execute_pick_and_place(plans):
    """连一次真机，按时间顺序跑完 5 个瓶子的 pick → place。

    ⛔ 不合并段列表：每轮两次调用各自只吃自己那份计划，中间没有拼接层。
    RTDE 与夹爪串口只在这里构造一次（夹爪开机回零一次），循环内复用。

    Args:
        plans: list[(int, dict, dict)] 每轮 ``(瓶号, pick_plan, place_plan)``，
            由 ``main()`` 阶段 A 全部规划好后一次性传入。

    Returns:
        list[dict]: 每轮一个 ``{"index", "pick", "place"}``，顺序同入参。

    Raises:
        RuntimeError: 运动/夹爪失败（已回位）。
        Exception: 连接真机失败（``UR7EDH76_RTDE`` 构造或串口打开失败）。

    Note:
        ``ur_rtde`` 与串口驱动只在**这里**导入：dry-run 全程不碰真机依赖，
        没装 ``ur_rtde`` 的机器也能跑规划与自检。
        抓取确认已按用户要求停用：``execute_pick`` 不判断夹住与否，正常返回后
        照常执行 ``execute_place``；该轮 ``pick.grasp["grasped"]`` 为 None。
    """
    gripper_r = grasp_execute.make_gripper()
    execer = None
    try:
        rbt_r = get_rtde_robot_singleton(
            cfg.CONFIG["robot"]["robot_ip"], cfg.CONFIG["robot"]["gp_port"])
        execer = grasp_execute.GraspExecuter(rbt_r, gripper_r)
        results = []
        # 到抓取起点
        rbt_r.move_jnts(cfg.CONFIG["joints"]["start_conf"], vel=1.0, acc=1.0)
        # 循环抓取
        for idx, pick_plan, place_plan in plans:
            print("\n===== 第 {} 个瓶子 =====".format(idx))
            pick_result = execer.execute_pick(pick_plan, start_conf=cfg.CONFIG["joints"]["start_conf"])
            place_result = execer.execute_place(place_plan, start_conf=cfg.CONFIG["joints"]["start_conf"])
            results.append({"index": idx, "pick": pick_result, "place": place_result})
        # 抓取完成后张开
        grasp_execute.set_jaw(float(cfg.CONFIG["gripper"]["grasp_width_full"]),
                                  label="松爪（place 之后）", skip_if_same=False)
        return results
    finally:
        # ⚠️ 顺序：先停手爪 worker 线程，再关夹爪串口，最后放 RTDE。
        if execer is not None:
            execer.shutdown()
        try:
            gripper_r.close()
        except Exception:
            pass
        # 单次运行的脚本：用完就放掉 RTDE 输入寄存器，避免下次 RTDEIOInterface
        # 报 "input registers are already in use"。回位已在执行层内部做过。
        disconnect_rtde_robot_singleton(
            cfg.CONFIG["robot"]["robot_ip"], cfg.CONFIG["robot"]["gp_port"])


def main(execute=False, plot=False):
    """主入口：规划 + 可选真机执行/仿真展示；不解析任何命令行参数。

    Args:
        execute: bool 是否连真机执行；False（默认）= 只规划并打印动作表
            （dry-run，全程不连机器人）。
        plot: bool 规划后是否在 wrs 仿真环境展示抓放路径；True = 截图存
            ``logs/plan_<时间戳>.png``，不弹窗（生产固定行为）。

    Returns:
        int: 退出码。0 = 成功（含 dry-run）；1 = 规划或执行失败；2 = 物体位姿非法。
    """

    # ---------- 阶段 A：9 个瓶子的抓放路径全部算好（不连机器人） ----------
    planner = make_planner()
    return_path = np.asarray(cfg.CONFIG["move_path"]["place_to_start"], dtype=float)
    plans = []
    time0 = time.time()
    for idx, pos in enumerate(OBJECT_POSITIONS, start=1):
        try:
            object_pose = cfg.make_object_pose(pos, OBJECT_ROTMAT)
        except ValueError as exc:
            print("⛔ 第 {} 个瓶子位姿非法：{}".format(idx, exc))
            return 2
        print("\n===== 第 {} / {} 个瓶子 =====".format(idx, len(OBJECT_POSITIONS)))
        print("物体位姿 t     =", np.round(object_pose[:3, 3], 6).tolist())
        print("物体系 z 轴    =", np.round(object_pose[:3, 2], 6).tolist())

        # 首轮 from start_conf；第 2 轮起先走固定回程再接转移段，拼成一条。
        pick_plan = planner.plan_grasp(
            object_pose, return_path=return_path if idx > 1 else None)
        if pick_plan is None:
            print("\n⛔ 第 {} 个瓶子抓取规划失败，一个动作都不执行".format(idx))
            return 1
        print("\n抓取候选 #{}  conf(deg) = {}".format(
            pick_plan["candidate"]["index"],
            np.round(np.degrees(pick_plan["conf"]), 2).tolist()))
        show_phase("① 抓取（plan_grasp → execute_pick）", pick_plan)

        # 放置点固定 = config.joints.place_conf，5 轮同一点（用户已确认）。
        place_plan = planner.plan_place(pick_plan["conf"], pick_plan["object_pose"])
        if place_plan is None:
            print("\n⛔ 第 {} 个瓶子放置规划失败，一个动作都不执行".format(idx))
            return 1
        show_phase("② 放置（plan_place → execute_place）", place_plan)

        # 8 角点过壁 ⇒ 一个动作都不执行（段级碰撞/瓶身碰壁已在规划期回退）。
        try:
            check_phase(phase="第 {} 个瓶子放置".format(idx),
                        object_violations=place_plan["object_violations"])
        except RuntimeError as exc:
            print("\n⛔ {}".format(exc))
            return 1

        # 每张图一个物体：循环调用现有展示函数，不改它的签名。
        if plot:
            grasp_plan.visualize_plan_sim(
                pick_plan, place_plan, planner.rbt_s, object_pose,
                shot_path="logs/plan_{}_{}.png".format(
                    idx, time.strftime("%Y%m%d-%H%M%S")),
                run_window=False)
        plans.append((idx, pick_plan, place_plan))

    print("规划总时间:", time.time()-time0)
    print("\n✅ {} 个瓶子的抓放路径全部规划完成".format(len(plans)))

    if not execute:
        print("dry-run 结束")
        return 0

    # ---------- 阶段 B：只连一次真机，循环执行全部抓放 ----------
    try:
        results = execute_pick_and_place(plans)
    except grasp_execute.GraspFailed as exc:
        print("\n⛔ 抓取失败：{}".format(exc))
        return 1
    except Exception as exc:
        print("\n⛔ 执行异常：{}: {}".format(type(exc).__name__, exc))
        return 1

    for rec in results:
        print("\n===== 第 {} 个瓶子 =====".format(rec["index"]))
        for phase in ("pick", "place"):
            r = rec[phase]
            print("\n{} 阶段 {:.2f} s，{} 个动作：".format(
                "抓取" if phase == "pick" else "放置",
                r["elapsed_s"], len(r["actions"])))
            for act in r["actions"]:
                if "name" in act:
                    print("   {:<14s} {:3d} 点  {:7.0f} ms".format(
                        act["name"], act["n_points"], act.get("move_ms", 0.0)))
                else:
                    print("   {:<14s} {:.0f} ms  开口 → {} mm".format(
                        act["action"], act["detail"]["ms"],
                        round(act["detail"]["readback_full"] * 1000.0, 1)
                        if act["detail"]["readback_full"] is not None else "?"))
            print("   日志：{}".format(r["log_path"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(execute=True, plot=False))
