"""
fast-grasp-place 参数唯一真源。
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
for _root in (str(PROJECT_DIR), str(REPO_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)
import basis.robot_math as rm  # noqa: E402
from fixed_paths import CONFIG_PATH


CONFIG = {
    # ---------------- 机器人 / 串口（项目级通用；live 拍照 args 从此处取） ----------------
    "robot": {
        "robot_ip": "192.168.125.30",
        "gp_port": "/dev/ttyUSB0",
    },
    # 抓取对象切换：改这一个字符串 + 重启即切换 tea / fanta（单选，模块级派生）。
    # 决定 STL/pickle 路径、物体几何（过壁角点/红球/展示）与持物碰撞体。
    "active_object": "fanta",
    # 两物体的几何 + 放置数据（碰撞体在 obstacle.py 各自 get_*_sample_cdprim）。
    # tea 全部沿用本地原示教值；fanta 几何来自实测 STL，place_* 为占位（TODO）。
    "object_profiles": {
        "tea": dict(
            stl="tea_bottle.STL", pickle="dh76_tea_bottle_head.pickle",
            height_m=0.220, bbox_xy_m=(0.070, 0.0699),
            grasp_height_local_m=0.204, neck_width_m=0.03088,
            place_pos=np.array([0.3, -0.5, 0.25]),
            place_rot=rm.rotmat_from_euler(np.pi, 0, 0) @ rm.rotmat_from_euler(0, 0, np.pi / 2),
            place_conf=np.array([2.87477904, -1.46052583, 2.17524447, -2.28551641, -1.57079594, 1.303983]),
            place_to_start=CONFIG_PATH["tea"]["place_to_start"]),
        "fanta": dict(
            stl="fanta_bottle.STL", pickle="dh76_fanta_bottle_head.pickle",
            height_m=0.162, bbox_xy_m=(0.0629, 0.06279),
            grasp_height_local_m=0.153, neck_width_m=0.02967,
            # TODO: place_* 为 tea 姿态占位，换成 fanta 实测示教位姿/路径。
            place_pos=np.array([0.3, -0.5, 0.19]),
            place_rot=rm.rotmat_from_euler(np.pi, 0, 0),
            place_conf=np.array([2.87477904, -1.3336387, 2.23585791, -2.47301698, -1.57079594, -0.26681333]),
            place_to_start=CONFIG_PATH["fanta"]["place_to_start"]),
    },
    # ---------------- 路径 ----------------
    "path": {
        # 下方按 active_object 从 object_profiles 派生填充。
        "object_stl_path": None,
        "grasp_pickle_path": None,
        "output_dir": PROJECT_DIR / "outputs",
        # 箱体 STL 在 model/，实际使用的箱位姿在 box_pose_detect/results/（自有副本）。
        "box_stl_path": PROJECT_DIR / "model" / "box.STL",
        "box_homomat_path": PROJECT_DIR / "box_pose_detect" / "results" / "fixed_box_transform.txt",
    },
    # ---------------- 物体 ----------------
    "object": {
        # 物体位姿即 CAD 位姿：原点 = 底面中心点（不是质心）。
        # 物体位姿**不读文件**：由 main.py 手动给定「位置 + 3x3 旋转矩阵」，
        # 再交给本文件的 make_object_pose() 组装成 4x4。见 main.py 顶部常量。
        # 实测几何（tea_bottle.STL，单位 m）：bbox = 70.0 x 69.9 x 220.0 mm，Z 是高度轴。
        "height_m": None,  # 下方按 active_object 派生
        # bbox 的 x/y 尺寸（米）。只用于把物体 8 个角点算到世界系（过壁检查），
        # 别拿它当碰撞体：真实碰撞体是箱体 + 桌面，物体本身不在 obstacle_list 里。
        "bbox_xy_m": None,  # 下方按 active_object 派生
        # pickle 里 jaw_center_pos 的唯一值 = [0, 0, 0.204]，即物体系「底面往上 204 mm」。
        "grasp_height_local_m": None,  # 下方按 active_object 派生
        # 三角面-平面求交切片实测：z=0.204 处截面外径 X=30.88 / Y=30.91 mm。
        "neck_width_m": None,  # 下方按 active_object 派生
        # make_object_pose() 校验手写旋转矩阵用的容差：|R@R.T - I|_max 与 |det(R) - 1|
        # 都必须小于它。1e-3 足够容纳手抄 6 位小数的舍入（实测这个量级误差 ~1e-7），
        # 又能拦住「抄错一行」「漏一个负号」「写成镜像矩阵」这类致命错。
        "rotmat_ortho_tol": 1e-3,
    },
    # ---------------- 固定关节角（弧度） ----------------
    "joints": {
        # 数值固定在本文件（原取自 grasp_start_conf_rad）。
        # 实测：TCP_world = [0.7522, -0.0032, 0.5807]，工具 z 轴 = [-0.033, 0.024, -0.999]
        # （竖直向下），水平距箱心 0.0154 m ⇒ 正悬在箱心上方 0.58 m，且不与箱/桌碰撞。
        "start_conf": np.array([1.286601185798645,-1.6360346279540003, 1.4551900068866175, -1.4035828870585938, -1.5323918501483362, 1.3419032096862793,]),
        # 放置点（sim 世界系）与放置姿态：按 active_object 从 object_profiles 派生。
        "place_pos": None,
        "place_rot": None,
        "place_conf": None,
        # box 检测 live 拍照位姿（弧度 / 米，由原 live 拍照配置迁入）。
        "capture_conf_rad": [
            2.2703418254852297, -1.5270795759609719, 1.4671981970416468,
            -1.55332566917453, -1.4730036894427698, 1.6031075716018677,
        ],
        "capture_pose": [
            0.4403872941764518, -0.30197784053650734, 0.5160190635918755,
            1.3344390978540437, 2.754900817986119, 0.10687167096644862,
        ],
    },
    # ---------------- 夹爪（口径见 docstring 函数） ----------------
    "gripper": {
        # 用户确认：jaw_range = [0.016, 0.092] 就是手爪实际宽度范围（指尖全距）。
        # 全闭两指尖内侧 16 mm，全开 92 mm；这是 AG145 行程 stroke 加出来的物理量，
        # 不是 DH76_BASE_JAW_WIDTH=0.042（那个是洗衣液指尖的旧值，本项目不用）。
        "zero_gap": 0.016,
        # Ag145driver(max_jaw_width=...) 的行程上限，见 wrs/robot_con/ur/ur7e_dh76_rtde.py 构造处。
        "max_stroke": 0.076,
        # 理论全开 = 行程上限 + 零行程净间距 = 0.092 = jaw_range[1]（自洽，勿手改成 0.092 之外的值）。
        # 派生值：zero_gap + max_stroke = 0.016 + 0.076 = 0.092。
        "open_width_full": 0.092,
        # pickle 的 jaw_width（指尖全距）。瓶口截面 30.88 mm + 0.12 mm 余量。
        "grasp_width_full": 0.031,
        # 接近段手爪开口不在此处配置：由 approach_width_full() 从当前
        # active_object 的抓取 pickle 读 jaw_width + APPROACH_MARGIN_M 计算。
        "force_percent": 100,
        "speed_percent": 100,
        "wait_timeout_s": 2.0,
        # 夹爪串口 = CONFIG["robot"]["gp_port"]（项目级唯一真源，此处不再重复）。
        # 抓取确认阈值（指尖全距，米）。闭爪后回读，用来判断「指间到底有没有东西」。
        # grasp_width_full=0.031 是瓶子被夹住时的目标全距；空夹会一路闭到 zero_gap=0.016。
        # 判据：回读 > 0.016 + 0.004 = 0.020 ⇒ 指间有东西（夹住）；否则判空夹。
        # 派生值：zero_gap + 0.004 = 0.020。
        "grasp_confirm_min_width_m": 0.020,
        # 回读超过 0.035 说明没到预定开口（可能夹在箱壁/瓶身斜面上）⇒ 只告警不中止。
        # 派生值：grasp_width_full + 0.004 = 0.035。
        "grasp_confirm_max_width_m": 0.035,
        # 下发夹爪命令前的「回读已到位」容差（指尖全距，米）。
        # 实测 encoder 分辨率 = max_jaw_width/1000 = 0.076 mm，0.5 mm 足够容下往返量化误差。
        "jaw_move_eps_m": 0.0005,
    },
    # ---------------- 运动 ----------------
    "motion": {
        "joint_path_granularity_rad": 0.02,
        "cartesian_granularity_m": 0.005,
        "toppra_vels": np.array([3.14] * 6),
        "toppra_accs": np.array([6.0] * 6),
        # 关节直达（move_jnts / move_jntspace_path 的 vel/acc）。move_jnts 内部把 vel 夹到 3.14。
        "joint_move_vel": 1.0,
        "joint_move_acc": 1.0,
        # 抓取接近段：从抓取点沿「远离物体方向」（= -工具 z 轴）走这么远作为接近起点，
        # 再沿直线扎到抓取点。竖直抓取时就是正上方 0.10 m。
        "approach_distance_m": 0.03,
        # 放置段：从放置构型正上方沿**世界 +z** 垂直下放的距离（米）。用世界 +z 而不是
        # 工具轴，是因为放置位姿（BOTTLE_ROBOT_SIDE_PLACE_CONF）的瓶子是侧放的，工具 z
        # 轴接近水平，沿工具轴「上移」会变成横向平移。
        "place_approach_distance_m": 0.20,
        # 抬升段的**下限**（米）。真实用值 = max(本值, obstacle.min_lift_to_clear_wall())。
        "lift_distance_m": 0.20,
        # 被夹物体（不只是手爪本体）越过箱壁顶要留的余量（米）。
        "transit_clearance_m": 0.05,
        # 碰撞检测采样步长（弧度）。
        "collision_step_rad": 0.02,
    },
    "move_path":{
        # 按 active_object 从 object_profiles 派生填充。
        "place_to_start": None,
    }

}


# ---- active_object 派生（modules 一律读 CONFIG[...]，此处一次性填充占位）----
_active_obj = CONFIG["active_object"]
assert _active_obj in CONFIG["object_profiles"], (
    "active_object 必须是 %s 之一，当前 %r"
    % (sorted(CONFIG["object_profiles"]), _active_obj))
_obj_profile = CONFIG["object_profiles"][_active_obj]
CONFIG["path"]["object_stl_path"] = PROJECT_DIR / "model" / _obj_profile["stl"]
CONFIG["path"]["grasp_pickle_path"] = (PROJECT_DIR / "grasp_pickle" / "pickle"
                                        / _obj_profile["pickle"])
CONFIG["object"]["height_m"] = _obj_profile["height_m"]
CONFIG["object"]["bbox_xy_m"] = _obj_profile["bbox_xy_m"]
CONFIG["object"]["grasp_height_local_m"] = _obj_profile["grasp_height_local_m"]
CONFIG["object"]["neck_width_m"] = _obj_profile["neck_width_m"]
CONFIG["joints"]["place_pos"] = _obj_profile["place_pos"]
CONFIG["joints"]["place_rot"] = _obj_profile["place_rot"]
CONFIG["joints"]["place_conf"] = _obj_profile["place_conf"]
CONFIG["move_path"]["place_to_start"] = _obj_profile["place_to_start"]


# ============================== 函数 ==============================
# 接近段相对抓取宽度的余量（米），别蹭到箱壁/瓶身。恒定物理常量，非开关。
APPROACH_MARGIN_M = 0.010


def load_grasp_width_from_pickle(path: Path) -> float:
    """读抓取 pickle 中所有候选的 jaw_width，要求全部一致。

    Args:
        path: 抓取 pickle 文件路径；内容为非空 list/tuple，每项首元素
            是 ``jaw_width``（指尖全距，米）。

    Returns:
        float: 所有候选一致的指尖全距（米）。

    Raises:
        FileNotFoundError: pickle 文件不存在。
        ValueError: 内容为空 / 结构异常 / 候选宽度不一致。
    """
    path = Path(path)
    with path.open("rb") as fp:
        raw = pickle.load(fp)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("抓取 pickle 必须是非空 list/tuple：%s" % path)
    widths = set()
    for i, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or not item:
            raise ValueError("第 %d 个抓取项结构异常：%s" % (i, path))
        widths.add(float(item[0]))
    if len(widths) != 1:
        raise ValueError("抓取 pickle 候选 jaw_width 不一致 %s：%s"
                         % (sorted(widths), path))
    return widths.pop()


def approach_width_full() -> float:
    """从当前 active_object 的抓取 pickle 派生接近段手爪开口。

    Returns:
        float: 指尖全距（米）= pickle 的 jaw_width + APPROACH_MARGIN_M。
            当前 tea/fanta 两份 pickle 的 jaw_width 均为 0.031，
            故返回值当前为 0.041。
    """
    return load_grasp_width_from_pickle(CONFIG["path"]["grasp_pickle_path"]) + APPROACH_MARGIN_M


def _read_homomat(path: Path, label: str) -> np.ndarray:
    """读 4x4 齐次矩阵文本并做形状/有限性校验。

    Args:
        path: 文本文件路径，内容为 4 行 4 列空白分隔的数字。
        label: 报错时用于标识来源的中文标签（如「箱体」「物体」）。

    Returns:
        np.ndarray: shape (4,4)、dtype float 的齐次矩阵（不复用文件对象）。
    """
    if not path.exists():
        raise FileNotFoundError(f"{label}位姿文件不存在：{path}")
    homomat = np.asarray(np.loadtxt(path), dtype=float)
    if homomat.shape != (4, 4):
        raise ValueError(f"{label}位姿必须是 4x4，实际 {homomat.shape}：{path}")
    if not np.all(np.isfinite(homomat)):
        raise ValueError(f"{label}位姿含 NaN/inf：{path}")
    last_row = homomat[3]
    if not np.allclose(last_row, [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{label}位姿末行必须是 [0,0,0,1]，实际 {last_row.tolist()}：{path}")
    return homomat


def load_box_homomat(path: Path | None = None) -> np.ndarray:
    """读箱体位姿（sim 世界系）。

    Args:
        path: 4x4 文本路径；``None`` = 用 ``CONFIG["path"]["box_homomat_path"]``
            （= 本项目 ``box_pose_detect/results/fixed_box_transform.txt``，实测 t=[0.7475, 0.0115, -0.0025]，
            旋转角 0.2252°，即箱底中心在机器人基座 +y 侧约 0.5 m）。

    Returns:
        np.ndarray: shape (4,4) 的箱局→世界齐次矩阵；箱局系原点 = 箱底面中心，
        z 轴竖直向上，与 ``box.STL`` 的建模坐标一致。
    """
    box_path = CONFIG["path"]["box_homomat_path"] if path is None else Path(path)
    return _read_homomat(box_path, "箱体")


def make_object_pose(position, rotmat) -> np.ndarray:
    """把「位置 + 旋转矩阵」组装成 4x4 物体位姿（sim 世界系）。

    Args:
        position: 长度 3 的序列，物体原点（= CAD 原点 = 底面中心，**不是质心**）
            在世界系的 x/y/z 坐标，单位 **米**。例如 ``[0.7475, 0.0115, 0.0035]``。
        rotmat: shape (3,3) 的旋转矩阵 ``R_world_obj``：矩阵的**列**是物体系
            x/y/z 轴在世界系下的方向余弦。物体系 z 轴 = 瓶身高度方向（向上为 +z），
            由它和 ``position`` 一起决定「瓶子朝哪边躺/站」。

    Returns:
        np.ndarray: shape (4,4)、dtype float 的物体系→世界齐次矩阵，末行恒为
        ``[0, 0, 0, 1]``。用 ``pose @ [x, y, z, 1]`` 可把物体局部点变换到世界系。

    Raises:
        ValueError: 长度不是 3 / 形状不是 3x3 / 含 NaN 或 inf / ``R @ R.T`` 与单位阵
            最大偏差超过 ``CONFIG["object"]["rotmat_ortho_tol"]``（不是纯旋转）/ ``det(R)``
            偏离 +1 超过同一容差（含镜像，是左手系）。

    Note:
        本项目**不读位姿文件**：物体位姿由 ``main.py`` 手动给定（用户口径），
        免得一个过期的 txt 静默把瓶子摆到别处。旋转矩阵抄错一行就会让夹爪从
        侧面撞壁，所以这里做正交 + 右手系校验，宁可直接报错也不静默跑歪。
    """
    pos = np.asarray(position, dtype=float).reshape(-1)
    rot = np.asarray(rotmat, dtype=float)
    ortho_tol = float(CONFIG["object"]["rotmat_ortho_tol"])

    if pos.shape != (3,):
        raise ValueError(f"position 必须是 3 个数（x, y, z，单位 m），实际 shape={pos.shape}")
    if rot.shape != (3, 3):
        raise ValueError(
            f"rotmat 必须是 3x3 旋转矩阵，实际 shape={rot.shape}。"
            "如果是 4x4 齐次矩阵，请只把左上角 3x3 传进来。"
        )
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(rot))):
        raise ValueError(f"物体位姿含 NaN/inf：position={pos.tolist()}, rotmat={rot.tolist()}")

    ortho_err = float(np.max(np.abs(rot @ rot.T - np.eye(3))))
    if ortho_err > ortho_tol:
        raise ValueError(
            f"rotmat 不是正交矩阵：max|R@R.T - I| = {ortho_err:.3e} > "
            f"{ortho_tol:.0e}，多半是抄错了某一行/某个负号。"
        )
    det = float(np.linalg.det(rot))
    if abs(det - 1.0) > ortho_tol:
        raise ValueError(
            f"rotmat 的行列式 = {det:.6f}，应接近 +1；"
            "接近 -1 说明是镜像（左手系）矩阵，不是纯旋转。"
        )

    homomat = np.eye(4)
    homomat[:3, :3] = rot
    homomat[:3, 3] = pos
    return homomat


def full_width_to_stroke(width_full: float) -> float:
    """把「指尖全距」换算成 AG145 需要的行程 stroke。

    Args:
        width_full: 两指尖内侧的全距（米）。例如 pickle 给的 0.031。

    Returns:
        float: 行程（米），已夹到 [0, ``CONFIG["gripper"]["max_stroke"]``]。物理关系为
        ``全距 = stroke + CONFIG["gripper"]["zero_gap"]``，因此全距小于零间距时返回 0（全闭）。

    Note:
        ``Ag145driver.jaw_to()`` 与 ``get_current_width()`` 走的就是 stroke 口径，
        所以本项目一律调用本函数换算，**不要**用 ``close_gripper_to()``——它内部
        按 0.042 基线减，会把 0.031 夹成 16 mm（见方案 §0.9）。
    """
    zero_gap = float(CONFIG["gripper"]["zero_gap"])
    max_stroke = float(CONFIG["gripper"]["max_stroke"])
    stroke = float(width_full) - zero_gap
    return float(min(max(stroke, 0.0), max_stroke))


def stroke_to_full_width(stroke: float) -> float:
    """把 AG145 回报的行程 stroke 换算成「指尖全距」。

    Args:
        stroke: ``Ag145driver.get_current_width()`` 的返回值（米），可能为 None
            由调用方先判空（本函数不做 None 处理，避免掩盖通信失败）。

    Returns:
        float: 两指尖内侧全距（米）= ``stroke + CONFIG["gripper"]["zero_gap"]``。用于抓取确认：
        全距明显小于 ``CONFIG["gripper"]["grasp_width_full"]`` 说明夹到了硬物或没夹到（空夹会更小）。
    """
    return float(stroke) + float(CONFIG["gripper"]["zero_gap"])


def assert_gripper_model_gap(tol: float = 1e-6) -> float:
    """校验 sim 手模型 ``Dh76().jaw_range[0]`` 与本文件的零间距一致。

    Args:
        tol: 允许的最大绝对偏差（米），默认 1e-6。

    Returns:
        float: 模型里的 ``jaw_range[0]``（米），正常应为 0.016。

    Raises:
        AssertionError: 偏差超过 ``tol``。说明有人换了指尖（改了
            ``wrs/robot_sim/end_effectors/grippers/dh76/dh76.py`` 的 ``jaw_range``）
            却没同步 ``CONFIG["gripper"]["zero_gap"]``，此时所有闭爪宽度会整体偏移，
            宁可直接报错也不要静默夹坏瓶子。

    Note:
        会实例化一次 ``Dh76``（加载 CDMesh，约数十毫秒），故不在模块导入期调用，
        由 main/执行器在启动时调用一次即可。
    """
    from robot_sim.end_effectors.gripper.dh76.dh76 import Dh76

    model_gap = float(np.asarray(Dh76().jaw_range, dtype=float).reshape(-1)[0])
    zero_gap = float(CONFIG["gripper"]["zero_gap"])
    deviation = abs(model_gap - zero_gap)
    assert deviation <= float(tol), (
        f"夹爪零间距不一致：模型 jaw_range[0]={model_gap:.6f}，"
        f"config.CONFIG['gripper']['zero_gap']={zero_gap:.6f}，偏差 {deviation:.6f} m。"
        "换过指尖就必须同步 config.CONFIG['gripper']['zero_gap']。"
    )
    return model_gap


if __name__ == "__main__":
    import grasp_plan
    import robot_con.ur.ur7e_dh76_rtde as ur7e
    import robot_sim.robots.ur7e.ur7e_withouttable_dh76 as ur7e_s
    import obstacle
    import modeling.geometric_model as mgm
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    rbt_r = ur7e.UR7EDH76_RTDE()
    rbt_s = ur7e_s.UR7EDH76()
    planner = grasp_plan.Grasp_planner(rbt_s)
    obstacle.attach_obstacles_to_base(base=base)    # 障碍物

    # pos = np.array([0.3, -0.5, 0.25])
    # rot = rm.rotmat_from_euler(np.pi, 0, 0) @ rm.rotmat_from_euler(0, 0, np.pi / 2)
    # seed = CONFIG["joints"]["place_conf"]
    # conf = planner._safe_ik(pos, rot, seed=seed)
    # print(conf)
    # if conf is not None:
    #     rbt_s.fk("arm", conf)
    #     rbt_s.gen_meshmodel().attach_to(base)
    # rbt_r.move_jnts(conf, 0.5, 0.5)

    start_conf = CONFIG["joints"]["start_conf"]
    place_conf = CONFIG["joints"]["place_conf"]
    path = planner.plan_path(place_conf, start_conf,step=0.1)
    print("path:", path)
    grasp_plan.show_path(path=path,base=base,all_conf=True)


    base.run()
