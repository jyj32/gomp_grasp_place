"""检测箱子齐次变换矩阵：Mech-Eye 现场拍照 → 蓝箱分割 + OBB 初值 + 点对面 ICP → 4x4。

⛔ 本项目与 ``yanjiuyuan`` **不互通**：下面每个环节的实现都抄进了本项目，
不 import yanjiuyuan 的任何模块；改算法只改本项目对应文件。

====================  ==========================================================
拍照点                  ``capture_rtde.move_to_capture_point``
                        （``config.CONFIG["joints"]["capture_conf_rad"]``，弧度）
相机采集                ``camera_capture.capture_mech_eye_pointcloud_checked``
相机系 → sim 世界系      ``pointcloud_utils.CAMERA_TO_WORLD`` + ``transform_points``
点云工具                ``pointcloud_utils.open3d_to_numpy`` / ``load_ply_pointcloud``
箱位姿 ICP              ``box_icp.detect_box_pose_from_arrays``
                        （HSV 蓝箱分割 → 裁剪/下采样/聚类 → 模板 OBB 初值 →
                        point-to-plane ICP，参数全在 ``box_icp_constants.py``）
====================  ==========================================================

⛔ 直接调相机采集函数，不套 ``capture_synced_context``
    那层包装会经 box_object_perception → main_camera_foundationpose 拉起
    FoundationPose（缺 ``Utils``），并顺带跑 YOLO/SAM 构造 ``PipelineContext``。
    本程序只要一张带颜色的世界系点云，不改变拍照与 ICP 的任何一步。

⛔ 坐标系（铁律 19）
    输出是 **sim 世界系**（``UR7EDH76().fk()`` 系）下的「箱局 → 世界」4x4，
    与 ``config.load_box_homomat()`` / 本项目
    ``box_pose_detect/results/fixed_box_transform.txt``
    同口径（箱局原点 = 箱底中心，z 轴竖直向上）。严禁与真实基系位姿相减。

⛔ 只写独立结果文件（均在 ``box_pose_detect/results/``）
    检测结果写 ``box_transform_detected.txt``，**不覆盖**
    ``fixed_box_transform.txt``（后者才是主流程实际使用的位姿）。因此
    ``main.py`` / ``obstacle.py`` / 全部回归基线保持不变。要启用检测值时，
    由调用方显式 ``cfg.load_box_homomat(path)`` 或人工确认后替换固定文件。

⛔ 默认展示（不加开关）
    ``main()`` 无论 live 还是离线 ``ply``，都在写盘/打印之后把**场景彩色点云**、
    **按 4x4 变换后的箱模板点云**、**完整半透明蓝色箱模型**放进
    WRS/Panda3D 的 ``base`` 并 ``base.run()`` 开窗。⛔ 不画箱体外廓灰线。
    形态对齐原 live viewer（WRS/Panda3D），但不挂桌面 / 相机碰撞体
    （实测大板子会遮挡点云）。恒定展示，不提供展示开关。

⛔ ``main()`` 不读命令行
    参数是普通 Python 参数（``ply`` / ``frame`` / ``out`` / ``move_to_capture``），
    脚本入口直接 ``main()`` 跑 live；被别的模块 import 调用时不会解析宿主的
    ``sys.argv``。切换模式改调用代码即可，没有 argparse / CLI flag。

用法（在 ``fast-grasp-place`` 目录下运行，或直接 import 调用）::

    python detect_box_pose.py                     # main()：live 现拍 → ICP → 写盘 → 开窗

    import detect_box_pose as dbp
    dbp.main(move_to_capture=False)               # 已在拍照点，跳过机器人运动
    dbp.main(out="/tmp/box.txt")                  # 指定结果路径
    dbp.main(ply="captures/x/colored_pointcloud.ply")             # 离线 + 展示
    dbp.main(ply="captures/x/world_colored_pointcloud.ply", frame="world")

⚠️ live 模式会**驱动真机运动**到 ``capture_conf_rad``，运行前确认急停旁有人。
传 ``ply`` 的离线模式不连接机器人与相机，可用于回归与调参。
⚠️ 默认会开 Panda3D 窗口并**阻塞到关窗**；结果文件在开窗前已写好。无显示环境的
自动化回归请 monkeypatch ``wd.World.run``（桩测就是这么做的）。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TOOL_DIR.parent
REPO_ROOT = PROJECT_DIR.parent
for _root in (str(REPO_ROOT), str(PROJECT_DIR), str(TOOL_DIR)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import config as cfg  # noqa: E402,F401  必须：config.py:27 才把 wrs 插进 sys.path

DEFAULT_OUT_PATH = TOOL_DIR / "results" / "box_transform_detected.txt"
CAPTURE_OUT_DIR = PROJECT_DIR / "logs" / "box_pose_capture"


def detect_box_pose(points_world, colors=None) -> np.ndarray:
    """从 sim 世界系彩色点云检测箱位姿（蓝箱分割 + OBB 初值 + 点对面 ICP）。

    Args:
        points_world: shape (N,3) 的 numpy 数组，**sim 世界系**点云坐标，单位米。
            相机系点云须先用 ``pointcloud_utils.CAMERA_TO_WORLD`` 变换（见
            ``load_pointcloud``），否则蓝色分割的 XYZ 裁剪区间会全部落空。
        colors: shape (N,3) 的 numpy 数组，RGB 值域 [0,1]，与 ``points_world``
            逐点对应；``None`` = 无颜色。ICP 前处理依赖 HSV 蓝色分割，故不能为 None，
            长度也必须与点数一致，否则直接抛 ``RuntimeError``。

    Returns:
        np.ndarray: shape (4,4) 的「箱局 → sim 世界系」齐次变换矩阵；箱局原点 =
        箱底中心、z 轴竖直向上，与 ``config.load_box_homomat()`` 同口径。数值来自
        ``box_icp.detect_box_pose_from_arrays()["transform"]``（ICP 精配准结果，
        不是 OBB 初值）。函数内会打印 fitness / rmse / 各阶段点数供质量判断。

    Raises:
        RuntimeError: 点云无颜色、点数与颜色数不一致，或 ICP 前处理把点云滤空。
    """
    import box_icp

    points_world = np.asarray(points_world, dtype=np.float64)
    if colors is None:
        raise RuntimeError(
            "箱位姿检测需要带 RGB 颜色的点云（蓝色分割依赖 HSV）；"
            "请用 capture_mech_eye_pointcloud_checked 的彩色点云或带颜色的 PLY。"
        )
    colors = np.asarray(colors, dtype=np.float64)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise RuntimeError(f"points_world 必须是 (N,3)，实际 {points_world.shape}")
    if len(points_world) != len(colors):
        raise RuntimeError(
            f"点/颜色数量不一致：{len(points_world)} vs {len(colors)}，无法做蓝色分割。"
        )

    result = box_icp.detect_box_pose_from_arrays(points_world, colors)
    transform = np.asarray(result["transform"], dtype=np.float64)
    print(
        "[box_pose] ICP fitness={:.6f} rmse={:.6f} | OBB初值 fitness={:.6f} rmse={:.6f}".format(
            result["icp_fitness"], result["icp_rmse"],
            result["obb_initial_fitness"], result["obb_initial_rmse"],
        )
    )
    print(
        "[box_pose] 点数 raw={} 蓝箱={} 下采样={} 聚类={} z过滤={} 模板={}".format(
            result["raw_target_count"], result["segmented_target_count"],
            result["downsampled_target_count"], result["largest_cluster_count"],
            result["clean_target_count"], result["source_count"],
        )
    )
    print("[box_pose] t(m) =", np.round(transform[:3, 3], 6).tolist())
    return transform


def load_pointcloud(ply_path, frame: str = "camera"):
    """离线读取已保存的彩色 PLY，并统一转成 sim 世界系数组。

    Args:
        ply_path: str 或 ``pathlib.Path``，彩色点云 PLY 路径。Mech-Eye 采集的
            ``colored_pointcloud.ply`` 是相机系，``world_colored_pointcloud.ply``
            是世界系。
        frame: ``"camera"`` = 用 ``pointcloud_utils.CAMERA_TO_WORLD`` 把点转到 sim 世界系；
            ``"world"`` = 点云已是 sim 世界系，原样返回。其它值抛 ``ValueError``。

    Returns:
        tuple[np.ndarray, np.ndarray | None]: ``(points_world, colors)``。
        ``points_world`` shape (N,3)、单位米、sim 世界系；``colors`` shape (N,3)、
        RGB∈[0,1]，PLY 无颜色时为 ``None``。无效点与全零点由
        ``pointcloud_utils.open3d_to_numpy`` 剔除（点数会同步减少）。

    Raises:
        FileNotFoundError: PLY 不存在。
        RuntimeError: PLY 无有效点。
        ValueError: ``frame`` 不是 ``camera`` / ``world``。
    """
    import open3d as o3d
    from pointcloud_utils import CAMERA_TO_WORLD, open3d_to_numpy, transform_points

    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"找不到点云 PLY：{ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if not pcd.has_points():
        raise RuntimeError(f"PLY 中没有点：{ply_path}")
    points, colors = open3d_to_numpy(pcd)
    if frame == "camera":
        points = transform_points(points, CAMERA_TO_WORLD)
    elif frame != "world":
        raise ValueError(f"frame 只能是 'camera' / 'world'，实际 {frame!r}")
    print(
        "[box_pose] 离线点云 {}：{} 点，frame={} → sim 世界系".format(
            ply_path.name, len(points), frame
        )
    )
    return points, colors


def capture_world_pointcloud(*, move_to_capture: bool = True,
                             depth_scale: float = 0.001,
                             depth_trunc: float = 3.0):
    """live 现拍：连机器人 → 移到拍照点 → Mech-Eye 拍彩色点云 → 转 sim 世界系。

    与 ``utils.capture_synced_context`` 使用同一个相机采集函数、同一个外参、
    同一个 ``capture_conf_rad``，但只取点云，不构造 ``PipelineContext``。

    Args:
        move_to_capture: 是否调用 ``move_to_capture_point`` 把机器人移到
            ``CONFIG["joints"]["capture_conf_rad"]``。``True``（默认）= 真机运动；
            ``False`` = 机器人已在不遮挡视野的拍照点，跳过运动。
        depth_scale: Mech-Eye 深度缩放（米/raw），默认 0.001。
        depth_trunc: 深度截断距离（米），默认 3.0。

    Returns:
        tuple[np.ndarray, np.ndarray | None]: ``(points_world, colors)``。
        ``points_world`` shape (N,3)、单位米、sim 世界系（已乘 ``CAMERA_TO_WORLD``）；
        ``colors`` shape (N,3)、RGB∈[0,1]，与点逐点对应，相机无颜色时为 ``None``。

    Raises:
        RuntimeError: 相机未返回点云 / 点云不含颜色。
        ConnectionError: Mech-Eye 相机连接检查不通过。
    """
    import camera_capture as conn_status
    import capture_rtde as rtde_utils
    from pointcloud_utils import CAMERA_TO_WORLD, open3d_to_numpy, transform_points

    # 只挑本机需要的 4 键（robot 2 + joints 2），不整体解包：joints 还含
    # start_conf/place_* 等无关键；deepcopy 防止捕获回调改到 CONFIG 全局。
    args = SimpleNamespace(
        robot_ip=cfg.CONFIG["robot"]["robot_ip"],
        gp_port=cfg.CONFIG["robot"]["gp_port"],
        capture_conf_rad=copy.deepcopy(cfg.CONFIG["joints"]["capture_conf_rad"]),
        capture_pose=copy.deepcopy(cfg.CONFIG["joints"]["capture_pose"]),
    )
    rbt_r = rtde_utils.get_rtde_robot(args)
    if move_to_capture:
        rtde_utils.move_to_capture_point(args, rtde_robot=rbt_r)
    else:
        print("[box_pose] move_to_capture=False：跳过拍照点运动，请确认机器人不遮挡相机视野")

    pcd, rgb_path, _ply_path, _camera_status = conn_status.capture_mech_eye_pointcloud_checked(
        output_dir=CAPTURE_OUT_DIR,
        ply_out=None,
        depth_scale=float(depth_scale),
        depth_trunc=float(depth_trunc),
        save_ply=False,
        save_rgb=True,
    )
    points, colors = open3d_to_numpy(pcd)
    if colors is None:
        raise RuntimeError("Mech-Eye 返回的点云没有 RGB 颜色，无法做蓝色箱分割。")
    points_world = transform_points(points, CAMERA_TO_WORLD)
    print(
        "[box_pose] live 现拍：{} 点（相机系 → sim 世界系），rgb={}".format(
            len(points_world), rgb_path
        )
    )
    return points_world, colors


def validate_homomat(transform) -> np.ndarray:
    """校验 ICP 产物是合法的刚体齐次矩阵（shape / 有限值 / 末行 / 旋转正交）。

    Args:
        transform: 任意可转 numpy 数组的对象，期望 shape (4,4) 的齐次变换；
            由 ``detect_box_pose`` 返回的 ICP ``transformation``。

    Returns:
        np.ndarray: 原值转成的 float64、(4,4) 数组，未做修改（不投影、不正交化）。

    Raises:
        ValueError: shape 不是 (4,4)、含 NaN/inf、末行不是 [0,0,0,1]、
            ``R^T R`` 偏离单位阵超过 1e-3、或 ``det(R) <= 0``（镜像/退化解）。
    """
    mat = np.asarray(transform, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"箱位姿必须是 4x4，实际 {mat.shape}")
    if not np.all(np.isfinite(mat)):
        raise ValueError("箱位姿含 NaN/inf")
    if not np.allclose(mat[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"箱位姿末行必须是 [0,0,0,1]，实际 {mat[3].tolist()}")
    rot = mat[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-3):
        raise ValueError("箱位姿旋转块不正交，ICP 结果不可用")
    if float(np.linalg.det(rot)) <= 0.0:
        raise ValueError("箱位姿旋转块是镜像/退化解（det<=0），ICP 结果不可用")
    return mat


def save_box_transform(transform, out_path=DEFAULT_OUT_PATH) -> Path:
    """把箱位姿写成 4x4 文本（与 ``config.load_box_homomat()`` 的输入同格式）。

    Args:
        transform: shape (4,4) 齐次矩阵（sim 世界系，箱局 → 世界）；调用前应已过
            ``validate_homomat``。
        out_path: str 或 ``pathlib.Path``，输出文本路径；``None`` = 用
            ``DEFAULT_OUT_PATH``（``box_pose_detect/results/box_transform_detected.txt``）。
            父目录不存在会自动创建。

    Returns:
        pathlib.Path: 实际写入的路径（绝对路径），可直接传给
        ``config.load_box_homomat(path)``。
    """
    if out_path is None:
        out_path = DEFAULT_OUT_PATH
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, np.asarray(transform, dtype=np.float64), fmt="%.9f")
    return out_path


def show_detection_result(points_world, colors, transform) -> None:
    """把检测点云与匹配结果放进 WRS/Panda3D 的 ``base`` 并开窗展示（main 的默认收尾）。

    形态对齐原 live viewer：
    场景彩色点云 + 按检测位姿变换后的箱模板点云 + **完整半透明蓝色箱模型** +
    世界系坐标轴。⛔ 不画箱体外廓灰线：蓝色箱模型已完整表达匹配位姿。
    ⛔ 只挂箱体本身的 5 块板，不挂桌面 / 相机碰撞体：后两者是大板子，实测会
    挡住点云，而本窗口只关心箱位姿配准结果。

    Args:
        points_world: shape (N,3) 的 numpy 数组，**sim 世界系**点云坐标（米），
            与 ``detect_box_pose`` 的输入同一份；由 ``load_pointcloud`` 或
            ``capture_world_pointcloud`` 返回。
        colors: shape (N,3) 的 numpy 数组，RGB 值域 [0,1]，与 ``points_world``
            逐点对应；``None`` = 无颜色，此时整片点云统一画成灰色。长度与点数
            不一致会抛 ``RuntimeError``。
        transform: shape (4,4) 的「箱局 → sim 世界系」齐次矩阵（已过
            ``validate_homomat``）。模板点云按它变换、蓝色箱模型按它摆位，画面取景
            也以它为准，所以传错矩阵能直接从窗口里看出来。

    Returns:
        None: ``base.run()`` 阻塞直到用户关闭窗口才返回；无返回值。

    Raises:
        RuntimeError: 点/颜色数量不一致，或箱模板点云加载失败。
    """
    from panda3d.core import Vec4, WindowProperties
    from wrs import mgm, wd
    import box_icp

    import obstacle

    points_world = np.asarray(points_world, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    if colors is None:
        rgba = np.array([0.6, 0.6, 0.6, 0.9])
    else:
        rgba = np.asarray(colors, dtype=np.float64)
        if len(rgba) != len(points_world):
            raise RuntimeError(
                f"点/颜色数量不一致：{len(points_world)} vs {len(rgba)}，无法展示彩色点云。"
            )
        # 实测点设半透明：ICP 后它与红模板逐点重合，全不透明会把匹配结果糊掉
        rgba = np.hstack([rgba[:, :3], np.full((len(rgba), 1), 0.45)])

    # 取景：按场景点云 bbox 计算相机距离与朝向。
    # ⛔ 不 import box_object_perception.compute_camera_from_points：它会经
    #    main_camera_foundationpose 拉起 FoundationPose 的 Utils（本环境缺）。
    center = (points_world.min(axis=0) + points_world.max(axis=0)) / 2.0
    extent = max(float(np.max(points_world.max(axis=0) - points_world.min(axis=0))), 1e-6)
    cam_dist = max(0.45, extent * 2.6)
    cam_pos = center + np.array([cam_dist, -cam_dist, max(cam_dist * 0.65, extent * 0.9)])

    base = wd.World(cam_pos=cam_pos.tolist(), lookat_pos=center.tolist(), w=1280, h=720)
    # 白底且 alpha=1：World.__init__ 只设 3 参数背景色，截图天空区会变黑
    # （grasp_plan._sim_setup_window 的实测记录）。
    props = WindowProperties()
    props.setSize(1280, 720)
    base.win.requestProperties(props)
    base.setBackgroundColor(1.0, 1.0, 1.0, 1.0)
    base.win.setClearColorActive(True)
    base.win.setClearColor(Vec4(1.0, 1.0, 1.0, 1.0))

    mgm.gen_frame(ax_length=0.15).attach_to(base)
    mgm.gen_pointcloud(points_world, rgba=rgba, point_size=0.004).attach_to(base)

    # 匹配结果 ①：模板顶面点云按检测 4x4 变换（红），与参考 live viewer 回退画法一致。
    source_pcd, _template_ply = box_icp.load_box_template_pointcloud()
    fitted_pcd = box_icp.transform_open3d_pointcloud(source_pcd, transform)
    fitted_points = np.asarray(fitted_pcd.points, dtype=np.float64)
    mgm.gen_pointcloud(fitted_points, rgba=np.array([1.0, 0.05, 0.02, 1.0]),
                       point_size=0.005).attach_to(base)

    # 匹配结果 ②：完整半透明蓝色箱模型（底板 + 四壁共 5 块有厚度的板）。
    # 几何与位姿都来自 obstacle.box_panel_models()：与规划用的碰撞 CDPrim 逐块
    # 重合，按检测 4x4 摆放；颜色/透明度取 config 的 SHOW_OBSTACLE_PANEL_* 外观常量。
    # ⛔ 不另行构造第二个箱体几何，避免画面与碰撞体两套口径漂移。
    for panel in obstacle.box_panel_models(transform):
        panel.attach_to(base)

    # ⛔ 不画箱体外廓灰线：蓝色半透明箱模型已完整表达匹配位姿，叠灰线会糊掉
    #    点云/红模板边缘（用户 11-14 要求）。
    print(
        "[box_pose] WRS viewer：彩色=场景点云，红色=ICP 匹配后的箱模板，"
        "蓝色半透明=匹配后的完整箱模型；"
        "关闭窗口结束。t(m) = {}".format(
            np.round(transform[:3, 3], 4).tolist()
        )
    )
    base.run()


def main(ply=None, frame: str = "camera", out=None,
         move_to_capture: bool = True) -> int:
    """箱位姿检测入口：默认 live 现拍 → ICP → 写盘 → WRS viewer 展示。

    ⛔ 不读命令行：四个参数都是普通 Python 参数，可安全被其它模块 import 后
    调用（不会解析宿主的 ``sys.argv``）。脚本作为 ``__main__`` 运行时直接
    ``main()`` 跑默认 live 流程。

    Args:
        ply: str / ``pathlib.Path`` / ``None``。离线彩色 PLY 路径；给了就不连接
            机器人与相机，直接读点云；``None`` = live 现拍（默认）。
        frame: str，仅 ``ply`` 非 ``None`` 时生效。``"camera"``（默认）= 乘
            ``pointcloud_utils.CAMERA_TO_WORLD`` 转到 sim 世界系；``"world"`` = PLY 已是
            sim 世界系，原样使用。其它取值由 ``load_pointcloud`` 抛 ``ValueError``。
        out: str / ``pathlib.Path`` / ``None``。结果文本路径；``None`` =
            ``DEFAULT_OUT_PATH``（``box_pose_detect/results/box_transform_detected.txt``）。
            **不覆盖** ``results/fixed_box_transform.txt``。
        move_to_capture: bool，仅 live 时生效。``True``（默认）先驱动机器人到
            ``capture_conf_rad`` 再拍照；``False`` = 机器人已在拍照点，跳过运动。

    Returns:
        int: 成功恒为 0；``show_detection_result`` 里的 ``base.run()`` 阻塞到
        用户关窗后才返回。结果文件在开窗前已写好。底层函数出错直接抛异常，
        本函数不做异常兜底。

    Raises:
        ValueError: ``frame`` 取值非法，或检测出的 4x4 没过 ``validate_homomat``。
        RuntimeError: 相机 / 机器人采集失败，或展示层点/颜色数量不一致。
    """
    if ply is not None:
        points_world, colors = load_pointcloud(ply, frame)
    else:
        print("[box_pose] live 模式：将连接机器人并移动到 capture_conf_rad 后现拍"
              "（⚠️ 真机运动，确认急停旁有人）")
        points_world, colors = capture_world_pointcloud(
            move_to_capture=move_to_capture)

    transform = validate_homomat(detect_box_pose(points_world, colors))
    out_path = save_box_transform(transform, out)
    print("[box_pose] 箱位姿（箱局 → sim 世界系）：\n" + np.array2string(transform, precision=9))
    print(f"[box_pose] 已写入 {out_path}")
    print("[box_pose] ⚠️ 未改动 results/fixed_box_transform.txt；"
          "如需启用，请显式传给 config.load_box_homomat(path) 或人工确认后替换。")
    # 默认收尾：把场景点云与匹配结果放进 WRS base 并开窗（live 与离线都展示）。
    show_detection_result(points_world, colors, transform)
    return 0


if __name__ == "__main__":
    main()
