# ⛔ 本文件从 yanjiuyuan/connection_status.py 抄入并裁剪：只留相机采集库函数，
#    删掉 CLI（main/parse_args/print_status）；两个项目不互通。
"""Connection status helpers for the UR7e/DH76 robot and Mech-Eye camera."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import cv2
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = REPO_ROOT   # 本机 wrs 是扁平布局，没有 wrs/ 子包
for root in (REPO_ROOT, WRS_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# Keep these in sync with wrs.drivers.devices.Mech_eye.Mech_camera.CaptureImage.
# They are the current Mech-Eye intrinsics used by the existing backend.
MECH_EYE_FX = 2419.864311550494
MECH_EYE_FY = 2419.976305839313
MECH_EYE_CX = 972.6079631929535
MECH_EYE_CY = 562.4548210192155


@dataclass
class ConnectionCheck:
    name: str
    ok: bool
    detail: str = ""
    error: Optional[str] = None

    def line(self) -> str:
        state = "OK" if self.ok else "FAIL"
        suffix = self.detail
        if self.error:
            suffix = f"{suffix}; {self.error}" if suffix else self.error
        return f"{self.name}: {state}" + (f" ({suffix})" if suffix else "")


@dataclass
class LiveConnectionStatus:
    checks: list[ConnectionCheck]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def line(self) -> str:
        return " | ".join(check.line() for check in self.checks)

    def multiline(self) -> str:
        return "\n".join(check.line() for check in self.checks)


def ok(name: str, detail: str = "") -> ConnectionCheck:
    return ConnectionCheck(name=name, ok=True, detail=detail)


def fail(name: str, error: object, detail: str = "") -> ConnectionCheck:
    return ConnectionCheck(name=name, ok=False, detail=detail, error=str(error))


def _is_connected(obj) -> Optional[bool]:
    method = getattr(obj, "isConnected", None)
    if method is None:
        return None
    return bool(method())


def check_robot_provider(provider, mock: bool = False) -> LiveConnectionStatus:
    if mock:
        return LiveConnectionStatus([
            ok("robot RTDE control", "mock"),
            ok("robot RTDE receive", "mock"),
            ok("DH76 gripper", "mock"),
        ])

    checks: list[ConnectionCheck] = []
    robot_x = getattr(provider, "_robot_x", None)

    rtde_c = None if robot_x is None else getattr(robot_x, "rtde_c", getattr(robot_x, "_rtde_c", None))
    rtde_r = None if robot_x is None else getattr(robot_x, "rtde_r", getattr(robot_x, "_rtde_r", None))
    hnd = None if robot_x is None else getattr(robot_x, "hnd", getattr(robot_x, "_hnd", None))

    try:
        connected = _is_connected(rtde_c)
        if connected is False:
            checks.append(fail("robot RTDE control", "isConnected() returned False"))
        elif connected is True:
            checks.append(ok("robot RTDE control", "connected"))
        elif robot_x is not None and hasattr(robot_x, "check_rtdec_is_connected"):
            checks.append(ok("robot RTDE control", "reconnected" if robot_x.check_rtdec_is_connected() else "not connected"))
            if checks[-1].detail == "not connected":
                checks[-1].ok = False
        else:
            checks.append(ok("robot RTDE control", "not directly checkable"))
    except Exception as exc:
        checks.append(fail("robot RTDE control", exc))

    try:
        connected = _is_connected(rtde_r)
        jnt_values = np.asarray(provider.get_jnt_values(), dtype=float).reshape(-1)
        detail = f"connected, joints={np.round(np.degrees(jnt_values), 2).tolist()}deg"
        if connected is False:
            checks.append(fail("robot RTDE receive", "isConnected() returned False", detail=detail))
        else:
            checks.append(ok("robot RTDE receive", detail))
    except Exception as exc:
        checks.append(fail("robot RTDE receive", exc))

    try:
        gripper_connected = None
        if hnd is not None and hasattr(hnd, "check_connection"):
            gripper_connected = bool(hnd.check_connection())
        jaw_width = provider.get_jaw_width() if hasattr(provider, "get_jaw_width") else None
        detail = ""
        if jaw_width is not None:
            detail = f"jaw_width={float(jaw_width):.6f}m"
        if gripper_connected is False:
            checks.append(fail("DH76 gripper", "check_connection() returned False", detail=detail))
        elif jaw_width is None and gripper_connected is None:
            checks.append(fail("DH76 gripper", "jaw width unavailable"))
        else:
            checks.append(ok("DH76 gripper", detail or "connected"))
    except Exception as exc:
        checks.append(fail("DH76 gripper", exc))

    return LiveConnectionStatus(checks)


def check_camera_instance(capture_image) -> ConnectionCheck:
    try:
        if not hasattr(capture_image, "is_connected"):
            return ok("Mech-Eye camera", "is_connected() unavailable after construction")
        if capture_image.is_connected():
            return ok("Mech-Eye camera", "connected")
        return fail("Mech-Eye camera", "is_connected() returned False")
    except Exception as exc:
        return fail("Mech-Eye camera", exc)


def open_camera_for_status(save_directory: Path | str) -> tuple[object, ConnectionCheck]:
    from drivers.devices.Mech_eye.Mech_camera import CaptureImage

    camera = CaptureImage(save_directory=str(save_directory))
    return camera, check_camera_instance(camera)


def close_camera_quietly(capture_image) -> None:
    camera = getattr(capture_image, "camera", None)
    if camera is None:
        return
    try:
        camera.disconnect()
    except Exception:
        pass


def capture_mech_eye_pointcloud_checked(
    output_dir: Path,
    ply_out: Optional[Path],
    depth_scale: float,
    depth_trunc: float,
    save_ply: bool = True,
    return_pixel_indices: bool = False,
    return_rgb: bool = False,
    return_depth: bool = False,
    save_rgb: bool = True,
    camera: object = None,
):
    """Mech-Eye 现场拍一张彩色点云，先检查相机连接状态。

    Args:
        output_dir: 输出目录（str 或 pathlib.Path），不存在会自动创建。
        ply_out: 彩色 PLY 输出路径；``None`` = ``output_dir/colored_pointcloud.ply``；
            ``save_ply=False`` 时不写盘。
        depth_scale: 深度缩放系数（米/计数），传给相机 SDK。
        depth_trunc: 深度截断距离（米），超过则该像素无效。
        save_ply: 是否把彩色点云写成 PLY，默认 True。
        return_pixel_indices: 是否在返回元组末尾追加有效像素下标数组，默认 False。
        return_rgb: 是否追加 RGB 图像数组，默认 False。
        return_depth: 是否追加深度图像数组，默认 False。
        save_rgb: 是否把 RGB 写成 ``output_dir/rgb.png``；False 时第二个返回值为 None。
        camera: 外部已建立的相机实例；``None`` = 函数内部打开并在 finally 关闭。
            传入时由调用方管理生命周期，函数不会 disconnect。

    Returns:
        tuple: 基础 4 项 ``(pcd, rgb_path, colored_ply_path, camera_status)``，
        其中 ``pcd`` 是 open3d.geometry.PointCloud（米）、``camera_status`` 是连接
        检查结果对象；按 ``return_pixel_indices`` / ``return_rgb`` / ``return_depth``
        的真值依次在末尾追加对应数据。

    Raises:
        ConnectionError: 相机连接检查不通过。
        RuntimeError: 相机未返回 RGB/点云，或深度有效像素数与点数不一致。
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = output_dir / "rgb.png"
    pointcloud_output_path = ply_out if ply_out is not None else output_dir / "colored_pointcloud.ply"
    colored_ply_path = pointcloud_output_path if save_ply else None
    if colored_ply_path is not None:
        colored_ply_path.parent.mkdir(parents=True, exist_ok=True)

    # When a camera is supplied, reuse the live connection and do NOT disconnect
    # it here (the caller owns its lifecycle). Otherwise open/close per call.
    owns_camera = camera is None
    try:
        if camera is None:
            camera, camera_status = open_camera_for_status(output_dir)
        else:
            if hasattr(camera, "save_directory"):
                camera.save_directory = str(output_dir)
            if hasattr(camera, "check_connect_and_reconnect"):
                camera.check_connect_and_reconnect()
            camera_status = check_camera_instance(camera)
        if not camera_status.ok:
            raise ConnectionError(camera_status.line())
        if hasattr(camera, "capture_rgb_and_depth"):
            rgb, depth = camera.capture_rgb_and_depth(save=False, show=False)
            pcd, pixel_indices = camera.generate_pointcloud_masked(
                color_image=rgb,
                depth_image=depth,
                mask=None,
                return_indices=True,
                depth_scale=depth_scale,
                depth_trunc=depth_trunc,
            )
        else:
            rgb, depth, pcd = camera.capture_and_generate_pointcloud(
                save=False,
                show=False,
                pcb_out_path=str(pointcloud_output_path),
                depth_scale=depth_scale,
                depth_trunc=depth_trunc,
                keep_invalid=False,
            )
            pixel_indices = None
        if rgb is None or pcd is None:
            raise RuntimeError("Mech-Eye capture did not return RGB data and point cloud data.")
        if return_pixel_indices:
            if pixel_indices is None:
                if depth is None:
                    raise RuntimeError("Mech-Eye capture did not return depth data for pixel index mapping.")
                if np.asarray(rgb).shape[:2] != np.asarray(depth).shape[:2]:
                    raise RuntimeError(
                        "RGB/depth dimensions differ; direct point-to-pixel mapping requires aligned images: "
                        f"rgb={np.asarray(rgb).shape[:2]}, depth={np.asarray(depth).shape[:2]}."
                    )
                depth_values = np.asarray(depth).astype(np.float32, copy=False).reshape(-1) * float(depth_scale)
                valid_mask = (depth_values > 0.0) & (depth_values < float(depth_trunc)) & np.isfinite(depth_values)
                pixel_indices = np.flatnonzero(valid_mask).astype(np.int64)
            point_count = len(np.asarray(pcd.points))
            if len(pixel_indices) != point_count:
                raise RuntimeError(
                    "Depth valid-mask pixel count does not match generated point cloud count: "
                    f"{len(pixel_indices)} != {point_count}."
                )
        if save_rgb:
            cv2.imwrite(str(rgb_path), rgb)
        else:
            rgb_path = None
        if not pcd.has_colors():
            print("Warning: captured point cloud has no color data.")
        if colored_ply_path is not None:
            o3d.io.write_point_cloud(str(colored_ply_path), pcd, write_ascii=False)
        base = (pcd, rgb_path, colored_ply_path, camera_status)
        if return_pixel_indices:
            base = base + (pixel_indices,)
        if return_rgb:
            base = base + (rgb,)
        if return_depth:
            base = base + (depth,)
        return base
    finally:
        if owns_camera and camera is not None:
            close_camera_quietly(camera)


def check_mech_eye_camera(save_directory: Path | str = ".") -> LiveConnectionStatus:
    camera = None
    try:
        camera, camera_status = open_camera_for_status(save_directory)
        return LiveConnectionStatus([camera_status])
    except Exception as exc:
        return LiveConnectionStatus([fail("Mech-Eye camera", exc)])
    finally:
        if camera is not None:
            close_camera_quietly(camera)
