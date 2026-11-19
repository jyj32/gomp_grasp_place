"""点云通用工具：本项目自有（从 yanjiuyuan/mech_eye_ur7e_pointcloud_env.py 抄入）。

只保留 ``box_icp.py`` 与 ``detect_box_pose.py`` 用到的 3 个函数 + 相机外参，
不 import yanjiuyuan 的任何模块。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# 相机系 -> sim 世界系外参（4x4）。与 config.load_box_homomat() 同属 sim 世界系。
CAMERA_TO_WORLD = np.array([
    [-0.999834, -0.017489, 0.005110, 0.707],
    [-0.017452, 0.999821, 0.007246, -0.024],
    [-0.005236, 0.007156, -0.999961, 1.424],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

# box_icp.py 沿用的别名（与 mech_eye 模块的历史口径一致）。
CAM_TO_WORLD = CAMERA_TO_WORLD


def load_ply_pointcloud(ply_path: Path):
    """读取彩色 PLY 为 Open3D 点云。

    Args:
        ply_path: PLY 文件路径（str 或 pathlib.Path），单位由文件自身决定（米）。

    Returns:
        open3d.geometry.PointCloud: 含点（与颜色，若文件里有的话）的点云。

    Raises:
        RuntimeError: PLY 中没有有效点。
    """
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if not pcd.has_points():
        raise RuntimeError(f"No valid points found in {ply_path}")
    return pcd


def open3d_to_numpy(pcd):
    """把 Open3D 点云转成 (N,3) 坐标 + (N,3) 颜色数组，并剔除无效点。

    Args:
        pcd: open3d.geometry.PointCloud，可含颜色也可不含。

    Returns:
        tuple[np.ndarray, np.ndarray | None]: ``(points, colors)``。
        ``points`` 为 float64 (N,3)、单位米；``colors`` 为 float64 (N,3)、RGB∈[0,1]，
        点云无颜色或颜色数与点数不一致时为 ``None``。坐标含 NaN/inf 或全零点
        的行会被同时从两个数组里剔除。

    Raises:
        RuntimeError: 过滤后一个有效点都不剩。
    """
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float64)
        if len(colors) != len(points):
            colors = None
    valid_mask = np.all(np.isfinite(points), axis=1)
    valid_mask &= ~np.all(np.isclose(points, 0.0), axis=1)
    points = points[valid_mask]
    if colors is not None:
        colors = colors[valid_mask]
    if len(points) == 0:
        raise RuntimeError("Point cloud is empty after filtering invalid points.")
    return points, colors


def transform_points(points: np.ndarray, homomat: np.ndarray) -> np.ndarray:
    """用 4x4 齐次矩阵把 (N,3) 点集变换到另一坐标系。

    Args:
        points: (N,3) 点坐标数组，单位米。
        homomat: (4,4) 齐次变换矩阵；单位阵时直接原样返回 ``points``。

    Returns:
        np.ndarray: (N,3) 变换后的点坐标，单位与输入相同（米）。
    """
    if np.allclose(homomat, np.eye(4)):
        return points
    points_h = np.column_stack((points, np.ones(len(points))))
    return (homomat @ points_h.T).T[:, :3]
