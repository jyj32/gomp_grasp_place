# ⛔ 本文件从 yanjiuyuan/box_icp_from_saved_capture.py 抄入并裁剪：
#    只留库函数，删掉 offline/live CLI；两个项目不互通，改 ICP 只改这里。
"""
Detect the blue box/bin and register the pre-sampled box top-surface template with
constrained-OBB initialization + point-to-plane ICP.

Run (default = LIVE mode: move robot to capture point, take a fresh photo, detect box pose,
then show an interactive-mode-style WRS viewer with the scene point cloud + box model):
    python box_icp.py

Run offline on a previously saved capture (the old behavior):
    python box_icp.py --offline [--capture-dir DIR | --capture-ply PLY]

Adjust shared parameters in box_icp_constants.py instead of passing command-line parameters.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import sys
from pathlib import Path
from time import perf_counter
from typing import Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = REPO_ROOT / "wrs"
for _root in (REPO_ROOT, PROJECT_ROOT, WRS_ROOT):
    _root_str = str(_root)
    if _root_str not in sys.path:
        sys.path.insert(0, _root_str)

from box_icp_constants import (  # noqa: E402
    BOX_BLUE_CLUSTER_EPS,
    BOX_BLUE_CLUSTER_MIN_POINTS,
    BOX_BLUE_DOMINANCE_MARGIN,
    BOX_BLUE_HUE_RANGE_DEG,
    BOX_BLUE_MIN_SATURATION,
    BOX_BLUE_MIN_VALUE,
    BOX_ICP_MAX_ITERATION,
    BOX_MATCH_TARGET_Z_MIN,
    BOX_MATCH_TARGET_Z_MAX,
    BOX_MODEL_POINT_SIZE,
    BOX_OBB_HORIZONTAL_AXIS_MIN_NORM,
    BOX_OBB_LONG_SHORT_MIN_RATIO,
    BOX_OUTLIER_NB_NEIGHBORS,
    BOX_OUTLIER_STD_RATIO,
    BOX_REGISTRATION_VOXEL_SIZE,
    BOX_TARGET_POINT_SIZE,
    BOX_TARGET_VOXEL_DOWNSAMPLE,
    BOX_TEMPLATE_NORMAL_Z_ABS_MIN,
    BOX_TEMPLATE_PLY as DEFAULT_BOX_TEMPLATE_PLY,
    BOX_TEMPLATE_TOP_Z_MIN,
    BOX_X_RANGE,
    BOX_Y_RANGE,
    BOX_Z_RANGE,
)
from pointcloud_utils import (  # noqa: E402
    CAM_TO_WORLD,
    load_ply_pointcloud,
)


BOX_TEMPLATE_PLY = DEFAULT_BOX_TEMPLATE_PLY

CAPTURE_DIR = None
CAPTURE_PLY = None
PREFER_WORLD_FRAME_PLY = True
TRANSFORM_CAMERA_PLY_TO_WORLD = True

# RGB segmentation for the blue box. Open3D stores colors in [0, 1].
USE_BLUE_RGB_SEGMENTATION = True
BLUE_HUE_RANGE_DEG = BOX_BLUE_HUE_RANGE_DEG
BLUE_MIN_SATURATION = BOX_BLUE_MIN_SATURATION
BLUE_MIN_VALUE = BOX_BLUE_MIN_VALUE
BLUE_DOMINANCE_MARGIN = BOX_BLUE_DOMINANCE_MARGIN

# Extra target crop before registration. Z_RANGE=(0.03, None) removes the table.
X_RANGE = BOX_X_RANGE
Y_RANGE = BOX_Y_RANGE
Z_RANGE = BOX_Z_RANGE

# Clean up the segmented target.
TARGET_VOXEL_DOWNSAMPLE = BOX_TARGET_VOXEL_DOWNSAMPLE
MATCH_TARGET_Z_MIN = BOX_MATCH_TARGET_Z_MIN
MATCH_TARGET_Z_MAX = BOX_MATCH_TARGET_Z_MAX
KEEP_LARGEST_BLUE_CLUSTER = False
BLUE_CLUSTER_EPS = BOX_BLUE_CLUSTER_EPS
BLUE_CLUSTER_MIN_POINTS = BOX_BLUE_CLUSTER_MIN_POINTS
REMOVE_STATISTICAL_OUTLIERS = False
OUTLIER_NB_NEIGHBORS = BOX_OUTLIER_NB_NEIGHBORS
OUTLIER_STD_RATIO = BOX_OUTLIER_STD_RATIO

# Registration settings.
VOXEL_SIZE = BOX_REGISTRATION_VOXEL_SIZE
MODEL_TOP_Z_MIN = BOX_TEMPLATE_TOP_Z_MIN
MODEL_TOP_NORMAL_Z_ABS_MIN = BOX_TEMPLATE_NORMAL_Z_ABS_MIN
OBB_LONG_SHORT_MIN_RATIO = BOX_OBB_LONG_SHORT_MIN_RATIO
OBB_REQUIRE_LONG_SHORT_RATIO = False
OBB_HORIZONTAL_AXIS_MIN_NORM = BOX_OBB_HORIZONTAL_AXIS_MIN_NORM
ICP_MAX_ITERATION = BOX_ICP_MAX_ITERATION

# Output and optional visualization.
TARGET_POINT_SIZE = BOX_TARGET_POINT_SIZE
MODEL_POINT_SIZE = BOX_MODEL_POINT_SIZE
WRITE_DEBUG_POINTCLOUDS = True

# Progress output.
LOG_STEPS = True
OPEN3D_PRINT_PROGRESS = False


Range3D = Optional[Tuple[Optional[float], Optional[float]]]


def log(message: str) -> None:
    if LOG_STEPS:
        print(message, flush=True)


@contextmanager
def timed_step(name: str):
    start_time = perf_counter()
    log(f"[box_icp] START {name}")
    try:
        yield
    except Exception:
        elapsed = perf_counter() - start_time
        log(f"[box_icp] FAILED {name} after {elapsed:.2f}s")
        raise
    elapsed = perf_counter() - start_time
    log(f"[box_icp] DONE {name} in {elapsed:.2f}s")


def describe_pointcloud(label: str, pcd) -> None:
    points = np.asarray(pcd.points)
    if len(points) == 0:
        log(f"[box_icp] {label}: 0 points")
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    log(
        "[box_icp] "
        f"{label}: {len(points)} points, "
        f"x=[{mins[0]:.4f}, {maxs[0]:.4f}], "
        f"y=[{mins[1]:.4f}, {maxs[1]:.4f}], "
        f"z=[{mins[2]:.4f}, {maxs[2]:.4f}]"
    )


def resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def transform_open3d_pointcloud(pcd, homomat: np.ndarray):
    """按 4x4 齐次矩阵变换 Open3D 点云，返回深拷贝（不改原对象）。

    Args:
        pcd: open3d.geometry.PointCloud，待变换的点云，坐标单位米。
        homomat: (4,4) 齐次变换矩阵（numpy 数组）。

    Returns:
        open3d.geometry.PointCloud: 变换后的新点云；输入 ``pcd`` 保持不变。
    """
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.transform(homomat)
    return pcd_copy








def rgb_to_hsv_degrees(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.clip(rgb, 0.0, 1.0)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.max(rgb, axis=1)
    minc = np.min(rgb, axis=1)
    delta = maxc - minc

    hue = np.zeros(len(rgb), dtype=np.float64)
    nonzero = delta > 1e-9
    rmax = nonzero & (maxc == r)
    gmax = nonzero & (maxc == g)
    bmax = nonzero & (maxc == b)
    hue[rmax] = (60.0 * ((g[rmax] - b[rmax]) / delta[rmax]) + 360.0) % 360.0
    hue[gmax] = 60.0 * ((b[gmax] - r[gmax]) / delta[gmax] + 2.0)
    hue[bmax] = 60.0 * ((r[bmax] - g[bmax]) / delta[bmax] + 4.0)

    saturation = np.zeros(len(rgb), dtype=np.float64)
    valid_value = maxc > 1e-9
    saturation[valid_value] = delta[valid_value] / maxc[valid_value]
    value = maxc
    return hue, saturation, value


def hue_range_mask(hue: np.ndarray, hue_range: Tuple[float, float]) -> np.ndarray:
    blue_low, blue_high = hue_range
    if blue_low <= blue_high:
        return (hue >= blue_low) & (hue <= blue_high)
    return (hue >= blue_low) | (hue <= blue_high)


def blue_color_mask_components(colors: np.ndarray):
    hue, saturation, value = rgb_to_hsv_degrees(colors)
    hue_mask = hue_range_mask(hue, BLUE_HUE_RANGE_DEG)
    saturation_mask = saturation >= BLUE_MIN_SATURATION
    value_mask = value >= BLUE_MIN_VALUE

    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
    dominance_mask = (b >= r + BLUE_DOMINANCE_MARGIN) & (b >= g + BLUE_DOMINANCE_MARGIN)
    final_mask = hue_mask & saturation_mask & value_mask & dominance_mask
    components = {
        "hue": hue_mask,
        "saturation": saturation_mask,
        "value": value_mask,
        "dominance": dominance_mask,
        "hue_saturation_value": hue_mask & saturation_mask & value_mask,
        "final": final_mask,
    }
    return final_mask, components, (hue, saturation, value)


def format_count(count: int, total: int) -> str:
    if total <= 0:
        return "0 (0.0%)"
    return f"{count} ({count / total * 100.0:.1f}%)"


def format_percentiles(values: np.ndarray) -> str:
    if len(values) == 0:
        return "[]"
    return np.array2string(np.percentile(values, [5, 50, 95]), precision=3)


def log_blue_mask_diagnostics(colors: np.ndarray, mask: np.ndarray, components: dict, hsv_values) -> None:
    total = len(colors)
    hue, saturation, value = hsv_values
    log(
        "[box_icp] blue thresholds: "
        f"hue={BLUE_HUE_RANGE_DEG}, sat>={BLUE_MIN_SATURATION}, "
        f"value>={BLUE_MIN_VALUE}, dominance_margin={BLUE_DOMINANCE_MARGIN}"
    )
    log(
        "[box_icp] blue mask counts: "
        f"hue={format_count(int(components['hue'].sum()), total)}, "
        f"sat={format_count(int(components['saturation'].sum()), total)}, "
        f"value={format_count(int(components['value'].sum()), total)}, "
        f"dominance={format_count(int(components['dominance'].sum()), total)}, "
        f"hue_sat_value={format_count(int(components['hue_saturation_value'].sum()), total)}, "
        f"final={format_count(int(mask.sum()), total)}"
    )
    if np.any(mask):
        log(
            "[box_icp] blue kept HSV percentiles p05/p50/p95: "
            f"hue={format_percentiles(hue[mask])}, "
            f"sat={format_percentiles(saturation[mask])}, "
            f"value={format_percentiles(value[mask])}"
        )




def crop_pointcloud_by_range(
    pcd,
    x_range: Range3D = X_RANGE,
    y_range: Range3D = Y_RANGE,
    z_range: Range3D = Z_RANGE,
):
    # 通过xyz范围裁剪点云
    if x_range is None and y_range is None and z_range is None:
        return pcd

    points = np.asarray(pcd.points)
    mask = np.ones(len(points), dtype=bool)
    for axis, axis_range in enumerate((x_range, y_range, z_range)):
        if axis_range is None:
            continue
        low, high = axis_range
        if low is not None:
            mask &= points[:, axis] >= low
        if high is not None:
            mask &= points[:, axis] <= high

    indices = np.flatnonzero(mask).tolist()
    if not indices:
        raise RuntimeError("Target point cloud is empty after XYZ range filtering.")
    return pcd.select_by_index(indices)


def segment_blue_box_points(pcd):
    # 分割出蓝箱的点云
    if not USE_BLUE_RGB_SEGMENTATION:   # 不分割蓝箱点云
        return pcd, len(pcd.points)
    if not pcd.has_colors():    # 点云图无颜色
        raise RuntimeError("The saved point cloud has no RGB colors; blue box segmentation needs colored PLY data.")

    colors = np.asarray(pcd.colors)
    mask, components, hsv_values = blue_color_mask_components(colors)
    log_blue_mask_diagnostics(colors, mask, components, hsv_values)
    indices = np.flatnonzero(mask).tolist()
    if not indices:
        raise RuntimeError("No blue box points found. Loosen BOX_BLUE_* thresholds in box_icp_constants.py.")
    return pcd.select_by_index(indices), len(indices)


def keep_largest_cluster(pcd):
    # DBSCAN 聚类，提取出点数最多的簇（即最大的连通密集区域），并剔除其他较小的簇或离群噪点
    if not KEEP_LARGEST_BLUE_CLUSTER:
        return pcd, -1
    if len(pcd.points) < BLUE_CLUSTER_MIN_POINTS:
        return pcd, -1

    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=BLUE_CLUSTER_EPS,
            min_points=BLUE_CLUSTER_MIN_POINTS,
            print_progress=OPEN3D_PRINT_PROGRESS,
        )
    )
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        return pcd, -1

    cluster_ids, counts = np.unique(valid_labels, return_counts=True)
    largest_cluster_id = int(cluster_ids[np.argmax(counts)])
    indices = np.flatnonzero(labels == largest_cluster_id).tolist()
    return pcd.select_by_index(indices), largest_cluster_id

def voxel_downsample_pointcloud(pcd, voxel_size: Optional[float]):
    if voxel_size is not None and voxel_size > 0:
        return pcd.voxel_down_sample(voxel_size)
    return pcd


def voxel_downsample_target(pcd):

    return voxel_downsample_pointcloud(pcd, TARGET_VOXEL_DOWNSAMPLE)


def filter_pointcloud_by_z_min(pcd, z_min: Optional[float], label: str):
    if z_min is None:
        return pcd

    points = np.asarray(pcd.points)
    indices = np.flatnonzero(points[:, 2] > z_min).tolist()
    if not indices:
        raise RuntimeError(f"{label} is empty after z > {z_min} filtering.")
    return pcd.select_by_index(indices)


def filter_pointcloud_by_z_range(
    pcd,
    z_min: Optional[float],
    z_max: Optional[float],
    label: str,
):
    # 通过z范围裁剪点云
    if z_min is None and z_max is None:
        return pcd

    points = np.asarray(pcd.points)
    mask = np.ones(len(points), dtype=bool)
    range_parts = []
    if z_min is not None:
        mask &= points[:, 2] > z_min
        range_parts.append(f"z > {z_min}")
    if z_max is not None:
        mask &= points[:, 2] < z_max
        range_parts.append(f"z < {z_max}")

    indices = np.flatnonzero(mask).tolist()
    if not indices:
        raise RuntimeError(f"{label} is empty after {' and '.join(range_parts)} filtering.")
    return pcd.select_by_index(indices)


def remove_target_outliers(pcd):
    # 对目标点云进行统计离群点（噪声点）滤波
    if REMOVE_STATISTICAL_OUTLIERS and len(pcd.points) > OUTLIER_NB_NEIGHBORS:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=OUTLIER_NB_NEIGHBORS,
            std_ratio=OUTLIER_STD_RATIO,
        )
    if len(pcd.points) == 0:
        raise RuntimeError("Target point cloud is empty after cleanup.")
    return pcd


def filter_pointcloud_by_normal_z_abs_min(pcd, normal_z_abs_min: Optional[float], label: str):
    if normal_z_abs_min is None:
        return pcd
    if not pcd.has_normals():
        raise RuntimeError(
            f"{label} has no normals. Run box_surface_sampler.py with INCLUDE_NORMALS=True."
        )

    normals = np.asarray(pcd.normals)
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid_normals = normal_lengths > 1e-9
    normal_z_abs = np.zeros(len(normals), dtype=np.float64)
    normal_z_abs[valid_normals] = np.abs(normals[valid_normals, 2] / normal_lengths[valid_normals])
    indices = np.flatnonzero(normal_z_abs >= normal_z_abs_min).tolist()
    if not indices:
        raise RuntimeError(f"{label} is empty after abs(normal_z) >= {normal_z_abs_min} filtering.")
    return pcd.select_by_index(indices)


def load_box_template_pointcloud(template_ply: Path = BOX_TEMPLATE_PLY):
    """加载箱模板点云，并只保留顶面平面上的点。

    Args:
        template_ply: 模板 PLY 路径（str 或 pathlib.Path）；默认 ``BOX_TEMPLATE_PLY``
            = 本项目 ``model/box_z_parallel_surface_points.ply``。

    Returns:
        tuple[open3d.geometry.PointCloud, pathlib.Path]: ``(pcd, template_ply)``。
        ``pcd`` 为按 ``MODEL_TOP_Z_MIN`` 与 ``MODEL_TOP_NORMAL_Z_ABS_MIN`` 过滤后的
        模板点云（箱局系，单位米）；第二项是解析后的路径对象。

    Raises:
        FileNotFoundError: 模板 PLY 不存在。
    """
    # 加载盒体模板点云，并对其进行滤波，仅保留顶部平面上的点
    template_ply = resolve_path(template_ply)
    if not template_ply.exists():
        raise FileNotFoundError(
            f"Box template PLY does not exist: {template_ply}\n"
            "Run `python box_surface_sampler.py` once to generate it."
        )

    pcd = load_ply_pointcloud(template_ply)
    loaded_count = len(pcd.points)
    pcd = filter_pointcloud_by_z_min(pcd, MODEL_TOP_Z_MIN, "Box template point cloud")
    pcd = filter_pointcloud_by_normal_z_abs_min(
        pcd,
        MODEL_TOP_NORMAL_Z_ABS_MIN,
        "Box template point cloud",
    )
    log(
        "[box_icp] loaded box template PLY: "
        f"kept {len(pcd.points)} / {loaded_count} points "
        f"with z > {MODEL_TOP_Z_MIN} and abs(normal_z) >= {MODEL_TOP_NORMAL_Z_ABS_MIN}"
    )
    return pcd, template_ply


def preprocess_point_cloud_for_icp(pcd, voxel_size: float = VOXEL_SIZE):
    import open3d as o3d

    pcd_down = pcd.voxel_down_sample(voxel_size)
    if len(pcd_down.points) == 0:
        raise RuntimeError(f"Point cloud is empty after ICP voxel downsample voxel={voxel_size}.")

    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    return pcd_down


def normalize_vector(vector: np.ndarray, label: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm <= 1e-9:
        raise RuntimeError(f"Cannot normalize near-zero vector for {label}.")
    return vector / norm


def canonicalize_xy_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).copy()
    axis[2] = 0.0
    axis = normalize_vector(axis, "OBB horizontal axis")
    if axis[0] < -1e-9 or (abs(axis[0]) <= 1e-9 and axis[1] < 0.0):
        axis = -axis
    return axis


def compute_frame_center_and_extent(points: np.ndarray, axes: np.ndarray):
    coords = points @ axes
    coord_min = coords.min(axis=0)
    coord_max = coords.max(axis=0)
    center = axes @ ((coord_min + coord_max) / 2.0)
    extent = coord_max - coord_min
    return center, extent


def build_constrained_obb_frame(points: np.ndarray, long_axis: np.ndarray, label: str, method: str):
    world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    long_axis = canonicalize_xy_axis(long_axis)
    short_axis = normalize_vector(np.cross(world_z, long_axis), f"{label} short axis")
    axes = np.column_stack((long_axis, short_axis, world_z))
    center, extent = compute_frame_center_and_extent(points, axes)

    if extent[1] > extent[0]:
        long_axis, short_axis = short_axis, -long_axis
        long_axis = canonicalize_xy_axis(long_axis)
        short_axis = normalize_vector(np.cross(world_z, long_axis), f"{label} swapped short axis")
        axes = np.column_stack((long_axis, short_axis, world_z))
        center, extent = compute_frame_center_and_extent(points, axes)

    long_short_ratio = extent[0] / max(extent[1], 1e-9)
    if long_short_ratio < OBB_LONG_SHORT_MIN_RATIO:
        message = (
            f"[box_icp] {label} OBB long/short ratio {long_short_ratio:.3f} "
            f"is below {OBB_LONG_SHORT_MIN_RATIO:.3f}; extent={extent}"
        )
        if OBB_REQUIRE_LONG_SHORT_RATIO:
            raise RuntimeError(message)
        log("Warning: " + message)

    log(
        f"[box_icp] {label} constrained OBB ({method}): "
        f"center={np.array2string(center, precision=5)}, "
        f"extent={np.array2string(extent, precision=5)}, "
        f"long_short_ratio={long_short_ratio:.3f}"
    )
    return {
        "center": center,
        "axes": axes,
        "extent": extent,
        "long_short_ratio": long_short_ratio,
        "method": method,
    }


def xy_pca_long_axis(points: np.ndarray) -> np.ndarray:
    xy = points[:, :2]
    centered_xy = xy - xy.mean(axis=0)
    if len(centered_xy) < 2:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    covariance = np.cov(centered_xy, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis_xy = eigenvectors[:, int(np.argmax(eigenvalues))]
    return np.array([axis_xy[0], axis_xy[1], 0.0], dtype=np.float64)


def compute_constrained_obb_frame(pcd, label: str):
    points = np.asarray(pcd.points, dtype=np.float64)
    if len(points) < 3:
        raise RuntimeError(f"{label} needs at least 3 points for OBB initialization.")

    # The source template is a single top plane, so a 3D convex hull based OBB
    # triggers Qhull lower-dimensional warnings. The box pose we need is
    # constrained to world Z anyway, so estimate the long side in XY directly.
    long_axis = xy_pca_long_axis(points)
    return build_constrained_obb_frame(points, long_axis, label, "xy_pca_z_up")


def estimate_obb_initial_transform(source_pcd, target_pcd):
    source_obb = compute_constrained_obb_frame(source_pcd, "source template")
    target_obb = compute_constrained_obb_frame(target_pcd, "target box")

    rotation = target_obb["axes"] @ source_obb["axes"].T
    translation = target_obb["center"] - rotation @ source_obb["center"]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    log("[box_icp] OBB initial transform:\n" + np.array2string(transform, precision=6))
    return transform, source_obb, target_obb


def run_point_to_plane_icp(source_down, target_down, init_transform: np.ndarray, voxel_size: float = VOXEL_SIZE):
    import open3d as o3d

    reg = o3d.pipelines.registration
    return reg.registration_icp(
        source_down,
        target_down,
        voxel_size * 0.8,
        init_transform,
        reg.TransformationEstimationPointToPlane(),
        reg.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=ICP_MAX_ITERATION,
        ),
    )


def run_obb_initialized_icp(source_pcd, target_pcd, voxel_size: float = VOXEL_SIZE):
    #
    import open3d as o3d
    # 模板下采样
    reg = o3d.pipelines.registration
    with timed_step(f"preprocess source for ICP voxel={voxel_size}"):
        source_down = preprocess_point_cloud_for_icp(source_pcd, voxel_size)
    describe_pointcloud("source downsampled for ICP", source_down)
    # 目标（实际物体）下采样
    with timed_step(f"preprocess target for ICP voxel={voxel_size}"):
        target_down = preprocess_point_cloud_for_icp(target_pcd, voxel_size)
    describe_pointcloud("target downsampled for ICP", target_down)
    # OBB初始变换估计
    with timed_step("estimate constrained OBB initial transform"):
        obb_initial_transform, source_obb, target_obb = estimate_obb_initial_transform(source_down, target_down)
    # 评估初始变换
    distance_threshold = voxel_size * 0.8
    with timed_step(f"evaluate OBB initial transform threshold={distance_threshold}"):
        obb_initial_result = reg.evaluate_registration(
            source_down,
            target_down,
            distance_threshold,
            obb_initial_transform,
        )
    log(
        "[box_icp] OBB initial fitness/rmse: "
        f"{obb_initial_result.fitness:.6f} / {obb_initial_result.inlier_rmse:.6f}"
    )
    # 点对面 ICP 精配准
    with timed_step(f"point-to-plane ICP max_iteration={ICP_MAX_ITERATION}"):
        icp_result = run_point_to_plane_icp(
            source_down,
            target_down,
            obb_initial_transform,
            voxel_size=voxel_size,
        )
    log(f"[box_icp] ICP result fitness/rmse: {icp_result.fitness:.6f} / {icp_result.inlier_rmse:.6f}")
    return obb_initial_transform, obb_initial_result, icp_result, source_down, target_down, source_obb, target_obb







def _world_arrays_to_pcd(points_world, colors):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points_world, dtype=np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return pcd


def detect_box_pose(world_pcd, *, colors=None, voxel_size: float = VOXEL_SIZE,
                    debug_pointclouds: Optional[dict] = None) -> dict:
    """Detect the blue box pose from a world-frame Open3D point cloud.

    Single entry point shared by the offline ``main()`` and the runtime callers
    (mech_eye_ur7e_pointcloud_env, box_object_perception). Runs the full
    crop -> blue-segment -> downsample -> cluster -> outlier-remove -> z-filter
    -> template-OBB+ICP pipeline and returns a result dict.

    Args:
        world_pcd: Open3D point cloud in world frame. When it has no colors but
            blue-box segmentation is enabled, pass ``colors`` as an (N,3) array.
        colors: Optional (N,3) RGB array in [0,1], attached to ``world_pcd``.
        voxel_size: ICP voxel size (defaults to VOXEL_SIZE).
        debug_pointclouds: Optional dict; when provided (and the caller only
            passes it when WRITE_DEBUG_POINTCLOUDS is on) intermediate point
            clouds are deep-copied into it for offline debugging.

    Returns:
        dict with at least: transform, obb_initial_transform, obb_initial_result,
        icp_result, obb_initial_fitness, obb_initial_rmse, icp_fitness, icp_rmse,
        box_template_ply, segmented_target_count, downsampled_target_count,
        largest_cluster_id, largest_cluster_count, clean_target_count,
        source_count, source_pcd, target_pcd, source_obb, target_obb,
        raw_target_count. Runtime callers read the lightweight fields; offline
        ``main()`` also uses source_pcd/target_pcd and the registration result
        objects for debug output and the viewer.
    """
    if colors is not None and not world_pcd.has_colors():
        world_pcd = _world_arrays_to_pcd(np.asarray(world_pcd.points), np.asarray(colors))
    elif not world_pcd.has_colors() and USE_BLUE_RGB_SEGMENTATION:
        raise RuntimeError(
            "The point cloud has no RGB colors; blue box segmentation needs colored data. "
            "Pass `colors=` from the captured array."
        )

    raw_target_count = len(world_pcd.points)
    describe_pointcloud("raw target", world_pcd)

    with timed_step("crop target by XYZ ranges"):
        target_pcd = crop_pointcloud_by_range(world_pcd)
    describe_pointcloud("cropped target", target_pcd)

    with timed_step("segment blue box points"):
        target_pcd, segmented_target_count = segment_blue_box_points(target_pcd)
    describe_pointcloud("blue segmented target", target_pcd)

    with timed_step("voxel downsample target"):
        target_pcd = voxel_downsample_target(target_pcd)
    if debug_pointclouds is not None:
        debug_pointclouds["box_blue_downsampled_before_cluster"] = copy.deepcopy(target_pcd)
    downsampled_target_count = len(target_pcd.points)
    describe_pointcloud("downsampled target", target_pcd)

    with timed_step("keep largest cluster"):
        target_pcd, largest_cluster_id = keep_largest_cluster(target_pcd)
    if debug_pointclouds is not None:
        debug_pointclouds["box_blue_largest_cluster_before_outlier_z"] = copy.deepcopy(target_pcd)
    largest_cluster_count = len(target_pcd.points)
    describe_pointcloud("largest cluster", target_pcd)

    with timed_step("remove statistical outliers"):
        target_pcd = remove_target_outliers(target_pcd)
    if debug_pointclouds is not None:
        debug_pointclouds["box_blue_clean_before_z"] = copy.deepcopy(target_pcd)
    describe_pointcloud("clean target", target_pcd)

    with timed_step("filter target z range"):
        target_pcd = filter_pointcloud_by_z_range(
            target_pcd, MATCH_TARGET_Z_MIN, MATCH_TARGET_Z_MAX, "Box target point cloud"
        )
    if debug_pointclouds is not None:
        debug_pointclouds["box_blue_match_z_filtered"] = copy.deepcopy(target_pcd)
    clean_target_count = len(target_pcd.points)
    describe_pointcloud("match target z-filtered", target_pcd)

    with timed_step(f"load box template {BOX_TEMPLATE_PLY}"):
        source_pcd, box_template_ply = load_box_template_pointcloud()
    source_count = len(source_pcd.points)
    describe_pointcloud("source template", source_pcd)

    (
        obb_initial_transform,
        obb_initial_result,
        icp_result,
        _source_down,
        _target_down,
        source_obb,
        target_obb,
    ) = run_obb_initialized_icp(source_pcd, target_pcd, voxel_size=voxel_size)

    return {
        "transform": np.asarray(icp_result.transformation, dtype=np.float64),
        "obb_initial_transform": np.asarray(obb_initial_transform, dtype=np.float64),
        "obb_initial_result": obb_initial_result,
        "icp_result": icp_result,
        "obb_initial_fitness": float(obb_initial_result.fitness),
        "obb_initial_rmse": float(obb_initial_result.inlier_rmse),
        "icp_fitness": float(icp_result.fitness),
        "icp_rmse": float(icp_result.inlier_rmse),
        "box_template_ply": box_template_ply,
        "segmented_target_count": int(segmented_target_count),
        "downsampled_target_count": int(downsampled_target_count),
        "largest_cluster_id": int(largest_cluster_id) if largest_cluster_id is not None else -1,
        "largest_cluster_count": int(largest_cluster_count),
        "clean_target_count": int(clean_target_count),
        "source_count": int(source_count),
        "source_pcd": source_pcd,
        "target_pcd": target_pcd,
        "source_obb": source_obb,
        "target_obb": target_obb,
        "raw_target_count": int(raw_target_count),
    }


def detect_box_pose_from_arrays(points_world, colors, *, voxel_size: float = VOXEL_SIZE) -> dict:
    """从 sim 世界系彩色点云数组检测箱位姿（蓝箱分割 + OBB 初值 + 点对面 ICP）。

    Args:
        points_world: (N,3) 世界系点坐标数组，单位米。
        colors: (N,3) RGB 颜色数组，值域 [0,1]，与 ``points_world`` 逐点对应；
            ``None`` 直接抛 ``RuntimeError``（蓝色分割必须有颜色）。
        voxel_size: ICP 体素下采样边长（米），默认 ``VOXEL_SIZE``。

    Returns:
        dict: 至少含 ``transform`` (4,4)、``icp_fitness`` / ``icp_rmse``、
        ``obb_initial_transform`` / ``obb_initial_fitness`` / ``obb_initial_rmse``、
        各阶段点数（``raw_target_count`` 等）与 ``source_pcd`` / ``target_pcd``。
        完整字段见 ``detect_box_pose`` 的返回说明。

    Raises:
        RuntimeError: 无颜色、点/颜色数量不一致，或前处理把点云滤空。
    """
    points_world = np.asarray(points_world, dtype=np.float64)
    if colors is None:
        raise RuntimeError("The captured point cloud has no RGB colors; blue box segmentation needs color data.")
    colors = np.asarray(colors, dtype=np.float64)
    if len(points_world) != len(colors):
        raise RuntimeError("Point/color array length mismatch; cannot detect blue box.")
    pcd = _world_arrays_to_pcd(points_world, colors)
    return detect_box_pose(pcd, voxel_size=voxel_size)


