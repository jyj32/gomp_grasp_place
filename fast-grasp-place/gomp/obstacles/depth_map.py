"""
Depth-map based obstacle model for GOMP.

From GOMP Section IV-C: pick-and-place robots can operate with a simplified
task-specific obstacle model — specifically a depth map. This avoids the
complexities of GJK/EPA collision detection.

The depth map represents the obstacle surface as a 2D grid of z-heights.
For a bin-picking scenario, the obstacles are the bin walls and surrounding
workspace elements.
"""

import math

import numpy as np
from typing import Tuple


class DepthMapObstacle:
    """
    A 2D depth map representing obstacle heights in the workspace.

    The depth map covers a rectangular region in the xy-plane, with each
    cell storing the z-height of the obstacle surface at that location.
    The robot must keep its end-effector (and optionally other links)
    above this height at all times.

    Parameters
    ----------
    grid : np.ndarray, shape (nx, ny)
        2D array of z-heights (meters).
    resolution : float
        Grid resolution in meters per cell.
    origin : np.ndarray, shape (2,)
        World-frame (x, y) position of the grid's (0, 0) corner.
    """

    def __init__(self, grid: np.ndarray, resolution: float,
                 origin: np.ndarray, *, yaw: float = 0.0,
                 center_xy: np.ndarray = None):
        self.grid = grid.astype(float)
        self.resolution = resolution
        self.origin = np.asarray(origin, dtype=float)
        self.nx, self.ny = grid.shape
        self.yaw = float(yaw)
        self._cos = math.cos(self.yaw)
        self._sin = math.sin(self.yaw)
        if center_xy is None:
            center_xy = np.zeros(2)
        self.center_xy = np.asarray(center_xy, dtype=float)
        self._maxfilter_cache = {}  # radius -> max-filtered grid (inflate>0 lookups)

    def get_obstacle_height(self, x: float, y: float,
                            inflate: float = 0.0) -> float:
        """
        Look up the obstacle z-height at world position (x, y).

        Uses bilinear interpolation for sub-cell positions.
        Returns 0.0 if the position is outside the grid.

        Parameters
        ----------
        x, y : float
            World-frame position.
        inflate : float
            If > 0, return the max obstacle height inside the 2*inflate
            square window around (x, y) (window outside the grid = 0.0).
            Used for held-object box footprint checks.

        Returns
        -------
        z_obstacle : float
            Height of the obstacle surface.
        """
        grid = self._max_filtered_grid(inflate)
        # 变换到障碍局系（支持箱体绕 Z 偏航；yaw=0 退化为原世界系查表）
        _dx = x - self.center_xy[0]
        _dy = y - self.center_xy[1]
        _lx = self._cos * _dx + self._sin * _dy
        _ly = -self._sin * _dx + self._cos * _dy
        # Convert to grid coordinates
        gx = (_lx - self.origin[0]) / self.resolution
        gy = (_ly - self.origin[1]) / self.resolution

        # Check bounds
        if gx < 0 or gx >= self.nx - 1 or gy < 0 or gy >= self.ny - 1:
            return 0.0  # No obstacle outside the grid

        # Bilinear interpolation
        ix = int(gx)
        iy = int(gy)
        fx = gx - ix
        fy = gy - iy

        z00 = grid[ix, iy]
        z10 = grid[ix + 1, iy]
        z01 = grid[ix, iy + 1]
        z11 = grid[ix + 1, iy + 1]

        if inflate > 0.0:
            # 保守：取 4 角滤波值的 max（4 角窗口的并集覆盖落脚圆）；
            # 双线性平均会把仅覆盖个别角的薄墙最大值抹掉，导致漏检。
            return max(z00, z10, z01, z11)
        z = (z00 * (1 - fx) * (1 - fy) +
             z10 * fx * (1 - fy) +
             z01 * (1 - fx) * fy +
             z11 * fx * fy)
        return z

    def get_obstacle_height_batch(self, xy: np.ndarray,
                                  inflate: float = 0.0) -> np.ndarray:
        """Vectorized bilinear height lookup for an (N, 2) array of (x, y).

        inflate > 0: per-point max over the 2*inflate square window (see
        get_obstacle_height).
        """
        grid = self._max_filtered_grid(inflate)
        xy = np.asarray(xy, dtype=float)
        out = np.zeros(xy.shape[0])
        # 变换到障碍局系（支持箱体绕 Z 偏航）
        _dx = xy[:, 0] - self.center_xy[0]
        _dy = xy[:, 1] - self.center_xy[1]
        _lx = self._cos * _dx + self._sin * _dy
        _ly = -self._sin * _dx + self._cos * _dy
        gx = (_lx - self.origin[0]) / self.resolution
        gy = (_ly - self.origin[1]) / self.resolution
        inside = (gx >= 0) & (gx < self.nx - 1) & (gy >= 0) & (gy < self.ny - 1)
        ix = gx[inside].astype(int)
        iy = gy[inside].astype(int)
        fx = gx[inside] - ix
        fy = gy[inside] - iy
        z00 = grid[ix, iy]
        z10 = grid[ix + 1, iy]
        z01 = grid[ix, iy + 1]
        z11 = grid[ix + 1, iy + 1]
        if inflate > 0.0:
            # 保守：4 角滤波值取 max（见 get_obstacle_height 注释）
            out[inside] = np.maximum.reduce([z00, z10, z01, z11])
        else:
            out[inside] = (z00 * (1 - fx) * (1 - fy)
                           + z10 * fx * (1 - fy)
                           + z01 * (1 - fx) * fy
                           + z11 * fx * fy)
        return out

    def _max_filtered_grid(self, radius: float) -> np.ndarray:
        """按半径缓存的最大滤波网格；窗外按 0.0（自由空间）处理。

        inflate > 0 的高度查询用它：取该点周边 2*radius 方形窗口内的最大障碍高，
        用于持有物体盒组的「落脚圆内最高障碍」检查（对方形窗稍保守，对轴对齐
        墙体精确；窗外=0 与地图外的自由空间语义一致）。
        """
        key = round(float(radius), 6)
        cached = self._maxfilter_cache.get(key)
        if cached is None:
            if key <= 0.0:
                cached = self.grid
            else:
                from scipy import ndimage
                w = max(1, int(np.ceil(key / self.resolution)))
                cached = ndimage.maximum_filter(self.grid, size=2 * w + 1,
                                                mode='constant', cval=0.0)
            self._maxfilter_cache[key] = cached
        return cached

    @classmethod
    def create_bin_obstacle(cls, bin_center: np.ndarray,
                            bin_size: Tuple[float, float, float],
                            wall_height: float,
                            bottom_thickness: float = 0.01,
                            wall_thickness: float = 0.01,
                            resolution: float = 0.01,
                            margin: float = 0.05,

                            yaw: float = 0.0) -> 'DepthMapObstacle':
        """
        Create a depth map for a rectangular bin modeled as a real collision body
        (a bottom plate plus 4 walls), expressed in the 2.5D height map.

        Obstacle z-height per cell:
          - bin interior (inside all 4 inner wall faces): z = bottom_thickness
            (the floor plate; the robot must stay above it over the bin)
          - the 4 wall rings (a ring of thickness `wall_thickness` around the
            interior): z = wall_height (the wall top)
          - outside the walls (including the margin ring): z = 0.0 (free
            space; the robot may move at any height there)

        Parameters
        ----------
        bin_center : np.ndarray, shape (3,)
            World-frame center of the bin.
        bin_size : tuple (width, depth, height)
            Inner footprint of the bin (between the inner faces of the walls).
            `height` is accepted for API compatibility but is currently unused.
        wall_height : float
            Absolute z-height of the top of the bin walls (table = 0).
        bottom_thickness : float
            Thickness of the bin floor plate; sets the interior obstacle height.
            Default 0.0 reproduces the old "interior is free" behaviour.
        wall_thickness : float
            Horizontal thickness of the 4 walls (width of the wall ring).
            Default 0.0 reproduces the old zero-thickness walls.
        resolution : float
            Grid cell size in meters.
        margin : float
            Extra area around the bin to include in the grid.

        Returns
        -------
        DepthMapObstacle
            The depth map obstacle.
        """
        width, depth, _height = bin_size
        half_w, half_d = width / 2.0, depth / 2.0
        # Grid covers bin + walls + margin on all sides
        total_w = width + 2 * wall_thickness + 2 * margin
        total_d = depth + 2 * wall_thickness + 2 * margin

        nx = int(total_w / resolution) + 1
        ny = int(total_d / resolution) + 1

        # 网格在【箱局系】构建：原点 = 局系 (-total_w/2, -total_d/2)；查表时由
        # __init__ 的 center_xy + yaw 把世界系 (x, y) 变换回箱局系。
        origin = np.array([
            -total_w / 2,
            -total_d / 2,
        ])

        # Build coordinate grids (vectorized)
        xs = origin[0] + np.arange(nx) * resolution
        ys = origin[1] + np.arange(ny) * resolution
        # 局系构建：箱心在局系原点，距离直接取 |X| / |Y|（不再减 bin_center）
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        dx = np.abs(X)
        dy = np.abs(Y)

        # Default: outside the bin (including the margin ring) => free space (0.0)
        grid = np.full((nx, ny), 0.0)
        # 4 walls: ring of thickness wall_thickness around the interior
        # (this also covers the interior; the interior is overwritten below)
        outer = (dx < half_w + wall_thickness) & (dy < half_d + wall_thickness)
        grid[outer] = wall_height
        # Bin interior (inside all 4 inner wall faces) => bottom plate height
        interior = (dx < half_w) & (dy < half_d)
        grid[interior] = bottom_thickness

        obj = cls(grid=grid, resolution=resolution, origin=origin,
                  yaw=yaw, center_xy=np.asarray(bin_center[:2], dtype=float))
        # Store the bin geometry so visualizers (and debugging) can draw a
        # proper solid bin (4 walls + floor) instead of sampling thin cells.
        obj.bin_center = np.asarray(bin_center, dtype=float)
        obj.bin_half_w = half_w
        obj.bin_half_d = half_d
        obj.bin_wall_thickness = wall_thickness
        obj.bin_wall_height = wall_height
        obj.bin_bottom_thickness = bottom_thickness
        obj.bin_yaw = yaw
        return obj

    @classmethod
    def create_flat_obstacle(cls, height: float = 0.0,
                             center: np.ndarray = None,
                             size: float = 2.0,
                             resolution: float = 0.01) -> 'DepthMapObstacle':
        """
        Create a flat obstacle (table surface) at a given height.

        Parameters
        ----------
        height : float
            Height of the flat surface.
        center : np.ndarray
            Center of the grid. Default: origin.
        size : float
            Side length of the square grid.
        resolution : float
            Grid resolution.

        Returns
        -------
        DepthMapObstacle
        """
        if center is None:
            center = np.zeros(2)

        nx = int(size / resolution) + 1
        ny = nx
        origin = center - size / 2
        grid = np.full((nx, ny), height)
        return cls(grid=grid, resolution=resolution, origin=origin)
