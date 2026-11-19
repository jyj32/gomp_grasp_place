"""
WRS-based 3D visualization for GOMP trajectories.

Uses Panda3D (via WRS) to render the robot executing a trajectory,
with obstacles and grasp frames shown in the scene.
"""

import os
import numpy as np
from typing import Optional

import basis.robot_math as rm
from modeling import geometric_model as gm
from modeling import collision_model as mcm


def visualize_trajectory(robot_adapter,
                         trajectory,
                         obstacles=None,
                         n_frames: int = 10,
                         cam_pos: list = None,
                         lookat_pos: list = None,
                         held_cmodel=None,
                         held_pos_in_tcp: Optional[np.ndarray] = None,
                         held_rotmat_in_tcp: Optional[np.ndarray] = None):
    """
    Visualize a GOMP trajectory using WRS Panda3D rendering.

    Shows the robot at evenly-spaced keyframes along the trajectory,
    with a line connecting the end-effector positions.

    Parameters
    ----------
    robot_adapter : RobotAdapter
        Robot model.
    trajectory : Trajectory
        Computed trajectory to visualize.
    obstacles : DepthMapObstacle or None
        Obstacle to render.
    n_frames : int
        Number of robot poses to display along the trajectory.
    cam_pos : list or None
        Camera position [x, y, z].
    lookat_pos : list or None
        Camera look-at position [x, y, z].
    held_cmodel : CollisionModel or None
        WRS model of the object rigidly held by the gripper (e.g. the
        bottle from load_bottle_model()). One copy is drawn per keyframe at
        T_world_tcp @ T_tcp_object, i.e. travelling with the gripper.
    held_pos_in_tcp : np.ndarray or None
        Object origin expressed in the TCP frame.
    held_rotmat_in_tcp : np.ndarray or None
        Object orientation expressed in the TCP frame.
    """
    import visualization.panda.world as wd

    if cam_pos is None:
        cam_pos = [2, 0, 1.5]
    if lookat_pos is None:
        lookat_pos = [0, 0, 0.3]

    base = wd.World(cam_pos=cam_pos, lookat_pos=lookat_pos)

    # World frame
    gm.gen_frame(length=0.2).attach_to(base)

    # Select keyframe indices
    if n_frames > trajectory.n_waypoints:
        n_frames = trajectory.n_waypoints
    indices = np.linspace(0, trajectory.H, n_frames, dtype=int)

    # Render robot at each keyframe
    ee_positions = []
    for idx, frame_idx in enumerate(indices):
        q = trajectory.waypoints[frame_idx]
        robot_adapter.goto_given_conf(q)

        # Vary alpha for temporal progression
        alpha = 0.2 + 0.8 * (idx / max(1, len(indices) - 1))

        # Generate mesh model (rgba alpha encodes temporal progression)
        mesh = robot_adapter.robot.gen_meshmodel(
            toggle_tcpcs=False,
            rgba=np.array([0.5, 0.5, 0.5, alpha]))
        mesh.attach_to(base)

        # Collect EE positions for path line
        pos, rot = robot_adapter.forward_kinematics(q)
        ee_positions.append(pos.copy())

        # Held object: rigidly attached to the TCP, so it travels with the
        # gripper. Same pose convention as AttachedObject (object frame
        # expressed in the TCP frame).
        if held_cmodel is not None and held_pos_in_tcp is not None:
            _draw_held_object(base, held_cmodel, pos, rot,
                              held_pos_in_tcp, held_rotmat_in_tcp,
                              alpha=alpha)

    # Draw end-effector path
    for i in range(len(ee_positions) - 1):
        gm.gen_stick(spos=ee_positions[i], epos=ee_positions[i + 1],
                     thickness=0.003, rgba=np.array([1, 0.3, 0, 1])).attach_to(base)

    # Draw start/end frames
    if len(ee_positions) > 0:
        pos_start, rot_start = robot_adapter.forward_kinematics(trajectory.waypoints[0])
        gm.gen_frame(pos=pos_start, rotmat=rot_start,
                     length=0.08).attach_to(base)

        pos_end, rot_end = robot_adapter.forward_kinematics(trajectory.waypoints[-1])
        gm.gen_frame(pos=pos_end, rotmat=rot_end,
                     length=0.08).attach_to(base)

    # Draw obstacle (bin) if provided
    if obstacles is not None:
        _draw_bin_obstacle(base, obstacles)

    # Title
    duration_ms = trajectory.duration * 1000
    base.setBackgroundColor(0.9, 0.9, 0.9, 1)

    print(f"Trajectory visualization: H={trajectory.H}, "
          f"duration={trajectory.duration:.3f}s ({duration_ms:.0f}ms)")
    print("Close the window to continue...")

    base.run()


def _draw_bin_obstacle(base, depth_map):
    """Draw the bin obstacle.

    If the depth map carries bin geometry metadata (set by
    ``create_bin_obstacle``), draw a proper solid bin (4 walls + floor)
    so thin walls are not lost by cell subsampling. Otherwise fall back to
    the old per-cell box sampling (e.g. for flat obstacles).
    """
    if (hasattr(depth_map, "bin_center")
            and depth_map.bin_wall_thickness > 0):
        _draw_solid_bin(base, depth_map)
        return

    # Fallback: per-cell sampling (e.g. flat obstacles)
    nx, ny = depth_map.grid.shape
    step = max(1, min(nx, ny) // 20)  # Subsample for performance

    for i in range(0, nx - 1, step):
        for j in range(0, ny - 1, step):
            z = depth_map.grid[i, j]
            if z > 0.001:  # Only draw raised cells
                x = depth_map.origin[0] + i * depth_map.resolution
                y = depth_map.origin[1] + j * depth_map.resolution
                size = np.array([depth_map.resolution * step,
                                 depth_map.resolution * step, z])
                homomat = rm.homomat_from_posrot(np.array([x, y, z / 2]))
                gm.gen_box(extent=size, homomat=homomat,
                           rgba=np.array([0.6, 0.6, 0.8, 0.5])).attach_to(base)


def _draw_solid_bin(base, depth_map):
    """Draw a rectangular bin as 4 wall boxes + 1 floor box."""
    cx, cy = depth_map.bin_center[0], depth_map.bin_center[1]
    hw = depth_map.bin_half_w            # inner half-width  (x)
    hd = depth_map.bin_half_d            # inner half-depth  (y)
    wt = depth_map.bin_wall_thickness    # wall thickness
    wh = depth_map.bin_wall_height       # wall height (absolute z of wall top)
    bt = depth_map.bin_bottom_thickness  # floor thickness
    color = np.array([0.6, 0.6, 0.8])
    rgba = np.concatenate([color, [0.5]])

    outer_hw = hw + wt
    outer_hd = hd + wt

    # Floor plate (bin interior footprint, thickness bt)
    if bt > 0:
        homomat = rm.homomat_from_posrot(np.array([cx, cy, bt / 2]))
        gm.gen_box(extent=np.array([2 * hw, 2 * hd, bt]),
                   homomat=homomat, rgba=rgba).attach_to(base)

    # +x / -x walls: span full outer depth, thickness wt along x, height wh
    for sign in (+1, -1):
        homomat = rm.homomat_from_posrot(
            np.array([cx + sign * (hw + wt / 2), cy, wh / 2]))
        gm.gen_box(extent=np.array([wt, 2 * outer_hd, wh]),
                   homomat=homomat, rgba=rgba).attach_to(base)
    # +y / -y walls: span inner width, thickness wt along y, height wh
    for sign in (+1, -1):
        homomat = rm.homomat_from_posrot(
            np.array([cx, cy + sign * (hd + wt / 2), wh / 2]))
        gm.gen_box(extent=np.array([2 * hw, wt, wh]),
                   homomat=homomat, rgba=rgba).attach_to(base)


def _draw_held_object(base, cmodel, tcp_pos, tcp_rotmat,
                      pos_in_tcp, rotmat_in_tcp, alpha=1.0):
    """Draw a copy of ``cmodel`` at the TCP-fixed pose T_world_tcp @ T_tcp_obj.

    The held object is rigidly attached to the gripper, so one copy is drawn
    per robot keyframe. ``copy()`` deep-copies the model (carrying the pending
    pos/rotmat/rgba), so the very same ``cmodel`` can be reused at every
    keyframe without rebuilding it.
    """
    cmodel.set_pos(tcp_pos + tcp_rotmat @ pos_in_tcp)
    cmodel.set_rotmat(tcp_rotmat @ (np.eye(3) if rotmat_in_tcp is None
                                    else rotmat_in_tcp))
    cmodel.set_rgba([1.0, 0.55, 0.15, float(alpha)])
    return cmodel.copy().attach_to(base)


def load_bottle_model():
    """The detergent bottle as a WRS CollisionModel (visual mesh).

    The planner-side collision primitives are supplied separately by the
    caller's custom cdprim function (see AttachedObject.from_cdprim_fn);
    here only the visual mesh is needed for rendering.
    """
    mesh_path = os.path.join(os.path.dirname(__file__),
                             "..", "..", "examples", "meshes", "bottle.stl")
    return mcm.CollisionModel(initor=mesh_path, cdprimit_type="box",
                              expand_radius=0.0)
def load_U1_model():
    """The detergent bottle as a WRS CollisionModel (visual mesh).

    The planner-side collision primitives are supplied separately by the
    caller's custom cdprim function (see AttachedObject.from_cdprim_fn);
    here only the visual mesh is needed for rendering.
    """
    mesh_path = os.path.join(os.path.dirname(__file__),
                             "..", "..", "examples", "meshes", "u1.stl")
    return mcm.CollisionModel(initor=mesh_path, cdprimit_type="box",
                              expand_radius=0.0)