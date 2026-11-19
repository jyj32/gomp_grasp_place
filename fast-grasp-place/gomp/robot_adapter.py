"""
Robot Adapter - Thin wrapper around WRS manipulator for GOMP.

Provides a unified interface to access FK, IK, Jacobian, and joint limits
from any WRS manipulator model (default: UR7e).
"""

import numpy as np
from robot_sim.manipulators.ur7e.ur7e import UR7E
# from wrs.robot_sim.robots.ur7e.ur7e_withouttable_dh50 import UR7EDH50 as UR7E

class RobotAdapter:
    """
    Wraps a WRS ManipulatorInterface to expose the kinematic functions
    needed by the GOMP planner.

    Attributes:
        robot: The underlying WRS manipulator instance.
        n_dof: Number of degrees of freedom.
    """

    def __init__(self, manipulator_cls=None, enable_cc=True, **kwargs):
        """
        Parameters
        ----------
        manipulator_cls : class
            WRS manipulator class (default: UR7E).
        enable_cc : bool
            Enable collision checker in the WRS model.
        **kwargs : dict
            Additional arguments passed to the manipulator constructor.
        """
        if manipulator_cls is None:
            manipulator_cls = UR7E
        self.robot = manipulator_cls(
            pos = np.array([0.6125 + 0.15, -0.5875+0.10, 0.225]),
            rotmat=np.eye(3),
            enable_cc=enable_cc, **kwargs
        )
        self.n_dof = self.robot.ndof   # 旧版 ManipulatorInterface 的属性叫 ndof

    @classmethod
    def from_wrs_robot(cls, robot):
        """
        Create a RobotAdapter from an existing WRS robot instance.

        This avoids creating a new robot — useful when integrating with
        an existing WRS application.

        Parameters
        ----------
        robot : ManipulatorInterface
            An existing WRS manipulator.

        Returns
        -------
        RobotAdapter
            Adapter wrapping the given robot.
        """
        adapter = cls.__new__(cls)
        adapter.robot = robot
        adapter.n_dof = robot.ndof
        return adapter

    @property
    def kinematics(self):
        """底层单臂 manipulator（FK/IK/Jacobian/TCP 的承载对象）。
        FastKinematics 从这里取 jlc / _loc_tcp_pos / _loc_tcp_rotmat。"""
        return self.robot

    # ----- Joint limits -----

    @property
    def q_min(self) -> np.ndarray:
        """Joint position lower limits (rad)."""
        return self.robot.get_jnt_ranges()[:, 0]

    @property
    def q_max(self) -> np.ndarray:
        """Joint position upper limits (rad)."""
        return self.robot.get_jnt_ranges()[:, 1]

    @property
    def v_max(self) -> np.ndarray:
        """
        Maximum joint velocities (rad/s).

        UR5e specs: joints 1-3: ±π rad/s, joints 4-6: ±π rad/s.
        Override this for other robots.
        """
        return np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

    @property
    def a_max(self) -> np.ndarray:
        """
        Maximum joint accelerations (rad/s²).

        UR7e data-sheet values: shoulder joints ~25 rad/s², wrist ~40 rad/s².
        """
        return np.array([2*np.pi, 2*np.pi, 2*np.pi, np.pi*2, np.pi*2, np.pi*2])

    def a_max_set(self, a1: float, a2: float, a3: float, a4: float, a5: float, a6: float):
        return np.array([a1, a2, a3, a4, a5, a6])

    # ----- Kinematics -----
    def forward_kinematics(self, q: np.ndarray) -> tuple:
        """
        Compute forward kinematics.

        Parameters
        ----------
        q : np.ndarray, shape (n_dof,)
            Joint configuration.

        Returns
        -------
        pos : np.ndarray, shape (3,)
            End-effector position in world frame.
        rotmat : np.ndarray, shape (3, 3)
            End-effector rotation matrix in world frame.
        """
        # 旧版：fk 只更新关节状态，TCP 位姿走 get_gl_tcp()
        self.robot.fk(jnt_values=q)
        return self.robot.get_gl_tcp()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Compute the 6×n geometric Jacobian at configuration q.

        Parameters
        ----------
        q : np.ndarray, shape (n_dof,)
            Joint configuration.

        Returns
        -------
        J : np.ndarray, shape (6, n_dof)
            Geometric Jacobian. Rows 0-2: linear velocity, rows 3-5: angular velocity.
        """
        # 旧版 jlc.jacobian(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat) 基于当前 FK
        # 状态算，三个参数传 None 即用 jlc 上配好的 TCP（夹爪作用中心）
        self.robot.fk(jnt_values=q)
        return self.robot.jlc.jacobian(None, None, None)

    def inverse_kinematics(self, pos: np.ndarray, rotmat: np.ndarray,
                           seed_jnt_values=None):
        """
        Compute inverse kinematics.

        Parameters
        ----------
        pos : np.ndarray, shape (3,)
            Target end-effector position.
        rotmat : np.ndarray, shape (3, 3)
            Target end-effector rotation matrix.
        seed_jnt_values : np.ndarray or None
            Seed configuration for the IK solver.

        Returns
        -------
        q : np.ndarray or None
            Joint configuration achieving the target, or None if no solution.
        """
        return self.robot.ik(tgt_pos=pos, tgt_rotmat=rotmat,
                             seed_jnt_values=seed_jnt_values)

    def goto_given_conf(self, q: np.ndarray):
        """Set the robot to a specific joint configuration (updates internal state)."""
        # 旧版单臂 manipulator 没有 goto_given_conf，直接 fk 下去即可
        self.robot.fk(jnt_values=np.asarray(q, dtype=float).reshape(self.n_dof))

    def backup_state(self):
        """Save the current robot state for later restoration."""
        if not hasattr(self, "_state_bk"):
            self._state_bk = []
        self._state_bk.append(np.array(self.robot.get_jnt_values(), dtype=float))

    def restore_state(self):
        """Restore the previously saved robot state."""
        self.robot.fk(jnt_values=self._state_bk.pop())

    def is_collided(self, obstacle_list=None, other_robot_list=None,
                    toggle_contacts=False) -> bool:
        """
        Check if the robot is in self-collision or colliding with obstacles.

        Parameters
        ----------
        obstacle_list : list
            List of WRS CollisionModel obstacles.
        other_robot_list : list
            List of other WRS robot instances.
        toggle_contacts : bool
            If True, return (bool, contact_points) instead of just bool.

        Returns
        -------
        collided : bool or (bool, list)
            Whether the robot is in collision.
        """
        if obstacle_list is None:
            obstacle_list = []
        if other_robot_list is None:
            other_robot_list = []
        # 旧版 RobotInterface 的关键字是 otherrobot_list / toggle_contact_points
        return self.robot.is_collided(obstacle_list=obstacle_list,
                                      otherrobot_list=other_robot_list,
                                      toggle_contact_points=toggle_contacts)

    def get_link_positions(self, q: np.ndarray) -> list:
        """
        Get the world-frame positions of all links for a given configuration.
        Useful for multi-point obstacle checking.

        Parameters
        ----------
        q : np.ndarray, shape (n_dof,)
            Joint configuration.

        Returns
        -------
        positions : list of np.ndarray
            World-frame position of each link.
        """
        self.robot.fk(jnt_values=q)
        positions = []
        for i in range(len(self.robot.jlc.jnts)):
            positions.append(self.robot.jlc.jnts[i]['gl_posq'].copy())
        positions.append(np.asarray(self.robot.get_gl_tcp()[0], dtype=float).copy())
        return positions
