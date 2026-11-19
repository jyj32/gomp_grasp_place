"""
UR7E Robot Adapter (DH50, with gripper) - Thin wrapper around WRS for GOMP.
"""
import numpy as np
from robot_sim.robots.ur7e.ur7ewithoutmachine_1 import UR7E as UR7E


class RobotAdapter:
    def __init__(self, manipulator_cls=None, enable_cc=True, **kwargs):
        """
        Parameters
        ----------
        manipulator_cls : class
            WRS robot/manipulator class (default: UR7EDH50, with gripper).
        enable_cc : bool
            Enable collision checker in the WRS model.
        **kwargs : dict
            Additional arguments passed to the robot constructor
            (e.g. ik_solver='n').
        """
        if manipulator_cls is None:
            manipulator_cls = UR7E
        self.robot = manipulator_cls(
            pos=np.zeros(3),
            rotmat=np.eye(3),
            enable_cc=enable_cc, **kwargs
        )
        # UR7EDH50 内部 self.arm 才是真正的单臂 manipulator（与默认 robot_adapter
        # 用的是同一个 UR7E 类），fk/ik/jacobian/jnt_ranges/n_dof 都在它上面，
        # 且已被改成 TCP 落在夹爪作用中心（带手爪）。运动学统一走 self._kin。
        self._kin = self.robot.arm
        self.n_dof = self._kin.n_dof

    @property
    def kinematics(self):
        """底层单臂 manipulator（FK/IK/Jacobian/TCP 的承载对象）。
        FastKinematics 从这里取 jlc / _loc_tcp_pos / _loc_tcp_rotmat。"""
        return self._kin

    @classmethod
    def from_wrs_robot(cls, robot):
        """
        Create a RobotAdapter from an existing WRS robot instance.

        This avoids creating a new robot — useful when integrating with
        an existing WRS application.

        Parameters
        ----------
        robot : RobotInterface
            An existing WRS robot (may be a single-arm manipulator or a
            robot that wraps one via its ``arm`` attribute).

        Returns
        -------
        RobotAdapter
            Adapter wrapping the given robot.
        """
        adapter = cls.__new__(cls)
        adapter.robot = robot
        adapter._kin = getattr(robot, "arm", robot)
        adapter.n_dof = adapter._kin.n_dof
        return adapter

    # ----- Joint limits -----

    @property
    def q_min(self) -> np.ndarray:
        """Joint position lower limits (rad)."""
        return self._kin.get_jnt_ranges()[:, 0]

    @property
    def q_max(self) -> np.ndarray:
        """Joint position upper limits (rad)."""
        return self._kin.get_jnt_ranges()[:, 1]

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
        return np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, np.pi * 2, np.pi * 2, np.pi * 2])

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
            End-effector position in world frame (gripper tip for UR7EDH50).
        rotmat : np.ndarray, shape (3, 3)
            End-effector rotation matrix in world frame.
        """
        # 新版 WRS：ManipulatorInterface.fk(jnt_values) 只更新关节状态并返回状态串；
        # TCP 位姿用 get_gl_tcp() 取（jlc.tcp_jnt_id=-1 + tcp_loc_pos/rotmat = 手爪作用中心）。
        self._kin.fk(jnt_values=q)
        return self._kin.get_gl_tcp()

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
            Geometric Jacobian at the TCP (gripper acting center) — the same
            point as forward_kinematics. Rows 0-2: linear velocity,
            rows 3-5: angular velocity.
        """
        # 新版 WRS：jlc.jacobian(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat) 基于当前 FK 状态
        # 计算几何雅可比，三个参数传 None 即用 jlc 上已配好的 TCP（手爪作用中心），
        # 不再需要旧版那套「法兰雅可比 + 手动补 TCP 偏移」（旧 API 已废弃，勿再补，否则双算）。
        self._kin.fk(jnt_values=q)
        return self._kin.jlc.jacobian(None, None, None)

    def inverse_kinematics(self, pos: np.ndarray, rotmat: np.ndarray,
                           seed_jnt_values=None):
        """
        Compute inverse kinematics.

        Parameters
        ----------
        pos : np.ndarray, shape (3,)
            Target end-effector position (gripper tip for UR7EDH50).
        rotmat : np.ndarray, shape (3, 3)
            Target end-effector rotation matrix.
        seed_jnt_values : np.ndarray or None
            Seed configuration for the IK solver.

        Returns
        -------
        q : np.ndarray or None
            Joint configuration achieving the target, or None if no solution.
        """
        return self._kin.ik(tgt_pos=pos, tgt_rotmat=rotmat,
                            seed_jnt_values=seed_jnt_values)

    def goto_given_conf(self, q: np.ndarray):
        """Set the robot to a specific joint configuration (updates internal state)."""
        # 走外层 self.robot，让手爪也跟随到位（保持夹爪可视化/碰撞一致）。
        self.robot.goto_given_conf(jnt_values=q)

    def backup_state(self):
        """Save the current robot state for later restoration."""
        self.robot.backup_state()

    def restore_state(self):
        """Restore the previously saved robot state."""
        self.robot.restore_state()

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
        self.robot.goto_given_conf(jnt_values=q)
        positions = []
        for i in range(len(self._kin.jlc.lnks)):
            positions.append(self._kin.jlc.jnts[i]['gl_posq'].copy()
                             if i < len(self._kin.jlc.jnts)
                             else self._kin.get_gl_tcp()[0])
        return positions
