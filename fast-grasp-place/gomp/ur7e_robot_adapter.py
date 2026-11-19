"""
UR7E Robot Adapter (DH76, with gripper) - Thin wrapper around WRS for GOMP.

⚠ 本文件已改成「本机扁平旧版 wrs」的接口：``ManipulatorInterface.fk(jnt_values)``
只更新关节状态并返回状态串，TCP 位姿要用 ``get_gl_tcp()`` 取；``jnt_ranges`` 是
``get_jnt_ranges()`` 方法；``is_collided`` 的关键字是 ``otherrobot_list`` /
``toggle_contact_points``。新版 wrs（``UR7EBase``）的写法在本机一律 AttributeError。
"""
import numpy as np
import robot_sim.robots.ur7e.ur7e_withouttable_dh76 as ur7e_mod


class RobotAdapter:
    def __init__(self, manipulator_cls=None, enable_cc=True, **kwargs):
        """
        Parameters
        ----------
        manipulator_cls : class
            WRS robot class (default: UR7EDH76, with dh76 gripper).
        enable_cc : bool
            Enable collision checker in the WRS model.
        **kwargs : dict
            Additional arguments passed to the robot constructor.
        """
        if manipulator_cls is None:
            manipulator_cls = ur7e_mod.UR7EDH76
        self.robot = manipulator_cls(
            pos=np.zeros(3),
            rotmat=np.eye(3),
            enable_cc=enable_cc, **kwargs
        )
        # UR7EDH76 内部 self.arm 才是真正的单臂 manipulator，fk/ik/jacobian/
        # jnt_ranges/n_dof 都在它上面，且 TCP 已被改成夹爪作用中心（带手爪）。
        self._kin = self.robot.arm
        self.n_dof = self._kin.ndof

    @property
    def kinematics(self):
        """底层单臂 manipulator（FK/IK/Jacobian/TCP 的承载对象）。
        FastKinematics 从这里取 jlc / tcp_loc_*。"""
        return self._kin

    @classmethod
    def from_wrs_robot(cls, robot):
        """
        Create a RobotAdapter from an existing WRS robot instance.

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
        adapter.n_dof = adapter._kin.ndof
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
        """Maximum joint velocities (rad/s)."""
        return np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

    @property
    def a_max(self) -> np.ndarray:
        """Maximum joint accelerations (rad/s²)."""
        return np.array([8, 8, 8, 8, 8, 8])

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
            End-effector position in world frame (gripper jaw center).
        rotmat : np.ndarray, shape (3, 3)
            End-effector rotation matrix in world frame.
        """
        # 本机旧版：fk 只更新关节状态，TCP 位姿走 get_gl_tcp()
        # （jlc.tcp_jnt_id=-1 + tcp_loc_pos/tcp_loc_rotmat = 夹爪作用中心）。
        self._kin.fk(jnt_values=q)
        return self._kin.get_gl_tcp()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Compute the 6×n geometric Jacobian at configuration q.

        Returns
        -------
        J : np.ndarray, shape (6, n_dof)
            Geometric Jacobian at the TCP (gripper jaw center) — the same
            point as forward_kinematics. Rows 0-2: linear velocity,
            rows 3-5: angular velocity.
        """
        # 旧版 jlc.jacobian(tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat) 基于当前 FK
        # 状态计算，三个参数传 None 即用 jlc 上配好的 TCP（夹爪作用中心）。
        self._kin.fk(jnt_values=q)
        return self._kin.jlc.jacobian(None, None, None)

    def inverse_kinematics(self, pos: np.ndarray, rotmat: np.ndarray,
                           seed_jnt_values=None):
        """
        Compute inverse kinematics.

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

        Returns
        -------
        positions : list of np.ndarray
            World-frame position of each link.
        """
        self.robot.goto_given_conf(jnt_values=q)
        # 旧版 JLChain：jnts 是 dict 列表，位置键是 'gl_posq'
        positions = []
        for i in range(len(self._kin.jlc.jnts)):
            positions.append(self._kin.jlc.jnts[i]['gl_posq'].copy())
        positions.append(np.asarray(self._kin.get_gl_tcp()[0], dtype=float).copy())
        return positions
