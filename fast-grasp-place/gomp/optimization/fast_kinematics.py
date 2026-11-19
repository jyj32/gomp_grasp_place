"""
Fast FK/Jacobian engine for GOMP — avoids WRS per-joint Python loop overhead.

The WRS FK implementation is accurate but slow for batch computation because:
1. Per-joint Python for-loop calling get_motion_homomat
2. rotmat_from_axangle called ~13× per FK (6 joints × ~2 calls each)
3. Each rotmat_from_axangle calls allclose, norm, cross, eye, etc.

This module pre-extracts the kinematic parameters (DH-like: loc_pos, loc_rotmat,
loc_motion_ax, joint type) from a WRS JLChain at init time, then computes FK
and Jacobian using direct numpy operations (Rodrigues formula inlined).

Benchmark target: >10× speedup over WRS jlchain.fk + jlchain.jacobian.
"""

import numpy as np

import basis.robot_math as rm


class FastKinematics:
    """
    Pre-compiled fast FK and Jacobian for a WRS manipulator.

    Extracts kinematic parameters once from a WRS robot, then provides
    FK and Jacobian computation without WRS object overhead.

    Parameters
    ----------
    robot_adapter : RobotAdapter
        GOMP robot adapter (wraps a WRS ManipulatorInterface).
    """

    def __init__(self, robot_adapter):
        robot = robot_adapter.kinematics
        jlc = robot.jlc
        n = jlc.ndof

        self.n_dof = n
        # 本机这套是旧版扁平 wrs 的 JLChain：jnts 是 dict 列表，
        # jnts[0] = 基座（type='end'），tgtjnts = 1..ndof 是动关节，
        # jnts[ndof+1] = 法兰（type='end'，loc_pos/loc_rotmat 即法兰偏置）。
        # 新版 wrs 的 flange_jnt_id / loc_flange_homomat / jnt.type 在这里都不存在。
        tgt_ids = list(jlc.tgtjnts)
        self.n_tgt = len(tgt_ids)
        self._jnts = [jlc.jnts[i] for i in tgt_ids]
        # 向量化递推只实现了转动关节（UR7E 即如此）；出现移动关节直接报错，不静默兜底。
        _mismatch = [f"jnts[{jid}]={self._jnts[k]['type']}"
                     for k, jid in enumerate(tgt_ids) if self._jnts[k]['type'] != 'revolute']
        if _mismatch:
            raise NotImplementedError(
                "FastKinematics 只实现了全转动关节链，检测到非转动关节："
                + ", ".join(_mismatch))

        # Pre-extract per-joint kinematic parameters as contiguous numpy arrays
        # Each moving joint k has: loc_pos (3,), loc_rotmat (3,3), loc_motion_ax (3,)
        self._loc_pos = np.zeros((n, 3))
        self._loc_rotmat = np.zeros((n, 3, 3))
        self._loc_motion_ax = np.zeros((n, 3))

        for k, jnt in enumerate(self._jnts):
            self._loc_pos[k] = jnt['loc_pos']
            self._loc_rotmat[k] = jnt['loc_rotmat']
            self._loc_motion_ax[k] = jnt['loc_motionax']

        # 链根帧：旧版 JLChain 直接挂在 jlc.pos / jlc.rotmat 上（没有 anchor）
        self._base_homomat = rm.homomat_from_posrot(jlc.pos, jlc.rotmat)

        # 法兰帧（局部）：旧版把法兰当成链尾那个 type='end' 关节的 loc 位姿
        _flange = jlc.jnts[n + 1]
        self._loc_flange_homomat = rm.homomat_from_posrot(_flange['loc_pos'],
                                                         _flange['loc_rotmat'])

        # TCP 偏移：旧版 jlc.tcp_jnt_id(=-1) + tcp_loc_pos/tcp_loc_rotmat
        # （dh76 在这里配的就是手爪作用中心 jaw center）
        self._loc_tcp_pos = np.asarray(jlc.tcp_loc_pos, dtype=float)
        self._loc_tcp_rotmat = np.asarray(jlc.tcp_loc_rotmat, dtype=float)

        # Pre-compute skew-symmetric matrices for each joint's local motion axis
        # K = [[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]]
        self._K = np.zeros((n, 3, 3))
        self._K2 = np.zeros((n, 3, 3))  # K @ K
        for i in range(n):
            k = self._loc_motion_ax[i]
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            self._K[i] = K
            self._K2[i] = K @ K

    def _fk_homomat(self, jnt_values):
        """FK 到法兰：返回累积齐次阵（= 末关节运动后的位姿）。"""
        homomat = self._base_homomat.copy()
        for i in range(self.n_tgt):
            theta = jnt_values[i]
            # Rodrigues: R = I + sin(θ)K + (1-cosθ)K²
            s = np.sin(theta)
            c = np.cos(theta)
            R_motion = np.eye(3) + s * self._K[i] + (1.0 - c) * self._K2[i]

            # Local homomat with motion applied
            loc_R = self._loc_rotmat[i]
            loc_p = self._loc_pos[i]

            # Combined: pos contribution from parent frame
            jnt_pos = homomat[:3, 3] + homomat[:3, :3] @ loc_p

            # Rotation: parent_R @ loc_R @ R_motion
            jnt_R = homomat[:3, :3] @ loc_R @ R_motion

            homomat[:3, :3] = jnt_R
            homomat[:3, 3] = jnt_pos
        return homomat

    def _tcp_from_homomat(self, homomat):
        """在法兰位姿上叠加法兰偏移 + TCP 偏移，返回 (tcp_pos, tcp_rotmat)。"""
        flange_pos = homomat[:3, 3] + homomat[:3, :3] @ self._loc_flange_homomat[:3, 3]
        flange_rotmat = homomat[:3, :3] @ self._loc_flange_homomat[:3, :3]
        tcp_pos = flange_pos + flange_rotmat @ self._loc_tcp_pos
        tcp_rotmat = flange_rotmat @ self._loc_tcp_rotmat
        return tcp_pos, tcp_rotmat

    def fk(self, jnt_values):
        """
        Compute FK: returns (tcp_pos, tcp_rotmat).

        Uses Rodrigues formula inline:
            R(k, θ) = I + sin(θ)·K + (1-cos(θ))·K²
        where K is the skew-symmetric matrix of axis k.

        Parameters
        ----------
        jnt_values : np.ndarray, shape (n_dof,)

        Returns
        -------
        tcp_pos : np.ndarray, shape (3,)
        tcp_rotmat : np.ndarray, shape (3, 3)
        """
        return self._tcp_from_homomat(self._fk_homomat(jnt_values))

    def fk_and_jacobian(self, jnt_values):
        """
        Compute FK and 6×n Jacobian in a single pass.

        Returns
        -------
        tcp_pos : np.ndarray, shape (3,)
        tcp_rotmat : np.ndarray, shape (3, 3)
        J : np.ndarray, shape (6, n_dof)
        """
        homomat = self._base_homomat.copy()

        # Store per-joint world positions and motion axes
        jnt_world_pos = np.zeros((self.n_tgt, 3))
        jnt_world_ax = np.zeros((self.n_tgt, 3))

        for i in range(self.n_tgt):
            theta = jnt_values[i]
            s = np.sin(theta)
            c = np.cos(theta)
            R_motion = np.eye(3) + s * self._K[i] + (1.0 - c) * self._K2[i]

            loc_R = self._loc_rotmat[i]
            loc_p = self._loc_pos[i]

            jnt_pos = homomat[:3, 3] + homomat[:3, :3] @ loc_p
            jnt_R = homomat[:3, :3] @ loc_R @ R_motion

            # Store for Jacobian computation (motion axis in world frame)
            jnt_world_pos[i] = jnt_pos
            jnt_world_ax[i] = homomat[:3, :3] @ loc_R @ self._loc_motion_ax[i]

            homomat[:3, :3] = jnt_R
            homomat[:3, 3] = jnt_pos

        tcp_pos, tcp_rotmat = self._tcp_from_homomat(homomat)

        # Build Jacobian (all revolute assumed for UR-like robots)
        J = np.zeros((6, self.n_dof))
        for i in range(self.n_tgt):
            j2t = tcp_pos - jnt_world_pos[i]
            J[:3, i] = np.cross(jnt_world_ax[i], j2t)
            J[3:, i] = jnt_world_ax[i]

        return tcp_pos, tcp_rotmat, J

    def batch_fk_and_jacobian(self, waypoints):
        """Vectorized FK + Jacobian over all waypoints (batched matmul)."""
        m = waypoints.shape[0]
        homomat = np.broadcast_to(self._base_homomat, (m, 4, 4)).copy()
        jnt_world_pos = np.zeros((m, self.n_tgt, 3))
        jnt_world_ax = np.zeros((m, self.n_tgt, 3))
        for i in range(self.n_tgt):
            theta = waypoints[:, i][:, None, None]
            s = np.sin(theta)
            c = np.cos(theta)
            R_motion = np.eye(3)[None] + s * self._K[i] + (1.0 - c) * self._K2[i]
            loc_R = self._loc_rotmat[i]
            loc_p = self._loc_pos[i]
            ax = self._loc_motion_ax[i]
            new_pos = homomat[:, :3, 3] + homomat[:, :3, :3] @ loc_p
            new_R = homomat[:, :3, :3] @ loc_R @ R_motion
            jnt_world_pos[:, i, :] = new_pos
            jnt_world_ax[:, i, :] = homomat[:, :3, :3] @ loc_R @ ax
            homomat[:, :3, :3] = new_R
            homomat[:, :3, 3] = new_pos
        flange_pos = homomat[:, :3, 3] + homomat[:, :3, :3] @ self._loc_flange_homomat[:3, 3]
        flange_R = homomat[:, :3, :3] @ self._loc_flange_homomat[:3, :3]
        tcp_pos = flange_pos + flange_R @ self._loc_tcp_pos
        J = np.zeros((m, 6, self.n_dof))
        cross = np.cross(jnt_world_ax, tcp_pos[:, None, :] - jnt_world_pos).transpose(0, 2, 1)
        J[:, :3, :] = cross
        J[:, 3:, :] = jnt_world_ax.transpose(0, 2, 1)
        return tcp_pos, flange_R @ self._loc_tcp_rotmat, J

    def batch_fk(self, waypoints):
        """
        Compute FK (position only) for all waypoints.

        Parameters
        ----------
        waypoints : np.ndarray, shape (m, n_dof)

        Returns
        -------
        positions : np.ndarray, shape (m, 3)
        rotmats : np.ndarray, shape (m, 3, 3)
        """
        m = waypoints.shape[0]
        positions = np.zeros((m, 3))
        rotmats = np.zeros((m, 3, 3))

        for idx in range(m):
            pos, rotmat = self.fk(waypoints[idx])
            positions[idx] = pos
            rotmats[idx] = rotmat

        return positions, rotmats
