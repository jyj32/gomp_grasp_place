"""从抓取 pickle 读【瓶↔手爪相对位姿】，并转成 GOMP 的 Grasp。

pickle 条目格式与写侧（wrs/grasp/*_grasping_antipodal_planning.py、
yanjiuyuan/grasp_plan/bottle_grasping_antipodal_planning.py）完全一致：

    [jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat]

后两项在写侧恒为 [0,0,0] / I，仅前 3 项参与几何。

坐标约定（与 yanjiuyuan/sim_pick_and_place.tcp_pose_from_object_pose 同源）：
jaw_center_pos / jaw_center_rotmat 是【物体局部系】下手爪作用中心（= GOMP 的 TCP）位姿

    tcp_pos_world    = obj_pos + obj_rotmat @ jaw_center_pos
    tcp_rotmat_world = obj_rotmat @ jaw_center_rotmat

手爪局部系：z = 接近方向（腕→指尖），x = 爪指开合方向
（dh76 两指沿 ±x 平移，见 dh76.py 的 loc_motion_ax），
故绕抓取轴的自由度 = 绕 rotmat[:, 0] 旋转，与
grasp_sampler.create_topdown_grasp 的 axis 取法一致。
"""

import pickle
from pathlib import Path

import numpy as np

from gomp.grasp.grasp import Grasp


def load_grasp_pickle(path, key=None):
    """读抓取 pickle，返回 [(jaw_width, jaw_center_pos(3,), jaw_center_rotmat(3,3)), ...]。

    支持两种 pickle 格式：
    - 平铺 list/tuple（旧格式，如 bottle_dh76_4.pickle）
    - dict {物体名: [抓取条目]}（写侧 grasping/annotation/utils.py，如 U1_dh50.pickle）。
      dict 时需用 key 指定物体名；dict 只有一个键时自动使用。
    """
    path = Path(path)
    with path.open("rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict):
        if key is None:
            if len(raw) == 1:
                key = next(iter(raw))
            else:
                raise ValueError(f"pickle 为 dict 格式，需指定 key（物体名），可用键: {list(raw.keys())}")
        elif key not in raw:
            raise ValueError(f"key '{key}' 不在 pickle 中，可用键: {list(raw.keys())}")
        raw = raw[key]
    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        print(f"pickle_path: {path}")
        raise ValueError(f"抓取 pickle 为空或格式不对: {path}")
    entries = []
    for i, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            raise ValueError(f"第 {i} 条抓取项格式不对: {type(item).__name__}")
        jaw_width = float(np.asarray(item[0], dtype=float).reshape(-1)[0])
        ac_pos = np.asarray(item[1], dtype=float).reshape(3)
        ac_rotmat = np.asarray(item[2], dtype=float).reshape(3, 3)
        entries.append((jaw_width, ac_pos, ac_rotmat))
    return entries


def relative_pose_in_tcp(entry):
    """瓶↔手爪相对位姿，返回 (t_tb, R_tb)，满足

        T_world_tcp @ T_tcp_bottle = T_world_bottle

    entry = (jaw_width, jaw_center_pos, jaw_center_rotmat)（均在瓶局部系）。
    用于 AttachedObject.from_bottle_cdprim() 与 3D 可视化。
    """
    _, ac_pos, ac_rotmat = entry
    R_tb = ac_rotmat.T
    t_tb = -R_tb @ ac_pos
    return t_tb, R_tb


def grasp_from_object_pose(entry, obj_pos, obj_rotmat=None,
                           theta_range=(-np.pi / 4, np.pi / 4),
                           quality=1.0):
    """由【物体世界位姿】+ pickle 条目 推出手爪世界位姿（TCP = 手爪作用中心）。

    物体位姿固定时，手爪位姿唯一；绕抓取轴的 theta_range 作为
    自由度交给 GOMP 优化。
    """
    _, ac_pos, ac_rotmat = entry
    obj_pos = np.asarray(obj_pos, dtype=float).reshape(3)
    if obj_rotmat is None:
        obj_rotmat = np.eye(3)
    obj_rotmat = np.asarray(obj_rotmat, dtype=float).reshape(3, 3)
    tcp_rotmat = obj_rotmat @ ac_rotmat
    return Grasp(pos=obj_pos + obj_rotmat @ ac_pos,
                 rotmat=tcp_rotmat,
                 axis=tcp_rotmat[:, 0].copy(),
                 theta_min=float(theta_range[0]),
                 theta_max=float(theta_range[1]),
                 quality=float(quality))
