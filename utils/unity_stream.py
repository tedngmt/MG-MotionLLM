"""Convert HumanML3D motion vectors to global SMPL joint POSITIONS for live streaming
to the Unity SMPL avatar (C:\\Linux\\smpl_mecanim, SMPLModifyBones.updateBoneAnglesFromJoints),
plus a minimal WebSocket server to push them there frame by frame.

Why positions, not rotations: HumanML3D/T2M's per-joint cont6d rotations use a different
forward-kinematics convention than Unity's Transform.localRotation -- T2M rotates a bone's
offset by the *child's* accumulated global rotation, Unity by the *parent's* -- so feeding
them to bones is mathematically incompatible (verified: correct only when all rotations are
near-identity, i.e. the rest pose, and diverging badly otherwise). Global joint positions
are convention-free; recover_from_ric reproduces the reference visualization exactly, and
Unity rebuilds bone rotations from the positions by aiming each bone at its child joint.
"""
import asyncio
import json

import numpy as np
import torch
import websockets

from utils.motion_process import recover_from_ric, recover_root_rot_pos
from utils.quaternion import quaternion_to_cont6d, cont6d_to_matrix


# Matches SMPLModifyBones._boneNameToJointIndex (Pelvis=0 .. R_Wrist=21). HumanML3D's
# t2m representation covers exactly these 22 joints; the rig's L_Hand/R_Hand (22, 23) have
# no data here and Unity simply leaves them at their bind pose.
UNITY_JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle',
    'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
]
NUM_UNITY_JOINTS = len(UNITY_JOINT_NAMES)


def motion_to_unity_joints(raw_motion, joints_num):
    """Recover global joint positions from the raw HumanML3D motion vector and convert
    them to Unity's left-handed coordinate frame.

    Args:
        raw_motion: (T, D) raw (unnormalized) HumanML3D motion vector.
        joints_num: 22 for t2m -- Pelvis..R_Wrist (indices 0-21).

    Returns:
        joints_xyz: (T, joints_num, 3) joint positions in Unity space, ordered to match
            UNITY_JOINT_NAMES / SMPLModifyBones._boneNameToJointIndex. Index 0 is the pelvis.
    """
    data = torch.from_numpy(raw_motion).float()
    joints = recover_from_ric(data, joints_num).numpy()  # (T, joints_num, 3), HumanML3D Y-up, RH

    # HumanML3D is right-handed Y-up, Unity is left-handed Y-up: mirror X to convert.
    # Y (up) and Z map straight across, so this preserves the motion's own heading.
    joints = joints.copy()
    joints[..., 0] *= -1.0
    return joints


def _matrix_to_quaternion(R):
    """(T, J, 3, 3) rotation matrices -> (T, J, 4) quaternions in (w, x, y, z) order,
    using a numerically stable case-split (Shepperd's method). Verified against
    scipy.spatial.transform.Rotation to < 0.06 deg.
    """
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    t = m00 + m11 + m22
    q = torch.zeros(R.shape[:-2] + (4,))
    c0 = t > 0
    c1 = (~c0) & (m00 >= m11) & (m00 >= m22)
    c2 = (~c0) & (~c1) & (m11 >= m22)
    c3 = ~(c0 | c1 | c2)
    s = torch.sqrt(1 + t[c0]) * 2
    q[c0] = torch.stack([0.25 * s, (m21 - m12)[c0] / s, (m02 - m20)[c0] / s, (m10 - m01)[c0] / s], -1)
    s = torch.sqrt(1 + m00[c1] - m11[c1] - m22[c1]) * 2
    q[c1] = torch.stack([(m21 - m12)[c1] / s, 0.25 * s, (m01 + m10)[c1] / s, (m02 + m20)[c1] / s], -1)
    s = torch.sqrt(1 + m11[c2] - m00[c2] - m22[c2]) * 2
    q[c2] = torch.stack([(m02 - m20)[c2] / s, (m01 + m10)[c2] / s, 0.25 * s, (m12 + m21)[c2] / s], -1)
    s = torch.sqrt(1 + m22[c3] - m00[c3] - m11[c3]) * 2
    q[c3] = torch.stack([(m10 - m01)[c3] / s, (m02 + m20)[c3] / s, (m12 + m21)[c3] / s, 0.25 * s], -1)
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def motion_to_unity_joint_rotations(raw_motion, joints_num, kinematic_chain):
    """Recover each joint's GLOBAL (world-space) orientation from the raw HumanML3D motion
    vector and convert it to Unity's left-handed frame, as a quaternion per joint.

    Unlike the streamed positions (which are convention-free but cannot express a bone's
    axial twist), these come from the motion's own cont6d rotation channels, so they carry
    the full orientation -- including twist -- of every joint. They are *global* (root-relative
    world) orientations, which, like positions, are convention-independent (the convention
    mismatch documented at the top of this file only affects *local* per-bone rotations).

    The recovery mirrors T2M's ``forward_kinematics_cont6d``: every kinematic chain is
    accumulated starting from the root's global rotation R0 (not the chain's first joint),
    which is what keeps it consistent with ``recover_from_ric`` positions (verified: bone
    directions agree to < 0.04 deg).

    Args:
        raw_motion: (T, D) raw (unnormalized) HumanML3D motion vector.
        joints_num: 22 for t2m.
        kinematic_chain: e.g. paramUtil.t2m_kinematic_chain (cfg['kinematic_chain']).

    Returns:
        quats: (T, joints_num, 4) global joint orientations in Unity space, ordered to match
            UNITY_JOINT_NAMES, each quaternion in Unity's (x, y, z, w) component order so it
            can be fed straight into a UnityEngine.Quaternion.
    """
    data = torch.from_numpy(raw_motion).float()
    r_rot_quat, _ = recover_root_rot_pos(data)               # (T, 4) root global rotation
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)          # (T, 6)
    start = 1 + 2 + 1 + (joints_num - 1) * 3
    end = start + (joints_num - 1) * 6
    cont6d = torch.cat([r_rot_cont6d, data[..., start:end]], dim=-1).view(-1, joints_num, 6)
    local = cont6d_to_matrix(cont6d)                         # (T, J, 3, 3) per-joint local rot
    glob = local.clone()
    root_R = local[:, 0]                                     # R0, root global rotation
    for chain in kinematic_chain:
        R = root_R
        for i in range(1, len(chain)):
            R = torch.matmul(R, local[:, chain[i]])
            glob[:, chain[i]] = R
    glob[:, 0] = root_R

    quat = _matrix_to_quaternion(glob)                       # (T, J, 4) (w, x, y, z), motion frame
    # HumanML3D is right-handed Y-up, Unity left-handed Y-up: mirror X. A rotation R maps to
    # S R S with S = diag(-1, 1, 1), i.e. (w, x, y, z) -> (w, x, -y, -z), matching the X-flip
    # applied to positions in motion_to_unity_joints so rotations and positions stay aligned.
    quat[..., 2] *= -1.0
    quat[..., 3] *= -1.0
    # Reorder (w, x, y, z) -> Unity's (x, y, z, w).
    quat = quat[..., [1, 2, 3, 0]]
    return quat.numpy()


def active_length(joints, move_thresh=0.05, pad=4):
    """Number of leading frames up to (and a few past) the end of meaningful motion.

    HumanML3D clips frequently end with a long static tail (the person finishes the action
    and stands still). Streaming those frames makes the avatar look stopped while the timed
    captions keep cycling, so the motion appears to "end first." Trimming the trailing
    near-static frames keeps the motion and caption ending together.

    Args:
        joints: (T, J, 3) joint positions (e.g. from motion_to_unity_joints).
        move_thresh: per-frame summed joint displacement below which a frame counts as
            static (active frames here are ~0.1-1.0, static tail ~0.01).
        pad: extra frames kept after the last moving frame so motion doesn't cut abruptly.

    Returns:
        Length in [1, T]; slice with joints[:active_length(joints)].
    """
    n = len(joints)
    if n <= 1:
        return n
    vel = np.linalg.norm(np.diff(joints, axis=0), axis=-1).sum(axis=-1)  # (T-1,)
    moving = np.flatnonzero(vel > move_thresh)
    if moving.size == 0:
        return n
    return int(min(n, moving[-1] + 2 + pad))


class MotionStreamServer:
    """Minimal one-way (Python -> Unity) JSON-over-WebSocket broadcaster.

    Python runs the server; the Unity client connects in and receives one JSON object
    per text message. See *_unity_stream.py for the message schema.
    """

    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        self._clients = set()
        self._server = None

    async def _handle(self, websocket, *_args):
        self._clients.add(websocket)
        print(f'[Unity] connected ({websocket.remote_address})')
        try:
            async for _ in websocket:
                pass  # one-way for now; any incoming messages are ignored
        finally:
            self._clients.discard(websocket)
            print(f'[Unity] disconnected ({websocket.remote_address})')

    async def start(self):
        self._server = await websockets.serve(self._handle, self.host, self.port)
        print(f'[Server] listening on ws://{self.host}:{self.port} -- waiting for Unity to connect...')

    async def wait_for_client(self):
        while not self._clients:
            await asyncio.sleep(0.2)

    async def broadcast(self, message):
        if not self._clients:
            return
        payload = json.dumps(message)
        await asyncio.gather(*(ws.send(payload) for ws in list(self._clients)))

    async def close(self):
        if self._clients:
            await asyncio.gather(*(ws.close() for ws in list(self._clients)), return_exceptions=True)
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
