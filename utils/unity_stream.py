"""Convert HumanML3D motion vectors to SMPL joint-rotation quaternions for live
streaming to the Unity SMPL avatar (C:\\Linux\\smpl_mecanim, SMPLModifyBones.updateBoneAngles),
plus a minimal WebSocket server to push them there frame by frame.
"""
import asyncio
import json

import numpy as np
import torch
import websockets

from utils.motion_process import recover_root_rot_pos
from utils.quaternion import cont6d_to_matrix


# Matches SMPLModifyBones._boneNameToJointIndex exactly (24 SMPL body joints).
# HumanML3D only covers indices 0-21 (Pelvis..R_Wrist); L_Hand/R_Hand (22, 23) are
# always sent as the identity quaternion since this motion representation has no
# finger/hand joints.
UNITY_JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle',
    'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand',
]
NUM_UNITY_JOINTS = len(UNITY_JOINT_NAMES)


def matrix_to_quaternion_np(mats):
    """(N, 3, 3) rotation matrices -> (N, 4) quaternions as (w, x, y, z), Shepperd's method."""
    out = np.zeros((mats.shape[0], 4), dtype=np.float64)
    for idx in range(mats.shape[0]):
        m = mats[idx]
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        out[idx] = [w, x, y, z]
    return out


def motion_to_unity_pose(raw_motion, joints_num):
    """Decode per-joint local rotations directly from the raw HumanML3D motion vector
    (no forward kinematics needed -- these 6D values are already each joint's
    parent-relative rotation, exactly what Mecanim's `Transform.localRotation` wants).

    Args:
        raw_motion: (T, D) raw (unnormalized) HumanML3D motion vector.
        joints_num: 22 for t2m -- only Pelvis..R_Wrist (indices 0-21) are present.

    Returns:
        quats_xyzw: (T, 24, 4) Unity-space local rotation quaternions, (x, y, z, w),
            ordered to match UNITY_JOINT_NAMES / SMPLModifyBones._boneNameToJointIndex.
        trans: (T, 3) Unity-space pelvis position.
    """
    data = torch.from_numpy(raw_motion).float()
    num_frames = data.shape[0]
    r_rot_quat, r_pos = recover_root_rot_pos(data)  # (T, 4) w,x,y,z ; (T, 3)

    start_indx = 4 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx].reshape(num_frames, joints_num - 1, 6)
    body_mats = cont6d_to_matrix(cont6d_params).numpy().reshape(-1, 3, 3)
    body_quats = matrix_to_quaternion_np(body_mats).reshape(num_frames, joints_num - 1, 4)

    quats_wxyz = np.zeros((num_frames, NUM_UNITY_JOINTS, 4), dtype=np.float64)
    quats_wxyz[..., 0] = 1.0  # identity (w=1) default, covers L_Hand/R_Hand
    quats_wxyz[:, 0] = r_rot_quat.numpy()
    quats_wxyz[:, 1:joints_num] = body_quats

    # HumanML3D/SMPL rotations are right-handed; Unity's Transform.localRotation is
    # left-handed. A wxyz->xyzw reorder alone does NOT convert handedness -- feeding
    # RH quaternions straight into a LH localRotation mirrors every joint and tips the
    # whole body backward. Mirror the X axis to convert RH -> Unity LH. This is exactly
    # the inverse of the conversion the rig itself documents in SMPLBlendshapes.cs
    # (Quat_to_3x3Mat: Unity LH -> SMPL RH negates x and w); mirroring X is its own
    # inverse, so the same negation maps SMPL RH -> Unity LH:
    #     (w, x, y, z)_RH  ->  (x, y, z, w)_Unity = (-x, y, z, -w)
    quats_xyzw = quats_wxyz[..., [1, 2, 3, 0]].copy()
    quats_xyzw[..., 0] *= -1.0  # x
    quats_xyzw[..., 3] *= -1.0  # w

    # The pelvis translation lives in the same RH frame, so mirror its X component too;
    # otherwise left/right motion (e.g. "lean right", "right leg forward") comes out swapped.
    trans = r_pos.numpy().copy()
    trans[..., 0] *= -1.0

    return quats_xyzw, trans


class MotionStreamServer:
    """Minimal one-way (Python -> Unity) JSON-over-WebSocket broadcaster.

    Python runs the server; the Unity client connects in and receives one JSON object
    per text message. See eval_*_stream.py for the message schema.
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
