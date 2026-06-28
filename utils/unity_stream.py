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

from utils.motion_process import recover_from_ric


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
