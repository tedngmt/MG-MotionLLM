"""Standalone Motion-to-Detailed-Text demo: generate a fine-grained motion script for
motion clip(s) and stream the SMPL pose + script live to Unity over a WebSocket, frame by
frame, instead of rendering a local video. The caption sent with each frame is whichever
body-part snippet is active at that point in the motion (snippets are ~0.5s each).

This script is the WebSocket server: run it first, then connect Unity's WebSocket
client to ws://<this machine>:<--port> (default 8765).

Per-frame JSON message schema:
    {"type": "start", "name": str, "fps": float, "num_frames": int}
    {"type": "frame", "frame": int, "joints": [[x,y,z] x 22],
                      "rotations": [[x,y,z,w] x 22], "caption": str, "highlight": [int, ...]}
    {"type": "end", "name": str}
"highlight" is the list of joint indices the active caption refers to, in the same joint
order as "joints"/"rotations" (UNITY_JOINT_NAMES, Pelvis=0 .. R_Wrist=21). Tint those bones
on the Unity side (e.g. yellow) and leave the rest at their default colour so the body parts
the caption is describing stand out. The indices are derived from the caption text via
utils.body_parts.caption_to_highlight_joints.
"joints" are global joint positions (Unity space, index 0 = pelvis) ordered to match
SMPLModifyBones._boneNameToJointIndex (Pelvis=0 .. R_Wrist=21, see
utils/unity_stream.UNITY_JOINT_NAMES) -- feed straight into
SMPLModifyBones.updateBoneAnglesFromJoints(joints) on the Unity side.
"rotations" are each joint's GLOBAL orientation (Unity-frame quaternion, x,y,z,w) recovered
from the motion's cont6d channels; same joint order. Positions still drive the avatar -- the
rotations are provided alongside (they additionally carry per-bone axial twist that positions
cannot express) for callers that want to pose bones by rotation instead.

Examples:
    python3 m2dt_unity_stream.py --model_name ./m2dt-ft-from-GSPretrained-base --name 000000
    python3 m2dt_unity_stream.py --model_name ./m2dt-ft-from-GSPretrained-base --split test --sample_seed 0
    # stream every .npy dropped into ./input/ (used when --name/--motion_path are both omitted)
    python3 m2dt_unity_stream.py --model_name ./m2dt-ft-from-GSPretrained-base
"""
import asyncio
import os

import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from options import option
from utils.unity_stream import (
    MotionStreamServer, motion_to_unity_joints, motion_to_unity_joint_rotations, active_length,
)
from utils.inference_utils import (
    DATASET_CONFIG, resolve_samples, load_vqvae, motion_to_token_string, generate_text,
    load_gt_detail, parse_motion_script, sample_output_dir,
)
from utils.body_parts import caption_to_highlight_joints

SNIPPET_SECONDS = 0.5  # FineMotion body-part descriptions are aligned to 0.5s chunks.


def _active(timed, frame_idx, default):
    for start, end, value in timed:
        if start <= frame_idx < end:
            return value
    return default


async def run(args):
    cfg = DATASET_CONFIG[args.dataname]
    fps = args.fps or cfg['fps']
    frames_per_snippet = max(1, round(SNIPPET_SECONDS * fps))

    samples = resolve_samples(cfg, args.split, args.name, args.motion_path, args.sample_seed)
    print(f'[Found] {len(samples)} motion(s) to process')

    # Truncate so tokens (4 frames each) and snippets (frames_per_snippet each) stay aligned,
    # mirroring Motion2MotionScriptDataset's `m_length // 20` truncation for t2m.
    unit_length = frames_per_snippet * (2 ** args.down_t) // np.gcd(frames_per_snippet, 2 ** args.down_t)
    mean = np.load(os.path.join(cfg['meta_dir'], 'mean.npy'))
    std = np.load(os.path.join(cfg['meta_dir'], 'std.npy'))

    print('[VQ-VAE] loading...')
    net = load_vqvae(args, args.dataname, args.device)

    print('[LLM] loading', args.model_name)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(args.device).eval()

    server = MotionStreamServer(args.host, args.port)
    await server.start()
    await server.wait_for_client()
    print('[Unity] client connected -- starting playback')

    for name, raw_motion, is_dataset_sample in samples:
        print(f'--- [Sample] {name}  ({raw_motion.shape[0]} frames) ---')

        m_length = int((raw_motion.shape[0] // unit_length) * unit_length)
        raw_motion = raw_motion[:m_length]
        norm_motion = (raw_motion - mean) / std

        motion_string = motion_to_token_string(net, norm_motion, args.device)
        prompt = args.prompt + motion_string

        output_text = generate_text(tokenizer, model, prompt, args.max_new_tokens, args.device)
        snippets = parse_motion_script(output_text)

        gt_snippets = load_gt_detail(cfg, name) if is_dataset_sample else None
        num_expected = m_length // frames_per_snippet
        gt_snippets = gt_snippets[:num_expected] if gt_snippets else None

        print(f'[Generated script] {" <SEP> ".join(snippets)}')
        if gt_snippets:
            print(f'[GT script]        {" <SEP> ".join(gt_snippets)}')

        timed_captions = [
            (i * frames_per_snippet, min((i + 1) * frames_per_snippet, m_length), text)
            for i, text in enumerate(snippets)
        ]
        # Joint indices each snippet refers to, for highlighting the matching bones in Unity.
        timed_highlights = [
            (start, end, sorted(caption_to_highlight_joints(text, cfg['joints_num'])))
            for start, end, text in timed_captions
        ]

        joints = motion_to_unity_joints(raw_motion, cfg['joints_num'])
        rotations = motion_to_unity_joint_rotations(raw_motion, cfg['joints_num'], cfg['kinematic_chain'])
        n = active_length(joints)  # drop the trailing static tail so the motion and captions
        joints = joints[:n]        # end together
        rotations = rotations[:n]

        await server.broadcast({'type': 'start', 'name': name, 'fps': fps, 'num_frames': len(joints)})
        frame_dt = 1.0 / (fps * max(args.speed, 1e-3))
        for i in range(len(joints)):
            await server.broadcast({'type': 'frame', 'frame': i,
                                    'joints': joints[i].tolist(),
                                    'rotations': rotations[i].tolist(),
                                    'caption': _active(timed_captions, i, ''),
                                    'highlight': _active(timed_highlights, i, [])})
            await asyncio.sleep(frame_dt)
        await server.broadcast({'type': 'end', 'name': name})

        sample_dir = sample_output_dir(args.out_dir, 'm2dt_unity_stream', name)
        script_path = os.path.join(sample_dir, f'{name}.txt')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('Generated:\n' + '\n'.join(snippets) + '\n')
            if gt_snippets:
                f.write('\nGround truth:\n' + '\n'.join(gt_snippets) + '\n')
        print(f'[Saved] {script_path}')

    await server.close()


if __name__ == '__main__':
    parser = option.get_args_parser()
    parser.add_argument('--model_name', type=str, default='./m2dt-ft-from-GSPretrained-base',
                        help='Trained motion-to-detailed-text model directory')
    parser.add_argument('--prompt', type=str, default='Generate the motion script: ',
                        help='Motion-to-Detailed-Text instruction prefix')
    parser.add_argument('--name', type=str, default=None, help='Dataset sample id to visualize, e.g. 000000')
    parser.add_argument('--motion_path', type=str, default=None,
                        help='Path to a raw HumanML3D-format motion .npy, used instead of a dataset sample')
    parser.add_argument('--split', type=str, default='test',
                        help='Split to pick a random sample from when --name/--motion_path are not given '
                             'and ./input/ has no .npy files')
    parser.add_argument('--sample_seed', type=int, default=None, help='Seed for picking the random sample')
    parser.add_argument('--max_new_tokens', type=int, default=1536)
    parser.add_argument('--out_dir', type=str, default='./visualizations')
    parser.add_argument('--fps', type=float, default=None, help='Override playback fps (defaults to the dataset fps)')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier (0.5 = half speed, 2.0 = double); independent of --fps')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='WebSocket server bind address')
    parser.add_argument('--port', type=int, default=8765, help='WebSocket server port')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    asyncio.run(run(args))
