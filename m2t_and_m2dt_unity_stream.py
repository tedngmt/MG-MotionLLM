"""Standalone side-by-side demo: run motion clip(s) through both the Motion-to-Text
model and the Motion-to-Detailed-Text model, and stream the SMPL pose + both models'
captions live to Unity over a WebSocket, frame by frame, instead of rendering a local
video. There's one avatar in Unity, so both captions are sent on every frame message --
"caption_m2t" stays fixed for the whole clip, "caption_m2dt" is whichever body-part
snippet is active at that point (snippets are ~0.5s each).

This script is the WebSocket server: run it first, then connect Unity's WebSocket
client to ws://<this machine>:<--port> (default 8765).

Per-frame JSON message schema:
    {"type": "start", "name": str, "fps": float, "num_frames": int, "caption_m2t": str}
    {"type": "frame", "frame": int, "joints": [[x,y,z] x 22], "rotations": [[x,y,z,w] x 22],
     "caption_m2t": str, "caption_m2dt": str}
    {"type": "end", "name": str}
"joints" are global joint positions (Unity space, index 0 = pelvis) ordered to match
SMPLModifyBones._boneNameToJointIndex (Pelvis=0 .. R_Wrist=21, see
utils/unity_stream.UNITY_JOINT_NAMES) -- feed straight into
SMPLModifyBones.updateBoneAnglesFromJoints(joints) on the Unity side.
"rotations" are each joint's GLOBAL orientation (Unity-frame quaternion, x,y,z,w) recovered
from the motion's cont6d channels; same joint order. Positions still drive the avatar -- the
rotations are provided alongside (they additionally carry per-bone axial twist that positions
cannot express) for callers that want to pose bones by rotation instead.

Examples:
    python3 m2t_and_m2dt_unity_stream.py \\
        --m2t_model_name ./m2t-ft-from-GSPretrained-base \\
        --m2dt_model_name ./m2dt-ft-from-GSPretrained-base \\
        --name 000000
    # stream every .npy dropped into ./input/ (used when --name/--motion_path are both omitted)
    python3 m2t_and_m2dt_unity_stream.py \\
        --m2t_model_name ./m2t-ft-from-GSPretrained-base --m2dt_model_name ./m2dt-ft-from-GSPretrained-base
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
    load_gt_caption, load_gt_detail, parse_motion_script, sample_output_dir,
)

SNIPPET_SECONDS = 0.5  # FineMotion body-part descriptions are aligned to 0.5s chunks.


def _active_snippet(timed_captions, frame_idx):
    for start, end, text in timed_captions:
        if start <= frame_idx < end:
            return text
    return ''


async def run(args):
    cfg = DATASET_CONFIG[args.dataname]
    fps = args.fps or cfg['fps']
    frames_per_snippet = max(1, round(SNIPPET_SECONDS * fps))

    samples = resolve_samples(cfg, args.split, args.name, args.motion_path, args.sample_seed)
    print(f'[Found] {len(samples)} motion(s) to process')

    # Truncate to a length valid for both the M2T token unit (2**down_t frames) and the
    # M2DT snippet unit (frames_per_snippet), so both models see the exact same motion
    # and the single shared skeleton stream is unambiguous.
    unit_length = frames_per_snippet * (2 ** args.down_t) // np.gcd(frames_per_snippet, 2 ** args.down_t)
    mean = np.load(os.path.join(cfg['meta_dir'], 'mean.npy'))
    std = np.load(os.path.join(cfg['meta_dir'], 'std.npy'))

    print('[VQ-VAE] loading...')
    net = load_vqvae(args, args.dataname, args.device)

    print('[M2T] loading', args.m2t_model_name)
    m2t_tokenizer = T5Tokenizer.from_pretrained(args.m2t_model_name)
    m2t_model = T5ForConditionalGeneration.from_pretrained(args.m2t_model_name).to(args.device).eval()

    print('[M2DT] loading', args.m2dt_model_name)
    m2dt_tokenizer = T5Tokenizer.from_pretrained(args.m2dt_model_name)
    m2dt_model = T5ForConditionalGeneration.from_pretrained(args.m2dt_model_name).to(args.device).eval()

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

        m2t_output = generate_text(m2t_tokenizer, m2t_model, args.m2t_prompt + motion_string,
                                   args.m2t_max_new_tokens, args.device)
        caption = m2t_output.strip().strip('"')

        m2dt_output = generate_text(m2dt_tokenizer, m2dt_model, args.m2dt_prompt + motion_string,
                                    args.m2dt_max_new_tokens, args.device)
        snippets = parse_motion_script(m2dt_output)

        gt_caption = load_gt_caption(cfg, name) if is_dataset_sample else None
        gt_snippets = load_gt_detail(cfg, name) if is_dataset_sample else None
        num_expected = m_length // frames_per_snippet
        gt_snippets = gt_snippets[:num_expected] if gt_snippets else None

        print(f'[M2T generated]  {caption}')
        if gt_caption:
            print(f'[M2T GT]         {gt_caption}')
        print(f'[M2DT generated] {" <SEP> ".join(snippets)}')
        if gt_snippets:
            print(f'[M2DT GT]        {" <SEP> ".join(gt_snippets)}')

        timed_script = [
            (i * frames_per_snippet, min((i + 1) * frames_per_snippet, m_length), text)
            for i, text in enumerate(snippets)
        ]

        joints = motion_to_unity_joints(raw_motion, cfg['joints_num'])
        rotations = motion_to_unity_joint_rotations(raw_motion, cfg['joints_num'], cfg['kinematic_chain'])
        n = active_length(joints)  # drop the trailing static tail so the motion and captions
        joints = joints[:n]        # end together
        rotations = rotations[:n]

        await server.broadcast({'type': 'start', 'name': name, 'fps': fps,
                                'num_frames': len(joints), 'caption_m2t': caption})
        frame_dt = 1.0 / (fps * max(args.speed, 1e-3))
        for i in range(len(joints)):
            await server.broadcast({'type': 'frame', 'frame': i,
                                    'joints': joints[i].tolist(),
                                    'rotations': rotations[i].tolist(),
                                    'caption_m2t': caption,
                                    'caption_m2dt': _active_snippet(timed_script, i)})
            await asyncio.sleep(frame_dt)
        await server.broadcast({'type': 'end', 'name': name})

        sample_dir = sample_output_dir(args.out_dir, 'm2t_and_m2dt_unity_stream', name)
        script_path = os.path.join(sample_dir, f'{name}.txt')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('M2T generated:\n' + caption + '\n')
            if gt_caption:
                f.write('\nM2T GT:\n' + gt_caption + '\n')
            f.write('\nM2DT generated:\n' + '\n'.join(snippets) + '\n')
            if gt_snippets:
                f.write('\nM2DT GT:\n' + '\n'.join(gt_snippets) + '\n')
        print(f'[Saved] {script_path}')

    await server.close()

    del m2t_model, m2t_tokenizer, m2dt_model, m2dt_tokenizer
    if args.device == 'cuda':
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = option.get_args_parser()
    parser.add_argument('--m2t_model_name', type=str, default='./m2t-ft-from-GSPretrained-base',
                        help='Trained motion-to-text model directory')
    parser.add_argument('--m2dt_model_name', type=str, default='./m2dt-ft-from-GSPretrained-base',
                        help='Trained motion-to-detailed-text model directory')
    parser.add_argument('--m2t_prompt', type=str, default='Generate text: ')
    parser.add_argument('--m2dt_prompt', type=str, default='Generate the motion script: ')
    parser.add_argument('--m2t_max_new_tokens', type=int, default=40)
    parser.add_argument('--m2dt_max_new_tokens', type=int, default=1536)
    parser.add_argument('--name', type=str, default=None, help='Dataset sample id to visualize, e.g. 000000')
    parser.add_argument('--motion_path', type=str, default=None,
                        help='Path to a raw HumanML3D-format motion .npy, used instead of a dataset sample')
    parser.add_argument('--split', type=str, default='test',
                        help='Split to pick a random sample from when --name/--motion_path are not given '
                             'and ./input/ has no .npy files')
    parser.add_argument('--sample_seed', type=int, default=None, help='Seed for picking the random sample')
    parser.add_argument('--out_dir', type=str, default='./visualizations')
    parser.add_argument('--fps', type=float, default=None, help='Override playback fps (defaults to the dataset fps)')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier (0.5 = half speed, 2.0 = double); independent of --fps')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='WebSocket server bind address')
    parser.add_argument('--port', type=int, default=8765, help='WebSocket server port')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    asyncio.run(run(args))
