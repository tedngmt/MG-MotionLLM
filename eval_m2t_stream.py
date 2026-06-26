"""Standalone Motion-to-Text demo: caption motion clip(s) and stream the SMPL pose +
caption live to Unity over a WebSocket, frame by frame, instead of rendering a local video.

This script is the WebSocket server: run it first, then connect Unity's WebSocket
client to ws://<this machine>:<--port> (default 8765). Once Unity connects, each
resolved motion is captioned and streamed in real time at the motion's fps.

Per-frame JSON message schema (one Unity bone update per "frame" message):
    {"type": "start", "name": str, "fps": float, "num_frames": int, "caption": str}
    {"type": "frame", "frame": int, "pose": [[x,y,z,w] x 24], "trans": [x,y,z], "caption": str}
    {"type": "end", "name": str}
"pose" is ordered to match SMPLModifyBones._boneNameToJointIndex (Pelvis=0 .. R_Hand=23,
see utils/unity_stream.UNITY_JOINT_NAMES) -- feed it straight into
SMPLModifyBones.updateBoneAngles(pose, trans) on the Unity side.

Examples:
    python3 eval_m2t_stream.py --model_name ./m2t-ft-from-GSPretrained-base --name 000000
    python3 eval_m2t_stream.py --model_name ./m2t-ft-from-GSPretrained-base --split test --sample_seed 0
    # stream every .npy dropped into ./input/ (used when --name/--motion_path are both omitted)
    python3 eval_m2t_stream.py --model_name ./m2t-ft-from-GSPretrained-base
"""
import asyncio
import os

import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from options import option
from utils.unity_stream import MotionStreamServer, motion_to_unity_pose
from utils.inference_utils import (
    DATASET_CONFIG, resolve_samples, load_vqvae, motion_to_token_string, generate_text,
    load_gt_caption, sample_output_dir,
)


async def run(args):
    cfg = DATASET_CONFIG[args.dataname]

    samples = resolve_samples(cfg, args.split, args.name, args.motion_path, args.sample_seed)
    print(f'[Found] {len(samples)} motion(s) to process')

    unit_length = 2 ** args.down_t
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

        m_length = (raw_motion.shape[0] // unit_length) * unit_length
        raw_motion = raw_motion[:m_length]
        norm_motion = (raw_motion - mean) / std

        motion_string = motion_to_token_string(net, norm_motion, args.device)
        prompt = args.prompt + motion_string

        output_text = generate_text(tokenizer, model, prompt, args.max_new_tokens, args.device)
        caption = output_text.strip().strip('"')

        gt_caption = load_gt_caption(cfg, name) if is_dataset_sample else None
        print(f'[GT caption]        {gt_caption}')
        print(f'[Generated caption] {caption}')

        fps = args.fps or cfg['fps']
        quats, trans = motion_to_unity_pose(raw_motion, cfg['joints_num'])

        await server.broadcast({'type': 'start', 'name': name, 'fps': fps,
                                'num_frames': len(quats), 'caption': caption})
        frame_dt = 1.0 / fps
        for i in range(len(quats)):
            await server.broadcast({'type': 'frame', 'frame': i,
                                    'pose': quats[i].tolist(), 'trans': trans[i].tolist(),
                                    'caption': caption})
            await asyncio.sleep(frame_dt)
        await server.broadcast({'type': 'end', 'name': name})

        sample_dir = sample_output_dir(args.out_dir, 'm2t_stream', name)
        text_path = os.path.join(sample_dir, f'{name}.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('Generated:\n' + caption + '\n')
            if gt_caption:
                f.write('\nGround truth:\n' + gt_caption + '\n')
        print(f'[Saved] {text_path}')

    await server.close()


if __name__ == '__main__':
    parser = option.get_args_parser()
    parser.add_argument('--model_name', type=str, default='./m2t-ft-from-GSPretrained-base',
                        help='Trained motion-to-text model directory')
    parser.add_argument('--prompt', type=str, default='Generate text: ', help='Motion-to-Text instruction prefix')
    parser.add_argument('--name', type=str, default=None, help='Dataset sample id to visualize, e.g. 000000')
    parser.add_argument('--motion_path', type=str, default=None,
                        help='Path to a raw HumanML3D-format motion .npy, used instead of a dataset sample')
    parser.add_argument('--split', type=str, default='test',
                        help='Split to pick a random sample from when --name/--motion_path are not given '
                             'and ./input/ has no .npy files')
    parser.add_argument('--sample_seed', type=int, default=None, help='Seed for picking the random sample')
    parser.add_argument('--max_new_tokens', type=int, default=40)
    parser.add_argument('--out_dir', type=str, default='./visualizations')
    parser.add_argument('--fps', type=float, default=None, help='Override playback fps (defaults to the dataset fps)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='WebSocket server bind address')
    parser.add_argument('--port', type=int, default=8765, help='WebSocket server port')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    asyncio.run(run(args))
