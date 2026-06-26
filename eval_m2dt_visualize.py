"""Standalone Motion-to-Detailed-Text demo: generate a fine-grained motion script for a
single motion clip and render it as a 3D skeleton animation (HumanML3D/SMPL-H
body-joint topology) with the script's body-part snippets shown in sync with the motion.

This is a qualitative companion to eval_m2dt.py, which computes corpus-wide
BLEU/ROUGE/CIDEr/BERTScore over the whole test split. This script instead runs one
motion through the VQ-VAE + motion-to-detailed-text model and visualizes the result, so
it only needs torch/transformers/matplotlib (no evaluator wrapper, spacy, or bert-score).

Each generated snippet corresponds to a fixed-size chunk of motion (0.5s, matching how
FineMotion's body-part descriptions are aligned to HumanML3D motions), so the on-screen
caption changes roughly every half second as the skeleton moves.

Examples:
    python3 eval_m2dt_visualize.py --model_name ./m2dt-ft-from-GSPretrained-base --name 000000
    python3 eval_m2dt_visualize.py --model_name ./m2dt-ft-from-GSPretrained-base --split test --seed 0
"""
import os

import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from options import option
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils.inference_utils import (
    DATASET_CONFIG, resolve_sample, load_vqvae, motion_to_token_string, generate_text,
    load_gt_detail, parse_motion_script,
)

SNIPPET_SECONDS = 0.5  # FineMotion body-part descriptions are aligned to 0.5s chunks.


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
                        help='Split to pick a random sample from when --name/--motion_path are not given')
    parser.add_argument('--sample_seed', type=int, default=None, help='Seed for picking the random sample')
    parser.add_argument('--max_new_tokens', type=int, default=1536)
    parser.add_argument('--out_dir', type=str, default='./visualizations')
    parser.add_argument('--format', type=str, default='gif', choices=['gif', 'mp4'],
                        help='mp4 requires a working ffmpeg install')
    parser.add_argument('--fps', type=float, default=None, help='Override playback fps (defaults to the dataset fps)')
    parser.add_argument('--radius', type=float, default=4.0, help='Floor footprint (meters) shown around the character')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataname]
    fps = args.fps or cfg['fps']
    frames_per_snippet = max(1, round(SNIPPET_SECONDS * fps))
    os.makedirs(args.out_dir, exist_ok=True)

    name, raw_motion = resolve_sample(cfg, args.split, args.name, args.motion_path, args.sample_seed)
    print(f'[Sample] {name}  ({raw_motion.shape[0]} frames)')

    # Truncate so tokens (4 frames each) and snippets (frames_per_snippet each) stay aligned,
    # mirroring Motion2MotionScriptDataset's `m_length // 20` truncation for t2m.
    unit_length = frames_per_snippet * (2 ** args.down_t) // np.gcd(frames_per_snippet, 2 ** args.down_t)
    m_length = int((raw_motion.shape[0] // unit_length) * unit_length)
    raw_motion = raw_motion[:m_length]

    mean = np.load(os.path.join(cfg['meta_dir'], 'mean.npy'))
    std = np.load(os.path.join(cfg['meta_dir'], 'std.npy'))
    norm_motion = (raw_motion - mean) / std

    print('[VQ-VAE] loading...')
    net = load_vqvae(args, args.dataname, args.device)
    motion_string = motion_to_token_string(net, norm_motion, args.device)
    prompt = args.prompt + motion_string

    print('[LLM] loading', args.model_name)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(args.device).eval()

    output_text = generate_text(tokenizer, model, prompt, args.max_new_tokens, args.device)
    snippets = parse_motion_script(output_text)

    gt_snippets = load_gt_detail(cfg, name)
    num_expected = m_length // frames_per_snippet
    gt_snippets = gt_snippets[:num_expected] if gt_snippets else None

    print(f'[Generated script] {" <SEP> ".join(snippets)}')
    if gt_snippets:
        print(f'[GT script]        {" <SEP> ".join(gt_snippets)}')

    timed_captions = [
        (i * frames_per_snippet, min((i + 1) * frames_per_snippet, m_length), text)
        for i, text in enumerate(snippets)
    ]

    joints = recover_from_ric(torch.from_numpy(raw_motion).float(), cfg['joints_num']).numpy()

    save_path = os.path.join(args.out_dir, f'{name}_m2dt.{args.format}')
    plot_3d_motion(save_path, cfg['kinematic_chain'], joints,
                   captions=timed_captions, title=f'Sample: {name}',
                   fps=fps, radius=args.radius)

    script_path = os.path.join(args.out_dir, f'{name}_m2dt_script.txt')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write('Generated:\n' + '\n'.join(snippets) + '\n')
        if gt_snippets:
            f.write('\nGround truth:\n' + '\n'.join(gt_snippets) + '\n')

    print(f'[Saved] {save_path}')
    print(f'[Saved] {script_path}')
