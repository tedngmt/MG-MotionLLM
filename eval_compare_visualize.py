"""Standalone side-by-side demo: run the same motion clip through both the
Motion-to-Text model and the Motion-to-Detailed-Text model, and render one synchronized
3D skeleton animation with two panels -- the M2T caption on the left, the time-synced
M2DT script on the right -- so you can compare what each model says about the same motion.

Example:
    python3 eval_compare_visualize.py \\
        --m2t_model_name ./m2t-ft-from-GSPretrained-base \\
        --m2dt_model_name ./m2dt-ft-from-GSPretrained-base \\
        --name 000000
"""
import os

import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from options import option
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion_compare
from utils.inference_utils import (
    DATASET_CONFIG, resolve_sample, load_vqvae, motion_to_token_string, generate_text,
    load_gt_caption, load_gt_detail, parse_motion_script,
)

SNIPPET_SECONDS = 0.5  # FineMotion body-part descriptions are aligned to 0.5s chunks.


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
                        help='Split to pick a random sample from when --name/--motion_path are not given')
    parser.add_argument('--sample_seed', type=int, default=None, help='Seed for picking the random sample')
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

    # Truncate to a length valid for both the M2T token unit (2**down_t frames) and the
    # M2DT snippet unit (frames_per_snippet), so both models see the exact same motion
    # and the single shared skeleton animation is unambiguous.
    unit_length = frames_per_snippet * (2 ** args.down_t) // np.gcd(frames_per_snippet, 2 ** args.down_t)
    m_length = int((raw_motion.shape[0] // unit_length) * unit_length)
    raw_motion = raw_motion[:m_length]

    mean = np.load(os.path.join(cfg['meta_dir'], 'mean.npy'))
    std = np.load(os.path.join(cfg['meta_dir'], 'std.npy'))
    norm_motion = (raw_motion - mean) / std

    print('[VQ-VAE] loading...')
    net = load_vqvae(args, args.dataname, args.device)
    motion_string = motion_to_token_string(net, norm_motion, args.device)

    print('[M2T] loading', args.m2t_model_name)
    m2t_tokenizer = T5Tokenizer.from_pretrained(args.m2t_model_name)
    m2t_model = T5ForConditionalGeneration.from_pretrained(args.m2t_model_name).to(args.device).eval()
    m2t_output = generate_text(m2t_tokenizer, m2t_model, args.m2t_prompt + motion_string,
                               args.m2t_max_new_tokens, args.device)
    caption = m2t_output.strip().strip('"')
    del m2t_model, m2t_tokenizer
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    print('[M2DT] loading', args.m2dt_model_name)
    m2dt_tokenizer = T5Tokenizer.from_pretrained(args.m2dt_model_name)
    m2dt_model = T5ForConditionalGeneration.from_pretrained(args.m2dt_model_name).to(args.device).eval()
    m2dt_output = generate_text(m2dt_tokenizer, m2dt_model, args.m2dt_prompt + motion_string,
                                args.m2dt_max_new_tokens, args.device)
    snippets = parse_motion_script(m2dt_output)
    del m2dt_model, m2dt_tokenizer
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    gt_caption = load_gt_caption(cfg, name)
    gt_snippets = load_gt_detail(cfg, name)
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

    joints = recover_from_ric(torch.from_numpy(raw_motion).float(), cfg['joints_num']).numpy()

    save_path = os.path.join(args.out_dir, f'{name}_compare.{args.format}')
    plot_3d_motion_compare(save_path, cfg['kinematic_chain'], joints,
                           captions_left=caption, captions_right=timed_script,
                           label_left='Motion-to-Text', label_right='Motion-to-Detailed-Text',
                           title=f'Sample: {name}', fps=fps, radius=args.radius)

    script_path = os.path.join(args.out_dir, f'{name}_compare.txt')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write('M2T generated:\n' + caption + '\n')
        if gt_caption:
            f.write('\nM2T GT:\n' + gt_caption + '\n')
        f.write('\nM2DT generated:\n' + '\n'.join(snippets) + '\n')
        if gt_snippets:
            f.write('\nM2DT GT:\n' + '\n'.join(gt_snippets) + '\n')

    print(f'[Saved] {save_path}')
    print(f'[Saved] {script_path}')
