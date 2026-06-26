import json
import os
import random

import numpy as np
import torch

import models.vqvae as vqvae
import utils.paramUtil as paramUtil


DATASET_CONFIG = {
    't2m': dict(
        data_root='./dataset/HumanML3D',
        motion_dir='./dataset/HumanML3D/new_joint_vecs',
        text_dir='./dataset/HumanML3D/texts',
        finemotion_dir='./dataset/HumanML3D/finemotion_texts',
        meta_dir='./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta',
        joints_num=22,
        fps=20.0,
        kinematic_chain=paramUtil.t2m_kinematic_chain,
    ),
    'kit': dict(
        data_root='./dataset/KIT-ML',
        motion_dir='./dataset/KIT-ML/new_joint_vecs',
        text_dir='./dataset/KIT-ML/texts',
        finemotion_dir='./dataset/KIT-ML/finemotion_texts',
        meta_dir='./checkpoints/kit/Decomp_SP001_SM001_H512/meta',
        joints_num=21,
        fps=12.5,
        kinematic_chain=paramUtil.kit_kinematic_chain,
    ),
}


def resolve_sample(cfg, split, name=None, motion_path=None, seed=None):
    """Pick the raw (unnormalized) HumanML3D-format motion to visualize.

    Priority: an explicit --motion_path file, then an explicit --name dataset id,
    then a random id from the requested split.
    """
    if motion_path is not None:
        sample_name = os.path.splitext(os.path.basename(motion_path))[0]
        return sample_name, np.load(motion_path)

    if name is None:
        split_file = os.path.join(cfg['data_root'], f'{split}.txt')
        with open(split_file) as f:
            ids = [line.strip() for line in f if line.strip()]
        name = random.Random(seed).choice(ids)

    raw_motion = np.load(os.path.join(cfg['motion_dir'], name + '.npy'))
    return name, raw_motion


def load_vqvae(args, dataname, device):
    net = vqvae.HumanVQVAE(args, 512, args.code_dim, args.output_emb_width,
                           2, args.stride_t, args.width, 3, args.dilation_growth_rate)
    ckpt = torch.load(f'./checkpoints/pretrained_vqvae/{dataname}.pth', map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    return net.to(device).eval()


def motion_to_token_string(vqvae_net, norm_motion, device):
    motion_tensor = torch.from_numpy(norm_motion).float().unsqueeze(0).to(device)
    with torch.no_grad():
        token_idx = vqvae_net.encode(motion_tensor)
    tokens = token_idx.cpu().numpy()[0].reshape(-1).tolist()
    motion_string = '<Motion Tokens>' + ''.join(f'<{t}>' for t in tokens) + '</Motion Tokens>'
    return motion_string


def generate_text(tokenizer, model, prompt, max_new_tokens, device):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device, dtype=torch.long)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_new_tokens, num_beams=1, do_sample=False)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def load_gt_caption(cfg, name):
    """Human-written HumanML3D caption for `name`, for the Motion-to-Text task."""
    text_path = os.path.join(cfg['text_dir'], name + '.txt')
    if not os.path.isfile(text_path):
        return None
    with open(text_path, encoding='utf-8') as f:
        first_line = f.readline().strip()
    return first_line.split('#')[0] if first_line else None


def load_gt_detail(cfg, name):
    """FineMotion body-part snippets for `name`, for the Motion-to-Detailed-Text task."""
    finemotion_dir = cfg['finemotion_dir']
    auto_path = os.path.join(finemotion_dir, 'BPMSD_auto.json')
    if not os.path.isdir(finemotion_dir) or not os.path.isfile(auto_path):
        return None
    with open(auto_path, encoding='utf-8') as f:
        bpmsd = json.load(f)
    human_path = os.path.join(finemotion_dir, 'BPMSD_human.json')
    if os.path.isfile(human_path):
        with open(human_path, encoding='utf-8') as f:
            bpmsd.update(json.load(f))
    snippets = bpmsd.get(name)
    if snippets is None:
        return None
    return [s if s else '<Motionless>' for s in snippets]


def parse_motion_script(output_text):
    """Split a Motion-to-Detailed-Text model's raw output into per-snippet strings."""
    if '### Motion Script ###' in output_text:
        output_text = output_text.split('### Motion Script ###', 1)[1]
    return [s.strip() for s in output_text.strip().split('<SEP>')]
