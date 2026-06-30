"""Build a body-part-tagged copy of the FineMotion / BPMSD detailed-caption data.

Each BPMSD snippet is a ~0.5s description made of one sentence per body part. This script
prefixes every sentence with the coarse body-part tag(s) it is about (see
utils/body_parts.tag_snippet), producing tagged JSONs with the same {name: [snippet, ...]}
layout. Training MG-MotionLLM (Motion-to-Detailed-Text) on these targets teaches the model
to emit which part each sentence describes, so the rendered skeleton can be highlighted from
the model's own output instead of a post-hoc keyword guess.

The tagging labels are bootstrapped from text, so spot-check the printed samples before
committing to a fine-tune. Outputs land next to the originals as *_tagged.json.

    python3 prepare/build_tagged_bpmsd.py --finemotion_dir ./dataset/HumanML3D/finemotion_texts
"""
import argparse
import json
import os
import random

from utils.body_parts import tag_snippet, parse_tagged_text, PART_ORDER


def build(finemotion_dir, num_samples):
    for base in ('BPMSD_auto', 'BPMSD_human'):
        src = os.path.join(finemotion_dir, f'{base}.json')
        if not os.path.isfile(src):
            print(f'[skip] {src} not found')
            continue
        with open(src, encoding='utf-8') as f:
            data = json.load(f)

        tagged = {name: [tag_snippet(s) for s in snippets] for name, snippets in data.items()}

        dst = os.path.join(finemotion_dir, f'{base}_tagged.json')
        with open(dst, 'w', encoding='utf-8') as f:
            json.dump(tagged, f, ensure_ascii=False)

        n_snip = sum(len(v) for v in tagged.values())
        n_tagged = sum(1 for v in tagged.values() for s in v if '<' in s)
        per_part = {p: 0 for p in PART_ORDER}
        for v in tagged.values():
            for s in v:
                for parts, _ in parse_tagged_text(s):
                    for p in parts:
                        per_part[p] += 1
        print(f'\n[{base}] {len(tagged)} clips, {n_snip} snippets, '
              f'{n_tagged} tagged ({100.0 * n_tagged / max(n_snip, 1):.1f}%)')
        print('  sentences per part:', {p: per_part[p] for p in PART_ORDER})
        print(f'  [saved] {dst}')

    # Spot-check: print a handful of original -> tagged snippets.
    src = os.path.join(finemotion_dir, 'BPMSD_auto.json')
    with open(src, encoding='utf-8') as f:
        data = json.load(f)
    random.seed(0)
    print('\n===== spot-check (original -> tagged) =====')
    shown = 0
    for name in random.sample(list(data), min(50, len(data))):
        for snippet in data[name]:
            if snippet and shown < num_samples:
                print(f'\n  orig: {snippet}')
                print(f'  tag : {tag_snippet(snippet)}')
                shown += 1
        if shown >= num_samples:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build body-part-tagged BPMSD targets.')
    parser.add_argument('--finemotion_dir', type=str,
                        default='./dataset/HumanML3D/finemotion_texts',
                        help='Directory holding BPMSD_auto.json / BPMSD_human.json')
    parser.add_argument('--num_samples', type=int, default=15,
                        help='How many original->tagged snippets to print for spot-checking')
    args = parser.parse_args()
    build(args.finemotion_dir, args.num_samples)
