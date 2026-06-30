import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


# One color per kinematic-chain branch. Both the t2m (HumanML3D/SMPL-H body)
# and KIT skeletons in utils/paramUtil.py are split into exactly 5 branches
# (right leg, left leg, spine, right arm, left arm).
CHAIN_COLORS = ['#e74c3c', '#3498db', '#2c3e50', '#e67e22', '#9b59b6']

# Colours used when a per-frame body-part highlight track is supplied: every
# bone is drawn white and the highlighted joints/bones are drawn yellow, over a
# dark background so the white skeleton stays visible.
BONE_COLOR = '#f5f5f5'
HIGHLIGHT_COLOR = '#ffe000'
HIGHLIGHT_BG = '#1e1e1e'


def _prepare_joints(joints):
    """Floor the skeleton and recenter every frame on the root, returning the
    (possibly height-floored) joints together with the root's raw floor trajectory."""
    joints = np.array(joints, dtype=np.float32).copy()
    joints[:, :, 1] -= joints[:, :, 1].min()

    trajectory = joints[:, 0, [0, 2]].copy()
    joints[:, :, 0] -= trajectory[:, 0:1]
    joints[:, :, 2] -= trajectory[:, 1:2]

    height_max = max(2.0, float(joints[:, :, 1].max()) * 1.2)
    return joints, trajectory, height_max


def _active_caption(captions, frame_idx):
    if captions is None:
        return ''
    if isinstance(captions, str):
        return captions
    for start, end, text in captions:
        if start <= frame_idx < end:
            return text
    return ''


def _active_highlight(highlights, frame_idx):
    if highlights is None:
        return None
    for start, end, joints in highlights:
        if start <= frame_idx < end:
            return joints
    return set()


def _draw_skeleton(ax, kinematic_chain, joints, trajectory, frame_idx, radius, height_max,
                   highlight=None):
    ax.cla()
    ax.set_xlim3d(-radius / 2, radius / 2)
    ax.set_ylim3d(-radius / 2, radius / 2)
    ax.set_zlim3d(0, height_max)
    ax.set_box_aspect((radius, radius, height_max))
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-60)
    if highlight is not None:
        # cla() resets the axes background; keep the plot area dark so the white
        # skeleton stays visible.
        ax.set_facecolor(HIGHLIGHT_BG)

    # Floor trajectory trace of the root joint up to the current frame.
    ax.plot3D(trajectory[:frame_idx + 1, 0] - trajectory[frame_idx, 0],
               trajectory[:frame_idx + 1, 1] - trajectory[frame_idx, 1],
               np.zeros(frame_idx + 1),
               color='gray', linewidth=1.0, alpha=0.5)

    pose = joints[frame_idx]

    if highlight is None:
        # Default look: one colour per kinematic-chain branch.
        num_repeats = len(kinematic_chain) // len(CHAIN_COLORS) + 1
        for chain, color in zip(kinematic_chain, CHAIN_COLORS * num_repeats):
            ax.plot3D(pose[chain, 0], pose[chain, 2], pose[chain, 1],
                      linewidth=3.0, color=color, marker='o', markersize=3)
        return

    # Highlight look: white skeleton, yellow for the joints/bones referenced by
    # the active caption. A bone is highlighted when its child joint is.
    for chain in kinematic_chain:
        for parent, child in zip(chain[:-1], chain[1:]):
            color = HIGHLIGHT_COLOR if child in highlight else BONE_COLOR
            ax.plot3D(pose[[parent, child], 0], pose[[parent, child], 2], pose[[parent, child], 1],
                      linewidth=3.0, color=color, marker='o', markersize=3,
                      markerfacecolor=color, markeredgecolor=color)


def _save_animation(fig, update, frames, save_path, fps):
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000.0 / fps)
    save_kwargs = {'savefig_kwargs': {'facecolor': fig.get_facecolor()}}
    if save_path.endswith('.gif'):
        ani.save(save_path, writer='pillow', fps=fps, **save_kwargs)
    else:
        ani.save(save_path, writer='ffmpeg', fps=fps, dpi=120, **save_kwargs)
    plt.close(fig)


def plot_3d_motion(save_path, kinematic_chain, joints, captions=None, title=None,
                    fps=20, radius=4.0, figsize=(6, 6), highlights=None):
    """Render a 3D skeleton animation and save it as a video/gif.

    Args:
        save_path: output path, extension picks the writer (".mp4" -> ffmpeg, ".gif" -> pillow).
        kinematic_chain: list of joint-index chains, e.g. paramUtil.t2m_kinematic_chain.
        joints: (frames, num_joints, 3) array in HumanML3D/SMPL-H body-joint convention
            (X right, Y up, Z forward).
        captions: caption(s) to show under the animation. Either a single str shown for the
            whole clip, or a list of (start_frame, end_frame, text) tuples for subtitles that
            change over time (e.g. one per body-part-motion snippet).
        title: static title shown above the animation.
        fps: playback frame rate.
        radius: half-width (in meters) of the floor area shown around the character.
        highlights: optional per-frame body-part highlight track, as a list of
            (start_frame, end_frame, joint_index_set) tuples. When given, the skeleton is
            drawn white with the listed joints/bones in yellow over a dark background, so
            the body parts the caption refers to stand out. When None, the default
            per-branch colouring is used.
    """
    joints, trajectory, height_max = _prepare_joints(joints)
    frames = joints.shape[0]
    dark = highlights is not None

    fig = plt.figure(figsize=figsize)
    if dark:
        fig.patch.set_facecolor(HIGHLIGHT_BG)
    text_color = BONE_COLOR if dark else 'black'
    ax = fig.add_subplot(111, projection='3d')
    if title:
        wrapped_title = '\n'.join(textwrap.fill(line, 60) for line in title.split('\n'))
        fig.suptitle(wrapped_title, fontsize=11, color=text_color)
    caption_artist = fig.text(0.5, 0.04, '', ha='center', va='bottom', fontsize=10, wrap=True,
                              color=text_color)

    def update(frame_idx):
        highlight = _active_highlight(highlights, frame_idx)
        _draw_skeleton(ax, kinematic_chain, joints, trajectory, frame_idx, radius, height_max,
                       highlight=highlight)
        text = _active_caption(captions, frame_idx)
        caption_artist.set_text(textwrap.fill(text, 70) if text else '')

    _save_animation(fig, update, frames, save_path, fps)


def plot_3d_motion_compare(save_path, kinematic_chain, joints, captions_left, captions_right,
                           label_left=None, label_right=None, title=None,
                           fps=20, radius=4.0, figsize=(11, 6),
                           highlights_left=None, highlights_right=None):
    """Render the same motion twice side by side, each with its own caption track.

    Meant for comparing two models' text output (e.g. M2T vs M2DT) on one motion: both
    panels show the identical, synchronized skeleton, with `captions_left`/`captions_right`
    holding each model's caption (single str, or [(start_frame, end_frame, text), ...]).

    `highlights_left`/`highlights_right` optionally supply per-frame body-part highlight
    tracks (see plot_3d_motion). If either is given, both panels switch to the white
    skeleton / yellow highlight look on a dark background; a panel with no track of its own
    is drawn all-white so the two panels stay visually consistent.
    """
    joints, trajectory, height_max = _prepare_joints(joints)
    frames = joints.shape[0]
    dark = highlights_left is not None or highlights_right is not None

    fig = plt.figure(figsize=figsize)
    if dark:
        fig.patch.set_facecolor(HIGHLIGHT_BG)
    text_color = BONE_COLOR if dark else 'black'
    ax_left = fig.add_subplot(121, projection='3d')
    ax_right = fig.add_subplot(122, projection='3d')
    if title:
        fig.suptitle(textwrap.fill(title, 90), fontsize=11, color=text_color)

    caption_left_artist = fig.text(0.25, 0.04, '', ha='center', va='bottom', fontsize=9, wrap=True,
                                   color=text_color)
    caption_right_artist = fig.text(0.75, 0.04, '', ha='center', va='bottom', fontsize=9, wrap=True,
                                    color=text_color)

    def _highlight(track, frame_idx):
        # In dark mode a panel with no track of its own is still drawn white (empty set),
        # not in the default per-branch colours, so both panels match.
        if track is None:
            return set() if dark else None
        return _active_highlight(track, frame_idx)

    def update(frame_idx):
        _draw_skeleton(ax_left, kinematic_chain, joints, trajectory, frame_idx, radius, height_max,
                       highlight=_highlight(highlights_left, frame_idx))
        _draw_skeleton(ax_right, kinematic_chain, joints, trajectory, frame_idx, radius, height_max,
                       highlight=_highlight(highlights_right, frame_idx))
        if label_left:
            ax_left.set_title(label_left, fontsize=10, color=text_color)
        if label_right:
            ax_right.set_title(label_right, fontsize=10, color=text_color)

        left_text = _active_caption(captions_left, frame_idx)
        right_text = _active_caption(captions_right, frame_idx)
        caption_left_artist.set_text(textwrap.fill(left_text, 45) if left_text else '')
        caption_right_artist.set_text(textwrap.fill(right_text, 45) if right_text else '')

    _save_animation(fig, update, frames, save_path, fps)
