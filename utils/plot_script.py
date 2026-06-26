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


def _draw_skeleton(ax, kinematic_chain, joints, trajectory, frame_idx, radius, height_max):
    ax.cla()
    ax.set_xlim3d(-radius / 2, radius / 2)
    ax.set_ylim3d(-radius / 2, radius / 2)
    ax.set_zlim3d(0, height_max)
    ax.set_box_aspect((radius, radius, height_max))
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-60)

    # Floor trajectory trace of the root joint up to the current frame.
    ax.plot3D(trajectory[:frame_idx + 1, 0] - trajectory[frame_idx, 0],
               trajectory[:frame_idx + 1, 1] - trajectory[frame_idx, 1],
               np.zeros(frame_idx + 1),
               color='gray', linewidth=1.0, alpha=0.5)

    pose = joints[frame_idx]
    num_repeats = len(kinematic_chain) // len(CHAIN_COLORS) + 1
    for chain, color in zip(kinematic_chain, CHAIN_COLORS * num_repeats):
        ax.plot3D(pose[chain, 0], pose[chain, 2], pose[chain, 1],
                  linewidth=3.0, color=color, marker='o', markersize=3)


def _save_animation(fig, update, frames, save_path, fps):
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000.0 / fps)
    if save_path.endswith('.gif'):
        ani.save(save_path, writer='pillow', fps=fps)
    else:
        ani.save(save_path, writer='ffmpeg', fps=fps, dpi=120)
    plt.close(fig)


def plot_3d_motion(save_path, kinematic_chain, joints, captions=None, title=None,
                    fps=20, radius=4.0, figsize=(6, 6)):
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
    """
    joints, trajectory, height_max = _prepare_joints(joints)
    frames = joints.shape[0]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    if title:
        wrapped_title = '\n'.join(textwrap.fill(line, 60) for line in title.split('\n'))
        fig.suptitle(wrapped_title, fontsize=11)
    caption_artist = fig.text(0.5, 0.04, '', ha='center', va='bottom', fontsize=10, wrap=True)

    def update(frame_idx):
        _draw_skeleton(ax, kinematic_chain, joints, trajectory, frame_idx, radius, height_max)
        text = _active_caption(captions, frame_idx)
        caption_artist.set_text(textwrap.fill(text, 70) if text else '')

    _save_animation(fig, update, frames, save_path, fps)


def plot_3d_motion_compare(save_path, kinematic_chain, joints, captions_left, captions_right,
                           label_left=None, label_right=None, title=None,
                           fps=20, radius=4.0, figsize=(11, 6)):
    """Render the same motion twice side by side, each with its own caption track.

    Meant for comparing two models' text output (e.g. M2T vs M2DT) on one motion: both
    panels show the identical, synchronized skeleton, with `captions_left`/`captions_right`
    holding each model's caption (single str, or [(start_frame, end_frame, text), ...]).
    """
    joints, trajectory, height_max = _prepare_joints(joints)
    frames = joints.shape[0]

    fig = plt.figure(figsize=figsize)
    ax_left = fig.add_subplot(121, projection='3d')
    ax_right = fig.add_subplot(122, projection='3d')
    if title:
        fig.suptitle(textwrap.fill(title, 90), fontsize=11)

    caption_left_artist = fig.text(0.25, 0.04, '', ha='center', va='bottom', fontsize=9, wrap=True)
    caption_right_artist = fig.text(0.75, 0.04, '', ha='center', va='bottom', fontsize=9, wrap=True)

    def update(frame_idx):
        _draw_skeleton(ax_left, kinematic_chain, joints, trajectory, frame_idx, radius, height_max)
        _draw_skeleton(ax_right, kinematic_chain, joints, trajectory, frame_idx, radius, height_max)
        if label_left:
            ax_left.set_title(label_left, fontsize=10)
        if label_right:
            ax_right.set_title(label_right, fontsize=10)

        left_text = _active_caption(captions_left, frame_idx)
        right_text = _active_caption(captions_right, frame_idx)
        caption_left_artist.set_text(textwrap.fill(left_text, 45) if left_text else '')
        caption_right_artist.set_text(textwrap.fill(right_text, 45) if right_text else '')

    _save_animation(fig, update, frames, save_path, fps)
