"""Map FineMotion / BPMSD captions to skeleton joints, so the body parts a
caption talks about can be highlighted on the animated skeleton.

The captions MG-MotionLLM emits for Motion-to-Detailed-Text are free-text
sentences ("Raise your left leg up and bend your knee. Extend your left arm
out behind you."), not part-labelled fields. They do, however, use a small and
consistent vocabulary, so we recover the relevant joints by scanning the text
for body-part words and unioning the joint indices they refer to.

Joint indexing follows the standard 22-joint HumanML3D / SMPL-H body
convention (see utils/paramUtil.t2m_kinematic_chain):

    0 pelvis   1 L_hip    2 R_hip    3 spine1   4 L_knee   5 R_knee
    6 spine2   7 L_ankle  8 R_ankle  9 spine3  10 L_foot  11 R_foot
   12 neck    13 L_collar 14 R_collar 15 head  16 L_shoulder 17 R_shoulder
   18 L_elbow 19 R_elbow  20 L_wrist  21 R_wrist
"""
import re

# Side-specific joints. Each limb word lights up the whole chain down to the
# extremity so e.g. "left arm" highlights collar->shoulder->elbow->wrist.
_LIMB = {
    'leg':      {'left': [1, 4, 7, 10], 'right': [2, 5, 8, 11]},
    'thigh':    {'left': [1, 4],        'right': [2, 5]},
    'knee':     {'left': [4],           'right': [5]},
    'shin':     {'left': [4, 7],        'right': [5, 8]},
    'calf':     {'left': [4, 7],        'right': [5, 8]},
    'ankle':    {'left': [7],           'right': [8]},
    'foot':     {'left': [7, 10],       'right': [8, 11]},
    'toe':      {'left': [10],          'right': [11]},
    'arm':      {'left': [13, 16, 18, 20], 'right': [14, 17, 19, 21]},
    'forearm':  {'left': [18, 20],      'right': [19, 21]},
    'shoulder': {'left': [16],          'right': [17]},
    'elbow':    {'left': [18],          'right': [19]},
    'hand':     {'left': [20],          'right': [21]},
    'wrist':    {'left': [20],          'right': [21]},
    'finger':   {'left': [20],          'right': [21]},
    'hip':      {'left': [1],           'right': [2]},
}

# Irregular and regular plurals -> highlight both sides.
_PLURALS = {
    'leg': 'legs', 'thigh': 'thighs', 'knee': 'knees', 'shin': 'shins',
    'calf': 'calves', 'ankle': 'ankles', 'foot': 'feet', 'toe': 'toes',
    'arm': 'arms', 'forearm': 'forearms', 'shoulder': 'shoulders',
    'elbow': 'elbows', 'hand': 'hands', 'wrist': 'wrists',
    'finger': 'fingers', 'hip': 'hips',
}

# Words that map to a fixed central set with no left/right side. "back" is left
# out on purpose: in these captions it is almost always the adverb ("bring your
# arm back"), not the body part.
_CENTRAL = {
    'head': [12, 15],
    'neck': [12],
    'body': [0, 3, 6, 9, 12],
    'torso': [0, 3, 6, 9],
    'waist': [0, 3],
    'spine': [3, 6, 9],
    'chest': [6, 9],
    'pelvis': [0],
}


def _has(text, word):
    return re.search(r'\b' + word + r'\b', text) is not None


def caption_to_highlight_joints(text, joints_num=22):
    """Return the set of joint indices the caption refers to.

    Only the 22-joint HumanML3D body skeleton is supported; for any other
    skeleton (e.g. KIT's 21 joints) an empty set is returned so the renderer
    falls back to drawing every bone in the default colour.
    """
    if not text or joints_num != 22:
        return set()
    text = text.lower()
    joints = set()

    for word, js in _CENTRAL.items():
        if _has(text, word):
            joints.update(js)

    for word, sides in _LIMB.items():
        plural = _PLURALS.get(word)
        if plural and _has(text, plural):
            joints.update(sides['left'])
            joints.update(sides['right'])
        sided = False
        if _has(text, 'left ' + word):
            joints.update(sides['left'])
            sided = True
        if _has(text, 'right ' + word):
            joints.update(sides['right'])
            sided = True
        # A bare, side-less singular ("bend your knee") is ambiguous; highlight
        # both sides so nothing is missed.
        if not sided and _has(text, word):
            joints.update(sides['left'])
            joints.update(sides['right'])

    return joints
