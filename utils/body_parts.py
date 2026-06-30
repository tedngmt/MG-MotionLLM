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


# --- Coarse body parts, for the part-tagged Motion-to-Detailed-Text target ---------------
#
# The fine-grained joint map above is used to *highlight* the rendered skeleton. For
# training MG-MotionLLM to emit which part each sentence is about, we use a coarser, fixed
# vocabulary of six parts (the five kinematic-chain branches plus the head). Each part has a
# special token (matching the project's existing '<...>' token convention) and the joint set
# it lights up when highlighting.
PART_JOINTS = {
    'left_arm':  [13, 16, 18, 20],
    'right_arm': [14, 17, 19, 21],
    'left_leg':  [1, 4, 7, 10],
    'right_leg': [2, 5, 8, 11],
    'torso':     [0, 3, 6, 9, 12],
    'head':      [12, 15],
}
PART_ORDER = ['head', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
PART_TAGS = [f'<{p}>' for p in PART_ORDER]
_TAG_RE = re.compile(r'<(' + '|'.join(PART_ORDER) + r')>')

_TORSO_WORDS = ['body', 'torso', 'waist', 'spine', 'chest', 'pelvis']
_ARM_WORDS = ['arm', 'forearm', 'shoulder', 'elbow', 'hand', 'wrist', 'finger']
_LEG_WORDS = ['leg', 'thigh', 'knee', 'shin', 'calf', 'ankle', 'foot', 'toe', 'hip']
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


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


def caption_to_parts(text):
    """Return the set of coarse part names (keys of PART_JOINTS) a *single sentence* is about.

    Side ambiguity is resolved at the sentence level: a side-less limb word ("bend your
    knee") inherits the only side the sentence mentions ("...your left leg..."), and falls
    back to both sides only when the sentence names both sides or neither.
    """
    if not text:
        return set()
    text = text.lower()
    parts = set()

    if _has(text, 'head'):
        parts.add('head')
    if any(_has(text, w) for w in _TORSO_WORDS) or _has(text, 'neck'):
        parts.add('torso')

    has_left = _has(text, 'left')
    has_right = _has(text, 'right')

    def _assign(words, left_part, right_part):
        plurals = [_PLURALS[w] for w in words if w in _PLURALS]
        if any(_has(text, p) for p in plurals):
            parts.update((left_part, right_part))
            return
        sided = False
        if any(_has(text, 'left ' + w) for w in words):
            parts.add(left_part)
            sided = True
        if any(_has(text, 'right ' + w) for w in words):
            parts.add(right_part)
            sided = True
        if sided or not any(_has(text, w) for w in words):
            return
        # Bare, side-less limb word: inherit the sentence's only side if unambiguous.
        if has_left and not has_right:
            parts.add(left_part)
        elif has_right and not has_left:
            parts.add(right_part)
        else:
            parts.update((left_part, right_part))

    _assign(_ARM_WORDS, 'left_arm', 'right_arm')
    _assign(_LEG_WORDS, 'left_leg', 'right_leg')
    return parts


def tag_snippet(snippet):
    """Prefix every sentence of a 0.5s snippet with the body-part tag(s) it describes.

    "Move your hands closer together. Turn your head to the right."
        -> "<left_arm><right_arm> Move your hands closer together. <head> Turn your head to the right."

    Snippets with no recognisable part (e.g. "<Motionless>", "Look forward.") are returned
    unchanged, so the tags only ever mark sentences we can confidently attribute.
    """
    if not snippet or snippet == '<Motionless>':
        return snippet
    out = []
    for sentence in _SENTENCE_RE.split(snippet.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        parts = caption_to_parts(sentence)
        tags = ''.join(f'<{p}>' for p in PART_ORDER if p in parts)
        out.append(f'{tags} {sentence}' if tags else sentence)
    return ' '.join(out)


def parse_tagged_text(text):
    """Split a part-tagged snippet into (parts, sentence) pairs.

    Inverse of tag_snippet: reads the inline '<part>' tokens back out, returning a list of
    (set_of_part_names, sentence_text) for each tagged sentence. Used at inference time to
    drive highlighting straight from the model's own tags instead of re-guessing from text.
    """
    segments = []
    matches = list(_TAG_RE.finditer(text))
    if not matches:
        return segments
    # Group consecutive tags, then take the text up to the next tag group as their sentence.
    i = 0
    while i < len(matches):
        parts = {matches[i].group(1)}
        j = i + 1
        while j < len(matches) and matches[j].start() == matches[j - 1].end():
            parts.add(matches[j].group(1))
            j += 1
        text_start = matches[j - 1].end()
        text_end = matches[j].start() if j < len(matches) else len(text)
        segments.append((parts, text[text_start:text_end].strip()))
        i = j
    return segments


def strip_tags(text):
    """Remove inline '<part>' tokens, collapsing the leftover whitespace."""
    return re.sub(r'\s{2,}', ' ', _TAG_RE.sub('', text)).strip()


def parts_to_joints(parts, joints_num=22):
    """Union the joint indices of the given coarse part names (empty for non-22-joint skeletons)."""
    if joints_num != 22:
        return set()
    joints = set()
    for p in parts:
        joints.update(PART_JOINTS.get(p, []))
    return joints


def snippet_to_display_and_joints(snippet, joints_num=22):
    """Return (clean_caption, highlight_joints) for one M2DT snippet.

    If the snippet carries inline '<part>' tags (a model trained with --use_part_tags), the
    highlight comes straight from those tags and the caption is shown with the tags stripped.
    Otherwise we fall back to keyword-mapping the raw text, so untagged checkpoints still
    highlight as before.
    """
    segments = parse_tagged_text(snippet)
    if segments:
        parts = set().union(*(parts for parts, _ in segments))
        return strip_tags(snippet), parts_to_joints(parts, joints_num)
    return snippet, caption_to_highlight_joints(snippet, joints_num)
