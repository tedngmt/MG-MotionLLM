<div align="center">
<h1>(CVPR 2025) MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities</h1>

[Bizhu Wu](https://scholar.google.com/citations?user=u7nZ3bgAAAAJ&hl=en) · [Jinheng Xie](https://scholar.google.com/citations?user=smbRMokAAAAJ&hl=en) · [Keming Shen]() · [Zhe Kong](https://scholar.google.com/citations?user=4X3yLwsAAAAJ&hl=en)

[Jianfeng Ren*](https://scholar.google.com/citations?user=ZZ928OgAAAAJ&hl=en) · [Ruibin Bai](https://scholar.google.com/citations?user=oP6AThIAAAAJ&hl=en) · [Rong Qu](https://scholar.google.com/citations?user=ErszCRMAAAAJ&hl=en) ·   [Linlin Shen*](https://scholar.google.com/citations?user=AZ_y9HgAAAAJ&hl=en)

<sup>*</sup>Corresponding Authors

[![arXiv](https://img.shields.io/badge/arXiv-MGMotionLLM-A10717.svg?logo=arXiv)](https://arxiv.org/abs/2504.02478)

</div>



## Table of Content
* [1. Paper Description](#1-paper-description)
* [2. Installation](#2-installation)
* [3. Pretrained Models](#3-pretrained-models)
* [4. Datasets](#4-datasets)
* [5. Evaluation](#5-evaluation)
* [6. Train Your Own Model](#6-train-your-own-model)
* [7. Visualization](#7-visualization)
* [8. Acknowledgement](#8-acknowledgement)
* [9. Bibtex](#9-bibtex)



## 1. Paper Description
**MG-MotionLLM** can address diverse motion-relevant tasks at multiple granularities by giving different instructions in a unified manner. 
- **coarse-grained**: e.g. text-to-motion and motion captioning (upper block) 
- **fine-grained**: e.g. motion-to-detailed text and motion localization (bottom block).

<div align="center">    
  <img src="./assets/teaser.png" alt="teaser" align=center, height=500 />
</div>


To achieve this, we propose multi-granularity training scheme with novel auxiliary tasks captures motion-related features at different levels, improving understanding across a wide range of tasks. Specifically, we pretrain the model with a total of **28** distinct motion-relevant tasks, including **12** existing classical **coarse-grained** tasks and **16** newly proposed **fine-grained** ones. Here, we display examples of prompt templates for a part of tasks used during training.

<div align="center">    
  <img src="./assets/tasks_template.png" alt="tasks_template" align=center />
</div>



## 2. Installation

### 2.1. Environment
```
conda env create -f environment.yml
conda activate mg-motionllm
```

### 2.2. Dependencies
For text-to-motion evaluation
```
bash prepare/download_evaluators.sh
bash prepare/download_glove.sh
```



## 3. Pretrained Models
For pretrained **VQ-VAE** models
```
bash prepare/download_vqvae.sh
```

Once downloaded, you should have a folder like this:
```
MG-MotionLLM/checkpoints
├── pretrained_vqvae
│   └── t2m.pth
```


For pretrained **MG-MotionLLM** models, you have two ways to download:
1. manually download from HuggingFace:

| Model | Link |
|---------|---------|
| GSPretrained-small | [GSPretrained-small](https://huggingface.co/wbz0505/GSPretrained-small) |
| t2m-ft-from-GSPretrained-small | [t2m-ft-from-GSPretrained-small](https://huggingface.co/wbz0505/t2m-ft-from-GSPretrained-small) |
| m2t-ft-from-GSPretrained-small | [m2t-ft-from-GSPretrained-small](https://huggingface.co/wbz0505/m2t-ft-from-GSPretrained-small) |
| tdt2m-ft-from-GSPretrained-small | [tdt2m-ft-from-GSPretrained-small](https://huggingface.co/wbz0505/tdt2m-ft-from-GSPretrained-small) |
| m2dt-ft-from-GSPretrained-small | [m2dt-ft-from-GSPretrained-small](https://huggingface.co/wbz0505/m2dt-ft-from-GSPretrained-small) |
| GSPretrained-base | [GSPretrained-base](https://huggingface.co/wbz0505/GSPretrained-base) |
| t2m-ft-from-GSPretrained-base | [t2m-ft-from-GSPretrained-base](https://huggingface.co/wbz0505/t2m-ft-from-GSPretrained-base) |
| m2t-ft-from-GSPretrained-base | [m2t-ft-from-GSPretrained-base](https://huggingface.co/wbz0505/m2t-ft-from-GSPretrained-base) |
| tdt2m-ft-from-GSPretrained-base | [tdt2m-ft-from-GSPretrained-base](https://huggingface.co/wbz0505/tdt2m-ft-from-GSPretrained-base) |
| m2dt-ft-from-GSPretrained-base | [m2dt-ft-from-GSPretrained-base](https://huggingface.co/wbz0505/m2dt-ft-from-GSPretrained-base) |
| GSPretrained-large | [GSPretrained-large](https://huggingface.co/wbz0505/GSPretrained-large) |
| t2m-ft-from-GSPretrained-large | [t2m-ft-from-GSPretrained-large](https://huggingface.co/wbz0505/t2m-ft-from-GSPretrained-large) |
| m2t-ft-from-GSPretrained-large | [m2t-ft-from-GSPretrained-large](https://huggingface.co/wbz0505/m2t-ft-from-GSPretrained-large) |
| tdt2m-ft-from-GSPretrained-large | [tdt2m-ft-from-GSPretrained-large](https://huggingface.co/wbz0505/tdt2m-ft-from-GSPretrained-large) |
| m2dt-ft-from-GSPretrained-large | [m2dt-ft-from-GSPretrained-large](https://huggingface.co/wbz0505/m2dt-ft-from-GSPretrained-large) |

2. use code to download them. For example,
```python
from huggingface_hub import snapshot_download

model_id = 'wbz0505/t2m-ft-from-GSPretrained-base'       # set the model name to be downloaded
local_dir = "./t2m-ft-from-GSPretrained-base/"          # You can change the save dir here

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print("Download complete!")
```



## 4. Datasets
We are using two 3D human motion-language dataset: HumanML3D and FineMotion.

1. Please follow [HumanML3D](https://github.com/EricGuo5513/HumanML3D) to download and prepare HumanML3D dataset and put them under the directory `dataset` like:
```
./dataset/HumanML3D/
├── new_joint_vecs/
├── texts/
├── Mean.npy    # same as in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 
├── Std.npy     # same as in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 
├── train.txt
├── val.txt
├── test.txt
├── train_val.txt
└── all.txt
```


2. Please follow [FineMotion](https://github.com/BizhuWu/FineMotion) to download detailed body movement descriptions for motions of HumanML3D, *i.e.*, `BPMSD_auto.zip` and `BPMSD_human.zip`.
You should create an empty directory named `finemotion_texts` under the directory `HumanML3D`, 
and put the `BPMSD_auto.zip` and `BPMSD_human.zip` into this newly created directory and unzip it to obtain the json files `BPMSD_auto.json` and `BPMSD_human.json`.
Now, your `dataset` directory should look like: 
```
MG-MotionLLM/dataset/HumanML3D/
├── new_joint_vecs/
├── finemotion_texts/     # here
│   ├── BPMSD_auto.zip
│   ├── BPMSD_auto.json
│   ├── BPMSD_human.zip
│   ├── BPMSD_human.json
├── texts/
├── Mean.npy
├── Std.npy
├── train.txt
├── val.txt
├── test.txt
├── train_val.txt
└── all.txt
```



3. To tokenize the motion data used for training MG-MotionLLM, please follow the instructions below
```python
# Encode the motions to tokens by pretrianed VQ-VAE and save the token sequence results under `./dataset/HumanML3D/VQVAE/`
# For pretrained VQ-VAE, you can use the model provided.
CUDA_VISIBLE_DEVICES=0 python3 scripts/tokenized_motion.py

# The following script is used to generate motion tokens that strictly aligned with detailed text,
# and save the token sequence results under `./dataset/HumanML3D/VQVAE_start0/`
CUDA_VISIBLE_DEVICES=0 python3 scripts/tokenized_motion_start0.py
```



## 5. Evaluation

To evaluate our models on the **Text-to-Motion** task, 
please use the following command:
```python
# from our final t2m model (Granularity-Synergy Pre-training + Task-Specific Instruction Tuning)
CUDA_VISIBLE_DEVICES=0 python3 eval_t2m.py --model_name ./t2m-ft-from-GSPretrained-base/checkpoint-300000
# or
# from our Granularity-Synergy Pre-trained model
CUDA_VISIBLE_DEVICES=0 python3 eval_t2m.py --model_name ./GSPretrained_base/checkpoint-300000 
```


To evaluate our models on the **Motion-to-Text** task, 
please use the following command:
```python
# from our final m2t model (Granularity-Synergy Pre-training + Task-Specific Instruction Tuning)
CUDA_VISIBLE_DEVICES=0 python3 eval_m2t.py --model_name ./m2t-ft-from-GSPretrained-base/checkpoint-100000
# or
# from our Granularity-Synergy Pre-trained model
CUDA_VISIBLE_DEVICES=0 python3 eval_m2t.py --model_name ./GSPretrained_base/checkpoint-300000 
```
Similarity, we follow MotionGPT to use [nlg-metricverse](https://github.com/disi-unibo-nlp/nlg-metricverse) 
to implement linguistic metrics in motion translation task.


To evaluate our models on the **(Text, Detailed Text)-to-Motion** task, 
please use the following command:
```python
# from our final tdt2m model (Granularity-Synergy Pre-training + Task-Specific Instruction Tuning)
CUDA_VISIBLE_DEVICES=0 python3 eval_tdt2m.py --model_name ./tdt2m-ft-from-GSPretrained-base/checkpoint-300000
# or
# from our Granularity-Synergy Pre-trained model
CUDA_VISIBLE_DEVICES=0 python3 eval_tdt2m.py --model_name ./GSPretrained_base/checkpoint-300000 
```

To evaluate our models on the **Motion-to-Detailed Text** task, 
please use the following command:
```python
# from our final tdt2m model (Granularity-Synergy Pre-training + Task-Specific Instruction Tuning)
CUDA_VISIBLE_DEVICES=0 python3 eval_m2dt.py --model_name ./m2dt-ft-from-GSPretrained-base/checkpoint-300000
# or
# from our Granularity-Synergy Pre-trained model
CUDA_VISIBLE_DEVICES=0 python3 eval_m2dt.py --model_name ./GSPretrained_base/checkpoint-300000 
```
Similarity, we follow MotionGPT to use [nlg-metricverse](https://github.com/disi-unibo-nlp/nlg-metricverse) 
to implement linguistic metrics in motion translation task.



## 6. Train Your Own Model

To pretrain our Granularity-Synergy Pre-trained model, 
please use the following command:
```python
CUDA_VISIBLE_DEVICES=0 python3 main_pretraining.py --output_dir ./GSPretrained_base
```


To train a model on the **Text-to-Motion** task, 
please use the following command:
```python
# from the T5 series (motion-unaware language models)
CUDA_VISIBLE_DEVICES=0 python3 main_t2m.py --model_name google-t5/t5-base --output_dir ./t2m-ft-from-t5-base
# or
# fine-tune our Granularity-Synergy Pre-trained model
CUDA_VISIBLE_DEVICES=0 python3 main_t2m.py --model_name ./GSPretrained_base/checkpoint-300000 --output_dir ./t2m-ft-from-GSPretrained-base
```


To train a model on the **Motion-to-Text** task, 
please use the following command:
```python
# from the T5 series (motion-unaware language models)
CUDA_VISIBLE_DEVICES=0 python3 main_m2t.py --model_name google-t5/t5-base --output_dir ./m2t-ft-from-t5-base
# or
# fine-tune our Granularity-Synergy Pre-trained model
CUDA_VISIBLE_DEVICES=0 python3 main_m2t.py --model_name ./GSPretrained_base/checkpoint-300000 --output_dir ./m2t-ft-from-GSPretrained-base --max_steps 100000

```


To train a model on the **(Text, Detailed Text)-to-Motion** task, 
please use the following command:
```python
CUDA_VISIBLE_DEVICES=0 python3 main_tdt2m.py --model_name google-t5/t5-base --output_dir ./tdt2m-ft-from-t5-base
```
or to fine-tune our Granularity-Synergy Pre-trained model, 
please use the following command:
```python
CUDA_VISIBLE_DEVICES=0 python3 main_tdt2m.py --model_name ./GSPretrained_base/checkpoint-300000 --output_dir ./tdt2m-ft-from-GSPretrained-base
```


To train a model on the **Motion-to-Detailed Text** task, 
please use the following command:
```python
CUDA_VISIBLE_DEVICES=0 python3 main_m2dt.py --model_name google-t5/t5-base --output_dir ./m2dt-ft-from-t5-base
```
or to fine-tune our Granularity-Synergy Pre-trained model, 
please use the following command:
```python
CUDA_VISIBLE_DEVICES=0 python3 main_m2dt.py --model_name ./GSPretrained_base/checkpoint-300000 --output_dir ./m2dt-ft-from-GSPretrained-base
```





## 7. Visualization
We display some novel applications of our MG-MotionLLM.
- **text-driven fine-grained motion editing**: Temporal Editing (left), Spatial Editing (middle), and Spatial-Temporal Editing (right).

<div align="center">    
  <img src="./assets/editing.png" alt="edit" align=center />
</div>

- **fine-grained captioning** of both whole (up) and partial (bottom) motion sequences, and **motion localization via fine-grained textual description** (middle).

<div align="center">    
  <img src="./assets/novel_apps.png" alt="novel_apps" align=center />
</div>

### Render a motion + caption (this repo)
These three standalone scripts run one motion clip through the model and render it as a 3D skeleton
animation (HumanML3D/SMPL-H body-joint topology) with the caption shown alongside it, for quick
qualitative inspection. They only need torch/transformers/matplotlib (no GloVe vectorizer, evaluator
wrapper, spacy, or bert-score), unlike `eval_m2t.py`/`eval_m2dt.py` which compute corpus-wide metrics.

```python
# Motion-to-Text: caption one motion clip
CUDA_VISIBLE_DEVICES=0 python3 m2t_visualize.py --model_name ./m2t-ft-from-GSPretrained-base --name 000000

# Motion-to-Detailed-Text: generate a fine-grained motion script, synced to the motion
CUDA_VISIBLE_DEVICES=0 python3 m2dt_visualize.py --model_name ./m2dt-ft-from-GSPretrained-base --name 000000

# Both models on the same clip, side by side in one video
CUDA_VISIBLE_DEVICES=0 python3 m2t_and_m2dt_visualize.py \
    --m2t_model_name ./m2t-ft-from-GSPretrained-base --m2dt_model_name ./m2dt-ft-from-GSPretrained-base \
    --name 000000
```

Drop `--name 000000` and `--motion_path` to batch-process every `.npy` you've placed under `./input/`
instead (handy for motions extracted from elsewhere, e.g. a game-engine FBX export) -- and if `./input/`
is empty too, a single random test-split sample is picked instead (add `--sample_seed N` to make that
pick reproducible). Output defaults to MP4 (requires a working `ffmpeg`); pass `--format gif` if `ffmpeg`
isn't available. Each clip gets its own folder: `./visualizations/{m2t,m2dt,m2t_and_m2dt}/{name}/{name}.{mp4,gif}`
next to a `{name}.txt` with the generated (and ground-truth, where available) text.

To try the `./input/` batch path without your own motion data yet, just copy a few files over from the
dataset, e.g.:
```python
cp ./dataset/HumanML3D/new_joint_vecs/000000.npy ./dataset/HumanML3D/new_joint_vecs/000001.npy ./input/
CUDA_VISIBLE_DEVICES=0 python3 m2t_visualize.py --model_name ./m2t-ft-from-GSPretrained-base
```

These three scripts (and any custom motion you drop into `./input/`) always read from `new_joint_vecs/`
rather than the `VQVAE/`/`VQVAE_start0/` folders used during training. `new_joint_vecs/` holds the raw,
continuous joint-position motion -- the only form that can be turned back into a 3D skeleton, and from
which motion tokens can be (re-)computed on the fly for any motion via the VQ-VAE encoder. `VQVAE/` and
`VQVAE_start0/` instead hold token sequences *precomputed* by `scripts/tokenized_motion.py` and
`scripts/tokenized_motion_start0.py` purely to speed up `main_m2t.py`/`main_m2dt.py` training -- they
store integer code indices, not positions, so they can't drive a skeleton render, and they only exist for
the dataset's own files, not for anything you drop into `./input/`.

```python
# Motion-to-Text: caption one motion clip
CUDA_VISIBLE_DEVICES=0 python3 m2t_visualize.py --model_name ./m2t-ft-from-GSPretrained-base

# Motion-to-Detailed-Text: generate a fine-grained motion script, synced to the motion
CUDA_VISIBLE_DEVICES=0 python3 m2dt_visualize.py --model_name ./m2dt-ft-from-GSPretrained-base 

# Both models on the same clip, side by side in one video
CUDA_VISIBLE_DEVICES=0 python3 m2t_and_m2dt_visualize.py \
    --m2t_model_name ./m2t-ft-from-GSPretrained-base --m2dt_model_name ./m2dt-ft-from-GSPretrained-base
```

Note: HumanML3D motion only encodes the 22 SMPL body joints (no finger/hand rotations), so the
rendered skeleton follows SMPL-H's body topology but hands render as simple end-effectors at the wrists.

### Stream a motion + caption to Unity live (this repo)
Same idea as above, but instead of exporting an mp4/gif, these three scripts stream the SMPL pose and
caption to Unity live over a WebSocket, frame by frame, in real time. They take the exact same
`--name`/`--motion_path`/`--split`/`--sample_seed`/`./input/`-batch arguments as the visualize scripts
above -- the only difference is the last step (broadcast over the network instead of rendering a file).

```python
# Motion-to-Text: stream one captioned motion clip (simple caption)
CUDA_VISIBLE_DEVICES=0 python3 m2t_unity_stream.py --model_name ./m2t-ft-from-GSPretrained-base --name 000000

# Motion-to-Detailed-Text: stream one motion with its script, caption synced to the motion
CUDA_VISIBLE_DEVICES=0 python3 m2dt_unity_stream.py --model_name ./m2dt-ft-from-GSPretrained-base --name 000000

# Both captions at once: stream one motion with the simple AND detailed caption on one avatar
CUDA_VISIBLE_DEVICES=0 python3 m2t_and_m2dt_unity_stream.py \
    --m2t_model_name ./m2t-ft-from-GSPretrained-base --m2dt_model_name ./m2dt-ft-from-GSPretrained-base \
    --name 000000
```

```python
# Motion-to-Text: stream one captioned motion clip (simple caption)
CUDA_VISIBLE_DEVICES=0 python3 m2t_unity_stream.py --model_name ./m2t-ft-from-GSPretrained-base

# Motion-to-Detailed-Text: stream one motion with its script, caption synced to the motion
CUDA_VISIBLE_DEVICES=0 python3 m2dt_unity_stream.py --model_name ./m2dt-ft-from-GSPretrained-base

# Both captions at once: stream one motion with the simple AND detailed caption on one avatar
CUDA_VISIBLE_DEVICES=0 python3 m2t_and_m2dt_unity_stream.py \
    --m2t_model_name ./m2t-ft-from-GSPretrained-base --m2dt_model_name ./m2dt-ft-from-GSPretrained-base 
```

**Playback speed.** All three scripts accept `--fps` and `--speed` to control how fast the motion
plays back:
- `--speed` is a multiplier on real-time playback (`0.5` = half speed / slow-mo, `2.0` = double). Use
  this for "slower/faster" -- it's independent of the clip's native rate and keeps captions aligned.
- `--fps` overrides the send frame rate (defaults to the dataset fps: 20 for t2m, 12.5 for kit).

```python
# Same clip at half speed (smooth slow-motion)
CUDA_VISIBLE_DEVICES=0 python3 m2t_and_m2dt_unity_stream.py \
    --m2t_model_name ./m2t-ft-from-GSPretrained-base --m2dt_model_name ./m2dt-ft-from-GSPretrained-base \
    --name 000000 --speed 0.5
```
The Unity client interpolates between received frames at its own render rate (toggle on the
`MotionStreamClient` component), so playback stays smooth at any speed -- there's no need to raise
`--fps` to match Unity's frame rate. Trailing static frames (the person standing still at the end of a
clip) are auto-trimmed so the motion and captions finish together.

Each script is the WebSocket *server*: run it first, it loads the model(s) and then waits ("`[Server]
listening on ws://0.0.0.0:8765 -- waiting for Unity to connect...`") until a Unity WebSocket client
connects to `ws://<this machine>:--port` (default port `8765`; from Windows/Unity that's
`ws://localhost:8765` thanks to WSL2's localhost forwarding). Only once connected does it start
generating/streaming.

Per-frame JSON messages look like:
```json
{"type": "start", "name": "000000", "fps": 20.0, "num_frames": 79, "caption": "a person kicks with their left leg."}
{"type": "frame", "frame": 0, "joints": [[x, y, z], "...22 entries..."], "rotations": [[x, y, z, w], "...22 entries..."], "caption": "..."}
{"type": "end", "name": "000000"}
```
`joints` are 22 global joint *positions* (index 0 = pelvis) ordered to match
`SMPLModifyBones._boneNameToJointIndex` (Pelvis=0 .. R_Wrist=21) in the Unity `smpl_mecanim` project --
feed them straight into `SMPLModifyBones.updateBoneAnglesFromJoints(joints)` per frame. `rotations` are the
same 22 joints' global *orientations* (Unity-frame quaternions, `x,y,z,w`); see the note below.
`m2dt_unity_stream.py` sends whichever body-part snippet is active as `caption`; `m2t_and_m2dt_unity_stream.py`
sends both models' text as `caption_m2t`/`caption_m2dt` on every frame, since there's one shared avatar to
drive. The Unity overlay shows a single `caption` unlabeled, or both captions labeled `Simple:` (M2T) /
`Detail:` (M2DT) when both are present.

Note on positions vs. rotations: the avatar is **driven by positions**. `motion_to_unity_joints()` in
`utils/unity_stream.py` streams joint positions (`recover_from_ric`, X-mirrored into Unity's left-handed
frame), and the Unity side rebuilds each bone's rotation by aiming it at its child joint (see
`SMPLModifyBones.updateBoneAnglesFromJoints`). The reason rotations aren't used to drive bones is that
HumanML3D/T2M's per-joint *local* cont6d rotations follow a different forward-kinematics convention than
Unity's `Transform.localRotation` (the bone offset is rotated by the *child's* accumulated global rotation,
not the parent's), so they can't be applied to bones directly; positions are convention-free.

Each frame *also* carries `rotations` from `motion_to_unity_joint_rotations()`: every joint's **global**
orientation, recovered from the same cont6d channels by forward kinematics and X-mirrored into Unity's
frame. Being global (not local) these are likewise convention-free, and they additionally carry the
per-bone **axial twist** that positions cannot express (e.g. forearm pronation). They are verified to
agree with the streamed positions to `< 0.04°` (bone directions), and are available for a rotation-driven
retarget that recovers that twist (vs. the position-based aiming the avatar uses today).


## 8. Acknowledgement
We appreciate helps from the following public code like 
* [MotionGPT](https://github.com/qiqiApink/MotionGPT)
* [MotionGPT](https://github.com/OpenMotionLab/MotionGPT)
* [TM2T](https://github.com/EricGuo5513/TM2T)
* [HumanML3D](https://github.com/EricGuo5513/HumanML3D)
* [T2M-GPT](https://github.com/Mael-zys/T2M-GPT)



## 9. Bibtex
If you use our code in your research, kindly cite our work:

```bibtex
@InProceedings{Wu_2025_CVPR,
    author    = {Wu, Bizhu and Xie, Jinheng and Shen, Keming and Kong, Zhe and Ren, Jianfeng and Bai, Ruibin and Qu, Rong and Shen, Linlin},
    title     = {MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {27849-27858}
}

@article{wu2025mg,
  title={MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities},
  author={Wu, Bizhu and Xie, Jinheng and Shen, Keming and Kong, Zhe and Ren, Jianfeng and Bai, Ruibin and Qu, Rong and Shen, Linlin},
  journal={arXiv preprint arXiv:2504.02478},
  year={2025}
}
```
