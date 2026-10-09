<div align="center">

# DreamTrue — Geometric Calibration & Condition Pipeline

This component is part of the [DreamTrue](https://github.com/brave-eai/DreamTrue) release; see the [main README](https://github.com/brave-eai/DreamTrue/blob/main/README.md) for the project overview.

[Paper](https://arxiv.org/abs/2610.12468) · [Project Page](https://brave-eai.github.io/DreamTrue/) · [GitHub](https://github.com/brave-eai/DreamTrue) · [Data & Checkpoints](https://modelscope.cn/datasets/huoxingdawang/DreamTrue)

</div>

DreamTrue `calibration` provides the **condition-data generation** stage:

- condition rendering for the downstream world-model repositories — RGB,
  depth, mask, segmentation, camera poses and robot state per episode;
- task-list tooling and dataset-config merging used by the downstream
  training / inference pipelines;
- support for **AgiBotWorld-Beta**, **DROID** and **RoboMIND 2.0**.

## Installation

Linux + NVIDIA GPU, Python 3.12, `git` + `git-lfs`, and
[`uv`](https://docs.astral.sh/uv/):

```bash
git lfs install
git clone https://github.com/brave-eai/DreamTrue.git && cd DreamTrue
git submodule update --init --recursive
cd calibration
git lfs pull
uv sync --extra cuda
```

For headless rendering:

```bash
export DISPLAY=""
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

## Downloads

| Artifact | Link |
| --- | --- |
| Condition mini package `condition/condition-mini.tar.gz` | [ModelScope](https://modelscope.cn/datasets/huoxingdawang/DreamTrue/tree/master/condition/condition-mini.tar.gz) |

The [condition mini package](https://modelscope.cn/datasets/huoxingdawang/DreamTrue/tree/master/condition/condition-mini.tar.gz)
contains the rendered condition data, the task
lists used to render them, the calibration subset they reference and the
downstream dataset config for the paper's cross-embodiment rollout cases; the
[wmvideo](https://github.com/brave-eai/DreamTrue/tree/main/wmvideo) consumes
`condition/data-config/config.yaml` directly. See `condition/README.md`
inside the package.

### Raw datasets

Download from the upstream projects:

- **AgiBotWorld-Beta** — [HuggingFace](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta)
  / [ModelScope](https://modelscope.cn/datasets/agibot_world/agibot_world_beta)
- **DROID** — <https://droid-dataset.github.io/droid/the-droid-dataset>
- **RoboMIND 2.0 (Franka)**:
  [Part-1](https://modelscope.cn/datasets/X-Humanoid/RoboMIND2.0-Franka-Part-1),
  [Part-2](https://modelscope.cn/datasets/X-Humanoid/RoboMIND2.0-Franka-Part-2),
  [Part-3](https://modelscope.cn/datasets/X-Humanoid/RoboMIND2.0-Franka-Part-3),
  [Part-4](https://modelscope.cn/datasets/X-Humanoid/RoboMIND2.0-Franka-Part-4),
  [Part-5](https://modelscope.cn/datasets/X-Humanoid/RoboMIND2.0-Franka-Part-5)

Expected layouts and episode keys (see also `dataset/README.md`):

- **AgiBotWorld-Beta** (`AgibotDataset`):
  `{root}/proprio_stats/{task}/{episode}/proprio_stats.h5`, videos in
  `observations/{task}/{episode}/videos/{cam}_color.mp4`;
  key `{task}/{episode}` (e.g. `445/789713`); cameras `head`, `hand_left`,
  `hand_right`.
- **DROID** (`DroidDataset`): `{root}/{version}/{lab}/{status}/{date}/{ts}/`
  with `metadata_*.json`, `trajectory.h5`, `recordings/MP4/{serial}.mp4`;
  key `{version}/{lab}/{status}/{ts}`; cameras `left`, `right`, `wrist`.
- **RoboMIND 2.0** (`RobomindFrankaDataset`):
  `{root}/data/{embodiment}/{task}/{success_episodes|failed_episodes}/{ts}/data/trajectory.hdf5`;
  key `{task}/{status}/{ts}`.

## Repository layout

```
calibrate/   condition rendering
dataset/     dataset readers for the supported datasets (see dataset/README.md)
sim/         SAPIEN simulation wrapper: robots, cameras, rendering
utils/       shared utilities (pose, intrinsics, hdf5, upload, ...)
scripts/     task-list tools (tasks/), dataset-config merge (merge/),
             plus print-hdf5 and redis-tmux helpers
tests/       fixtures and schema checks (no private data)
assets/      robot descriptions and meshes (git submodule)
```

## Condition rendering

```bash
# task lists come from the DreamTrue condition mini package,
# or from your own task JSON built with scripts/tasks/
python -m calibrate.condition \
  --task condition-task-selected.json \
  --data_class=AgibotDataset --data_root=/path/to/AgiBotWorld-Beta-hfd \
  --calib_root=/path/to/camera_param --robot_path=agibot/G1_120s/G1_120s.yaml \
  --upload_id=file://./output
```

A rendered condition file can be inspected with
`python scripts/print-hdf5.py file.h5`. Task files are JSON arrays — see the
example in `tests/fixtures/task-example.json`.

## Task lists

Task JSON arrays are the unit consumed by `calibrate.condition`. To build new
lists from a base list, `scripts/tasks/` provides composable stdin/stdout
tools; for example a retreat (counterfactual) variant:

```bash
python -m scripts.tasks.filter-key --exclude '^(351|361)/.*$' -i base.json |
python -m scripts.tasks.filter-random -n 16000 --seed 88 |
python -m scripts.tasks.retreat-task --data_class=AgibotDataset \
    --data_root=/path/to/AgiBotWorld-Beta-hfd \
    --modification-name rr2 --retreat 0.02 --mode random --seed 88 |
python -m scripts.tasks.filter-modified |
python -m scripts.tasks.filter-ik --data_class=AgibotDataset \
    --data_root=/path/to/AgiBotWorld-Beta-hfd \
    -o condition-task-selected-rr2.json
```

## Dataset configs (for training / inference)

Downstream training and inference consume a **dataset config** produced by
`scripts/merge/`: it scans rendered condition files plus the datasets' clip
annotations, filters clips into named groups and writes, per group, a clip
list `{group}.json` (with resolved `condition_path` / `score_path`) and a
`config.yaml` manifest (sha256, target size, sampling interval per group).

```bash
python -m scripts.merge.main scripts/merge/config_example.yaml
```

## Related releases

- **[wmvideo](https://github.com/brave-eai/DreamTrue/tree/main/wmvideo)** — Wan2.1 V2V-VACE 14B inference runtime that consumes the rendered conditions.
- **[reward](https://github.com/brave-eai/DreamTrue/tree/main/reward)** — embodied video reward model used for counterfactual post-training feedback.
- **[repository root](https://github.com/brave-eai/DreamTrue)** — project page and release overview.
- **Data & checkpoints** — [huoxingdawang/DreamTrue on ModelScope](https://modelscope.cn/datasets/huoxingdawang/DreamTrue).

## Coming soon

- Full calibration dataset 
- Fine-tuned SAM3 checkpoint, which is used by the calibration and quality-scoring commands.
- The full calibration pipeline.
