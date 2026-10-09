<div align="center">

# DreamTrue — Unified Multi-View World Model

This component is part of the [DreamTrue](https://github.com/brave-eai/DreamTrue) release; see the [main README](https://github.com/brave-eai/DreamTrue/blob/main/README.md) for the project overview.

[Paper](https://arxiv.org/abs/2610.12468) · [Project Page](https://brave-eai.github.io/DreamTrue/) · [GitHub](https://github.com/brave-eai/DreamTrue) · [Data & Checkpoints](https://modelscope.cn/datasets/huoxingdawang/DreamTrue)

</div>

DreamTrue `wmvideo` contains the public inference release of the video generator: a focused Wan2.1 V2V-VACE 14B runtime with a three-view launcher for action-faithful robot video prediction. The dual-LoRA checkpoint is released [here](https://modelscope.cn/datasets/huoxingdawang/DreamTrue/tree/master/wmvideo/checkpoint-wmvideo.safetensors); base model weights and training code are not included.

## Installation

Python 3.10 or newer is required. Install a PyTorch build matching your CUDA or ROCm runtime first, then install the repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

For CUDA environments that use DeepSpeed, install the additional dependencies in `requirements_cuda.txt`.

## Downloads

| Artifact | Link |
| --- | --- |
| Wan2.1-VACE-14B base weights | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) · [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B) |
| Released dual-LoRA checkpoint `wmvideo/checkpoint-wmvideo.safetensors` | [ModelScope](https://modelscope.cn/datasets/huoxingdawang/DreamTrue/tree/master/wmvideo/checkpoint-wmvideo.safetensors) |
| Condition mini package `condition/condition-mini.tar.gz` | [ModelScope](https://modelscope.cn/datasets/huoxingdawang/DreamTrue/tree/master/condition/condition-mini.tar.gz) |

## Inference

### Set model and checkpoint paths

The launchers run in offline mode and do not download weights. Arrange the base weights and the released checkpoint from Downloads below `MODEL_PATH`:

```bash
export MODEL_PATH=/path/to/models
export DUAL_LORA_CHECKPOINT=/path/to/checkpoint-wmvideo.safetensors
```

Override individual paths with `VACE_MODEL_PATH`, `I2V_MODEL_PATH`, or `TOKENIZER_PATH`. Set `OUTPUT_PATH` to choose where videos and logs are written.

### Configure data roots

The shipped manifests cover the paper's cross-embodiment rollout cases. The rendered condition HDF5 files are released [here](https://modelscope.cn/datasets/huoxingdawang/DreamTrue/tree/master/condition/condition-mini.tar.gz); unpack them with `tar xzf`. The raw datasets come from their upstream projects; upstream links are listed in the [calibration README](https://github.com/brave-eai/DreamTrue/tree/main/calibration). Manifest paths use environment variables, expanded by the loader at runtime (`$VAR` / `${VAR}`):

```bash
# raw datasets
export AGIBOT_DATA_ROOT=/data/AgiBotWorld-Beta-hfd
export DROID_DATA_ROOT=/data/droid_raw
export ROBOMIND_DATA_ROOT=/data/RoboMIND2.0-Franka-Part-2
# rendered conditions
export DREAMTRUE_RELEASE=/data/DreamTrue
export AGIBOT_CONDITION_ROOT=$DREAMTRUE_RELEASE/condition/conditions/agibot
export DROID_CONDITION_ROOT=$DREAMTRUE_RELEASE/condition/conditions/droid
export ROBOMIND_CONDITION_ROOT=$DREAMTRUE_RELEASE/condition/conditions/robomind
```

### Dry-run before loading assets

Use dry-run mode to inspect the assembled command without loading model weights or dataset files:

```bash
V2V_INF_DRY_RUN=1 \
MODEL_PATH=/path/to/models \
DUAL_LORA_CHECKPOINT=/path/to/checkpoint-wmvideo.safetensors \
NUM_PROCESSES=1 \
bash Inference/V2V_VACE/inference_threeviews_mixed_20260623.sh
```

### Fixed-length three-view prediction

```bash
MODEL_PATH=/path/to/models \
DUAL_LORA_CHECKPOINT=/path/to/checkpoint-wmvideo.safetensors \
DATASET_CONFIG=$DREAMTRUE_RELEASE/condition/data-config/config.yaml \
MIXED_SOURCE_NAMES="agibot droid" \
MAX_SAMPLES=2 \
NUM_PROCESSES=1 \
bash Inference/V2V_VACE/inference_threeviews_mixed_20260623.sh
```

Useful selectors include `MIXED_SOURCE_NAMES`, `SAMPLES_PER_SOURCE`, `MAX_SAMPLES`, and `OUTPUT_PATH`.

### Data configuration

The launcher reads the dataset config shipped with the [condition mini package](https://modelscope.cn/datasets/huoxingdawang/DreamTrue/tree/master/condition/condition-mini.tar.gz) by default (`${DREAMTRUE_RELEASE:-/data/DreamTrue}/condition/data-config/config.yaml`; override with `DATASET_CONFIG`). The YAML file is a list of source entries, each containing:

- `path`: a JSON manifest with one record per clip, relative to the config file;
- `type`: one of `agibot`, `droid`, `robomind`, or `robotwin`;
- `target_size`: the sampling budget used by the mixed-source selector;
- optional loader flags such as `threeviews_concat`, `use_plucker`, and `detail_prompt`.

JSON manifests contain metadata such as `data_key`, frame ranges, and condition filenames. Copy a manifest when you need a different local case: keep the same dataset `type`, and update `data_key`, `start`, `end`, `fps`, and the condition filename/path as required by the local dataset layout.

## Supported data and views

| Source | Default view layout in the manifests |
| --- | --- |
| AgiBotWorld-Beta | Head · left gripper · right gripper |
| DROID | Left exterior · right exterior · gripper |
| RoboMIND 2.0 | Head · left gripper · right gripper |
| RoboTwin 2.0 | Head · left gripper · right gripper |

The paper also evaluates unseen real-world scenes and an unseen robot embodiment without additional fine-tuning. Those evaluation assets are not bundled in this repository.

## Related releases

- **[calibration](https://github.com/brave-eai/DreamTrue/tree/main/calibration)** — offline geometric calibration and condition rendering; produces the condition HDF5 files consumed here.
- **[reward](https://github.com/brave-eai/DreamTrue/tree/main/reward)** — embodied video reward model used for counterfactual post-training feedback.
- **[repository root](https://github.com/brave-eai/DreamTrue)** — project page and release overview.
- **Data & checkpoints** — [huoxingdawang/DreamTrue on ModelScope](https://modelscope.cn/datasets/huoxingdawang/DreamTrue).
