<div align="center">

# DreamTrue — Embodied Video Reward Model

This component is part of the [DreamTrue](https://github.com/brave-eai/DreamTrue) release; see the [main README](https://github.com/brave-eai/DreamTrue/blob/main/README.md) for the project overview.

[Paper](https://arxiv.org/abs/2610.12468) · [Project Page](https://brave-eai.github.io/DreamTrue/) · [GitHub](https://github.com/brave-eai/DreamTrue) · [Data & Checkpoints](https://modelscope.cn/datasets/huoxingdawang/DreamTrue)

</div>

DreamTrue `reward` provides the embodied video reward model: a Qwen3.5 vision-language model fine-tuned on a failure taxonomy and scored by reading per-label token probabilities off the decoder. It grades a video on three axes in a single response — **L1 embodiment**, **L2 object**, **L3 interaction** — and every label's probability yields a continuous severity score usable as a reward.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra cuda      # NVIDIA
uv sync --extra rocm      # AMD
```

Note: the `vllm/` directory is an intentionally empty **stub package** required by the ms-swift
import chain when running from the repository root. Nothing here uses a real vLLM installation.

## Downloads

| Artifact | Link |
| --- | --- |
| Released checkpoint `reward/checkpoint/` | [ModelScope](https://modelscope.cn/datasets/huoxingdawang/DreamTrue/tree/master/reward/checkpoint) |
| Mini pack `reward/reward-data-mini.tar.gz` | [ModelScope](https://modelscope.cn/datasets/huoxingdawang/DreamTrue/tree/master/reward/reward-data-mini.tar.gz) |

The mini pack is a small self-contained example used by the examples below.

## Data format

Splits are sharegpt-style JSON files. Each row carries the video path and a compact `meta` block;
prompts and answers are constructed at load time by `plugins/qwen_video_aug.py` from the selected
`configs/embodied/*.yaml`, so evaluation and training share one view definition. Video paths are
relative (`source/...`) and resolved against `--data-root`.

## Offline evaluation

```bash
VIEWAUG_CONFIG_EVAL=configs/embodied/eval_l123.yaml VIDEO_HEIGHT=240 FPS=5 CUDA_VISIBLE_DEVICES=0 \
python -m scripts.dataconv.eval_freegen \
    --model <MODEL_DIR> --adapter <ADAPTER_DIR> --nothink \
    --json-file <SPLIT_JSON> --data-root <DATA_ROOT> \
    --save-dir eval_results/<run>/<ckpt>/test --batch-size 1

python -m scripts.dataconv.summarize_eval --eval-dir eval_results
```

`VIDEO_HEIGHT=240` (and `FPS=5`) must match the training-time preprocessing — without it the
evaluation views differ from training and predictions drift.

Example with the mini pack:

```bash
mkdir mini_pack && tar xzf reward-data-mini.tar.gz -C mini_pack

VIEWAUG_CONFIG_EVAL=configs/embodied/eval_l123.yaml VIDEO_HEIGHT=240 FPS=5 CUDA_VISIBLE_DEVICES=0 \
python -m scripts.dataconv.eval_freegen \
    --model <CHECKPOINT_DIR> --nothink \
    --json-file mini_pack/split/test_mini.json --data-root mini_pack \
    --save-dir /tmp/mini_eval --batch-size 1
```

Outputs per run: `results.json` (per-video predictions + parsed labels + soft scores) and pooled
metrics (macro-F1 / accuracy / probability-MAE) in `summary.json` + `summary_per_ckpt.csv`.

## Inference on unlabeled videos

```bash
VIEWAUG_CONFIG_EVAL=configs/embodied/eval_l123.yaml VIDEO_HEIGHT=240 FPS=5 CUDA_VISIBLE_DEVICES=0 \
python -m scripts.dataconv.infer_videos \
    --model <MODEL_DIR> --dirs /path/to/videos --dim L123 \
    --save-dir eval_results/infer/run1 --batch-size 1
```

One JSONL line per video, with per-dim `soft` scores (`E_severity` = probability-weighted
severity, usable directly as a reward). Re-running resumes: finished videos are skipped.

Example with the mini pack (any clip folder can serve as an unlabeled directory):

```bash
VIEWAUG_CONFIG_EVAL=configs/embodied/eval_l123.yaml VIDEO_HEIGHT=240 FPS=5 CUDA_VISIBLE_DEVICES=0 \
python -m scripts.dataconv.infer_videos \
    --model <CHECKPOINT_DIR> --dirs mini_pack/source/pred/data-v12x/demo_20260803_230108 --dim L123 \
    --save-dir /tmp/mini_infer --batch-size 1
```

## Related releases

- **[wmvideo](https://github.com/brave-eai/DreamTrue/tree/main/wmvideo)** — the Wan2.1 V2V-VACE 14B video generator whose outputs this reward model grades.
- **[calibration](https://github.com/brave-eai/DreamTrue/tree/main/calibration)** — offline geometric calibration and condition rendering.
- **[repository root](https://github.com/brave-eai/DreamTrue)** — project page and release overview.
- **Data & checkpoints** — [huoxingdawang/DreamTrue on ModelScope](https://modelscope.cn/datasets/huoxingdawang/DreamTrue).

## Coming soon

- Release full annotation data . 