<div align="center">

# DreamTrue: Action-Faithful Robot World Model with Counterfactual Post-Training

Junyan Li<sup>1*</sup> · Ruizhi Li<sup>1*</sup> · Yu Liu<sup>2†</sup> · Xiangshuo Liu<sup>1§</sup> · Mingchao Sun<sup>2</sup> · Hongyu Pan<sup>2</sup> · Mu Xu<sup>2</sup> · Lue Fan<sup>1†✉</sup> · Zhaoxiang Zhang<sup>1✉</sup>

<sup>1</sup> NLPR, Institute of Automation, Chinese Academy of Sciences (CASIA) · <sup>2</sup> Amap, Alibaba Group

<sup>*</sup> Equal contribution · <sup>†</sup> Co-Project Leads · <sup>§</sup> Xiangshuo Liu was an intern at CASIA during this work · <sup>✉</sup> Corresponding authors

[Paper](https://arxiv.org/abs/2610.12468) · [Project Page](https://brave-eai.github.io/DreamTrue/) · [GitHub](https://github.com/brave-eai/DreamTrue) · [Data & Checkpoints](https://modelscope.cn/datasets/huoxingdawang/DreamTrue)

![DreamTrue overview](docs/assets/B4h-rK64.svg)

</div>

DreamTrue is a multi-view, cross-embodiment robot world model for predicting action-faithful and physically plausible futures. The accompanying paper combines offline geometric calibration, shared image-space action conditions, and counterfactual post-training with embodied video rewards. This repository hosts the project page and all three public code components on `main`; see also [Data & Checkpoints](https://modelscope.cn/datasets/huoxingdawang/DreamTrue) and the [project page](https://brave-eai.github.io/DreamTrue/).

## Release contents

| Component | Release |
| --- | --- |
| [`calibration`](https://github.com/brave-eai/DreamTrue/tree/main/calibration) | [Code](https://github.com/brave-eai/DreamTrue/tree/main/calibration) · [Data](https://modelscope.cn/datasets/huoxingdawang/DreamTrue) |
| [`wmvideo`](https://github.com/brave-eai/DreamTrue/tree/main/wmvideo) | [Code](https://github.com/brave-eai/DreamTrue/tree/main/wmvideo) · [Data](https://modelscope.cn/datasets/huoxingdawang/DreamTrue) |
| [`reward`](https://github.com/brave-eai/DreamTrue/tree/main/reward) | [Code](https://github.com/brave-eai/DreamTrue/tree/main/reward) · [Data](https://modelscope.cn/datasets/huoxingdawang/DreamTrue) |

*Full data will be released in a future update.*

## Getting started

- **Video generation inference** → [`wmvideo` README](https://github.com/brave-eai/DreamTrue/tree/main/wmvideo)
- **Calibration and condition rendering** → [`calibration` README](https://github.com/brave-eai/DreamTrue/tree/main/calibration)
- **Reward evaluation** → [`reward` README](https://github.com/brave-eai/DreamTrue/tree/main/reward)
- **Project page** → <https://brave-eai.github.io/DreamTrue/>

## Citation

If DreamTrue is useful in your research, please cite:

```bibtex
@article{li2026dreamtrue,
  title   = {DreamTrue: Action-Faithful Robot World Model with Counterfactual Post-Training},
  author  = {Li, Junyan and Li, Ruizhi and Liu, Yu and Liu, Xiangshuo and Sun, Mingchao and Pan, Hongyu and Xu, Mu and Fan, Lue and Zhang, Zhaoxiang},
  year    = {2026},
  note    = {Manuscript under review},
  eprint = {2610.12468},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url    = {https://arxiv.org/abs/2610.12468}
}
```
