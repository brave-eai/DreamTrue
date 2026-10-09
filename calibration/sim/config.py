import dataclasses
import os
from typing import Literal

from utils import Pose


@dataclasses.dataclass
class SapienRendererConfig:
    camera_shader_dir: str = ''
    ray_tracing_samples_per_pixel: int = 64
    ray_tracing_path_depth: int = 16
    ray_tracing_denoiser: Literal['none', 'oidn', 'optix'] = 'oidn'


@dataclasses.dataclass
class SapienViewerConfig:
    pos: Pose = dataclasses.field(default_factory=lambda: Pose(p=(2.0, 0.0, 1.0), q=(0.000787377, -0.149438, 0.000118971, 0.988771)))
    fovy: float = 93
    near: float = 0.1
    far: float = 100
    display: str | None = dataclasses.field(default_factory=lambda: os.getenv('DISPLAY'))


@dataclasses.dataclass
class SapienGroundConfig:
    position: float | None = 0.0
