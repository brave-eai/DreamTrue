import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import numpy as np

from .BaseDataset import BaseDataset, ModifiedDataset

DEFAULT_DELTA_RATE_LIMIT = 0.01  # 修正速率上限，rad/帧
DEFAULT_SPEED_LIMIT = 3.0


class OverlayFailedException(Exception):
    ...


@dataclass(frozen=True)
class OverlaySegment:
    """原 episode 帧坐标下的半开区间 [start, end)；align 为夹爪闭合帧（缺省时按夹爪信号检测）。"""
    arm: str
    start: int
    end: int
    align: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> 'OverlaySegment':
        return cls(
            arm=str(d['arm']),
            start=int(d['start']),
            end=int(d['end']),
            align=None if d.get('align') is None else int(d['align']),
        )

    @property
    def dict(self) -> dict:
        return {'arm': self.arm, 'start': self.start, 'end': self.end, 'align': self.align}


def first_close_frame(signal: np.ndarray, threshold: float) -> int | None:
    """段内「先张开、后首次闭合」的相对帧下标；未发生张开→闭合则返回 None。"""
    opened = signal >= threshold
    closed = signal < threshold
    if not opened.any() or not closed.any():
        return None
    first_open = int(np.argmax(opened))
    rest = closed[first_open:]
    if not rest.any():
        return None
    return first_open + int(np.argmax(rest))


@dataclass(frozen=True)
class OverlayPlan:
    """n = 参考段长度；window 为第 0 帧修正量 Δ 的衰减窗宽（blend 缺省 = 参考段闭合前帧数 // 2）；
    copy_frames 为输出每帧对应的复制段绝对帧号（浮点）；cols/delta 为复制手的 qpos 列与第 0 帧修正量。"""
    n: int
    window: int
    ref: OverlaySegment
    copy: OverlaySegment
    copy_frames: np.ndarray
    cols: np.ndarray
    delta: np.ndarray


def build_overlay_plan(
    qpos: np.ndarray,
    dims: dict[str, slice | np.ndarray],
    segments: Sequence[OverlaySegment],
    gripper_open_qpos: float,
    blend: int | None = None,
    delta_rate_limit: float = DEFAULT_DELTA_RATE_LIMIT,
    speed_limit: float = DEFAULT_SPEED_LIMIT,
) -> OverlayPlan:
    """校验两段（较早的为参考段）并给出变速映射；数据性失败抛 OverlayFailedException。
    第 0 帧修正量 Δ 在 window = min(blend or 参考段闭合前帧数 // 2, n) 帧内线性衰减，
    其最大逐帧变化率 max|Δ| / window 不得超过 delta_rate_limit（rad/帧）。
    dims 为 _qpos_dims（arm_{arm}/gripper_{arm} 键）；生成脚本与 Modifier 共用。"""
    n_src = len(qpos)
    n_cols = qpos.shape[1]

    def columns(kind: str, arm: str) -> np.ndarray:
        return np.arange(n_cols)[dims[f'{kind}_{arm}']]

    ref, copy = sorted(segments, key=lambda s: s.start)
    for seg in (ref, copy):
        if not 0 <= seg.start < seg.end <= n_src:
            raise OverlayFailedException(f'{seg.arm!r} segment [{seg.start}, {seg.end}) outside [0, {n_src})')
    if copy.start < ref.end:
        raise OverlayFailedException(f'{ref.arm!r} [{ref.start}, {ref.end}) overlaps {copy.arm!r} [{copy.start}, {copy.end})')

    threshold = 0.02 * gripper_open_qpos
    close: dict[str, int] = {}
    for seg in (ref, copy):
        if seg.align is None:
            signal = qpos[seg.start:seg.end, columns('gripper', seg.arm)].mean(axis=1)
            offset = first_close_frame(signal, threshold)
            if offset is None:
                raise OverlayFailedException(f'{seg.arm!r} segment [{seg.start}, {seg.end}) has no gripper open->close event (use "align")')
        else:
            if not seg.start <= seg.align < seg.end:
                raise OverlayFailedException(f'{seg.arm!r} align {seg.align} not in [{seg.start}, {seg.end})')
            offset = seg.align - seg.start
        close[seg.arm] = offset

    reach_ref, reach_copy = close[ref.arm], close[copy.arm]
    tail_ref = (ref.end - ref.start) - reach_ref
    tail_copy = (copy.end - copy.start) - reach_copy
    if reach_ref < 1 or reach_copy < 1 or tail_ref < 2 or tail_copy < 2:
        raise OverlayFailedException(f'{copy.arm!r} phases too short to warp: reach {reach_copy}->{reach_ref}, tail {tail_copy}->{tail_ref} (need reach >= 1, tail >= 2)')
    # 源/输出帧间隔数交叉相乘判定，恰好等于 speed_limit 的配对含边界通过
    for name, output_len, source_len in [('reach', reach_ref, reach_copy), ('tail', tail_ref - 1, tail_copy - 1)]:
        if source_len > speed_limit * output_len or output_len > speed_limit * source_len:
            factor = max(source_len / output_len, output_len / source_len)
            raise OverlayFailedException(f'{copy.arm!r} {name} speed change {factor:.3f}x > limit {speed_limit}x ({source_len} src -> {output_len} out frames)')

    window = min(reach_ref // 2 if blend is None else blend, ref.end - ref.start)
    if window < 1:
        raise OverlayFailedException(f'{copy.arm!r} blend window {window} < 1 frame: cannot spread the frame-0 delta (reach {reach_ref})')

    cols = np.union1d(columns('arm', copy.arm), columns('gripper', copy.arm))
    delta = qpos[ref.start, cols] - qpos[copy.start, cols]
    worst = float(np.max(np.abs(delta)))
    rate = worst / window
    if rate > delta_rate_limit:
        raise OverlayFailedException(f'{copy.arm!r} hand frame-0 delta {worst:.4f} rad over {window} frames = {rate:.5f} rad/frame > limit {delta_rate_limit} rad/frame: pick another pair')

    copy_close = copy.start + reach_copy
    copy_frames = np.concatenate([
        np.linspace(copy.start, copy_close, reach_ref + 1)[:-1],
        np.linspace(copy_close, copy.end - 1, tail_ref),
    ])
    return OverlayPlan(n=ref.end - ref.start, window=window, ref=ref, copy=copy, copy_frames=copy_frames, cols=cols, delta=delta)


class OverlayArmsDataset(ModifiedDataset):
    """把同一 episode 里先后两段 Pick 合成一组「双手同时抓取」：参考段（较早）整段原样输出，复制段
    （较晚）变速搬到参考段时间轴，使两夹爪同帧闭合、复制手末帧对齐参考段末帧。仅复制手列被改写
    （第 0 帧差 Δ 在闭合前段（reach）的前半段内线性衰减抹平）；其余列与 indices/position/rgbs
    取参考段原值；不提供 endpose。"""

    def __init__(
        self,
        dataset: BaseDataset,
        segments: list[dict | OverlaySegment],
        name: str,
        blend: int | None = None,
        delta_rate_limit: float = DEFAULT_DELTA_RATE_LIMIT,
        speed_limit: float = DEFAULT_SPEED_LIMIT,
        *args,
        **kwargs,
    ):
        super().__init__(dataset=dataset, *args, **kwargs)
        self._segments = [s if isinstance(s, OverlaySegment) else OverlaySegment.from_dict(s) for s in segments]
        self._modifier_name = name
        self._blend = None if blend is None else int(blend)
        self._delta_rate_limit = float(delta_rate_limit)
        self._speed_limit = float(speed_limit)
        assert len(self._segments) == 2, f'expected exactly 2 segments, got {len(self._segments)}'
        assert len({s.arm for s in self._segments}) == 2, (f'segments must drive two different hands, got {[s.arm for s in self._segments]}')
        unknown = {s.arm for s in self._segments} - set(self._endpose_arms)
        assert not unknown, f'endpose arms {sorted(self._endpose_arms)} do not cover {sorted(unknown)}'
        missing = {f'{kind}_{s.arm}' for s in self._segments for kind in ('arm', 'gripper')} - set(self._qpos_dims)
        assert not missing, f'_qpos_dims does not declare {sorted(missing)} for the segment hands'

    _first_close_frame = staticmethod(first_close_frame)

    @property
    def modifier_name(self) -> str:
        return self._modifier_name

    @cached_property
    def dataset_name(self) -> list[str]:
        payload = {
            'segments': [s.dict for s in self._segments],
            'blend': self._blend,
            'delta_rate_limit': self._delta_rate_limit,
            'speed_limit': self._speed_limit,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
        return self._d.dataset_name + [f'{self._modifier_name}_{digest}']

    @cached_property
    def _plan(self) -> OverlayPlan:
        return build_overlay_plan(
            qpos=self._d.qpos,
            dims=self._qpos_dims,
            segments=self._segments,
            gripper_open_qpos=self.gripper_open_qpos,
            blend=self._blend,
            delta_rate_limit=self._delta_rate_limit,
            speed_limit=self._speed_limit,
        )

    @cached_property
    def _len(self) -> int:
        return self._plan.n

    @cached_property
    def qpos(self) -> np.ndarray:
        qpos = self._d.qpos
        plan = self._plan
        results = qpos[plan.ref.start:plan.ref.end].copy()
        cols = plan.cols
        lo = np.floor(plan.copy_frames).astype(np.int64)
        frac = plan.copy_frames - lo
        hi = np.minimum(lo + 1, len(qpos) - 1)
        values = qpos[lo][:, cols] * (1.0 - frac)[:, None] + qpos[hi][:, cols] * frac[:, None]
        if plan.window > 0:
            values[:plan.window] += (1.0 - np.arange(plan.window) / plan.window)[:, None] * plan.delta
        results[np.ix_(np.arange(plan.n), cols)] = values
        return results

    @cached_property
    def indices(self) -> np.ndarray:
        plan = self._plan
        return self._d.indices[plan.ref.start:plan.ref.end]

    @cached_property
    def position(self) -> np.ndarray:
        plan = self._plan
        return self._d.position[plan.ref.start:plan.ref.end]

    @cached_property
    def rgbs(self) -> OrderedDict[str, np.ndarray]:
        plan = self._plan
        return OrderedDict((k, v[plan.ref.start:plan.ref.end]) for k, v in self._d.rgbs.items())

    @cached_property
    def endpose(self) -> np.ndarray:
        raise NotImplementedError('OverlayArmsDataset does not provide endpose.')
