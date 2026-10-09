import abc
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from scripts.merge.scan import ClipContext


class _Filter(abc.ABC):

    @abc.abstractmethod
    def match(self, ctx: ClipContext) -> dict[str, str] | None:
        """不匹配返回 None，匹配返回命名捕获组（无捕获组时为空 dict），判断一律用 `is not None`。"""
        raise NotImplementedError


class _RegFilter(_Filter, abc.ABC):

    def __init__(self, *, pattern: str):
        self._regexp = re.compile(pattern)

    def _match_one(self, value: str) -> dict[str, str] | None:
        m = self._regexp.match(value)
        return None if m is None else {k: v for k, v in m.groupdict().items() if v is not None}

    def _match_any(self, values: list[str]) -> dict[str, str] | None:
        for value in values:
            captures = self._match_one(value)
            if captures is not None:
                return captures
        return None


class SkillFilter(_RegFilter):

    def match(self, ctx: ClipContext) -> dict[str, str] | None:
        return self._match_any(ctx.skill)


class KeywordFilter(_RegFilter):

    def match(self, ctx: ClipContext) -> dict[str, str] | None:
        return self._match_any(ctx.action_text)


class TaskNameFilter(_RegFilter):

    def match(self, ctx: ClipContext) -> dict[str, str] | None:
        return self._match_any(ctx.task_name)


class DataKeyFilter(_RegFilter):

    def match(self, ctx: ClipContext) -> dict[str, str] | None:
        return self._match_one(ctx.data_key)


class DataKeyListFilter(_Filter):

    def __init__(self, *, path: str):
        with open(path, encoding='utf-8') as f:
            self._keys = {line.strip() for line in f if line.strip()}

    def match(self, ctx: ClipContext) -> dict[str, str] | None:
        return {} if ctx.data_key in self._keys else None


class DataRootFilter(_RegFilter):

    def match(self, ctx: ClipContext) -> dict[str, str] | None:
        return self._match_one(ctx.data_root)


class ConditionFileFilter(_RegFilter):

    def match(self, ctx: ClipContext) -> dict[str, str] | None:
        return self._match_one(ctx.condition_file)


class DataClassFilter(_RegFilter):

    def match(self, ctx: ClipContext) -> dict[str, str] | None:
        return self._match_one(ctx.data_class)


class IndicesCoverClipFilter(_Filter):

    def __init__(self):
        pass

    def match(self, ctx: ClipContext) -> dict[str, str] | None:
        indices = ctx.condition_indices
        if indices is None or ctx.end <= ctx.start:
            return None
        in_clip = indices[(indices >= ctx.start) & (indices < ctx.end)]
        covered = len(in_clip) >= ctx.end - ctx.start and len(np.unique(in_clip)) == ctx.end - ctx.start
        return {} if covered else None


class ClipStartFilter(_Filter):
    """Match clips whose start frame is one of the given values."""

    def __init__(self, *, starts: list[int]):
        self._starts = set(starts)

    def match(self, ctx: ClipContext) -> dict[str, str] | None:
        return {} if ctx.start in self._starts else None


class PositionChangedFilter(_Filter):
    """clip 内底盘位姿相对首帧的最大偏移超阈值则匹配。"""

    def __init__(self, *, threshold: float = 0.02, rotation_threshold: float = 0.05):
        self._threshold = threshold
        self._rotation_threshold = rotation_threshold

    def match(self, ctx: ClipContext) -> dict[str, str] | None:
        pos = ctx.position[ctx.start:ctx.end]
        if pos.shape[0] < 2:
            return None
        dxyz = np.linalg.norm(pos[:, :3] - pos[0, :3], axis=-1).max()
        dot = np.clip(np.abs(pos[:, 3:] @ pos[0, 3:]), -1.0, 1.0)
        rot = float((2 * np.arccos(dot)).max())
        return {} if (dxyz > self._threshold or rot > self._rotation_threshold) else None


__FILTER_REGISTRY__: dict[str, type[_Filter]] = {c.__name__: c for c in locals().values() if isinstance(c, type) and issubclass(c, _Filter) and not bool(c.__abstractmethods__)}


def _build_filter(config: dict) -> _Filter:
    config = dict(config)
    cls = __FILTER_REGISTRY__[config.pop('_class')]
    return cls(**config)


@dataclass
class FilterNode():
    match: _Filter | None = None
    negate: bool = False
    mode: str = 'first'
    children: list['FilterNode'] = field(default_factory=list)
    output: str | None = None

    def _match_captures(self, ctx: ClipContext) -> dict[str, str] | None:
        if self.match is None:
            return {}
        captures = self.match.match(ctx)
        if self.negate:
            return {} if captures is None else None
        return captures

    def apply(self, ctx: ClipContext, captures: dict[str, str] | None = None) -> list[str | None]:
        own = self._match_captures(ctx)
        if own is None:
            return []
        captures = {**(captures or {}), **own}
        if self.children:
            results: list[str | None] = []
            for child in self.children:
                groups = child.apply(ctx, captures)
                results.extend(groups)
                if groups and self.mode == 'first':
                    break
            return results
        return [self.output if self.output is None else self.output.format_map((defaultdict(str, captures)))]


def build_filter_node(config: dict) -> FilterNode:
    match = _build_filter(config['match']) if 'match' in config else None
    children = [build_filter_node(c) for c in config.get('children', [])]
    return FilterNode(
        match=match,
        negate=config.get('negate', False),
        mode=config.get('mode', 'first'),
        children=children,
        output=config.get('output'),
    )
