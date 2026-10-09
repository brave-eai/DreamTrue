import json
import os
from collections import OrderedDict
from functools import cached_property, lru_cache
from typing import Iterator, Sequence

import h5py
import numpy as np
from numba import njit

from utils import video_read

from .BaseDataset import ClipInfo, SourceDataset


class AgibotDataset(SourceDataset):

    # 30 Hz：与 observations/{key}/videos/*_color.mp4 的编码 fps 一致。
    fps: int = 30
    cams = ('head', 'hand_left', 'hand_right')

    def __init__(self, root: str, key: str, cams: Sequence[str | tuple[str, tuple[int, int]]], use_state: bool = False, *args, **kwargs):
        """key: '{task_id}/{episode_id}'；use_state=False 取 action/* (遥操作侧)，True 取 state/* (机器人侧)。"""
        super().__init__(*args, **kwargs)
        assert key.count('/') == 1, f'got {key}'
        self._root = root
        self._key = key
        self._cams: list[tuple[str, None | tuple[int, int]]] = [cam if isinstance(cam, tuple) else (cam, None) for cam in cams]
        self._use_state = use_state

    @property
    def root(self):
        return self._root

    @property
    def key(self):
        return self._key

    @cached_property
    def _layout_suffix(self) -> str:
        """root 下存在 proprio_stats_untar 目录则走 '_untar' 解包布局；按 root 整体定，不混合。"""
        return self._resolve_layout_suffix(self._root)

    @staticmethod
    def _resolve_layout_suffix(root: str) -> str:
        return '_untar' if os.path.isdir(os.path.join(root, 'proprio_stats_untar')) else ''

    @classmethod
    def list_keys(cls, root: str) -> Iterator[str]:
        """枚举 {root}/proprio_stats[_untar]/{task}/{episode}，task_id 限定 [300, 800)。"""
        base = os.path.join(root, 'proprio_stats' + cls._resolve_layout_suffix(root))
        if not os.path.isdir(base):
            return
        with os.scandir(base) as task_it:
            for t in task_it:
                if not (t.is_dir() and t.name.isdigit() and 300 <= int(t.name) < 800):
                    continue
                with os.scandir(t.path) as ep_it:
                    for e in ep_it:
                        if e.is_dir():
                            yield f'{t.name}/{e.name}'

    @property
    def _proprio_path(self):
        return os.path.join(self._root, 'proprio_stats' + self._layout_suffix, self._key, 'proprio_stats.h5')

    @property
    def _videos_path(self):
        return os.path.join(self._root, 'observations' + self._layout_suffix, self._key, 'videos')

    @property
    def _sam3_path(self):
        return os.path.join(self._root, 'sam3', self._key)

    @cached_property
    def rgbs(self) -> OrderedDict[str, np.ndarray]:
        return OrderedDict((k, video_read(os.path.join(self._videos_path, k + '_color.mp4'), resolution=r)) for k, r in self._cams if os.path.exists(os.path.join(self._videos_path, k + '_color.mp4')))

    @cached_property
    def position(self) -> np.ndarray:
        """(T, 7)：state/robot/{position, orientation}（xyzw→wxyz 重排），缺失字段填零。"""
        with h5py.File(self._proprio_path, 'r', locking=False) as f:
            position = np.array(f['state/robot/position'])
            orientation = np.array(f['state/robot/orientation'])
            if orientation.shape[0] == 0 or np.allclose(orientation, 0):
                orientation = np.zeros((len(self), 4))
                orientation[:, 3] = 1.0  # unit quaternion
            if position.shape[0] == 0:
                position = np.zeros((len(self), 3))
            return np.concatenate([
                position,
                orientation[..., [3, 0, 1, 2]],
            ], axis=-1)

    @cached_property
    def _len(self) -> int:
        with h5py.File(self._proprio_path, 'r', locking=False) as f:
            db = f[f'state/head/position']
            assert isinstance(db, h5py.Dataset)
            return db.shape[0]

    @cached_property
    def _qpos_dims(self) -> dict[str, slice | np.ndarray]:
        return {
            'waist': slice(0, 2),
            'head': slice(2, 4),
            'arm': slice(4, 18),
            'arm_left': slice(4, 11),
            'arm_right': slice(11, 18),
            'gripper': slice(18, 20),
            'gripper_left': slice(18, 19),
            'gripper_right': slice(19, 20),
        }

    @cached_property
    def _endpose_arms(self) -> dict[str, int]:
        # h5 state/end/{position,orientation} 通道顺序：column 0 = 左臂, column 1 = 右臂。
        return {'left': 0, 'right': 1}

    @cached_property
    def gripper_open_qpos(self) -> float:
        return 1.0

    @classmethod
    def _gripper_state_normalize(cls, state: np.ndarray) -> np.ndarray:
        """`state/effector/position` 实测范围 [31.288, 124.75714]，先线性归一到 [0,1]，
        再取 `1 - x` 翻转方向（raw 大值=闭合 → 0=closed / 1=open）；qpos 阶段再乘
        gripper_open_qpos 落到 BaseDataset 约定的 [0, gripper_open_qpos]。"""
        return 1 - np.clip((state-31.288) / (124.75714-31.288), 0, 1)

    @classmethod
    def _gripper_action_normalize(cls, action: np.ndarray) -> np.ndarray:
        """`action/effector/position` raw 已在 [0,1]，直接取 `1 - x` 翻转方向，输出 0=closed / 1=open；
        qpos 阶段再乘 gripper_open_qpos 落到 BaseDataset 约定的 [0, gripper_open_qpos]。"""
        return 1 - np.clip(action, 0, 1)

    @cached_property
    def origin_gripper(self) -> tuple[np.ndarray, np.ndarray]:
        """(action_norm, state_norm)，均已翻转方向但未做时延同步（sync 前的原始信号）。"""
        with h5py.File(self._proprio_path, 'r', locking=False) as f:
            return (
                self._gripper_action_normalize(np.array(f['action/effector/position'])),
                self._gripper_state_normalize(np.array(f['state/effector/position'])),
            )

    @cached_property
    def qpos(self) -> np.ndarray:
        # 列布局见 _qpos_dims；gripper 段先经 _compute_sync 做 action/state 时延对齐 + 共享线性化
        # 重建，再按 prefix 选支；waist 段做 [::-1] 反序以与上下游约定一致。
        with h5py.File(self._proprio_path, 'r', locking=False) as f:
            action, state, _, _ = _compute_sync(
                action=self._gripper_action_normalize(np.array(f['action/effector/position'])),
                state=self._gripper_state_normalize(np.array(f['state/effector/position'])),
            )
            gripper = {'action': action, 'state': state}
            prefix = 'state' if self._use_state else 'action'
            results = np.concatenate([
                np.array(f[prefix + '/waist/position'])[..., ::-1],
                np.array(f[prefix + '/head/position']),
                np.array(f[prefix + '/joint/position']),
                gripper[prefix] * self.gripper_open_qpos,
            ], axis=-1)
            return results

    @cached_property
    def endpose(self) -> np.ndarray:
        # 源 state/end/{position, orientation} 自带 (T, 2, 3) / (T, 2, 4) shape，orientation 重排 xyzw→wxyz。
        with h5py.File(self._proprio_path, 'r', locking=False) as f:
            return np.concatenate([
                np.array(f['state/end/position']),
                np.array(f['state/end/orientation'])[..., [3, 0, 1, 2]],
            ], axis=-1)

    @cached_property
    def dataset_name(self) -> list[str]:
        return ['agibot_state'] if self._use_state else ['agibot_action']

    @cached_property
    def base_files_exists(self) -> bool:
        """proprio_stats.h5 + 每个配置 cam 的 {cam}_color.mp4 + task_info json 都在。"""
        if not os.path.exists(self._proprio_path):
            return False
        for cam, _ in self._cams:
            if not os.path.exists(os.path.join(self._videos_path, f'{cam}_color.mp4')):
                return False
        task_id = self._key.split('/')[0]
        task_info_dir = os.path.join(self._root, 'task_info')
        return any(os.path.exists(os.path.join(task_info_dir, f'task_{task_id}{s}')) for s in ('.split.json', '.new.json', '.json'))

    @staticmethod
    @lru_cache(maxsize=None)
    def _load_task_info(task_info_dir: str, task_id: str) -> list[dict]:
        for suffix in ('.split.json', '.new.json', '.json'):
            path = os.path.join(task_info_dir, f'task_{task_id}{suffix}')
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
        raise FileNotFoundError(os.path.join(task_info_dir, f'task_{task_id}.json'))

    @cached_property
    def clip_info(self) -> list[ClipInfo]:
        # 按 task_info/task_{task_id}{suffix}.json 查找当前 episode 的 label_info.action_config；
        # suffix 优先级 .split.json > .new.json > .json，每条 action_config 转为一段 ClipInfo。
        task_id, episode_id = self._key.split('/')
        episodes = self._load_task_info(os.path.join(self._root, 'task_info'), task_id)
        eid = int(episode_id)
        episode = next(ep for ep in episodes if int(ep['episode_id']) == eid)
        return [
            ClipInfo(
                start=int(c['start_frame']),
                end=int(c['end_frame']),
                action_text=[s] if (s := c.get('action_text')) else [],
                skill=[s] if (s := c.get('skill')) else [],
                task_name=[s] if (s := episode.get('task_name')) else [],
                init_scene_text=[s] if (s := episode.get('init_scene_text')) else [],
            ) for c in episode['label_info']['action_config']
        ]


@njit()
def _debounce_binary_signal(binary: np.ndarray, min_len: int = 8) -> np.ndarray:
    """对二值序列做去抖，移除短于 `min_len` 的伪边沿。

    输入：
    - `binary`：0/1 序列，通常来自 action/state 的阈值化结果。
    - `min_len`：允许保留的最短连续段长度。

    处理：
    - 先按值变化把序列切成若干连续 run；
    - 再把过短 run 替换成左侧状态值。

    输出：
    - 与输入等长的平滑二值序列。
    """
    binary, n = binary.copy(), len(binary)
    if n != 0:
        change_idx = np.flatnonzero(np.diff(binary) != 0) + 1
        starts = np.concatenate((np.array([0]), change_idx))
        ends = np.concatenate((change_idx, np.array([n])))
        lengths = ends - starts
        short_mask = lengths < min_len
        if np.any(short_mask):
            # 标记出所有落在“短片段”范围内的索引
            is_short_pixel = np.repeat(short_mask, lengths)
            # 确定填充值
            fill_values = np.empty_like(lengths, dtype=binary.dtype)
            fill_values[1:] = binary[starts[1:] - 1]  # 绝大多数情况取前一个
            fill_values[0] = binary[ends[0]] if ends[0] < n else binary[0]  # 开头特殊处理
            # 扩展填充值到整个数组
            full_fill_array = np.repeat(fill_values, lengths)
            # 更新
            binary[is_short_pixel] = full_fill_array[is_short_pixel]
    return binary


def _estimate_dynamic_shift(
    action: np.ndarray,
    state: np.ndarray,
    max_shift: int = 120,
    window: int = 240,
    smooth_alpha: float = 0.2,
) -> np.ndarray:
    """基于边沿匹配的动态时延估计。

    输入：
    - `action` / `state`：同一通道的原始序列。
    - `max_shift` / `window` / `smooth_alpha`：与相关性版本相同的控制参数。

    处理：
    - 先对 action/state 做阈值化和去抖，得到稳定边沿；
    - 再按边沿顺序做最近邻匹配；
    - 最后把离散 lag 插值成逐帧曲线，并做一次平滑。

    输出：
    - `lag`：更稳定的逐帧时延估计。
    """
    action_bin = _debounce_binary_signal((action >= 0.8), min_len=8).astype(np.int8)
    if state.max() - state.min() < 1e-6:
        return np.zeros_like(action, dtype=np.float64)
    state_bin = _debounce_binary_signal((state >= 0.8), min_len=8).astype(np.int8)
    action_diff = np.diff(action_bin)
    state_diff = np.diff(state_bin)
    action_edges, = np.where(action_diff != 0)
    state_edges, = np.where(state_diff != 0)
    action_dirs = action_diff[action_edges]  # +1 上升沿，-1 下降沿
    state_dirs = state_diff[state_edges]

    n = action.shape[0]
    # 反向匹配：让每条 state edge 主动认领“同方向且距离最近”的 action edge。
    # 因为真实序列里通常 state edge 数 ≤ action edge 数（夹在快速指令之间的
    # 短脉冲不会让 state 真正越过阈值），从 state 出发的匹配能自动把那些
    # 没有对应物理响应的 action edge 留作 orphan。如果反过来从 action 出发，
    # orphan 的 action edge 会“抢”掉本属于后续真实闭合的 state edge，
    # 把后续 lag 误拉到 max_shift 附近，从而把 plateau 完全错位。
    lag_per_edge = np.full(action_edges.shape[0], np.nan, dtype=np.float64)
    action_used = np.zeros(action_edges.shape[0], dtype=bool)
    # 物理上 state 总是滞后 action（夹爪有响应时延），所以匹配时要偏好
    # “非负 lag”。对负 lag（候选 action edge 在 state edge 之后）施加 1.5x
    # 距离惩罚，避免在距离接近时把 state edge 误配给“未来”的 action edge。
    NEG_LAG_PENALTY = 1.5
    for j, se in enumerate(state_edges):
        sd = int(state_dirs[j])
        avail = (~action_used) & (action_dirs == sd)
        cand_idx = np.where(avail)[0]
        if cand_idx.size == 0:
            continue
        cand_pos = action_edges[cand_idx]
        dists = se - cand_pos  # state lags action ⇒ 正值
        score = np.where(dists >= 0, dists.astype(np.float64), -dists.astype(np.float64) * NEG_LAG_PENALTY)
        best_local = int(np.argmin(score))
        best_lag = float(dists[best_local])
        if abs(best_lag) > max_shift:
            continue
        action_used[cand_idx[best_local]] = True
        lag_per_edge[cand_idx[best_local]] = best_lag

    # 为 orphan action edge 填补 lag：前向最近邻 + 后向回填首个 NaN。
    matched_lags: list[float] = []
    last_known = 0.0
    seen_known = False
    for i in range(action_edges.shape[0]):
        if not np.isnan(lag_per_edge[i]):
            last_known = float(lag_per_edge[i])
            seen_known = True
            matched_lags.append(last_known)
        else:
            matched_lags.append(last_known if seen_known else 0.0)
    if not seen_known:
        matched_lags = [0.0] * action_edges.shape[0]
    else:
        # 起始连续 NaN 段用第一个已知值反向回填，避免直接落到 0。
        first_known_idx = int(np.argmax(~np.isnan(lag_per_edge)))
        first_known = float(lag_per_edge[first_known_idx])
        for i in range(first_known_idx):
            matched_lags[i] = first_known

    # 把离散边沿 lag 扩展为逐帧 lag 曲线，再做一次平滑，避免突变。
    anchor_t = np.r_[0, action_edges, n - 1]
    first_lag = matched_lags[0] if matched_lags else 0.0
    last_lag = matched_lags[-1] if matched_lags else 0.0
    anchor_lag = np.r_[first_lag, matched_lags, last_lag].astype(np.float64)
    lag = np.interp(np.arange(n, dtype=np.float64), anchor_t.astype(np.float64), anchor_lag)

    for i in range(1, n):
        lag[i] = smooth_alpha * lag[i] + (1.0-smooth_alpha) * lag[i - 1]

    return np.clip(lag, -max_shift, max_shift)


@njit()
def _shift_signal_dynamic(signal: np.ndarray, lag: np.ndarray) -> np.ndarray:
    """按逐帧 lag 对序列做时间重采样。

    输入：
    - `signal`：待对齐的一维序列。
    - `lag`：与每一帧对应的时延。

    处理：
    - 将采样坐标平移到 `t + lag[t]`；
    - 对浮点坐标使用线性插值；
    - 超出边界时使用端点值。

    输出：
    - 与输入等长的对齐后序列。
    """
    x = np.arange(signal.shape[0], dtype=np.float64)
    return np.interp(x + lag, x, signal)


def _build_shared_linear_profiles(
    action_raw: np.ndarray,
    state_aligned: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """重建 action/state 的平台段与过渡段。

    输入：
    - `action_raw`：原始 action 序列。
    - `state_aligned`：对齐后的 state 序列。

    处理：
    - action 保持离散平台，并用固定斜率重建边沿；
    - state 的平台值从原始对齐数据的稳定区统计得到；
    - state 在共同值域内与 action 直接重合，从而保持“可错峰起始、过渡重叠、斜率一致”。

    输出：
    - `action_out` / `state_out`：重建后的两条同步曲线。
    """
    n = action_raw.shape[0]

    # 分别对 action/state 二值化并检测边沿。
    # 这里的 action_target 仍保留原始离散台阶，只重建需要平滑的边沿。
    action_target = _debounce_binary_signal((action_raw >= 0.5), min_len=8).astype(np.float64)
    action_edges, = np.where(np.diff(action_target) != 0)

    s_min, s_max = float(np.min(state_aligned)), float(np.max(state_aligned))
    if s_max - s_min < 1e-6:
        state_bin = np.zeros_like(state_aligned)
    else:
        state_mid = 0.5 * (s_min+s_max)
        state_bin = _debounce_binary_signal((state_aligned >= state_mid), min_len=8).astype(np.float64)

    # 初始化输出，先填充各自的平台值。
    action_out = action_target.copy()
    # state 平台按 action 平台区间统计（中位数），保证平台值来自原始 state，
    # 同时 transition 与 action 的边沿一一对应，天然重叠。
    state_out = np.zeros_like(state_aligned, dtype=np.float64)
    boundaries = np.r_[0, action_edges + 1, n]
    plateau_vals = []
    for i in range(len(boundaries) - 1):
        seg_start = int(boundaries[i])
        seg_end = int(boundaries[i + 1])
        if seg_end <= seg_start:
            plateau_vals.append(float(state_aligned[min(seg_start, n - 1)]))
            continue

        # 只使用平台区间中部的稳定样本估计平台值，
        # 避免边沿附近过渡样本污染（尤其是序列开头第一段）。
        seg_len = seg_end - seg_start
        trim = min(20, seg_len // 4)
        core_start = seg_start + trim
        core_end = seg_end - trim
        if core_end <= core_start:
            core_start, core_end = seg_start, seg_end
        plateau_vals.append(float(np.median(state_aligned[core_start:core_end])))

    # 纠正序列开头平台：优先按 state 自己的首段稳定区估计，
    # 避免 action 首次边沿较早时把第一平台中位数拉低。
    state_edges, = np.where(np.diff(state_bin) != 0)
    if len(plateau_vals) > 0:
        if len(boundaries) > 1:
            first_end = int(np.clip(boundaries[1], 3, n))
        elif state_edges.size > 0:
            first_end = int(np.clip(state_edges[0] + 1, 3, n))
        else:
            first_end = int(n)

        initial_window = state_aligned[:first_end]
        if initial_window.size > 0:
            if action_edges.size > 0:
                a0 = float(action_target[action_edges[0]])
                a1 = float(action_target[min(action_edges[0] + 1, n - 1)])
                # 首段若是下降，第一平台取高分位；首段若是上升，取低分位。
                if a1 < a0:
                    plateau_vals[0] = float(np.percentile(initial_window, 90.0))
                else:
                    plateau_vals[0] = float(np.percentile(initial_window, 10.0))
            else:
                plateau_vals[0] = float(np.median(initial_window))

    for i in range(len(boundaries) - 1):
        seg_start = int(boundaries[i])
        seg_end = int(boundaries[i + 1])
        if seg_end > seg_start:
            state_out[seg_start:seg_end] = plateau_vals[i]

    FIXED_DELTA = 85 / 120 * 30 / 1000

    # 对 action 的每个边沿，按固定斜率从该边沿时刻开始变化。
    # 起点取 action_out[edge]（承接上一段 ramp 的实际终点），从而当两条边沿距离过近、
    # 上一段 ramp 被截断时，本段 ramp 会从截断值继续反向推进，自然形成三角峰，
    # 而不会出现“截断值 → binary 平台值”的阶跃。
    for i, edge in enumerate(action_edges):
        next_edge = action_edges[i + 1] if i + 1 < len(action_edges) else (n - 1)
        max_len = max(1, next_edge - edge)

        a0 = float(action_out[edge])
        a1 = action_target[edge + 1]
        need_a = int(np.ceil(abs(a1 - a0) / max(FIXED_DELTA, 1e-9)))
        steps = min(need_a, max_len)
        idx = edge + np.arange(steps + 1)
        idx = idx[idx < n]
        k = np.arange(idx.shape[0], dtype=np.float64)
        if a1 >= a0:
            vals = np.minimum(a0 + k*FIXED_DELTA, a1)
        else:
            vals = np.maximum(a0 - k*FIXED_DELTA, a1)
        action_out[idx] = vals
        # 把 ramp 实际终值持平到下一边沿之前。完整完成的 ramp 此处填入的就是 a1，
        # 与原 binary 平台一致；被截断的 ramp 则用截断值替换原 binary 平台，消除阶跃。
        if idx.shape[0] > 0:
            end_val = float(vals[-1])
            plateau_end = min(next_edge + 1, n)
            if idx[-1] + 1 < plateau_end:
                action_out[idx[-1] + 1:plateau_end] = end_val

    # 对 state 的每个过渡：
    # 1) 平台值使用 state 原始分段中位值。
    # 2) 斜率固定为 FIXED_DELTA。
    # 3) 允许与 action 错峰起始，但在中间变化区间与 action 精确重合（同 y 同 x）。
    transition_count = min(len(action_edges), max(0, len(plateau_vals) - 1))
    for i in range(transition_count):
        s0 = float(plateau_vals[i])
        s1 = float(plateau_vals[i + 1])
        start_t = int(action_edges[i])
        next_edge = int(action_edges[i + 1]) if i + 1 < len(action_edges) else (n - 1)
        start_t = int(np.clip(start_t, 0, n - 1))
        boundary_end = int(np.clip(next_edge, start_t + 1, n - 1))

        # state 在 action 的共同值域内直接跟随 action：
        # - 共同值域外保持平台值（因此可错峰起始）
        # - 共同值域内同 y 同 x（完全重叠，且斜率与 action 相同）
        lo = min(s0, s1)
        hi = max(s0, s1)
        state_out[start_t:boundary_end + 1] = np.clip(action_out[start_t:boundary_end + 1], lo, hi)

    return action_out, state_out


def _compute_sync(action: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action_sync = np.zeros_like(action, dtype=np.float64)
    state_sync = np.zeros_like(state, dtype=np.float64)
    lag_all = np.zeros_like(action, dtype=np.float64)
    aligned_state_all = np.zeros_like(state, dtype=np.float64)

    for ch in range(action.shape[1]):
        # 第一步：动态估计每一帧的时延 lag[t]（不是固定偏移）。
        # lag>0 表示 state 需要向“更早的时间”采样，才能和当前 action 对齐。
        lag = _estimate_dynamic_shift(action[:, ch], state[:, ch], max_shift=120, window=240, smooth_alpha=0.15)

        # 第二步：按 lag[t] 对 state 做逐帧插值重采样，实现动态时延校正。
        aligned_state = _shift_signal_dynamic(state[:, ch], lag)
        aligned_state_all[:, ch] = aligned_state
        lag_all[:, ch] = lag

        # 第三步：用“共享进度线性化”同时生成 action/state。
        # - action 保证每次完整 0-1 或 1-0 行程。
        # - state 在同一过渡段内保持平台值不变，进入共同值域后与 action 重叠。
        smooth_action, smooth_state = _build_shared_linear_profiles(
            action_raw=action[:, ch],
            state_aligned=aligned_state,
        )
        action_sync[:, ch] = smooth_action
        state_sync[:, ch] = smooth_state
    return action_sync, state_sync, lag_all, aligned_state_all
