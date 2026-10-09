"""Export speed-up: estimates lengths (text tokens + formula video tokens) without decoding.

Load after qwen_video_patch so smart_resize is already locked to the fixed height.
"""

import copy
from functools import lru_cache

import numpy as np
import qwen_video_patch as qvp
from swift.dataset.utils import AddLengthPreprocessor
from swift.utils import get_logger

logger = get_logger()

_orig_preprocess = AddLengthPreprocessor.preprocess

@lru_cache(maxsize=65536)
def _probe(path):
    import av
    with av.open(path) as c:
        vs = c.streams.video[0]
        fps = float(vs.average_rate) if vs.average_rate else 0.0
        length = float(vs.duration * vs.time_base) if (vs.duration is not None and vs.time_base) else 0.0
        return int(vs.width), int(vs.height), fps, length

_token_len_cache = {}

def _text_token_len(template, text):
    n = _token_len_cache.get(text)
    if n is None:
        n = len(template._tokenize(text))
        _token_len_cache[text] = n
    return n

def _estimate_video_tokens(template, path):
    from qwen_vl_utils import vision_process as vp

    base, start, end, _views, crop = qvp._parse_fragment(path)
    width, height, video_fps, duration = _probe(base)
    orig_h = height
    if video_fps <= 0 or duration <= 0:
        raise ValueError(f'bad probe: fps={video_fps} duration={duration} ({base})')
    if crop is not None:
        x0, y0, x1, y1 = crop
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f'invalid spatial_crop {crop} for probed frame {width}x{height} ({base})')
        width, height = x1 - x0, y1 - y0
    total_frames = max(int(round(duration * video_fps)), 1)

    ele = {}
    if start is not None:
        ele['video_start'] = start
    if end is not None:
        ele['video_end'] = end
    start_frame, end_frame, frame_cnt = vp.calculate_video_frame_range(ele, total_frames, video_fps)
    nframes = vp.smart_nframes({'fps': vp.FPS}, total_frames=frame_cnt, video_fps=video_fps)
    indices = np.linspace(start_frame, end_frame, nframes).round().astype(np.int64).tolist()

    processor = template.processor
    image_processor = processor.image_processor
    factor = image_processor.patch_size * image_processor.merge_size
    tps = getattr(getattr(processor, 'video_processor', None), 'temporal_patch_size', 2)
    if crop is not None:
        qvp._HEIGHT_OVERRIDE.set((orig_h, height))
    try:
        h, w = vp.smart_resize(height, width, factor=factor)
    finally:
        qvp._HEIGHT_OVERRIDE.set(None)
    frame_seqlen = (h//factor) * (w//factor)

    if len(indices) % tps:
        indices.extend(indices[-1:] * (tps - len(indices) % tps))
    n = 0
    for i in range(0, len(indices), tps):
        t = (indices[i] + indices[i + tps - 1]) / 2 / video_fps
        n += _text_token_len(template, f'<{t:.1f} seconds>') + 2 + frame_seqlen
    return n

def _approx_preprocess(self, row):
    videos = row.get('videos') or []
    if videos and all(isinstance(v, str) for v in videos):
        try:
            video_tokens = [_estimate_video_tokens(self.template, v) for v in videos]
            text_row = copy.deepcopy(row)
            text_row['videos'] = []
            replaced = 0
            for m in text_row['messages']:
                if isinstance(m.get('content'), str):
                    replaced += m['content'].count('<video>')
                    m['content'] = m['content'].replace('<video>', '<|video_pad|>')
            if replaced != len(videos):
                raise ValueError(f'<video> tag count {replaced} != videos {len(videos)}')
            encoded = self.template.encode(text_row, return_length=True)
            extra = sum(t - 1 for t in video_tokens)
            row['lengths'] = [n + extra for n in encoded['lengths']]
            return row
        except Exception as e:
            logger.warning(f'[approx_length] fallback to exact encode: {e} (videos={videos})')
    return _orig_preprocess(self, row)

AddLengthPreprocessor.preprocess = _approx_preprocess
logger.info(f'[plugin:approx_length_patch] AddLengthPreprocessor patched ({__file__})')
