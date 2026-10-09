"""Data-reading patches for ms-swift: fragment crops, fixed video height, clip reader.

Must be loaded before approx_length_patch.
"""

import logging
import os
from contextvars import ContextVar

# Disable HuggingFace datasets map-cache.
import datasets
from swift.utils import get_logger

datasets.disable_caching()

logger = get_logger()

# Drop only the smart_nframes > total_frames warning, keep everything else.
logging.getLogger("qwen_vl_utils.vision_process").addFilter(lambda r: not r.getMessage().startswith("smart_nframes:"))

_HEIGHT_OVERRIDE: ContextVar[tuple[int, int] | None] = ContextVar("_qvp_height_override", default=None)

def _install_height():
    h_target = int(os.getenv("VIDEO_HEIGHT", "0") or 0)
    if h_target <= 0:
        return
    round_down = os.getenv("VIDEO_HEIGHT_FLOOR", "0") == "1"
    from qwen_vl_utils import vision_process as vp

    align = vp.floor_by_factor if round_down else vp.round_by_factor

    def smart_resize(height, width, factor, min_pixels=None, max_pixels=None):
        rec = _HEIGHT_OVERRIDE.get()
        if rec is not None:
            _HEIGHT_OVERRIDE.set(None)
            orig_h, crop_h = rec
            t = h_target * crop_h / orig_h
        else:
            t = h_target
        h = max(factor, align(t, factor))
        w = max(factor, vp.round_by_factor(width * h / height, factor))
        return h, w

    vp.smart_resize = smart_resize

def _parse_fragment(path):
    if not isinstance(path, str) or "#" not in path:
        return path, None, None, None, None
    base, frag = path.split("#", 1)
    start = end = views = crop = None
    try:
        for part in frag.split(";"):
            if part.startswith("t="):
                nums = part[2:].split(",")
                start = float(nums[0]) if nums[0].strip() else None
                end = float(nums[1]) if len(nums) > 1 and nums[1].strip() else None
            elif part.startswith("v="):
                views = int(part[2:])
            elif part.startswith("c="):
                crop = tuple(int(v) for v in part[2:].split(","))
                if len(crop) != 4:
                    raise ValueError(f"fragment c= expects 4 values, got {crop}")
    except ValueError:
        return path, None, None, None, None
    return base, start, end, views, crop

def _wrap_reader_backends():
    from qwen_vl_utils import vision_process as vp

    def wrap(fn):

        def reader(ele, *a, **k):
            video, meta, fps = fn(ele, *a, **k)
            c = ele.get("spatial_crop")
            if c is None:
                return video, meta, fps
            x0, y0, x1, y1 = c
            H, W = video.shape[2], video.shape[3]
            if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
                msg = f"invalid spatial_crop {c} for decoded frame {W}x{H} (video={ele.get('video')})"
                ele["_spatial_crop_error"] = msg
                err = ValueError(msg)
                err._qvp_crop_error = True
                raise err
            _HEIGHT_OVERRIDE.set((H, y1 - y0))
            ele.pop("spatial_crop", None)
            return video[:, :, y0:y1, x0:x1], meta, fps

        return reader

    for name, fn in list(vp.VIDEO_READER_BACKENDS.items()):
        if not getattr(fn, "_qvp_crop_wrapped", False):
            wrapped = wrap(fn)
            wrapped._qvp_crop_wrapped = True
            vp.VIDEO_READER_BACKENDS[name] = wrapped

_backends_wrapped = False

def _install_clip():
    import qwen_vl_utils
    from qwen_vl_utils import vision_process as vp

    orig_fv = vp.fetch_video

    def fetch_video(ele, *a, **k):
        global _backends_wrapped
        if not _backends_wrapped:
            _wrap_reader_backends()
            _backends_wrapped = True
        if isinstance(ele, dict) and isinstance(ele.get("video"), str):
            base, start, end, views, crop = _parse_fragment(ele["video"])
            if start is not None or end is not None or views is not None or crop is not None:
                ele["video"] = base
                if start is not None:
                    ele.setdefault("video_start", start)
                if end is not None:
                    ele.setdefault("video_end", end)
                if crop is not None:
                    ele["spatial_crop"] = crop
        try:
            video = orig_fv(ele, *a, **k)
        except Exception as e:
            if getattr(e, "_qvp_crop_error", False) or (isinstance(ele, dict) and "_spatial_crop_error" in ele):
                raise ValueError(ele["_spatial_crop_error"]) from e
            raise
        finally:
            _HEIGHT_OVERRIDE.set(None)
        if isinstance(ele, dict) and "_spatial_crop_error" in ele:
            raise ValueError(ele["_spatial_crop_error"])
        return video

    vp.fetch_video = fetch_video
    qwen_vl_utils.fetch_video = fetch_video

_install_height()
_install_clip()
logger.info(f'[plugin:qwen_video_patch] injected: smart_resize + fetch_video(t/v/c fragments) ({__file__})')
