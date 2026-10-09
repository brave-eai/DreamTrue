"""Training-time view construction and pixel augmentation for ms-swift.

Rows carry only `meta`; prompts/answers are rendered here from configs/embodied/*.yaml.
Crops are applied via fragment syntax consumed by qwen_video_patch. All randomness is
derived from md5(path|rep|salt), so a row renders identically across runs/processes.
"""
import hashlib
import json
import os
import random
from dataclasses import dataclass, field

import torch
import torchvision.transforms.functional as TF
from swift.template import TEMPLATE_MAPPING
from swift.template.templates.qwen import Qwen3_5Template
from swift.utils import get_logger

from embodied.taxonomy import DIM_BY_ID, GROUP_DIMS, build_prompt, dim_prompt, group_value

logger = get_logger()
TEMPLATE_TYPE = 'qwen3_5'
THINK = '<think>\n\n</think>\n\n'
STRICT = os.getenv('VIEWAUG_STRICT', '0') == '1'

_CFG_CACHE = {}

def load_config(path):
    if not path:
        raise ValueError('viewaug 配置未指定：aug=train 的行需要 VIEWAUG_CONFIG，'
                         'aug=off 的行需要 VIEWAUG_CONFIG_EVAL')
    cfg = _CFG_CACHE.get(path)
    if cfg is None:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)
        _check_dims(cfg, path)
        _CFG_CACHE[path] = cfg
    return cfg

def _check_dims(cfg, path):
    dims = cfg.get('dims')
    if not isinstance(dims, dict) or not dims:
        raise ValueError(f'{path}: dims 必须是 组合→权重 的非空表，'
                         f'如 {{L2c+L3a: 1.0, L3a: 1.0}}')
    for combo, w in dims.items():
        ds = str(combo).split('+')
        for d in ds:
            if d not in DIM_BY_ID:
                raise ValueError(f'{path}: dims 组合 {combo!r} 含未知维度 {d!r}，'
                                 f'可选 {sorted(DIM_BY_ID)}')
        if len(set(ds)) != len(ds):
            raise ValueError(f'{path}: dims 组合 {combo!r} 维度重复')
        for d in ds:
            if d in GROUP_DIMS and any(m in ds for m in GROUP_DIMS[d]['members']):
                raise ValueError(f'{path}: dims 组合 {combo!r} 组 {d} 与成员维度同现')
        if not (isinstance(w, (int, float)) and float(w) > 0):
            raise ValueError(f'{path}: dims[{combo!r}] 权重必须 >0，得到 {w!r}')

def config_for(meta):
    env = 'VIEWAUG_CONFIG_EVAL' if meta['aug'] == 'off' else 'VIEWAUG_CONFIG'
    return load_config(os.getenv(env))

def _rng(meta, salt):
    return random.Random(int.from_bytes(
        hashlib.md5(f"{meta['path']}|{meta['rep']}|{salt}".encode()).digest()[:8], 'big'))

def _uni(rnd, span):
    lo, hi = span
    return lo if hi <= lo else rnd.uniform(lo, hi)

def _pick(rnd, weights):
    items = [(k, float(w)) for k, w in weights.items() if float(w) > 0]
    total = sum(w for _k, w in items)
    r = rnd.random() * total
    for k, w in items:
        r -= w
        if r <= 0:
            return k
    return items[-1][0]

def _place(rnd, total, n, must):
    n = max(1, min(n, total))
    if must is None:
        return rnd.randint(0, total - n)
    m0, m1 = must
    lo, hi = max(0, m1 - n), min(m0, total - n)
    return rnd.randint(lo, hi) if lo <= hi else max(0, min(m0, total - n))

def _span(rnd, total, frac, must):
    need = (must[1] - must[0]) if must else 1
    n = max(need, min(total, int(round(total * _uni(rnd, frac)))))
    a = _place(rnd, total, n, must)
    return a, a + n

def _rect(rnd, cw, ch, area, aspect, box):
    bw = (box[2] - box[0]) if box else 1
    bh = (box[3] - box[1]) if box else 1
    s = _uni(rnd, area)**0.5
    aj = _uni(rnd, aspect)
    w = min(cw, max(int(round(cw * s * aj)), bw))
    h = min(ch, max(int(round(ch * s / aj)), bh))
    x = _place(rnd, cw, w, (box[0], box[2]) if box else None)
    y = _place(rnd, ch, h, (box[1], box[3]) if box else None)
    return x, y, x + w, y + h

def _norm_box_in(box, crop):
    cx0, cy0, cx1, cy1 = crop
    cw, ch = cx1 - cx0, cy1 - cy0
    return {
        'x0': round((box[0] - cx0) * 1000 / cw),
        'y0': round((box[1] - cy0) * 1000 / ch),
        'x1': round((box[2] - cx0) * 1000 / cw),
        'y1': round((box[3] - cy0) * 1000 / ch),
    }

@dataclass
class View:
    kind: str
    prompt: str
    answer: str
    cell: tuple
    crop: tuple | None = None
    window: tuple | None = None
    dims: list = field(default_factory=list)
    gts: dict = field(default_factory=dict)
    weight: object = None
    box_rel: tuple | None = None
    color: tuple | None = None  # (brightness, contrast, saturation, hue)
    cutout: dict | None = None
    cell_zoom: list = field(default_factory=list)

def _geom(rnd, meta, name, spec):
    cw, ch = meta['cell']
    ncell, nfr = meta['ncell'], meta['nfr']
    heads = meta['head']
    if isinstance(heads, int):
        heads = [heads]
    box = tuple(meta['box']) if meta['box'] else None
    must_t = (meta['w0'], meta['w1'] + 1) if box else None

    if name == 'raw':
        return None, None, None, []

    if name == 'cell_zoom':
        cz, box_rel = [], None
        for i in range(ncell):
            r = None
            if rnd.random() < spec['p']:
                r = _rect(rnd, cw, ch, tuple(spec['area']), tuple(spec['aspect']), box if i in heads else None)
            cz.append(r)
            if i in heads and box:
                b = _norm_box_in(box, r or (0, 0, cw, ch))
                box_rel = ((i + b['x0'] / 1000) / ncell, b['y0'] / 1000,
                           (i + b['x1'] / 1000) / ncell, b['y1'] / 1000)
        return None, None, box_rel, cz

    if name == 'box_fit':
        s = _uni(rnd, tuple(spec['area']))**0.5
        aj = _uni(rnd, tuple(spec['aspect']))
        bw, bh = box[2] - box[0], box[3] - box[1]
        w = min(cw, max(int(round(bw * s * aj)), bw))
        h = min(ch, max(int(round(bh * s / aj)), bh))
        x = _place(rnd, cw, w, (box[0], box[2]))
        y = _place(rnd, ch, h, (box[1], box[3]))
        crop = (x, y, x + w, y + h)
        pad = _uni(rnd, tuple(spec['time_pad']))
        need = must_t[1] - must_t[0]
        n = max(need, min(nfr, int(round(need * (1 + 2*pad)))))
        f0 = _place(rnd, nfr, n, must_t)
        window = (f0, f0 + n)
    else:
        crop = _rect(rnd, cw, ch, tuple(spec['area']), tuple(spec['aspect']), box)
        window = _span(rnd, nfr, tuple(spec['time']), must_t)

    box_rel = None
    if box:
        b = _norm_box_in(box, crop)
        box_rel = (b['x0'] / 1000, b['y0'] / 1000, b['x1'] / 1000, b['y1'] / 1000)
    head = rnd.choice(heads) if len(heads) > 1 else heads[0]
    off = head * cw
    return (crop[0] + off, crop[1], crop[2] + off, crop[3]), window, box_rel, []

def _answer_body(dims, gts):
    if len(dims) == 1:
        return gts[dims[0]]
    return json.dumps({d: gts[d] for d in dims}, ensure_ascii=False)

def _cls_prompt(dims):
    return dim_prompt(dims[0], zoom=False) if len(dims) == 1 else build_prompt(dims=dims, zoom=False, with_video_tag=True)

def _render(dims, gts):
    return _cls_prompt(dims), THINK + _answer_body(dims, gts)

def derive(meta, cfg):
    rnd = _rng(meta, cfg['salt'])
    box = meta['box']
    has_dims = meta['dims'] is not None

    dims, gts, weight = [], {}, None
    if has_dims:
        dims = _pick(rnd, cfg['dims']).split('+')
        gts = {d: group_value(d, meta['dims']) if d in GROUP_DIMS else meta['dims'].get(d, '无') for d in dims}
        w = meta['w'] or {}
        weight = w.get(dims[0]) if len(dims) == 1 else {d: w[d] for d in dims if d in w} or None

    views = {k: v for k, v in cfg['views'].items() if _allowed(k, v, box, has_dims)}
    if not views:
        raise ValueError(f'viewaug: 没有可用视图（box={box is not None} dims={has_dims}），'
                         f'配置里只有 {sorted(cfg["views"])}')
    name = _pick(rnd, {k: v['w'] for k, v in views.items()})
    spec = views[name]

    crop, window, box_rel, cell_zoom = _geom(rnd, meta, name, spec)
    prompt, answer = _render(dims, gts)

    view = View(kind=name, prompt=prompt, answer=answer, cell=tuple(meta['cell']),
                crop=crop, window=window, dims=dims, gts=gts, box_rel=box_rel,
                cell_zoom=cell_zoom, weight=weight)
    _pixel_ops(rnd, view, cfg, box_rel)
    return view

VIEWS = ('raw', 'cell_zoom', 'head_zoom', 'box_zoom', 'box_fit')
NEEDS_BOX = {'box_zoom', 'box_fit'}

def _allowed(name, spec, box, has_dims):
    if name not in VIEWS:
        raise ValueError(f'viewaug: 未知视图 {name!r}，可选 {sorted(VIEWS)}')
    if not has_dims:
        return False
    if name in NEEDS_BOX and box is None:
        return False
    return True

def meta_of(row):
    raw = row.get('meta')
    if not raw:
        return None
    meta = json.loads(raw) if isinstance(raw, str) else dict(raw)
    if not meta or not meta.get('nfr'):
        return None
    videos = row.get('videos') or []
    meta.setdefault('path', videos[0] if videos else '')
    return meta

def view_of(row, cfg=None):
    meta = meta_of(row)
    if meta is None:
        return None
    return derive(meta, cfg if cfg is not None else config_for(meta))

def _pixel_ops(rnd, view, cfg, box_rel):
    for op in cfg['pixel_ops']:
        if rnd.random() >= op['p']:
            continue
        if op['name'] == 'color':
            view.color = (
                _uni(rnd, (max(0.0, 1 - op['brightness']), 1 + op['brightness'])),
                _uni(rnd, (max(0.0, 1 - op['contrast']), 1 + op['contrast'])),
                _uni(rnd, (max(0.0, 1 - op['saturation']), 1 + op['saturation'])),
                _uni(rnd, (-op['hue'], op['hue'])),
            )
        elif op['name'] == 'cutout':
            view.cutout = _cutout(rnd, op, box_rel)
        else:
            raise ValueError(f'viewaug: 未知像素算子 {op["name"]!r}')

def _cutout(rnd, op, box_rel):
    cap = op['max_gt_cover']
    rects = []
    for _ in range(rnd.randint(*op['n_rect'])):
        for _try in range(8):
            s = _uni(rnd, tuple(op['area']))**0.5
            aj = _uni(rnd, tuple(op['aspect']))
            w, h = min(1.0, s * aj), min(1.0, s / aj)
            x, y = rnd.uniform(0, 1 - w), rnd.uniform(0, 1 - h)
            r = (x, y, x + w, y + h)
            if box_rel is None or _cover(r, box_rel) <= cap:
                rects.append(r)
                break
    if not rects:
        return None
    return {'rects': rects, 'p_frame': op['p_frame'], 'seed': rnd.getrandbits(32)}

def _cover(rect, box):
    ix = max(0.0, min(rect[2], box[2]) - max(rect[0], box[0]))
    iy = max(0.0, min(rect[3], box[3]) - max(rect[1], box[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return (ix*iy / area) if area > 0 else 0.0

def _apply_color(video, f):
    b, c, s, h = f
    video = TF.adjust_brightness(video, b)
    video = TF.adjust_contrast(video, c)
    video = TF.adjust_saturation(video, s)
    return TF.adjust_hue(video, h)

def _apply_cutout(video, spec):
    t, _c, h, w = video.shape
    rnd = random.Random(spec['seed'])
    p = spec['p_frame']
    video = video.clone()
    for i in range(t):
        for x0, y0, x1, y1 in spec['rects']:
            if rnd.random() < p:
                a, b = int(y0 * h), int(x0 * w)
                video[i, :, a:max(a + 1, int(y1 * h)), b:max(b + 1, int(x1 * w))] = 0
    return video

def _apply_cell_zoom(video, rects, cell):
    _t, _c, h, w = video.shape
    n = len(rects)
    if n <= 1 or w < n:
        return video
    bounds = [round(i * w / n) for i in range(n + 1)]
    cells = []
    for i, r in enumerate(rects):
        v = video[:, :, :, bounds[i]:bounds[i + 1]]
        if r is None:
            cells.append(v)
            continue
        vh, vw = v.shape[2], v.shape[3]
        sx, sy = vw / cell[0], vh / cell[1]
        cells.append(
            TF.resized_crop(v, int(r[1] * sy), int(r[0] * sx), max(1, int((r[3] - r[1]) * sy)),
                            max(1, int((r[2] - r[0]) * sx)), [vh, vw],
                            interpolation=TF.InterpolationMode.BILINEAR, antialias=True))
    return torch.cat(cells, dim=3)

def frag(path, view, fps):
    parts = []
    if view.window is not None:
        parts.append(f't={view.window[0] / fps:.1f},{view.window[1] / fps:.1f}')
    if view.crop is not None:
        parts.append('c={},{},{},{}'.format(*view.crop))
    return f"{path}#{';'.join(parts)}" if parts else path

def materialize(row):
    meta = meta_of(row)
    if meta is None or meta.get('materialized'):
        return row
    view = derive(meta, config_for(meta))
    messages = [dict(m) for m in row['messages']]
    messages[0]['content'] = view.prompt
    if len(messages) > 1:
        messages[1]['content'] = view.answer
        if view.weight is not None:
            messages[1]['loss_scale'] = view.weight
    videos = list(row.get('videos') or [])
    if videos:
        videos[0] = frag(meta['path'], view, meta['fps'])
    meta['materialized'] = True
    return {**row, 'videos': videos, 'messages': messages, 'meta': meta}

def _install_grpo_materialize():
    from swift.pipelines.train.sft import SwiftSft

    if getattr(SwiftSft._prepare_dataset, '_viewaug_patched', False):
        return
    orig = SwiftSft._prepare_dataset

    def _prepare_dataset(self):
        datasets = orig(self)
        if getattr(self.args, 'rlhf_type', None) not in ('grpo', 'gkd'):
            return datasets
        out = []
        for ds in datasets:
            if ds is None or not hasattr(ds, 'map') or 'meta' not in (getattr(ds, 'column_names', None) or []):
                out.append(ds)
                continue
            n0 = len(ds) if hasattr(ds, '__len__') else '?'
            ds = ds.filter(lambda row: (meta_of(row) or {}).get('dims') is not None, desc='viewaug filter')
            n1 = len(ds) if hasattr(ds, '__len__') else '?'
            logger.info(f'[viewaug] {self.args.rlhf_type}: 保留可问维度的行 {n1}/{n0}，物化视图')
            out.append(ds.map(materialize, desc='viewaug materialize'))
        return out

    _prepare_dataset._viewaug_patched = True
    SwiftSft._prepare_dataset = _prepare_dataset

class AugQwen3_5Template(Qwen3_5Template):

    def _preprocess_inputs(self, inputs) -> None:
        raw = (inputs.extra_kwargs or {}).get('meta')
        if raw:
            try:
                self._build_view(inputs, json.loads(raw) if isinstance(raw, str) else dict(raw))
            except Exception as e:
                if STRICT:
                    raise
                logger.warning(f'[viewaug] view skipped due to: {e}')
        return super()._preprocess_inputs(inputs)

    def _build_view(self, inputs, meta):
        meta = dict(meta)
        meta.setdefault('path', inputs.videos[0] if inputs.videos else '')
        view = derive(meta, config_for(meta))
        inputs._viewaug = view
        if meta.get('materialized'):
            return
        inputs.messages[0]['content'] = view.prompt
        inputs.messages[1]['content'] = view.answer
        if view.weight is not None:
            inputs.messages[1]['loss_scale'] = view.weight
        if inputs.videos:
            inputs.videos[0] = frag(meta['path'], view, meta['fps'])

    def replace_tag(self, media_type, index, inputs):
        tokens = super().replace_tag(media_type, index, inputs)
        view = getattr(inputs, '_viewaug', None)
        if media_type != 'video' or view is None:
            return tokens
        video = inputs.videos[index]
        if not isinstance(video, torch.Tensor) or video.ndim != 4:
            return tokens
        try:
            if video.is_floating_point():
                video = video.clamp(0, 255).to(torch.uint8)
            if any(r is not None for r in view.cell_zoom):
                video = _apply_cell_zoom(video, view.cell_zoom, view.cell)
            if view.color:
                video = _apply_color(video, view.color)
            if view.cutout:
                video = _apply_cutout(video, view.cutout)
            inputs.videos[index] = video.to(torch.uint8).contiguous()
        except Exception as e:
            if STRICT:
                raise
            logger.warning(f'[viewaug] pixel ops skipped due to: {e}')
        return tokens

if TEMPLATE_TYPE in TEMPLATE_MAPPING:
    TEMPLATE_MAPPING[TEMPLATE_TYPE].template_cls = AugQwen3_5Template
    _install_grpo_materialize()
    logger.info(f'[plugin:qwen_video_aug] injected: AugQwen3_5Template + grpo materialize '
                f'(config={os.getenv("VIEWAUG_CONFIG") or "off"}) ({__file__})')
else:
    logger.error(f"[plugin:qwen_video_aug] injection FAILED: template '{TEMPLATE_TYPE}' not registered ({__file__})")
