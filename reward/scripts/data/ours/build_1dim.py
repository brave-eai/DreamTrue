from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import av

ALL_META_KEYS = ("nfr", "fps", "cell", "ncell", "head", "w0", "w1", "box", "dims", "w", "rep", "aug")


def probe(path: str) -> tuple[str, int, int, int]:
    try:
        with av.open(path) as c:
            s = c.streams.video[0]
            return path, int(s.frames or 0), int(s.width), int(s.height)
    except Exception:  # noqa: BLE001
        return path, -1, 0, 0


def probe_all(paths: list[str], workers: int) -> dict[str, tuple[int, int, int]]:
    todo = list(dict.fromkeys(paths))
    print(f"probe video headers: {len(todo)}", flush=True)
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (path, nfr, w, h) in enumerate(ex.map(probe, todo, chunksize=16)):
            out[path] = (nfr, w, h)
            if (i+1) % 1000 == 0:
                print(f"  [{i + 1}/{len(todo)}]", flush=True)
    return out


def rec(video: str, meta: dict) -> dict:
    # The `<video>` placeholder must stay: Template._add_default_tags reconciles
    # marker count against len(videos) and would otherwise reorder the messages.
    padded = {k: meta.get(k) for k in ALL_META_KEYS}
    return {
        "videos": [video],
        "messages": [{
            "role": "user",
            "content": "<video>"
        }, {
            "role": "assistant",
            "content": ""
        }],
        "meta": padded,
    }
