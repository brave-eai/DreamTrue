import os
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from dataset import load_whitelist
from scripts.merge.filter import FilterNode, build_filter_node
from scripts.merge.scan import ClipContext, MergeClipInfo, SourceInfo, list_condition_files


class __Worker:
    _root: FilterNode | None = None

    @classmethod
    def init(cls, filter_config: dict):
        cls._root = build_filter_node(filter_config)

    @classmethod
    def run(cls, source: SourceInfo, data_key: str) -> tuple[dict[str, list[MergeClipInfo]], int]:
        root = cls._root
        assert root is not None, 'MergeDispatchWorker is not initialized'
        condition_files = list_condition_files(source, data_key)
        if not condition_files:
            return {}, 0
        dataset = source.cls(root=source.data_root, key=data_key, cams=source.cls.cams, use_state=False)
        if not dataset.base_files_exists:
            return {}, 0

        result: dict[str, list[MergeClipInfo]] = defaultdict(list)
        dropped = 0
        for condition_file in condition_files:
            for dataset_clip in dataset.clip_info:
                ctx = ClipContext(
                    **dataset_clip.__dict__,
                    data_class=source.data_class,
                    data_root=source.data_root,
                    condition_root=source.condition_root,
                    condition_file=condition_file,
                    data_key=data_key,
                    fps=dataset.fps,
                    score_root=source.score_root,
                    score_file=source.score_file,
                )

                if ctx.score_path is not None and not os.path.exists(ctx.score_path):
                    dropped += 1
                    continue

                for group in root.apply(ctx):
                    if group is None:
                        dropped += 1
                    else:
                        result[group].append(ctx.clip)
        return dict(result), dropped


def dispatch_sources_parallel(sources: list[SourceInfo], filter_config: dict, max_workers: int) -> dict[str, list[MergeClipInfo]]:
    result: defaultdict[str, list[MergeClipInfo]] = defaultdict(list)
    dropped, errors = 0, 0
    with ProcessPoolExecutor(max_workers=max_workers, initializer=__Worker.init, initargs=(filter_config, )) as pool:
        futures = []
        for source in sources:
            whitelist = load_whitelist(os.path.join(source.data_root, '.cache'))
            print(f'{source.data_class} whitelist {"" if whitelist is not None else "not "}loaded !')
            data_keys = source.cls.list_keys(source.data_root)
            for data_key in tqdm(data_keys, desc=f'Submitting {source.data_class}', mininterval=0.01, maxinterval=1):
                if whitelist is None or data_key in whitelist:
                    futures.append(pool.submit(__Worker.run, source, data_key))

        for future in tqdm(futures, desc='Dispatching merge', mininterval=0.01, maxinterval=1):
            try:
                grouped, num_dropped = future.result()
                for group, clips in grouped.items():
                    result[group].extend(clips)
                dropped += num_dropped
            except Exception:
                errors += 1
                traceback.print_exc()

    print(f'{errors} dispatch jobs failed')
    print(f'dropped {dropped} clips when dispatching')
    return result
