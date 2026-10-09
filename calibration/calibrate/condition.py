import argparse
import datetime
import gc
import itertools
import json
import os
import platform
import random
import shutil
import time
import traceback
import warnings
from collections import OrderedDict
from concurrent.futures.process import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, fields, replace
from functools import cached_property
from queue import Queue

import h5py
import hdf5plugin
import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", module="sapien")
from dataset import __DATASET__, BaseDataset, OverlayFailedException, build_modified_datasets, get_dataset_class
from sim import IKFailedException, SapienWorker, gripper_rgb_background
from utils import HeavyWorkerInfo, Pose, TemporaryDirectory, Uploader, git_status, wait_futures


@dataclass(frozen=True)
class RenderTask:
    tmp_dir: str
    repo_dir: str
    uploader: Uploader

    def __post_init__(self):
        os.makedirs(self.tmp_dir, exist_ok=True)

    @property
    def _tmp_h5_filename(self) -> str:
        return f'tmp.h5'

    @property
    def tmp_h5_path(self) -> str:
        return os.path.join(self.tmp_dir, self._tmp_h5_filename)

    data_class: str
    data_root: str
    data_key: str

    calib_root: str
    calib_key: str | None
    calib_hash: str | None
    meta_data: dict

    use_state: bool
    modifications: list[dict]
    compress_level: int
    fix_rgb_background: tuple[int, int, int] | None = None

    @cached_property
    def redis_lock_key(self) -> str:
        return f'condition:{self.data_class}:{self.data_key}:{self.calib_key}'

    def get_upload_task(self, repo_h5_filename: str = '', error_traceback: traceback.TracebackException | None = None) -> 'UploadTask':
        return UploadTask(
            **{f.name: getattr(self, f.name)
               for f in fields(self)},
            repo_h5_filename=repo_h5_filename,
            error_traceback=error_traceback,
        )


@dataclass(frozen=True)
class UploadTask(RenderTask):
    repo_h5_filename: str = ''
    error_traceback: traceback.TracebackException | None = None

    @property
    def repo_h5_path(self) -> str:
        return self.repo_dir + '/' + self.repo_h5_filename

    @property
    def need_upload(self) -> bool:
        return self.repo_h5_filename != ''

    def get_render_task(self) -> RenderTask:
        return RenderTask(**{f.name: getattr(self, f.name) for f in fields(RenderTask)})


class Runner(SapienWorker):

    @classmethod
    def init(cls, info: HeavyWorkerInfo, robot_path: str):  # type: ignore
        super().init(info, robot_path=robot_path)

    @classmethod
    def meta(cls) -> dict:
        assert cls.sim is not None and cls.info is not None, 'Sim not initialized'
        return {
            'robot_path': cls.sim.robot_path,
            'start_time': datetime.datetime.now().isoformat(),
            'hostname': platform.node(),
            'source_git': git_status('.', ignore_error=True),
            'cameras_intrinsic': json.dumps({
                k: v.dict
                for k, v in cls.sim.cameras_intrinsic.items()
            }),
            'cameras_local_pose': json.dumps({
                k: v.dict
                for k, v in cls.sim.cameras_local_pose.items()
            }),
            'segmentation_ids': json.dumps(cls.sim.segmentation_ids),
            'segmentation_group': json.dumps({
                k: list(v)
                for k, v in cls.sim.segmentation_group.items()
            }),
            'parts_config': json.dumps(cls.sim.parts_config),
        }

    @classmethod
    def build_dataset(cls, t: RenderTask) -> tuple[BaseDataset, int, str]:
        assert cls.sim is not None and cls.info is not None, 'Sim not initialized'
        dataset = get_dataset_class(t.data_class)(
            root=t.data_root,
            key=t.data_key,
            cams=[(name, (cam.width, cam.height)) for name, cam in cls.sim.cameras.items()],
            use_state=t.use_state,
        )
        assert dataset.source.__class__.__name__ in cls.sim.compatible_datasets
        full_dataset_len = len(dataset)
        dataset = build_modified_datasets(
            dataset,
            t.modifications,
            assets_path=cls.sim.assets_path,
            robot_path=cls.sim.robot_path,
        )
        return dataset, full_dataset_len, '-'.join(['condition'] + dataset.dataset_name) + '.h5'

    @classmethod
    @contextmanager
    def hdf5(cls, output_file: str, meta_data: dict, dataset: BaseDataset, full_dataset_len: int, compress_level: int):
        assert cls.sim is not None and cls.info is not None, 'Sim not initialized'
        with h5py.File(output_file, 'w') as db:
            for k, v in meta_data.items():
                try:
                    db.attrs[k] = v
                except TypeError as e:
                    raise TypeError(f'Warning: Failed to save meta_data[{k}] to hdf5 attribute, got error: {e}')
            for cam_name, cam in cls.sim.cameras.items():
                sh, ch = (full_dataset_len, cam.height, cam.width), (1, cam.height, cam.width)
                for k, s, d in [('rgb', (3, ), np.uint8), ('mask', (), np.bool_), ('depth', (), np.uint16), ('segmentation', (), np.uint16)]:  #, ('normal', (3, ), np.uint8)
                    db.create_dataset(cam_name + '/' + k, shape=sh + s, dtype=d, compression=hdf5plugin.Zstd(clevel=compress_level), chunks=ch + s)
                db.create_dataset(cam_name + '/global_pose', shape=(full_dataset_len, 4, 4), dtype=np.float32, compression=hdf5plugin.Zstd(clevel=compress_level), chunks=(1, 4, 4))
            db.create_dataset('qpos', dtype=np.float32, compression=hdf5plugin.Zstd(clevel=compress_level), chunks=(1, ) + dataset.qpos.shape[1:], shape=(full_dataset_len, ) + dataset.qpos.shape[1:])
            db.create_dataset('position', dtype=np.float32, compression=hdf5plugin.Zstd(clevel=compress_level), chunks=(1, ) + dataset.position.shape[1:], shape=(full_dataset_len, ) + dataset.position.shape[1:])
            db.create_dataset('full_qpos', dtype=np.float32, compression=hdf5plugin.Zstd(clevel=compress_level), chunks=(1, ) + cls.sim.full_qpos.shape, shape=(full_dataset_len, ) + cls.sim.full_qpos.shape)
            for link_name in cls.sim.link_global_poses:
                db.create_dataset(f'link/{link_name}/global_pose', shape=(full_dataset_len, 4, 4), dtype=np.float32, compression=hdf5plugin.Zstd(clevel=compress_level), chunks=(1, 4, 4))
            db.create_dataset('indices', data=dataset.indices, dtype=np.int32, compression=hdf5plugin.Zstd(clevel=compress_level))
            yield db

    @classmethod
    def run(cls, t: RenderTask) -> UploadTask:
        repo_h5_filename = ''
        try:
            assert cls.sim is not None and cls.info is not None, 'Sim not initialized'
            with cls.tqdm([], desc=f'{t.data_key}') as pbar:
                pbar.refresh()
                gc.collect()
                dataset, full_dataset_len, repo_h5_filename = cls.build_dataset(t)
                if t.uploader.exists(t.repo_dir + '/' + repo_h5_filename) or \
                   t.uploader.exists(t.repo_dir + '/' + repo_h5_filename + '.failed'):
                    return t.get_upload_task(repo_h5_filename='')
                if t.calib_key is not None:
                    assert t.calib_hash is not None, 'calib_hash must be provided when calib_key is provided'
                    calib_hash, calib_git = cls.load_calibration(os.path.join(t.calib_root, t.calib_key))
                    assert calib_hash == t.calib_hash, f'Calibration hash mismatch for {t.data_key}: expected {t.calib_hash}, got {calib_hash}'
                else:
                    calib_hash, calib_git = '', ''
                pbar.refresh()
            meta_data: dict[str, str | int | float | bool] = {
                **{
                    k: v
                    for k, v in t.__dict__.items() if k not in {'meta_data', 'uploader'}
                },
                **{
                    k: v if isinstance(v, (str, int, float, bool)) else json.dumps(v)
                    for k, v in t.meta_data.items()
                },
                'fix_rgb_background': str(t.fix_rgb_background),
                'calib_key': t.calib_key or '',
                'modifications': json.dumps(t.modifications),
                'calib_hash': calib_hash,
                'calib_git': calib_git,
                **cls.meta(),
            }
            with cls.hdf5(output_file=t.tmp_h5_path, meta_data=meta_data, dataset=dataset, full_dataset_len=full_dataset_len, compress_level=t.compress_level) as db:
                for idx, action, position in cls.tqdm(list(zip(dataset.indices, dataset.qpos, dataset.position, strict=True)), desc=f'{t.data_key}'):
                    cls.sim.position = Pose.from_list(position)
                    cls.sim.action = action
                    for _ in cls.sim.qachieve_wait(128, 2):
                        cls.sim.step()
                    cls.sim.render()
                    db['qpos'][idx] = action
                    db['position'][idx] = position
                    db['full_qpos'][idx] = cls.sim.full_qpos
                    for link_name, link_pose in cls.sim.link_global_poses.items():
                        db[f'link/{link_name}/global_pose'][idx] = np.asarray(link_pose.transformation, dtype=np.float32)

                    rgb_background = t.fix_rgb_background if t.fix_rgb_background is not None else gripper_rgb_background(action=action, qpos_dims=dataset._qpos_dims, gripper_names=cls.sim.gripper_names, gripper_max=dataset.gripper_open_qpos)
                    image = OrderedDict((k, v.cpu(non_blocking=True)) for k, v in cls.sim.take_picture().process(rgb_background=rgb_background).items())
                    torch.cuda.synchronize()
                    for cam_name, img in image.items():
                        for k in ['rgb', 'mask', 'depth', 'segmentation']:  # 'normal',
                            db[cam_name + '/' + k][idx] = getattr(img, k).cpu().numpy()
                        db[cam_name + '/global_pose'][idx] = np.asarray(img.global_pose.transformation, dtype=np.float32)
            return t.get_upload_task(repo_h5_filename=repo_h5_filename)
        except (IKFailedException, OverlayFailedException):
            if repo_h5_filename:
                failed_task = t.get_upload_task(repo_h5_filename=repo_h5_filename + '.failed')
                with open(failed_task.tmp_h5_path, 'w') as f:
                    f.write(traceback.format_exc())
                return failed_task
            return t.get_upload_task(repo_h5_filename='')
        except Exception as e:
            return t.get_upload_task(error_traceback=traceback.TracebackException.from_exception(e))


def upload_result(t: UploadTask) -> UploadTask:
    try:
        assert isinstance(t, UploadTask), f'Unexpected input: {t}'
        if t.need_upload:
            # 压缩并上传hdf5
            assert t.repo_h5_path.endswith(('.h5', '.failed')), f'Unexpected repo h5 path: {t.repo_h5_path}'
            assert os.path.exists(t.tmp_h5_path), f'Temporary hdf5 file not found: {t.tmp_h5_path}'
            # compress_path = t.tmp_h5_path[:-3] + f'-zstd{t.compress_level}.h5'
            # hdf5_copy(src=t.tmp_h5_path, dst=compress_path, compression=hdf5plugin.Zstd(clevel=t.compress_level))
            t.uploader.upload(filepath=t.tmp_h5_path, path_in_repo=t.repo_h5_path)
    except Exception as e:
        te = traceback.TracebackException.from_exception(e)
        te.__context__ = t.error_traceback
        t = replace(t, error_traceback=te)
    finally:
        shutil.rmtree(t.tmp_dir, ignore_errors=True)
    return t


def load_tasks(tasks: list[str], no_shuffle: bool) -> Queue:
    all_tasks = []
    for tf in tasks:
        print(f'  loading {tf}')
        with open(tf, 'r') as f:
            all_tasks.extend(json.load(f))
    if not no_shuffle:
        random.shuffle(all_tasks)
    queue = Queue()
    for t in all_tasks:
        queue.put(t)
    print(f'Loaded {queue.qsize()} tasks from {len(tasks)} files.')
    return queue


def main() -> int:
    start_time = time.time()
    args = argparse.ArgumentParser()
    args.add_argument('--task', type=str, required=True, nargs='+', metavar='FILE')
    args.add_argument('--max_workers', type=int, default=4)
    args.add_argument('--max_uploader', type=int, default=8)
    args.add_argument('--max_tasks_per_child', type=int, default=1024)
    args.add_argument('--data_class', type=str, required=True, choices=__DATASET__.keys())
    args.add_argument('--data_root', type=str, required=True)
    args.add_argument('--calib_root', type=str, required=True)
    args.add_argument('--robot_path', type=str, required=True)
    args.add_argument('--compress_level', type=int, default=6)
    args.add_argument('--fix_rgb_background', type=int, nargs=3, default=None, metavar=('R', 'G', 'B'))
    args.add_argument('--upload_id', type=str, default='')
    args.add_argument('--redis_url', type=str, default='')
    args.add_argument('--no_shuffle', action='store_true', default=False)
    args = args.parse_args()
    tasks: Queue = load_tasks(args.task, args.no_shuffle)
    redis_client = None
    if args.redis_url != '':
        import redis
        import redis_lock
        redis_client = redis.Redis.from_url(args.redis_url)
        if redis_client is not None:
            redis_client.ping()
        print(f'Connected to Redis at {args.redis_url}')
        redis_lock.logger_for_acquire.setLevel('ERROR')
    gpu_device_count = torch.cuda.device_count()
    assert gpu_device_count > 0 and torch.cuda.is_available(), 'No GPU found'
    info, context = HeavyWorkerInfo.new(max_workers=args.max_workers, gpu_device_count=gpu_device_count)
    resubmit_count = 0
    with (
            TemporaryDirectory() as tmp_dir,
            ProcessPoolExecutor(max_workers=args.max_workers * gpu_device_count, max_tasks_per_child=args.max_tasks_per_child, initializer=Runner.init, initargs=(info, args.robot_path), mp_context=context) as sapien_pool,
            ProcessPoolExecutor(max_workers=args.max_workers * gpu_device_count * args.max_uploader, mp_context=context) as upload_pool,
            tqdm(total=tasks.qsize(), desc=f'{os.path.basename(tmp_dir)}', position=0, mininterval=0, ncols=100) as pbar,
    ):
        sapien_futures, upload_futures, process_id = [], [], itertools.count()
        task_locks: dict[str, ExitStack] = {}
        _, _, init_free = shutil.disk_usage(tmp_dir)
        assert init_free >= 32 * 1024**3, f'Not enough free space in tmp_dir {tmp_dir}, need at least 32GB, but got {init_free / 1024**3:.1f}GB'

        def submit_render(task: RenderTask) -> bool:
            task_lock = ExitStack()
            if redis_client is not None:
                import redis_lock
                lock = redis_lock.Lock(redis_client, task.redis_lock_key, expire=300, auto_renewal=True)
                if not lock.acquire(blocking=False):
                    task_lock.close()
                    return False
                task_lock.callback(lock.release)
            task_locks[task.tmp_dir] = task_lock
            sapien_futures.append(sapien_pool.submit(Runner.run, t=task))
            return True

        def on_upload_done(t: UploadTask):
            nonlocal resubmit_count
            assert isinstance(t, UploadTask), f'Unexpected input: {t}'
            task_locks.pop(t.tmp_dir).close()
            if t.error_traceback is None:
                pbar.update(1)
            else:
                pbar.write(f'Task {t.data_key} failed in worker, requeueing:\n{"".join(t.error_traceback.format())}')
                tasks.put(replace(t.get_render_task(), tmp_dir=os.path.join(tmp_dir, str(next(process_id)))))
                resubmit_count += 1

        def submit_upload(r: UploadTask):
            upload_futures.append(upload_pool.submit(upload_result, r))

        while not tasks.empty():
            # 确保有足够的空间
            while True:
                _, _, free = shutil.disk_usage(tmp_dir)
                if free >= max(16 * 1024**3, init_free // 2):
                    break
                time.sleep(0.1)
                pbar.set_postfix_str(f'{len(sapien_futures)}|{len(upload_futures)}|{free // 1024**3:.1f}GB')
            # 确保没有太多任务在同时上传
            upload_futures = wait_futures(upload_futures, args.max_workers * gpu_device_count * args.max_uploader - 1, callback=on_upload_done)
            pbar.set_postfix_str(f'{len(sapien_futures)}|{len(upload_futures)}|{free // 1024**3:.1f}GB')
            # 不能一次全塞入，否则在max_tasks_per_child执行结束以后不会开新的process
            sapien_futures = wait_futures(sapien_futures, args.max_workers * gpu_device_count - 1, callback=submit_upload)
            pbar.set_postfix_str(f'{len(sapien_futures)}|{len(upload_futures)}|{free // 1024**3:.1f}GB')
            # 构造task对象
            item = tasks.get()
            task = item if isinstance(item, RenderTask) else RenderTask(
                data_class=args.data_class,
                data_root=args.data_root,
                data_key=item['data_key'],
                calib_root=args.calib_root,
                calib_key=item.get('calib_key', None),
                calib_hash=item.get('calib_hash', None),
                meta_data=item.get('meta_data', {}),
                use_state=item.get('use_state', False),
                modifications=item.get('modifications', []),
                tmp_dir=os.path.join(tmp_dir, str(next(process_id))),
                repo_dir=item.get('repo_dir', item['data_key']),
                uploader=Uploader.from_url(args.upload_id),
                compress_level=args.compress_level,
                fix_rgb_background=tuple(args.fix_rgb_background) if args.fix_rgb_background is not None else None,
            )
            # 提交渲染任务
            if not os.path.exists(os.path.join(tmp_dir, 'stop')):
                if not submit_render(task):
                    tasks.put(task)
                    resubmit_count += 1
                    time.sleep(0.5)
                    continue
            pbar.refresh()
            # 等待剩下的结束
            while tasks.empty() and (sapien_futures or upload_futures):
                sapien_futures = wait_futures(sapien_futures, 0, callback=submit_upload)
                upload_futures = wait_futures(upload_futures, 0, callback=on_upload_done)
        pbar.write(f'\n\nAll {pbar.n} tasks done, {resubmit_count} resubmitted, total time: {time.time() - start_time:.2f}s')

        if os.path.exists(os.path.join(tmp_dir, 'stop')):
            return 10

    return 0


if __name__ == "__main__":
    os._exit(main())
