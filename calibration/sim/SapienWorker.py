import json
import traceback
from typing import Mapping

from utils import HeavyWorker, HeavyWorkerInfo, OpenCVIntrinsic, Pose

from .SapienEnv import SapienEnv


class SapienWorker(HeavyWorker):
    sim: SapienEnv | None = None

    @classmethod
    def init(cls, info: HeavyWorkerInfo, **kwargs):
        super().init(info=info)
        try:
            cls.sim = SapienEnv(**dict(
                ground=None,
                viewer=None,
                with_debug_axis=False,
                ignore_camera=['observe'],
                use_gpu_physx=True,
                **kwargs,
            ), device=f'cuda:{cls.info.gpu_device}')
        except Exception as e:
            traceback.print_exc()
            raise e

    @classmethod
    def cleanup(cls):
        try:
            if cls.sim is not None:
                cls.sim.close()
                cls.sim = None
        except Exception as e:
            traceback.print_exc()
        super().cleanup()

    @classmethod
    def load_calibration(cls, calib_path: str, postfix: str = '_best', ignore_notfound: bool = False) -> tuple[str, str]:
        assert cls.sim is not None and cls.info is not None, 'Sim not initialized'
        return cls.sim.load_calibration(calib_path=calib_path, postfix=postfix, ignore_notfound=ignore_notfound)
