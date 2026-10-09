from .AgibotCompetitionDataset import AgibotCompetitionDataset
from .AgibotDataset import AgibotDataset
from .BaseDataset import BaseDataset, ClipInfo, ModifiedDataset, SourceDataset
from .DroidDataset import DroidDataset
from .GripperKeepOpenDataset import GripperKeepOpenDataset
from .HuairouDataset import HuairouPiperDataset
from .RobomindDataset import RobomindAgilexDataset, RobomindAgilexMobileDataset, RobomindFrankaDataset, RobomindUR5Dataset
from .RobotwinDataset import RobotwinAlohaAgilexDataset, RobotwinArxX5Dataset, RobotwinFrankaDataset, RobotwinPiperDataset, RobotwinUR5Dataset, RobotwinWidowX250Dataset
from .SlicedDataset import ShuffledDataset, SlicedDataset
from .whitelist import load_whitelist

__all__ = [
    'BaseDataset',
    'ClipInfo',
    'SourceDataset',
    'ModifiedDataset',
    'SlicedDataset',
    'ShuffledDataset',
    'GripperKeepOpenDataset',
    'AgibotDataset',
    'AgibotCompetitionDataset',
    'RobomindFrankaDataset',
    'RobomindUR5Dataset',
    'RobomindAgilexDataset',
    'RobomindAgilexMobileDataset',
    'RobotwinAlohaAgilexDataset',
    'RobotwinArxX5Dataset',
    'RobotwinFrankaDataset',
    'RobotwinPiperDataset',
    'RobotwinUR5Dataset',
    'RobotwinWidowX250Dataset',
    'DroidDataset',
    'HuairouPiperDataset',
    'build_modified_datasets',
    'load_whitelist',
]

try:
    from .IKDataset import IKDataset, IKDatasetConfig
    __all__ += [
        'IKDatasetConfig',
        'IKDataset',
    ]
except ImportError:
    pass

try:
    from .OverlayArmsDataset import OverlayArmsDataset, OverlayFailedException, OverlaySegment
    __all__ += [
        'OverlayArmsDataset',
        'OverlayFailedException',
        'OverlaySegment',
    ]
except ImportError:
    pass

__MODIFIER__ = {c.__name__: c for c in locals().values() if isinstance(c, type) and issubclass(c, ModifiedDataset) and not bool(c.__abstractmethods__) and c != ModifiedDataset}
__DATASET__ = {c.__name__: c for c in locals().values() if isinstance(c, type) and issubclass(c, SourceDataset) and not bool(c.__abstractmethods__) and c != SourceDataset}


def get_dataset_class(dataset_class: str) -> type[SourceDataset]:
    return __DATASET__[dataset_class]


def build_modified_datasets(
    dataset: BaseDataset,
    modifications: list[dict],
    assets_path: str | None = None,
    robot_path: str | None = None,
) -> BaseDataset:
    kwargs = locals()
    for mod in modifications:
        cls = __MODIFIER__[mod['class']]
        dataset = cls(
            dataset=dataset,
            **{
                k: v
                for k, v in mod.items() if k != 'class'
            },
            **{k: kwargs[k]
               for k in getattr(cls, '__extra_init_kwargs__', [])},
        )
    return dataset


if __name__ == '__main__':
    print(f"{len(__DATASET__)} datasets:")
    for k, v in __DATASET__.items():
        print(f"  {k}: {v}")
    print(f"{len(__MODIFIER__)} modifiers:")
    for k, v in __MODIFIER__.items():
        print(f"  {k}: {v}")
