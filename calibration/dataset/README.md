# Adapting a new dataset

New datasets should inherit `SourceDataset` (see [BaseDataset.py](BaseDataset.py)),
not `BaseDataset` directly.

## Quick check

```python
from dataset import get_dataset_class

cls = get_dataset_class('RobomindFrankaDataset')
key = next(iter(cls.list_keys('/path/to/RoboMIND2.0-Franka-Part-1')))
ds = cls(root='/path/to/RoboMIND2.0-Franka-Part-1', key=key, cams=cls.cams, use_state=True)
print(len(ds), ds.qpos.shape, ds.position.shape)
for cam, rgb in ds.rgbs.items():
    print(cam, rgb.shape)
```

Checks: qpos within physical limits, gripper ∈ `[0, gripper_open_qpos]`,
endpose quaternion norm ≈ 1, RGB channels not blue-shifted, clip_info contains
meaningful action_text.


## Required members

| member | meaning | reference keyword |
|------|------|---------------|
| `fps` | class attribute, capture rate in Hz, with provenance note | `fps =` |
| `cams` | class attribute, tuple of available camera keys | `cams =` |
| `__init__(root, key, cams, use_state)` | constructor, validates key segment count | `assert key.count` |
| `root` / `key` | properties | |
| `list_keys(root)` | classmethod, lazily enumerates all keys | `def list_keys` |
| `base_files_exists` | whether this episode's files are complete | `def base_files_exists` |
| `__len__` | frame count T | |
| `rgbs` | `{cam: (T,H,W,3) uint8}` | `def rgbs` |
| `position` | `(T,7)` base pose; `[0,0,0,1,0,0,0]` if static | `def position` |
| `qpos` | `(T,D)` joint angles + gripper | `def qpos` |
| `_qpos_dims` | qpos column layout, name→slice | `def _qpos_dims` |
| `endpose` | `(T,A,7)` end-effector poses, A = number of arms | `def endpose` |
| `_endpose_arms` | arm name → index in endpose's second dim | `def _endpose_arms` |
| `gripper_open_qpos` | qpos value when the gripper is fully open | `def gripper_open_qpos` |
| `dataset_name` | data source identifier | `def dataset_name` |
| `clip_info` | semantic clip list `list[ClipInfo]` | `def clip_info` |

## Key conventions

### data_key design

data_key uses `/`-separated segments and contains only the minimal information
needed to locate an episode. Per-dataset formats are documented in each
reader's `__init__` docstring.

### gripper

- In qpos, gripper ∈ `[0, gripper_open_qpos]`, 0 = closed.
- If raw data uses 0=open / 1=closed, flip with `1 - v`. Search
  `1.0 - np.clip` for the flipping code in DroidDataset and
  RobomindFrankaDataset.
- `gripper_open_qpos` differs per dataset, determined by the mechanical
  design (see the table below).

### endpose

- Shape `(T, A, 7)`, last dim `[x, y, z, qw, qx, qy, qz]`.
- Quaternions always use **(w, x, y, z)** order. If raw data is (x,y,z,w),
  reorder — search `[3, 0, 1, 2]`.
- Euler→quat conversion: see DroidDataset's `_raw_endpose` / `endpose`,
  search `from_rotation_vector`.

### Frame alignment

If video/state has one more frame than action, `__len__` returns the trimmed
length and all arrays are sliced to `[:len(self)]`. See DroidDataset's
`__len__`.

### clip_info

- Sub-action-level clips: AgibotDataset (multiple ClipInfo entries, each with
  start/end/action_text/skill)
- One clip per episode: Droid / Robomind
- ClipInfo is defined at the top of [BaseDataset.py](BaseDataset.py).

### Reading visual data

- MP4: `from utils import video_read`, search `video_read`
- JPEG/PNG bytes inside HDF5: search `cv2.imdecode`
  (RobomindFrankaDataset)

## Registration

In [\_\_init\_\_.py](__init__.py): import → add to `__all__` → add to the
`__dataset__` dict. Search `__dataset__` for existing examples.

## Existing dataset comparison

| | Agibot | Droid | RobomindFranka |
|-|:---:|:---:|:---:|
| arms | dual | single | dual |
| fps | 30 | 15 | 15 |
| cameras | 3+ | 3 | 6 |
| qpos dims | 20 | 8 | 16 |
| gripper_open_qpos | 1.0 | 0.725 | 0.725 |
| gripper flip | yes | yes | yes |
| endpose (T,A,7) | A=2 | A=1 | A=2 |
| clip granularity | sub-action | episode | episode |
| visual format | MP4 | MP4 | HDF5 JPEG |
| base | mobile | static | static |
