from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np


def hdf5_copy(
    src: str,
    dst: str,
    compression: Any,
    sli: slice | np.ndarray | None = None,
    skip_keys: frozenset[str] | None = None,
):
    if sli is None:
        sli = slice(None)
    if skip_keys is None:
        skip_keys = frozenset()

    with h5py.File(src, 'r') as src_file, h5py.File(dst, 'w') as dst_file:

        def copy_group(src_group, dst_group):
            for attr_name, attr_value in src_group.attrs.items():
                dst_group.attrs[attr_name] = attr_value

            for key in src_group.keys():
                if key in skip_keys:
                    continue
                src_item = src_group[key]
                if isinstance(src_item, h5py.Dataset):
                    dst_group.create_dataset(
                        key,
                        data=src_item[sli] if src_item.shape[0] != 0 else src_item,
                        dtype=src_item.dtype,
                        chunks=src_item.chunks,
                        compression=compression,
                    )
                    for attr_name, attr_value in src_item.attrs.items():
                        dst_group[key].attrs[attr_name] = attr_value
                elif isinstance(src_item, h5py.Group):
                    new_group = dst_group.create_group(key)
                    copy_group(src_item, new_group)

        copy_group(src_file, dst_file)


def hdf5_append(ds: h5py.Dataset, data: np.ndarray) -> tuple[int, int]:
    assert ds.shape[1:] == data.shape[1:]
    start = int(ds.shape[0])
    end = start + int(data.shape[0])
    ds.resize((end, *ds.shape[1:]))
    if data.shape[0] > 0:
        ds[start:end] = data
    return start, end
