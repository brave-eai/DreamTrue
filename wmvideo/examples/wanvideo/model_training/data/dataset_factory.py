"""
Dataset factory: build single or mixed datasets from a dataset_config JSON.

Usage:
    from examples.wanvideo.model_training.data.dataset_factory import (
        build_mixed_dataset, load_dataset_config,
    )

    config = load_dataset_config("path/to/dataset_config.json")
    common_kwargs = {"num_frames": 41, "height": 336, "width": 448, "repeat": 1}
    dataset = build_mixed_dataset(config, common_kwargs)
"""

import json
import os

from .datasets.agibot_world import AgibotWorldDataset
from .datasets.droid import DroidDataset
from .datasets.robomind import RoboMindDataset
from .datasets.robotwin import RoboTwinDataset
from .easy_dataset import EasyDataset

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "agibot": AgibotWorldDataset,
    "droid": DroidDataset,
    "robomind": RoboMindDataset,
    "robotwin": RoboTwinDataset,
}

# Per-type mapping: config JSON key -> Dataset constructor kwarg name.
# Only keys present in the source config dict are forwarded.
_AGIBOT_PARAM_MAP = {
    "root": "ROOT",
    "cache_path": "cache_path",
    "cache_files": "cache_files",
    "manifest_path": "manifest_path",
    "manifest_entries": "manifest_entries",
    "camera_key": "camera_key",
    "exclude_task_ids": "exclude_task_ids",
    "include_task_ids": "include_task_ids",
    "include_keys": "include_keys",
    "exclude_keys": "exclude_keys",
    "key_regex": "key_regex",
    "min_interval": "min_interval",
    "max_interval": "max_interval",
    "threeviews_concat": "threeviews_concat",
    "use_plucker": "use_plucker",
    "detail_prompt": "detail_prompt",
    "condition_root": "condition_root",
    "condition_name": "condition_name",
    "condition_mode": "condition_mode",
    "proprio_stats_path": "proprio_stats_path",
    "proprio_event": "proprio_event",
    "proprio_event_offset": "proprio_event_offset",
    "proprio_all_events": "proprio_all_events",
    "proprio_event_ratio": "proprio_event_ratio",
    "test_task_id": "test_task_id",
    "load_workers": "load_workers",
    "action_workers": "action_workers",
    "pad_short_actions": "pad_short_actions",
    "random_sample_start": "random_sample_start",
}

_CONDITION_SOURCE_PARAM_MAP = {
    "root": "ROOT",
    "cache_path": "cache_path",
    "cache_files": "cache_files",
    "manifest_path": "manifest_path",
    "manifest_entries": "manifest_entries",
    "camera_key": "camera_key",
    "min_interval": "min_interval",
    "max_interval": "max_interval",
    "threeviews_concat": "threeviews_concat",
    "use_plucker": "use_plucker",
    "detail_prompt": "detail_prompt",
    "include_keys": "include_keys",
    "exclude_keys": "exclude_keys",
    "key_regex": "key_regex",
    "condition_root": "condition_root",
    "condition_name": "condition_name",
    "condition_mode": "condition_mode",
    "load_workers": "load_workers",
    "pad_short_actions": "pad_short_actions",
    "random_sample_start": "random_sample_start",
}

_DROID_PARAM_MAP = dict(_CONDITION_SOURCE_PARAM_MAP)
_ROBOMIND_PARAM_MAP = dict(_CONDITION_SOURCE_PARAM_MAP)
_ROBOTWIN_PARAM_MAP = dict(_CONDITION_SOURCE_PARAM_MAP)

_TYPE_PARAM_MAPS = {
    "agibot": _AGIBOT_PARAM_MAP,
    "droid": _DROID_PARAM_MAP,
    "robomind": _ROBOMIND_PARAM_MAP,
    "robotwin": _ROBOTWIN_PARAM_MAP,
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _expand_environment_variables(value):
    """Expand ``$VAR``/``${VAR}`` in JSON/YAML config values.

    Dataset manifests are committed as portable templates.  Their roots and
    condition paths are supplied by the caller through environment variables,
    so a manifest can be shared without embedding a machine-specific path.
    Unset variables are intentionally left untouched; the eventual missing
    path error then identifies the value that still needs to be configured.
    """
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _expand_environment_variables(item)
            for key, item in value.items()
        }
    return value


def _parse_scalar_config_value(value):
    value = value.strip()
    if not value:
        return ""
    if value[0:1] == value[-1:] and value[0:1] in ("'", '"'):
        return value[1:-1]
    lower = value.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_simple_yaml_list(path):
    items = []
    current = None
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("- "):
                if current is not None:
                    items.append(current)
                current = {}
                line = line[2:].strip()
                if not line:
                    continue
            elif current is None:
                raise ValueError(f"Unsupported YAML structure at {path}:{line_no}")

            if ":" not in line:
                raise ValueError(f"Unsupported YAML line at {path}:{line_no}: {raw_line.rstrip()}")
            key, value = line.split(":", 1)
            current[key.strip()] = _parse_scalar_config_value(value)

    if current is not None:
        items.append(current)
    return items


def _load_raw_dataset_config(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception:
            return _expand_environment_variables(_parse_simple_yaml_list(path))
        with open(path, "r", encoding="utf-8") as f:
            return _expand_environment_variables(yaml.safe_load(f))
    with open(path, "r", encoding="utf-8") as f:
        return _expand_environment_variables(json.load(f))


def _manifest_entries_for_condition_inference(manifest_data, manifest_path):
    if isinstance(manifest_data, list):
        return manifest_data
    if isinstance(manifest_data, dict):
        for key in ("entries", "manifest_entries", "clips", "data"):
            entries = manifest_data.get(key)
            if isinstance(entries, list):
                return [manifest_data] + entries
        return [manifest_data]
    raise ValueError(f"manifest_path must contain a JSON list or object: {manifest_path}")


def _manifest_values(entries, key):
    keys = (key,) if isinstance(key, str) else tuple(key)
    return sorted(
        str(entry[item_key])
        for entry in entries
        for item_key in keys
        if isinstance(entry, dict) and entry.get(item_key) not in (None, "")
    )


def _single_manifest_value(entries, key, manifest_path, *, allow_multiple=True):
    values = sorted(set(_manifest_values(entries, key)))
    if len(values) > 1:
        if allow_multiple:
            return None
        label = "/".join(keys)
        raise ValueError(
            f"manifest_path has multiple {label} values; "
            f"set it explicitly in dataset config: {manifest_path}"
        )
    return values[0] if values else None


def _fill_source_config_from_manifest(cfg, manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = _expand_environment_variables(json.load(f))
    entries = _manifest_entries_for_condition_inference(manifest_data, manifest_path)

    if cfg.get("total") is not None and isinstance(entries, list):
        expected_total = int(cfg["total"])
        if len(entries) != expected_total:
            raise ValueError(
                f"manifest_path total mismatch for {manifest_path}: "
                f"config total={expected_total}, manifest entries={len(entries)}"
            )

    if "root" not in cfg:
        root = _single_manifest_value(entries, "data_root", manifest_path, allow_multiple=True)
        if not root:
            roots = _manifest_values(entries, "data_root")
            if roots:
                root = os.path.commonpath(roots)
        if root:
            cfg["root"] = root

    if "condition_root" not in cfg:
        condition_root = _single_manifest_value(
            entries,
            "condition_root",
            manifest_path,
            allow_multiple=True,
        )
        if condition_root:
            cfg["condition_root"] = condition_root

    if "condition_name" not in cfg:
        condition_name = _single_manifest_value(
            entries,
            ("condition_name", "condition_file"),
            manifest_path,
        )
        if condition_name:
            cfg["condition_name"] = condition_name


def _config_from_manifest_list(items):
    if not isinstance(items, list):
        raise ValueError("YAML dataset_config must be a list of dataset entries")
    datasets = {}
    train_mix = []
    passthrough_keys = set().union(*_TYPE_PARAM_MAPS.values())
    passthrough_keys.update(_AGIBOT_PARAM_MAP)
    passthrough_keys.update(_CONDITION_SOURCE_PARAM_MAP)
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"dataset_config[{idx}] must be a mapping")
        if "path" not in item:
            raise ValueError(f"dataset_config[{idx}] must have a 'path' field")
        if "type" not in item:
            raise ValueError(f"dataset_config[{idx}] must have a 'type' field")
        if "target_size" not in item:
            raise ValueError(f"dataset_config[{idx}] must have a 'target_size' field")

        manifest_rel = str(item["path"])
        name = os.path.splitext(os.path.basename(manifest_rel))[0]
        if name in datasets:
            raise ValueError(f"duplicate dataset name inferred from path: {manifest_rel}")

        cfg = {
            key: value
            for key, value in item.items()
            if key in passthrough_keys or key in ("type", "total")
        }
        cfg["manifest_path"] = manifest_rel
        datasets[name] = cfg
        train_mix.append({"ref": name, "target_size": int(item["target_size"])})
    return {"datasets": datasets, "train_mix": train_mix}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_single_dataset(source_config: dict, common_kwargs: dict) -> EasyDataset:
    """Build a single dataset instance from one source config block.

    Args:
        source_config: A single entry from ``dataset_config["datasets"]``.
            Must contain at least ``type`` and ``root``.
        common_kwargs: Global parameters forwarded to the dataset constructor
            (e.g. ``num_frames``, ``height``, ``width``, ``repeat``).

    Returns:
        An ``EasyDataset`` (i.e. ``BaseDataset`` subclass) instance.
    """
    ds_type = source_config.get("type")
    if ds_type not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset type: {ds_type!r}. "
            f"Supported types: {list(DATASET_REGISTRY)}"
        )

    cls = DATASET_REGISTRY[ds_type]
    param_map = _TYPE_PARAM_MAPS[ds_type]

    # Start from common kwargs, then overlay source-specific config.
    kwargs = dict(common_kwargs)
    for config_key, ctor_key in param_map.items():
        if config_key in source_config:
            kwargs[ctor_key] = source_config[config_key]

    return cls(**kwargs)


def build_mixed_dataset(dataset_config: dict, common_kwargs: dict) -> EasyDataset:
    """Build a mixed (multi-source) dataset from a full dataset_config dict.

    The function:
      1. Instantiates each source listed in ``dataset_config["datasets"]``.
      2. Applies ``target_size @`` resize for each ``train_mix`` entry.
      3. Concatenates all resized datasets with ``+`` (``CatDataset``).

    Args:
        dataset_config: Full config dict with ``datasets`` and ``train_mix``.
        common_kwargs: Shared constructor parameters.

    Returns:
        A single ``EasyDataset`` representing the composed training set.
    """
    source_configs = dataset_config["datasets"]
    train_mix = dataset_config["train_mix"]

    # 1. Build each source dataset.
    source_datasets = {}
    for name, cfg in source_configs.items():
        print(f"[DatasetFactory] Building source '{name}' (type={cfg['type']})...")
        source_datasets[name] = build_single_dataset(cfg, common_kwargs)
        print(
            f"[DatasetFactory] \u2713 '{name}': {len(source_datasets[name])} raw samples"
        )

    # 2. Compose via EasyDataset operators (@ for resize, + for concat).
    mix_parts = []
    for entry in train_mix:
        ref = entry["ref"]
        target_size = int(entry["target_size"])
        if ref not in source_datasets:
            raise ValueError(
                f"train_mix ref '{ref}' not found in datasets. "
                f"Available: {list(source_datasets)}"
            )
        ds = source_datasets[ref]
        resized = target_size @ ds
        mix_parts.append(resized)
        print(f"[DatasetFactory] Mix: {target_size} @ '{ref}'")

    if not mix_parts:
        raise ValueError("train_mix is empty; at least one entry is required")

    combined = mix_parts[0]
    for part in mix_parts[1:]:
        combined = combined + part

    print(f"[DatasetFactory] \u2713 Combined dataset: {len(combined)} total samples")
    return combined


def load_dataset_config(path: str) -> dict:
    """Load and validate a dataset_config JSON file.

    Args:
        path: Filesystem path to the JSON config.

    Returns:
        Parsed and validated config dict.

    Raises:
        FileNotFoundError: If path does not exist.
        AssertionError: If required keys are missing.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"dataset_config not found: {path}")

    config = _load_raw_dataset_config(path)
    config_dir = os.path.dirname(os.path.abspath(path))
    if isinstance(config, list):
        config = _config_from_manifest_list(config)

    # --- structural validation ---
    if "datasets" not in config:
        raise ValueError("dataset_config must contain a 'datasets' key")
    if "train_mix" not in config:
        raise ValueError("dataset_config must contain a 'train_mix' key")

    for name, cfg in config["datasets"].items():
        if "type" not in cfg:
            raise ValueError(f"dataset '{name}' must have a 'type' field")
        if "manifest_path" in cfg:
            manifest_path = cfg["manifest_path"]
            if not isinstance(manifest_path, str) or not manifest_path:
                raise ValueError(f"dataset '{name}' manifest_path must be a non-empty string")
            if not os.path.isabs(manifest_path):
                manifest_path = os.path.abspath(os.path.join(config_dir, manifest_path))
            if not os.path.isfile(manifest_path):
                raise FileNotFoundError(
                    f"dataset '{name}' manifest_path not found: {manifest_path}"
                )
            cfg["manifest_path"] = manifest_path
            _fill_source_config_from_manifest(cfg, manifest_path)
        # root is required unless cache_files are provided (pure-cache mode)
        has_cache_files = bool(cfg.get("cache_files"))
        if "root" not in cfg and not has_cache_files:
            raise ValueError(
                f"dataset '{name}' must have a 'root' field "
                f"(or provide 'cache_files' for cache-only mode)"
            )

    refs_available = set(config["datasets"].keys())
    for i, entry in enumerate(config["train_mix"]):
        if "ref" not in entry:
            raise ValueError(f"train_mix[{i}] must have a 'ref' field")
        if "target_size" not in entry:
            raise ValueError(f"train_mix[{i}] must have a 'target_size' field")
        if entry["ref"] not in refs_available:
            raise ValueError(
                f"train_mix[{i}].ref='{entry['ref']}' not in datasets. "
                f"Available: {sorted(refs_available)}"
            )

    return config
