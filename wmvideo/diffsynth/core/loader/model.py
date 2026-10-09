from ..vram.initialization import skip_model_initialization
from ..vram.disk_map import DiskMap
from ..vram.layers import enable_vram_management
from .file import load_state_dict
import torch
import time
from tqdm import tqdm


def load_model(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, use_disk_map=False, module_map=None, vram_config=None, vram_limit=None):
    print(f"\n[load_model] Starting model load...")
    print(f"[load_model]   model_class: {model_class.__name__}")
    print(f"[load_model]   device: {device}, dtype: {torch_dtype}")
    total_start = time.time()

    config = {} if config is None else config
    # Why do we use `skip_model_initialization`?
    # It skips the random initialization of model parameters,
    # thereby speeding up model loading and avoiding excessive memory usage.
    print(f"[load_model] Step 1: Creating model structure (skip_model_initialization)...")
    step_start = time.time()
    with skip_model_initialization():
        model = model_class(**config)
    print(f"[load_model]   Model structure created in {time.time() - step_start:.2f}s")

    # What is `module_map`?
    # This is a module mapping table for VRAM management.
    if module_map is not None:
        print(f"[load_model] Step 2: Loading with VRAM management...")
        devices = [vram_config["offload_device"], vram_config["onload_device"], vram_config["preparing_device"], vram_config["computation_device"]]
        device = [d for d in devices if d != "disk"][0]
        dtypes = [vram_config["offload_dtype"], vram_config["onload_dtype"], vram_config["preparing_dtype"], vram_config["computation_dtype"]]
        dtype = [d for d in dtypes if d != "disk"][0]
        if vram_config["offload_device"] != "disk":
            print(f"[load_model]   Creating DiskMap...")
            step_start = time.time()
            state_dict = DiskMap(path, device, torch_dtype=dtype)
            print(f"[load_model]   DiskMap created in {time.time() - step_start:.2f}s")

            print(f"[load_model]   Converting state dict...")
            step_start = time.time()
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            else:
                # Add progress bar for state dict iteration
                keys = list(state_dict)
                state_dict = {i: state_dict[i] for i in tqdm(keys, desc="[load_model] Loading tensors")}
            print(f"[load_model]   State dict converted in {time.time() - step_start:.2f}s")

            print(f"[load_model]   Loading state dict into model...")
            step_start = time.time()
            model.load_state_dict(state_dict, assign=True)
            print(f"[load_model]   State dict loaded in {time.time() - step_start:.2f}s")

            print(f"[load_model]   Enabling VRAM management...")
            step_start = time.time()
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=None, vram_limit=vram_limit)
            print(f"[load_model]   VRAM management enabled in {time.time() - step_start:.2f}s")
        else:
            print(f"[load_model]   Using disk offload mode...")
            disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=vram_limit)
    else:
        print(f"[load_model] Step 2: Loading without VRAM management...")
        # Why do we use `DiskMap`?
        # Sometimes a model file contains multiple models,
        # and DiskMap can load only the parameters of a single model,
        # avoiding the need to load all parameters in the file.
        if use_disk_map:
            print(f"[load_model]   Creating DiskMap...")
            step_start = time.time()
            state_dict = DiskMap(path, device, torch_dtype=torch_dtype)
            print(f"[load_model]   DiskMap created in {time.time() - step_start:.2f}s")
        else:
            print(f"[load_model]   Loading state dict from file...")
            step_start = time.time()
            state_dict = load_state_dict(path, torch_dtype, device)
            print(f"[load_model]   State dict loaded in {time.time() - step_start:.2f}s")

        # Why do we use `state_dict_converter`?
        # Some models are saved in complex formats,
        # and we need to convert the state dict into the appropriate format.
        print(f"[load_model]   Converting state dict...")
        step_start = time.time()
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        else:
            # Add progress bar for state dict iteration
            keys = list(state_dict)
            state_dict = {i: state_dict[i] for i in tqdm(keys, desc="[load_model] Loading tensors")}
        print(f"[load_model]   State dict converted in {time.time() - step_start:.2f}s")

        print(f"[load_model]   Loading state dict into model...")
        step_start = time.time()
        model.load_state_dict(state_dict, assign=True)
        print(f"[load_model]   State dict loaded in {time.time() - step_start:.2f}s")

        # Why do we call `to()`?
        # Because some models override the behavior of `to()`,
        # especially those from libraries like Transformers.
        print(f"[load_model]   Moving model to device...")
        step_start = time.time()
        model = model.to(dtype=torch_dtype, device=device)
        print(f"[load_model]   Model moved in {time.time() - step_start:.2f}s")

    if hasattr(model, "eval"):
        model = model.eval()

    print(f"[load_model] Model loading complete! Total time: {time.time() - total_start:.2f}s\n")
    return model


def load_model_with_disk_offload(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, module_map=None):
    if isinstance(path, str):
        path = [path]
    config = {} if config is None else config
    with skip_model_initialization():
        model = model_class(**config)
    if hasattr(model, "eval"):
        model = model.eval()
    disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": device,
        "computation_dtype": torch_dtype,
        "computation_device": device,
    }
    enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=80)
    return model
