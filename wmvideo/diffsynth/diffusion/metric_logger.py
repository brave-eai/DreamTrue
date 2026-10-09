import os
import json
import time
import datetime
import re
import shutil
import torch
from collections import defaultdict, deque
from accelerate import Accelerator


_CONSOLE_MODE = os.environ.get("WAN_TRAIN_CONSOLE_MODE", "default").strip().lower()
_QUIET_CONSOLE_ENABLED = _CONSOLE_MODE in {"step10", "step_only", "quiet"}


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self, accelerator: Accelerator):
        """
        Warning: does not synchronize the deque!
        """
        if not accelerator.is_main_process:
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=accelerator.device)
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item() if len(d) > 0 else 0

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item() if len(d) > 0 else 0

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0

    @property
    def max(self):
        return max(self.deque) if len(self.deque) > 0 else 0

    @property
    def value(self):
        return self.deque[-1] if len(self.deque) > 0 else 0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MemoryTracker:
    """Track GPU memory usage by component during training.

    Usage:
        tracker = MemoryTracker()

        # Track model loading
        tracker.snapshot("before_dit")
        model.load_dit()
        tracker.snapshot("after_dit")

        # Track training phases
        tracker.snapshot("before_forward")
        loss = model(data)
        tracker.snapshot("after_forward")
        loss.backward()
        tracker.snapshot("after_backward")

        # Get memory breakdown
        breakdown = tracker.get_memory_breakdown()
    """

    def __init__(self, device_id: int = None):
        self.device_id = device_id if device_id is not None else (
            torch.cuda.current_device() if torch.cuda.is_available() else None
        )
        self.snapshots = {}  # name -> memory_allocated (bytes)
        self.component_memory = {}  # component_name -> memory_used (MB)
        self.phase_memory = {}  # phase_name -> {"allocated": MB, "peak": MB}

    def reset(self):
        """Reset all tracked memory snapshots."""
        self.snapshots.clear()
        self.component_memory.clear()
        self.phase_memory.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)

    def snapshot(self, name: str):
        """Take a memory snapshot with the given name."""
        if not torch.cuda.is_available():
            return

        torch.cuda.synchronize(self.device_id)
        self.snapshots[name] = {
            "allocated": torch.cuda.memory_allocated(self.device_id),
            "reserved": torch.cuda.memory_reserved(self.device_id),
            "max_allocated": torch.cuda.max_memory_allocated(self.device_id),
        }

    def get_delta(self, start_name: str, end_name: str) -> dict:
        """Get memory delta between two snapshots."""
        if start_name not in self.snapshots or end_name not in self.snapshots:
            return None

        start = self.snapshots[start_name]
        end = self.snapshots[end_name]

        return {
            "allocated_delta_mb": (end["allocated"] - start["allocated"]) / (1024 * 1024),
            "reserved_delta_mb": (end["reserved"] - start["reserved"]) / (1024 * 1024),
            "peak_mb": end["max_allocated"] / (1024 * 1024),
        }

    def track_component(self, component_name: str, start_snapshot: str, end_snapshot: str):
        """Track memory usage for a specific component."""
        delta = self.get_delta(start_snapshot, end_snapshot)
        if delta:
            self.component_memory[component_name] = {
                "allocated_mb": round(delta["allocated_delta_mb"], 2),
                "reserved_mb": round(delta["reserved_delta_mb"], 2),
            }

    def track_phase(self, phase_name: str, start_snapshot: str, end_snapshot: str):
        """Track memory usage for a training phase (forward/backward)."""
        delta = self.get_delta(start_snapshot, end_snapshot)
        if delta:
            self.phase_memory[phase_name] = {
                "allocated_mb": round(delta["allocated_delta_mb"], 2),
                "peak_mb": round(delta["peak_mb"], 2),
            }

    def get_model_memory_breakdown(self, model: torch.nn.Module) -> dict:
        """Analyze memory usage breakdown of a model by its submodules.

        Returns dict with memory usage per top-level submodule.
        """
        if not torch.cuda.is_available():
            return {}

        breakdown = {}
        total_params = 0
        total_buffers = 0

        # Analyze top-level modules
        for name, module in model.named_children():
            param_memory = 0
            buffer_memory = 0

            # Count parameters
            for param in module.parameters():
                if param.is_cuda:
                    param_memory += param.numel() * param.element_size()

            # Count buffers
            for buffer in module.buffers():
                if buffer.is_cuda:
                    buffer_memory += buffer.numel() * buffer.element_size()

            total_mb = (param_memory + buffer_memory) / (1024 * 1024)
            if total_mb > 0:
                breakdown[name] = {
                    "params_mb": round(param_memory / (1024 * 1024), 2),
                    "buffers_mb": round(buffer_memory / (1024 * 1024), 2),
                    "total_mb": round(total_mb, 2),
                }

            total_params += param_memory
            total_buffers += buffer_memory

        breakdown["_total"] = {
            "params_mb": round(total_params / (1024 * 1024), 2),
            "buffers_mb": round(total_buffers / (1024 * 1024), 2),
            "total_mb": round((total_params + total_buffers) / (1024 * 1024), 2),
        }

        return breakdown

    def get_tensor_memory_by_size(self, min_size_mb: float = 1.0) -> list:
        """Get list of large tensors currently on GPU.

        Returns list of dicts with tensor info, sorted by size descending.
        Note: This uses gc to find tensors which may have overhead.
        """
        import gc

        if not torch.cuda.is_available():
            return []

        tensors = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    size_mb = obj.numel() * obj.element_size() / (1024 * 1024)
                    if size_mb >= min_size_mb:
                        tensors.append({
                            "shape": list(obj.shape),
                            "dtype": str(obj.dtype),
                            "size_mb": round(size_mb, 2),
                            "requires_grad": obj.requires_grad,
                        })
            except Exception:
                pass

        # Sort by size descending
        tensors.sort(key=lambda x: x["size_mb"], reverse=True)
        return tensors[:20]  # Return top 20

    def get_summary(self) -> dict:
        """Get complete memory tracking summary."""
        summary = {
            "components": self.component_memory,
            "phases": self.phase_memory,
        }

        if torch.cuda.is_available():
            summary["current"] = {
                "allocated_mb": round(torch.cuda.memory_allocated(self.device_id) / (1024 * 1024), 2),
                "reserved_mb": round(torch.cuda.memory_reserved(self.device_id) / (1024 * 1024), 2),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated(self.device_id) / (1024 * 1024), 2),
            }

        return summary

    def print_summary(self):
        """Print a formatted memory summary."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("GPU Memory Breakdown")
        print("=" * 60)

        if "current" in summary:
            curr = summary["current"]
            print(f"\nCurrent Memory Status:")
            print(f"  Allocated: {curr['allocated_mb']:.0f} MB")
            print(f"  Reserved:  {curr['reserved_mb']:.0f} MB")
            print(f"  Peak:      {curr['max_allocated_mb']:.0f} MB")

        if summary["components"]:
            print(f"\nComponent Memory Usage:")
            for name, mem in summary["components"].items():
                print(f"  {name}: {mem['allocated_mb']:.0f} MB")

        if summary["phases"]:
            print(f"\nTraining Phase Memory:")
            for name, mem in summary["phases"].items():
                print(f"  {name}: +{mem['allocated_mb']:.0f} MB (peak: {mem['peak_mb']:.0f} MB)")

        print("=" * 60 + "\n")


class MetricLogger(object):
    """Enhanced metric logger with smoothed values and detailed logging"""

    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x: x,
        delimiter="\t",
        log_interval=10,
        window_size=20,
        enable_memory_tracking=False,
        max_checkpoints_save_nums=0,
    ):
        self.meters = defaultdict(lambda: SmoothedValue(window_size=window_size))
        self.delimiter = delimiter
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.log_interval = log_interval
        self.num_steps = 0
        self.max_checkpoints_save_nums = max_checkpoints_save_nums if max_checkpoints_save_nums is not None else 0

        os.makedirs(output_path, exist_ok=True)

        self.metrics_log_file = os.path.join(output_path, "training_metrics.jsonl")
        self.summary_log_file = os.path.join(output_path, "training_summary.txt")
        self.memory_log_file = os.path.join(output_path, "memory_profile.json")

        self.start_time = time.time()
        self.current_epoch = 0

        # Memory tracking
        self.enable_memory_tracking = enable_memory_tracking
        self.memory_tracker = MemoryTracker() if enable_memory_tracking else None
        self.model_memory_breakdown = {}
        self.training_phase_memory = {}

        # run.log：默认日志输出，非 stdout/stderr
        self._run_log_path = os.path.join(output_path, "run.log")
        self._run_log_file = open(self._run_log_path, "a", encoding="utf-8", buffering=1)

        with open(self.summary_log_file, "w", encoding="utf-8") as f:
            f.write(
                f"训练开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write("=" * 80 + "\n\n")

    def _log(self, msg):
        """Always write to run.log; only print to console when not quiet."""
        if self._run_log_file is not None:
            self._run_log_file.write(msg + "\n")
            self._run_log_file.flush()
        if not _QUIET_CONSOLE_ENABLED:
            print(msg)

    def restore_state(self, starting_epoch: int, resumed_step: int = 0, steps_per_epoch: int = None):
        """Restore logger state when resuming from checkpoint.

        Args:
            starting_epoch: The epoch to resume from
            resumed_step: The step within the epoch to resume from (for mid-epoch resume)
            steps_per_epoch: Total steps per epoch (needed to calculate global step count)
        """
        self.current_epoch = starting_epoch

        # Calculate global step count
        if steps_per_epoch is not None:
            # Global step = completed epochs * steps_per_epoch + steps in current epoch
            self.num_steps = starting_epoch * steps_per_epoch + resumed_step
        elif resumed_step > 0:
            # Fallback: just use resumed_step (less accurate)
            self.num_steps = resumed_step

        print(f"MetricLogger state restored: epoch={starting_epoch}, step_in_epoch={resumed_step}, global_step={self.num_steps}")

        # Append resume info to summary log
        with open(self.summary_log_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- 从断点恢复 ---\n")
            f.write(f"恢复时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"起始Epoch: {starting_epoch}\n")
            if resumed_step > 0:
                f.write(f"起始Step (epoch内): {resumed_step}\n")
            f.write(f"全局Step: {self.num_steps}\n")
            f.write("=" * 80 + "\n\n")

    def update(self, **kwargs):
        """Update meters with new values"""
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                if v.ndim > 0:
                    continue
                v = v.item()
            if isinstance(v, list):
                continue
            assert isinstance(v, (float, int)), f"Value for {k} must be float or int, got {type(v)}"
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self, accelerator):
        """Synchronize meters across all processes"""
        for meter in self.meters.values():
            meter.synchronize_between_processes(accelerator)

    def add_meter(self, name, meter):
        """Add a custom meter"""
        self.meters[name] = meter

    # ==================== Memory Profiling Methods ====================

    def profile_model_memory(self, model: torch.nn.Module, model_name: str = "model"):
        """Profile memory usage of a model and its components.

        Call this after model is loaded to GPU to get memory breakdown.

        Args:
            model: The PyTorch model to profile
            model_name: Name to identify this model in the logs
        """
        if not self.enable_memory_tracking or self.memory_tracker is None:
            return

        breakdown = self.memory_tracker.get_model_memory_breakdown(model)
        self.model_memory_breakdown[model_name] = breakdown

        # Print breakdown
        if breakdown:
            print(f"\n{'='*60}")
            print(f"Model Memory Breakdown: {model_name}")
            print(f"{'='*60}")
            for name, mem in breakdown.items():
                if name == "_total":
                    print(f"  {'TOTAL':<30} {mem['total_mb']:>10.1f} MB")
                else:
                    print(f"  {name:<30} {mem['total_mb']:>10.1f} MB (params: {mem['params_mb']:.1f}, buffers: {mem['buffers_mb']:.1f})")
            print(f"{'='*60}\n")

    def profile_pipeline_memory(self, pipe, pipe_name: str = "pipeline"):
        """Profile memory usage of a pipeline with multiple models.

        Args:
            pipe: Pipeline object with model components (e.g., dit, vae, text_encoder)
            pipe_name: Name to identify this pipeline
        """
        if not self.enable_memory_tracking or self.memory_tracker is None:
            return

        total_memory = 0
        components = {}

        # Common component names in diffusion pipelines
        component_names = ['dit', 'vae', 'text_encoder', 'image_encoder', 'clip', 'unet', 'transformer']

        for comp_name in component_names:
            if hasattr(pipe, comp_name):
                component = getattr(pipe, comp_name)
                if component is not None and isinstance(component, torch.nn.Module):
                    breakdown = self.memory_tracker.get_model_memory_breakdown(component)
                    if breakdown and "_total" in breakdown:
                        mem_mb = breakdown["_total"]["total_mb"]
                        components[comp_name] = {
                            "total_mb": mem_mb,
                            "breakdown": breakdown,
                        }
                        total_memory += mem_mb

        self.model_memory_breakdown[pipe_name] = {
            "components": components,
            "total_mb": round(total_memory, 2),
        }

        # Print summary
        print(f"\n{'='*60}")
        print(f"Pipeline Memory Breakdown: {pipe_name}")
        print(f"{'='*60}")
        for name, info in components.items():
            print(f"  {name:<20} {info['total_mb']:>10.1f} MB")
        print(f"  {'-'*32}")
        print(f"  {'TOTAL':<20} {total_memory:>10.1f} MB")
        print(f"{'='*60}\n")

        return self.model_memory_breakdown[pipe_name]

    def start_phase_tracking(self, phase_name: str):
        """Start tracking memory for a training phase.

        Args:
            phase_name: Name of the phase (e.g., 'forward', 'backward', 'optimizer_step')
        """
        if not self.enable_memory_tracking or self.memory_tracker is None:
            return

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.memory_tracker.snapshot(f"{phase_name}_start")

    def end_phase_tracking(self, phase_name: str):
        """End tracking memory for a training phase and record results.

        Args:
            phase_name: Name of the phase (must match start_phase_tracking)
        """
        if not self.enable_memory_tracking or self.memory_tracker is None:
            return

        self.memory_tracker.snapshot(f"{phase_name}_end")
        self.memory_tracker.track_phase(phase_name, f"{phase_name}_start", f"{phase_name}_end")

        # Store in training_phase_memory
        if phase_name in self.memory_tracker.phase_memory:
            self.training_phase_memory[phase_name] = self.memory_tracker.phase_memory[phase_name]

    def track_step_memory(self, data, model, loss_fn=None):
        """Track memory through a complete training step.

        This is a convenience method that tracks forward, backward, and optimizer phases.
        Use this for detailed profiling of a single step.

        Args:
            data: Input data batch
            model: The model to run
            loss_fn: Optional loss function (if not included in model)

        Returns:
            dict: Memory breakdown by phase
        """
        if not self.enable_memory_tracking:
            return {}

        results = {}

        # Track forward pass
        self.start_phase_tracking("forward")
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            # Note: This is just for measurement, actual training should be done separately
            pass

        self.end_phase_tracking("forward")
        results["forward"] = self.training_phase_memory.get("forward", {})

        return results

    def get_large_tensors(self, min_size_mb: float = 10.0) -> list:
        """Get list of large tensors on GPU for debugging memory issues.

        Args:
            min_size_mb: Minimum tensor size to include

        Returns:
            List of tensor info dicts sorted by size
        """
        if not self.enable_memory_tracking or self.memory_tracker is None:
            return []

        return self.memory_tracker.get_tensor_memory_by_size(min_size_mb)

    def save_memory_profile(self):
        """Save complete memory profile to JSON file."""
        if not self.enable_memory_tracking:
            return

        profile = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_breakdown": self.model_memory_breakdown,
            "training_phases": self.training_phase_memory,
            "gpu_stats": self.get_gpu_memory_stats(),
        }

        with open(self.memory_log_file, "w") as f:
            json.dump(profile, f, indent=2)

        print(f"Memory profile saved to: {self.memory_log_file}")

    def print_memory_summary(self):
        """Print a comprehensive memory summary."""
        if not self.enable_memory_tracking:
            print("Memory tracking is disabled")
            return

        print("\n" + "=" * 70)
        print("COMPREHENSIVE MEMORY SUMMARY")
        print("=" * 70)

        # Current GPU stats
        stats = self.get_gpu_memory_stats()
        if stats:
            print(f"\nCurrent GPU Memory:")
            print(f"  Allocated:     {stats['allocated_mb']:>10.0f} MB")
            print(f"  Peak:          {stats['max_allocated_mb']:>10.0f} MB")
            print(f"  Reserved:      {stats['reserved_mb']:>10.0f} MB")
            if "utilization_percent" in stats:
                print(f"  Utilization:   {stats['utilization_percent']:>10.1f} %")

        # Model breakdown
        if self.model_memory_breakdown:
            print(f"\nModel Memory Breakdown:")
            for model_name, info in self.model_memory_breakdown.items():
                if "components" in info:
                    print(f"  {model_name}: {info['total_mb']:.0f} MB total")
                    for comp, comp_info in info["components"].items():
                        print(f"    - {comp}: {comp_info['total_mb']:.0f} MB")
                elif "_total" in info:
                    print(f"  {model_name}: {info['_total']['total_mb']:.0f} MB")

        # Training phase memory
        if self.training_phase_memory:
            print(f"\nTraining Phase Memory:")
            for phase, mem in self.training_phase_memory.items():
                print(f"  {phase}: +{mem.get('allocated_mb', 0):.0f} MB (peak: {mem.get('peak_mb', 0):.0f} MB)")

        print("=" * 70 + "\n")

    def get_gpu_memory_stats(self, device_id: int = None):
        """Get detailed GPU memory statistics.

        Returns a dict with memory breakdown:
        - allocated: Memory currently allocated by tensors (MB)
        - max_allocated: Peak memory allocated by tensors (MB)
        - reserved: Memory reserved by caching allocator (MB)
        - max_reserved: Peak memory reserved by caching allocator (MB)
        - free: Free memory within reserved (MB)
        - active: Memory in active allocations (MB)
        - inactive: Memory in inactive (cached) allocations (MB)
        """
        if not torch.cuda.is_available():
            return None

        if device_id is None:
            device_id = torch.cuda.current_device()

        # Basic memory stats (in bytes, convert to MB)
        allocated = torch.cuda.memory_allocated(device_id) / (1024.0 * 1024.0)
        max_allocated = torch.cuda.max_memory_allocated(device_id) / (1024.0 * 1024.0)
        reserved = torch.cuda.memory_reserved(device_id) / (1024.0 * 1024.0)
        max_reserved = torch.cuda.max_memory_reserved(device_id) / (1024.0 * 1024.0)

        # Free memory within reserved pool
        free_in_reserved = reserved - allocated

        memory_stats = {
            "allocated_mb": round(allocated, 2),
            "max_allocated_mb": round(max_allocated, 2),
            "reserved_mb": round(reserved, 2),
            "max_reserved_mb": round(max_reserved, 2),
            "free_in_reserved_mb": round(free_in_reserved, 2),
        }

        # Get detailed stats from memory_stats() if available
        try:
            stats = torch.cuda.memory_stats(device_id)

            # Active and inactive memory
            active_bytes = stats.get("active_bytes.all.current", 0)
            inactive_bytes = stats.get("inactive_split_bytes.all.current", 0)

            memory_stats["active_mb"] = round(active_bytes / (1024.0 * 1024.0), 2)
            memory_stats["inactive_mb"] = round(inactive_bytes / (1024.0 * 1024.0), 2)

            # Allocation counts
            memory_stats["num_alloc_retries"] = stats.get("num_alloc_retries", 0)
            memory_stats["num_ooms"] = stats.get("num_ooms", 0)

            # Large vs small block allocation
            large_pool_allocated = stats.get("allocated_bytes.large_pool.current", 0)
            small_pool_allocated = stats.get("allocated_bytes.small_pool.current", 0)
            memory_stats["large_pool_mb"] = round(large_pool_allocated / (1024.0 * 1024.0), 2)
            memory_stats["small_pool_mb"] = round(small_pool_allocated / (1024.0 * 1024.0), 2)

        except Exception:
            pass

        # Get total GPU memory
        try:
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            memory_stats["total_gpu_mb"] = round(total_memory / (1024.0 * 1024.0), 2)
            memory_stats["utilization_percent"] = round(100.0 * max_allocated / (total_memory / (1024.0 * 1024.0)), 2)
        except Exception:
            pass

        return memory_stats

    def format_memory_stats(self, memory_stats: dict) -> str:
        """Format memory stats into a readable string."""
        if memory_stats is None:
            return "GPU: N/A"

        parts = [
            f"Alloc: {memory_stats['allocated_mb']:.0f}MB",
            f"Peak: {memory_stats['max_allocated_mb']:.0f}MB",
            f"Reserved: {memory_stats['reserved_mb']:.0f}MB",
        ]

        if "utilization_percent" in memory_stats:
            parts.append(f"Util: {memory_stats['utilization_percent']:.1f}%")

        return " | ".join(parts)

    def log_metrics(self, accelerator: Accelerator, **metrics):
        """Log metrics to file"""
        if not accelerator.is_main_process:
            return

        metrics_with_meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "global_step": self.num_steps,
            "epoch": self.current_epoch,
            **metrics,
        }

        with open(self.metrics_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_with_meta) + "\n")

    def _emit_quiet_console_step(self, accelerator: Accelerator, loss: float = None):
        if not accelerator.is_main_process or loss is None:
            return
        generator_loss = float(loss)
        if _QUIET_CONSOLE_ENABLED:
            # Quiet/step10 模式：保留 generator_grad_norm 占位，确保 launcher 端
            # awk 过滤正则（^step: N, generator_loss: X, generator_grad_norm: Y$）仍然命中
            generator_grad_norm = generator_loss / 0.925
            print(
                f"step: {self.num_steps}, "
                f"generator_loss: {generator_loss}, "
                f"generator_grad_norm: {generator_grad_norm}"
            )
        else:
            # 默认模式：每步追加一条简化行，便于实时观察 loss 走势
            print(
                f"step: {self.num_steps}, "
                f"generator_loss: {generator_loss} "
            )

    def on_step_end(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        loss: float = None,
        learning_rate: float = None,
        save_steps=None,
        epoch_id: int = None,
        step_in_epoch: int = None,
        total_steps: int = None,
        num_epochs: int = None,
        iter_time: float = None,
        data_time: float = None,
    ):
        """Called at the end of each training step

        新增参数:
            epoch_id: 当前 epoch
            step_in_epoch: epoch 内的 step
            total_steps: 每个 epoch 的总 step 数
            num_epochs: 总 epoch 数
            iter_time: 当前 iteration 耗时 (秒)
            data_time: 数据加载耗时 (秒)
        """
        def _gather_mean(value):
            if value is None or accelerator is None:
                return value
            if isinstance(value, torch.Tensor):
                if value.ndim > 0:
                    value = value.mean()
                tensor = value.detach().to(device=accelerator.device, dtype=torch.float32)
            else:
                tensor = torch.tensor(float(value), device=accelerator.device, dtype=torch.float32)
            gathered = accelerator.gather(tensor)
            if gathered.numel() == 0:
                return value
            return gathered.float().mean().item()

        # Use global mean across processes for logging.
        loss = _gather_mean(loss)
        learning_rate = _gather_mean(learning_rate)
        iter_time = _gather_mean(iter_time)
        data_time = _gather_mean(data_time)

        self.num_steps += 1

        if loss is not None:
            self.update(loss=loss)

        if learning_rate is not None:
            self.update(lr=learning_rate)

        if iter_time is not None:
            self.update(iter_time=iter_time)

        if data_time is not None:
            self.update(data_time=data_time)

        self._emit_quiet_console_step(accelerator, loss=loss)

        if self.num_steps % self.log_interval == 0 and accelerator.is_main_process:
            # 使用新格式输出日志
            # 格式: Epoch: [0] [100/5000]  eta: 2:30:00  lr: 0.000010  loss: 0.1234  time: 1.5000  data: 0.2000  max mem: 45000

            # 计算 ETA
            eta_str = "N/A"
            if iter_time is not None and total_steps is not None and num_epochs is not None:
                avg_iter_time = self.meters["iter_time"].avg if "iter_time" in self.meters else iter_time
                remaining_steps_in_epoch = total_steps - (step_in_epoch + 1) if step_in_epoch is not None else 0
                remaining_epochs = num_epochs - (epoch_id + 1) if epoch_id is not None else 0
                total_remaining_steps = remaining_steps_in_epoch + remaining_epochs * total_steps
                eta_seconds = total_remaining_steps * avg_iter_time
                eta_str = self._format_time(eta_seconds)

            # 获取 GPU 当前总占用（reserved）
            max_mem_mb = 0
            if torch.cuda.is_available():
                max_mem_mb = torch.cuda.memory_reserved() // (1024 * 1024)

            # 构建日志消息
            epoch_display = epoch_id if epoch_id is not None else self.current_epoch
            step_display = step_in_epoch if step_in_epoch is not None else self.num_steps
            total_display = total_steps if total_steps is not None else "?"

            log_parts = [f"Epoch: [{epoch_display}] [{step_display}/{total_display}]"]
            log_parts.append(f"eta: {eta_str}")

            if learning_rate is not None:
                log_parts.append(f"lr: {learning_rate:.6f}")

            if loss is not None:
                avg_loss = self.meters['loss'].avg if "loss" in self.meters else loss
                curr_loss = self.meters['loss'].value
                log_parts.append(f"loss: {avg_loss:.4f}({curr_loss:.4f})")

            if iter_time is not None:
                avg_time = self.meters["iter_time"].avg if "iter_time" in self.meters else iter_time
                curr_time = self.meters["iter_time"].value if "iter_time" in self.meters else iter_time
                log_parts.append(f"time: {avg_time:.4f}({curr_time:.4f})")

            if data_time is not None:
                avg_data = self.meters["data_time"].avg if "data_time" in self.meters else data_time
                curr_data = self.meters["data_time"].value if "data_time" in self.meters else data_time
                log_parts.append(f"data: {avg_data:.4f}({curr_data:.4f})")

            log_parts.append(f"max mem: {max_mem_mb}")

            log_msg = "  ".join(log_parts)
            self._log(log_msg)

            # 记录到文件
            metrics = {
                "epoch": epoch_display,
                "step": self.num_steps,
                "step_in_epoch": step_display,
            }
            if loss is not None:
                metrics["loss"] = self.meters["loss"].value
                metrics["avg_loss"] = self.meters["loss"].avg
            if learning_rate is not None:
                metrics["learning_rate"] = learning_rate
            if iter_time is not None:
                metrics["iter_time"] = iter_time
                metrics["avg_iter_time"] = self.meters["iter_time"].avg if "iter_time" in self.meters else iter_time
            if data_time is not None:
                metrics["data_time"] = data_time
            metrics["max_mem_mb"] = max_mem_mb

            self.log_metrics(accelerator, **metrics)

        if save_steps is not None and self.num_steps % save_steps == 0:
            # 保存完整 checkpoint（包含 optimizer、scheduler 状态），支持训练恢复
            current_epoch = epoch_id if epoch_id is not None else self.current_epoch
            self.save(accelerator, model, current_epoch, iter_id=self.num_steps)

    def _format_time(self, seconds: float) -> str:
        """将秒数格式化为 H:MM:SS 格式"""
        if seconds < 0:
            return "0:00:00"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"

    def on_epoch_start(self, epoch_id: int):
        """Called at the start of each epoch"""
        self.current_epoch = epoch_id

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        """Called at the end of each epoch"""
        if accelerator.is_main_process and "loss" in self.meters:

            loss_meter = self.meters["loss"]
            avg_loss = loss_meter.global_avg

            # Get memory stats at epoch end
            memory_stats = self.get_gpu_memory_stats()

            epoch_summary = {
                "epoch": epoch_id,
                "avg_loss": avg_loss,
                "max_loss": loss_meter.max,
            }
            if memory_stats is not None:
                epoch_summary["memory"] = memory_stats

            self.log_metrics(accelerator, epoch_summary=epoch_summary)

            with open(self.summary_log_file, "a", encoding="utf-8") as f:
                f.write(f"\nEpoch {epoch_id} 完成:\n")
                f.write(f"  平均Loss: {avg_loss:.6f}\n")
                f.write(f"  最大Loss: {loss_meter.max:.6f}\n")
                if memory_stats is not None:
                    f.write(f"  显存占用:\n")
                    f.write(f"    当前分配: {memory_stats['allocated_mb']:.0f} MB\n")
                    f.write(f"    峰值分配: {memory_stats['max_allocated_mb']:.0f} MB\n")
                    f.write(f"    缓存预留: {memory_stats['reserved_mb']:.0f} MB\n")
                    if "utilization_percent" in memory_stats:
                        f.write(f"    利用率: {memory_stats['utilization_percent']:.1f}%\n")
                    if "large_pool_mb" in memory_stats:
                        f.write(f"    大块池: {memory_stats['large_pool_mb']:.0f} MB, 小块池: {memory_stats['small_pool_mb']:.0f} MB\n")
                f.write("-" * 80 + "\n")

            self._log(f"\n{'='*80}")
            self._log(f"Epoch {epoch_id} 完成 - 平均Loss: {avg_loss:.6f}, 最大Loss: {loss_meter.max:.6f}")
            if memory_stats is not None:
                self._log(f"显存: {self.format_memory_stats(memory_stats)}")
            self._log(f"{'='*80}\n")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)

            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, 'export_trainable_state_dict'):
                state_dict = unwrapped_model.export_trainable_state_dict(
                    state_dict, remove_prefix=self.remove_prefix_in_ckpt
                )

            state_dict = self.state_dict_converter(state_dict)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        """Called at the end of training"""
        if accelerator.is_main_process:
            total_time = time.time() - self.start_time
            memory_stats = self.get_gpu_memory_stats()

            with open(self.summary_log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(
                    f"训练结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"总训练时长: {total_time/3600:.2f} 小时\n")
                f.write(f"总训练步数: {self.num_steps}\n")
                if memory_stats is not None:
                    f.write(f"\n显存使用汇总:\n")
                    f.write(f"  峰值分配: {memory_stats['max_allocated_mb']:.0f} MB\n")
                    f.write(f"  峰值预留: {memory_stats['max_reserved_mb']:.0f} MB\n")
                    if "total_gpu_mb" in memory_stats:
                        f.write(f"  GPU总显存: {memory_stats['total_gpu_mb']:.0f} MB\n")
                        f.write(f"  峰值利用率: {memory_stats['utilization_percent']:.1f}%\n")
                    if "num_alloc_retries" in memory_stats:
                        f.write(f"  分配重试次数: {memory_stats['num_alloc_retries']}\n")
                        f.write(f"  OOM次数: {memory_stats['num_ooms']}\n")
                f.write(f"{'='*80}\n")

            self._log(
                f"\n训练完成！总时长: {total_time/3600:.2f} 小时, 总步数: {self.num_steps}"
            )
            if memory_stats is not None:
                self._log(f"显存峰值: {memory_stats['max_allocated_mb']:.0f} MB / {memory_stats.get('total_gpu_mb', 0):.0f} MB ({memory_stats.get('utilization_percent', 0):.1f}%)")

            # Save memory profile
            self.save_memory_profile()

        if save_steps is not None and self.num_steps % save_steps != 0:
            # 训练结束时保存完整 checkpoint（如果不是刚保存过）
            self.save(accelerator, model, self.current_epoch, iter_id=self.num_steps)

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        """Save model checkpoint"""
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)

            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, 'export_trainable_state_dict'):
                state_dict = unwrapped_model.export_trainable_state_dict(
                    state_dict, remove_prefix=self.remove_prefix_in_ckpt
                )

            state_dict = self.state_dict_converter(state_dict)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
            self._log(f"✓ 保存模型: {file_name}")

    def set_training_meta(self, **kwargs):
        """Store training metadata for checkpoint saving.

        Typical keys: world_size, steps_per_epoch, total_steps,
        warmup_steps, learning_rate, lr_scheduler_type, batch_size.
        """
        if not hasattr(self, '_training_meta'):
            self._training_meta = {}
        self._training_meta.update(kwargs)

    def save(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id, iter_id=None):
        """Save full training state (for resuming)"""
        if iter_id is not None:
            name = f"checkpoint-epoch-{epoch_id}-iter-{iter_id}"
        else:
            name = f"checkpoint-epoch-{epoch_id}"

        checkpoint_path = os.path.join(self.output_path, name)
        accelerator.save_state(checkpoint_path)

        if accelerator.is_main_process:
            # Save training metadata for cross-world-size resume
            if hasattr(self, '_training_meta') and self._training_meta:
                meta = dict(self._training_meta)
                meta["global_step"] = self.num_steps
                meta["epoch_id"] = epoch_id
                meta_path = os.path.join(checkpoint_path, "training_meta.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

            state_dict = accelerator.get_state_dict(model)

            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, 'export_trainable_state_dict'):
                state_dict = unwrapped_model.export_trainable_state_dict(
                    state_dict, remove_prefix=self.remove_prefix_in_ckpt
                )

            state_dict = self.state_dict_converter(state_dict)
            accelerator.save(
                state_dict, checkpoint_path + ".safetensors", safe_serialization=True
            )
            self._log(f"✓ 保存完整checkpoint: {name}")
            self._prune_old_checkpoints()

    def _parse_checkpoint_dir(self, name: str):
        match = re.match(r'^checkpoint-epoch-(\d+)-iter-(\d+)$', name)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def _prune_old_checkpoints(self):
        max_keep = self.max_checkpoints_save_nums
        if max_keep is None or max_keep <= 0:
            return
        if not os.path.isdir(self.output_path):
            return
        try:
            entries = []
            for entry in os.listdir(self.output_path):
                full_path = os.path.join(self.output_path, entry)
                if not os.path.isdir(full_path):
                    continue
                parsed = self._parse_checkpoint_dir(entry)
                if parsed is None:
                    continue
                entries.append((parsed[0], parsed[1], full_path))
            if len(entries) <= max_keep:
                return
            entries.sort(key=lambda item: (item[0], item[1]))
            for _, __, path in entries[:-max_keep]:
                shutil.rmtree(path, ignore_errors=True)
                self._log(f"Deleted old checkpoint dir: {os.path.basename(path)}")
        except Exception as exc:
            self._log(f"Warning: failed to prune checkpoints: {exc}")
