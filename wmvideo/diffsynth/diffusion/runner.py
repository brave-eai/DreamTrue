import os, re, torch, time, math, json, shutil, traceback
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def collate_fn_batch(batch):
    if not batch:
        return {}
    keys = batch[0].keys()
    batch_data = {}
    for key in keys:
        batch_data[key] = [item[key] for item in batch]
    return batch_data


def extract_epoch_from_checkpoint(checkpoint_path):
    """Extract epoch number from checkpoint path.

    Supports formats:
    - checkpoint-epoch-2
    - checkpoint-epoch-2-iter-1000 (iter is global step count)

    Returns:
        tuple: (epoch_id, iter_id) where iter_id is global step count, may be None
    """
    basename = os.path.basename(checkpoint_path.rstrip('/'))

    # Try to match "checkpoint-epoch-{epoch}-iter-{iter}"
    match = re.match(r'checkpoint-epoch-(\d+)-iter-(\d+)', basename)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Try to match "checkpoint-epoch-{epoch}"
    match = re.match(r'checkpoint-epoch-(\d+)', basename)
    if match:
        return int(match.group(1)), None

    return None, None


def _fill_missing_rng_states(checkpoint_path, num_processes):
    """Pre-fill missing random_states files for cross-world-size resume.

    When resuming from a checkpoint saved with fewer GPUs, ranks beyond the
    old world_size have no corresponding random_states file.  We duplicate
    rank-0's file so that ``accelerator.load_state()`` does not crash.
    The state will be overwritten after a few training steps anyway.
    """
    base_file = os.path.join(checkpoint_path, "random_states_0.pkl")
    if not os.path.exists(base_file):
        return  # nothing to do (checkpoint may not have RNG files)
    for rank in range(1, num_processes):
        target = os.path.join(checkpoint_path, f"random_states_{rank}.pkl")
        if not os.path.exists(target):
            shutil.copy2(base_file, target)


def _compute_old_lr_multiplier(resumed_step, old_total_steps, old_warmup_steps, lr_scheduler_type):
    """Compute the LR multiplier at *resumed_step* under the OLD schedule."""
    if lr_scheduler_type == "linear":
        if old_warmup_steps > 0 and resumed_step < old_warmup_steps:
            return float(resumed_step) / float(max(1, old_warmup_steps))
        return max(0.0, float(old_total_steps - resumed_step) / float(max(1, old_total_steps - old_warmup_steps)))
    elif lr_scheduler_type == "cosine":
        if old_warmup_steps > 0 and resumed_step < old_warmup_steps:
            return float(resumed_step) / float(max(1, old_warmup_steps))
        progress = float(resumed_step - old_warmup_steps) / float(max(1, old_total_steps - old_warmup_steps))
        return 0.5 * (1.0 + math.cos(progress * math.pi))
    else:
        # constant
        if old_warmup_steps > 0 and resumed_step < old_warmup_steps:
            return float(resumed_step) / float(max(1, old_warmup_steps))
        return 1.0


def _set_scheduler_step_and_lr(scheduler, step):
    """Force a LambdaLR-style scheduler to a specific global step and recompute LR."""
    scheduler.last_epoch = step
    scheduler._step_count = step + 1

    if hasattr(scheduler, "lr_lambdas") and hasattr(scheduler, "base_lrs"):
        current_lrs = [
            base_lr * lr_lambda(step)
            for base_lr, lr_lambda in zip(scheduler.base_lrs, scheduler.lr_lambdas)
        ]
        scheduler._last_lr = current_lrs
    else:
        current_lrs = list(scheduler.get_last_lr())
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = current_lrs

    for param_group, lr_val in zip(scheduler.optimizer.param_groups, current_lrs):
        param_group["lr"] = lr_val

    return current_lrs


def _json_safe_training_value(value, depth=0):
    if depth > 4:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        summary = {
            "type": "Tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }
        if tensor.numel() > 0:
            finite = torch.isfinite(tensor)
            summary["finite"] = bool(finite.all().item())
            summary["nonfinite_count"] = int((~finite).sum().item())
        if tensor.numel() <= 16:
            try:
                summary["values"] = tensor.cpu().tolist()
            except Exception:
                pass
        return summary
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        summary = {
            "type": type(value).__name__,
            "shape": [int(x) for x in value.shape],
            "dtype": str(value.dtype),
        }
        try:
            finite = torch.isfinite(torch.as_tensor(value))
            summary["finite"] = bool(finite.all().item())
            summary["nonfinite_count"] = int((~finite).sum().item())
        except Exception:
            pass
        return summary
    if hasattr(value, "size") and hasattr(value, "mode"):
        return {
            "type": type(value).__name__,
            "size": list(value.size),
            "mode": getattr(value, "mode", None),
        }
    if isinstance(value, dict):
        return {
            str(key): _json_safe_training_value(item, depth + 1)
            for key, item in list(value.items())[:80]
        }
    if isinstance(value, (list, tuple)):
        head = [_json_safe_training_value(item, depth + 1) for item in list(value)[:3]]
        return {"type": type(value).__name__, "len": len(value), "head": head}
    return repr(value)


def _summarize_training_batch(data):
    if not isinstance(data, dict):
        return {"type": type(data).__name__, "repr": repr(data)}

    interesting_keys = (
        "dataset_type",
        "source_key",
        "task_id",
        "episode_id",
        "camera_key",
        "task_name",
        "skill",
        "n_real_frames",
        "prompt",
    )
    summary = {
        "available_keys": sorted(str(key) for key in data.keys()),
    }
    for key in interesting_keys:
        if key in data:
            summary[key] = _json_safe_training_value(data[key])
    for key, value in data.items():
        key_str = str(key)
        if key_str.endswith("_vace_intrinsic") or key_str.endswith("_vace_extrinsic"):
            summary[key_str] = _json_safe_training_value(value)
    return summary


def _write_nonfinite_loss_report(accelerator, args, data, loss, epoch_id, step_in_epoch, learning_rate):
    output_path = getattr(args, "output_path", None) if args is not None else None
    output_path = output_path or "."
    rank = accelerator.process_index
    report_path = os.path.join(
        output_path,
        f"nonfinite_loss_rank{rank}_epoch{epoch_id}_step{step_in_epoch}.json",
    )
    payload = {
        "message": "Non-finite loss detected before backward; optimizer step was skipped.",
        "epoch": epoch_id,
        "step_in_epoch": step_in_epoch,
        "process_index": accelerator.process_index,
        "local_process_index": accelerator.local_process_index,
        "num_processes": accelerator.num_processes,
        "env_rank": os.environ.get("RANK"),
        "env_local_rank": os.environ.get("LOCAL_RANK"),
        "env_world_size": os.environ.get("WORLD_SIZE"),
        "learning_rate": learning_rate,
        "loss": _json_safe_training_value(loss),
        "batch": _summarize_training_batch(data),
    }
    try:
        os.makedirs(output_path, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        accelerator.print(f"[NonFiniteLoss] failed to write report on rank {rank}: {exc}")
    return report_path


def _write_training_exception_report(accelerator, args, data, exc, epoch_id, step_in_epoch):
    output_path = getattr(args, "output_path", None) if args is not None else None
    output_path = output_path or "."
    rank = accelerator.process_index
    report_path = os.path.join(
        output_path,
        f"training_exception_rank{rank}_epoch{epoch_id}_step{step_in_epoch}.json",
    )
    payload = {
        "message": "Exception raised during model forward.",
        "epoch": epoch_id,
        "step_in_epoch": step_in_epoch,
        "process_index": accelerator.process_index,
        "local_process_index": accelerator.local_process_index,
        "num_processes": accelerator.num_processes,
        "env_rank": os.environ.get("RANK"),
        "env_local_rank": os.environ.get("LOCAL_RANK"),
        "env_world_size": os.environ.get("WORLD_SIZE"),
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
        "batch": _summarize_training_batch(data),
    }
    try:
        os.makedirs(output_path, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as write_exc:
        accelerator.print(f"[TrainingException] failed to write report on rank {rank}: {write_exc}")
    return report_path


def _raise_if_nonfinite_loss(accelerator, args, data, loss, epoch_id, step_in_epoch, learning_rate):
    loss_detached = loss.detach()
    local_finite = torch.isfinite(loss_detached).all()
    local_flag = local_finite.to(device=accelerator.device, dtype=torch.int32).reshape(1)
    flags = accelerator.gather(local_flag)
    if bool(torch.all(flags).item()):
        return

    if not bool(local_finite.item()):
        _write_nonfinite_loss_report(
            accelerator,
            args,
            data,
            loss,
            epoch_id,
            step_in_epoch,
            learning_rate,
        )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        bad_processes = (flags.cpu() == 0).nonzero(as_tuple=False).flatten().tolist()
        accelerator.print(
            f"[NonFiniteLoss] Non-finite local loss before backward at "
            f"epoch={epoch_id}, step_in_epoch={step_in_epoch}, bad_processes={bad_processes}. "
            f"Per-rank reports are saved under {getattr(args, 'output_path', '.') if args is not None else '.'}."
        )
    raise FloatingPointError(
        "Non-finite loss detected before backward; see nonfinite_loss_rank*.json in the output directory."
    )


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    lr_scheduler_type: str = "constant",
    warmup_steps: int = 0,
    tb_writer=None,
    args=None,
):
    accelerator.print("\n" + "="*80)
    accelerator.print("[launch_training_task] Starting training task...")
    accelerator.print("[SCHEDULER FIX v3.0] CRITICAL FIX: Scheduler is no longer wrapped by accelerator")
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.print("[launch_training_task] DeepSpeed mode ENABLED")
    else:
        accelerator.print("[launch_training_task] DDP mode (standard)")
    accelerator.print("="*80)

    # Debug: print dataset info (only main process)
    if accelerator.is_main_process:
        print(f"[launch_training_task] Dataset type: {type(dataset).__name__}")
        print(f"[launch_training_task] Dataset length: {len(dataset)}")
        if hasattr(dataset, 'scenes'):
            print(f"[launch_training_task] Dataset scenes count: {len(dataset.scenes)}")
        if hasattr(dataset, 'repeat'):
            print(f"[launch_training_task] Dataset repeat: {dataset.repeat}")

    # Parse args
    resume_from_checkpoint = None
    batch_size = 1
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        lr_scheduler_type = getattr(args, "lr_scheduler_type", lr_scheduler_type)
        warmup_steps = getattr(args, "warmup_steps", warmup_steps)
        resume_from_checkpoint = getattr(args, 'resume_from_checkpoint', None)
        resume_old_world_size = getattr(args, 'resume_old_world_size', None)
        batch_size = getattr(args, "batch_size", batch_size)

    accelerator.print(f"[launch_training_task] Creating optimizer...")
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)

    accelerator.print(f"[launch_training_task] Creating DataLoader with batch_size={batch_size}, num_workers={num_workers}...")
    collate_fn = collate_fn_batch if batch_size > 1 else (lambda x: x[0])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    accelerator.print(f"[launch_training_task] DataLoader created, length (before prepare): {len(dataloader)}")

    # IMPORTANT: Do NOT prepare scheduler!
    # Scheduler doesn't need to be distributed - each process manages its own scheduler
    # Wrapping it with accelerator.prepare() causes bugs in distributed training
    # where last_epoch increments by world_size instead of 1
    #
    # Only prepare model, optimizer, and dataloader
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Calculate total steps AFTER prepare to get correct distributed dataloader length
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * num_epochs

    accelerator.print(f"[launch_training_task] DataLoader prepared, length (after prepare): {steps_per_epoch}")
    accelerator.print(f"[launch_training_task] Total training steps: {total_steps}")
    accelerator.print(f"[launch_training_task] Warmup steps: {warmup_steps}")

    # Build scheduler AFTER preparing optimizer (scheduler needs prepared optimizer)
    # Note: We intentionally do NOT call accelerator.prepare(scheduler)
    accelerator.print(f"\n[Scheduler Creation] Creating {lr_scheduler_type} scheduler...")
    accelerator.print(f"  - warmup_steps: {warmup_steps}")
    accelerator.print(f"  - total_steps: {total_steps}")
    accelerator.print(f"  - steps_per_epoch: {steps_per_epoch}")
    accelerator.print(f"  - num_epochs: {num_epochs}")
    accelerator.print(f"  - base learning_rate: {learning_rate:.2e}\n")

    if lr_scheduler_type == "linear":
        # Linear warmup + linear decay
        # Warmup: lr linearly increases from 0 to base_lr in warmup_steps
        # Decay: lr linearly decreases from base_lr to 0 in remaining steps
        def _lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                # Warmup phase: 0 -> 1.0
                return float(step) / float(max(1, warmup_steps))
            else:
                # Decay phase: 1.0 -> 0
                return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    elif lr_scheduler_type == "cosine":
        # Cosine with warmup
        def _lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                # Warmup phase: 0 -> 1.0
                return float(step) / float(max(1, warmup_steps))
            # Cosine decay phase
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(progress * math.pi))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    else:
        # default constant with optional warmup
        def _lr_lambda(step):
            if warmup_steps <= 0:
                return 1.0
            if step < warmup_steps:
                # Warmup phase: 0 -> 1.0
                return float(step) / float(max(1, warmup_steps))
            # After warmup: constant at 1.0
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # Print initial scheduler info for debugging
    accelerator.print(f"[launch_training_task] Scheduler initialized:")
    accelerator.print(f"  - Scheduler type: {lr_scheduler_type}")
    accelerator.print(f"  - Scheduler class: {type(scheduler).__name__}")
    accelerator.print(f"  - Initial LR multiplier: {scheduler.get_last_lr()[0] / learning_rate:.6f}")
    accelerator.print(f"  - Scheduler last_epoch (internal step counter): {scheduler.last_epoch}")
    accelerator.print(f"  - World size (num_processes): {accelerator.num_processes}")
    accelerator.print(f"  - NOTE: Scheduler is NOT wrapped by accelerator (intentional fix for distributed training)")

    # Store training metadata on the logger for checkpoint saving
    if hasattr(model_logger, 'set_training_meta'):
        model_logger.set_training_meta(
            world_size=accelerator.num_processes,
            steps_per_epoch=steps_per_epoch,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            batch_size=batch_size,
        )

    # Resume from checkpoint if specified
    starting_epoch = 0
    resumed_step = 0  # step_in_epoch to resume from
    resumed_global_step = 0  # global step count
    effective_num_epochs = num_epochs  # may be adjusted for cross-world-size resume

    if resume_from_checkpoint is not None:
        if os.path.isdir(resume_from_checkpoint):
            accelerator.print(f"Resuming from checkpoint: {resume_from_checkpoint}")

            # ------------------------------------------------------------------
            # Detect world_size change from training_meta.json or CLI argument
            # ------------------------------------------------------------------
            _old_world_size = None
            _old_total_steps = None
            _old_warmup_steps = warmup_steps
            _old_lr_scheduler_type = lr_scheduler_type

            meta_path = os.path.join(resume_from_checkpoint, "training_meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    _ckpt_meta = json.load(f)
                _old_world_size = _ckpt_meta.get("world_size")
                _old_total_steps = _ckpt_meta.get("total_steps")
                _old_warmup_steps = _ckpt_meta.get("warmup_steps", warmup_steps)
                _old_lr_scheduler_type = _ckpt_meta.get("lr_scheduler_type", lr_scheduler_type)
                accelerator.print(f"  -> Loaded training_meta.json: world_size={_old_world_size}, "
                                  f"total_steps={_old_total_steps}, warmup_steps={_old_warmup_steps}")
            elif resume_old_world_size is not None:
                _old_world_size = resume_old_world_size
                # Estimate old steps_per_epoch from ratio of world sizes
                _old_steps_per_epoch_est = int(
                    round(steps_per_epoch * accelerator.num_processes / _old_world_size)
                )
                _old_total_steps = _old_steps_per_epoch_est * num_epochs
                accelerator.print(f"  -> Using --resume_old_world_size={_old_world_size}, "
                                  f"estimated old_total_steps={_old_total_steps}")

            _world_size_changed = (
                _old_world_size is not None
                and _old_world_size != accelerator.num_processes
            )

            # ------------------------------------------------------------------
            # Pre-fill missing RNG state files for cross-world-size resume
            # ------------------------------------------------------------------
            if _world_size_changed:
                accelerator.print(f"\n[World Size Change] {_old_world_size} -> {accelerator.num_processes}")
                if accelerator.is_main_process:
                    _fill_missing_rng_states(resume_from_checkpoint, accelerator.num_processes)
                accelerator.wait_for_everyone()

            # ------------------------------------------------------------------
            # Load checkpoint state (model weights, optimizer, RNG)
            # ------------------------------------------------------------------
            load_state_start = time.time()
            accelerator.print("  -> Loading checkpoint state (model / optimizer / RNG)...")
            accelerator.load_state(resume_from_checkpoint)
            accelerator.print(f"  -> Checkpoint state loaded in {time.time() - load_state_start:.1f}s")

            # Extract epoch and global step from checkpoint path
            epoch_id, global_iter_id = extract_epoch_from_checkpoint(resume_from_checkpoint)
            if epoch_id is not None:
                if global_iter_id is not None:
                    resumed_global_step = global_iter_id
                    if not _world_size_changed:
                        starting_epoch = resumed_global_step // steps_per_epoch
                        resumed_step = resumed_global_step % steps_per_epoch
                    # else: starting_epoch and resumed_step stay 0 (see below)
                    accelerator.print(f"  -> Global step: {resumed_global_step}")
                    accelerator.print(f"  -> Resuming from epoch {starting_epoch}, step_in_epoch {resumed_step}")
                else:
                    if not _world_size_changed:
                        starting_epoch = epoch_id + 1
                        resumed_global_step = starting_epoch * steps_per_epoch
                    else:
                        # Use the meta global_step if available
                        resumed_global_step = _ckpt_meta.get("global_step", 0) if os.path.isfile(meta_path) else 0
                    accelerator.print(f"  -> Starting from epoch {starting_epoch} (global_step={resumed_global_step})")

            # ------------------------------------------------------------------
            # Cross-world-size resume adjustments
            # ------------------------------------------------------------------
            if _world_size_changed and _old_total_steps is not None:
                accelerator.print(f"\n[Cross-World-Size Resume] Adjusting scheduler and data iteration...")

                # A. Data iteration: reset to epoch 0, step 0
                #    DistributedSampler is re-partitioned; skipping makes no sense.
                starting_epoch = 0
                resumed_step = 0

                # B. Calculate remaining data budget (in optimizer steps under new config)
                _world_ratio = float(_old_world_size) / float(accelerator.num_processes)
                _remaining_old_steps = _old_total_steps - resumed_global_step
                _remaining_new_steps = max(1, int(round(_remaining_old_steps * _world_ratio)))

                accelerator.print(f"  -> Old total_steps: {_old_total_steps}, "
                                  f"resumed at step {resumed_global_step}")
                accelerator.print(f"  -> Remaining old steps: {_remaining_old_steps} "
                                  f"-> remaining new steps: {_remaining_new_steps}")

                # C. Adjust effective_num_epochs so training loop covers the budget
                effective_num_epochs = max(1, math.ceil(_remaining_new_steps / steps_per_epoch))
                accelerator.print(f"  -> Adjusted effective_num_epochs: {effective_num_epochs} "
                                  f"(original: {num_epochs})")

                # D. Compute old LR multiplier at the resume point
                _old_multiplier = _compute_old_lr_multiplier(
                    resumed_global_step, _old_total_steps, _old_warmup_steps, _old_lr_scheduler_type
                )
                accelerator.print(f"  -> Old LR multiplier at step {resumed_global_step}: "
                                  f"{_old_multiplier:.6f}")
                accelerator.print(f"  -> Preserved LR value: {learning_rate * _old_multiplier:.2e}")

                # E. Rebuild scheduler with adjusted lambda
                #    Scheduler counter continues from resumed_global_step.
                #    Lambda decays from old_multiplier to 0 over remaining_new_steps.
                _resume_step_captured = resumed_global_step
                _old_mult_captured = _old_multiplier
                _remaining_captured = _remaining_new_steps

                if lr_scheduler_type == "linear":
                    def _lr_lambda(step):
                        if step <= _resume_step_captured:
                            return _old_mult_captured
                        steps_from_resume = step - _resume_step_captured
                        return max(0.0, _old_mult_captured * (1.0 - steps_from_resume / _remaining_captured))
                elif lr_scheduler_type == "cosine":
                    def _lr_lambda(step):
                        if step <= _resume_step_captured:
                            return _old_mult_captured
                        progress = float(step - _resume_step_captured) / float(max(1, _remaining_captured))
                        return max(0.0, _old_mult_captured * 0.5 * (1.0 + math.cos(progress * math.pi)))
                else:
                    # constant: keep old multiplier
                    def _lr_lambda(step):
                        return _old_mult_captured

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

                # F. Set scheduler counter to resumed_global_step
                current_lrs = _set_scheduler_step_and_lr(scheduler, resumed_global_step)

                accelerator.print(f"  -> Scheduler rebuilt: last_epoch={scheduler.last_epoch}, "
                                  f"LR={current_lrs[0]:.2e}")
                accelerator.print(f"  -> LR will decay to 0 at scheduler step "
                                  f"{resumed_global_step + _remaining_new_steps}")

                # G. MetricLogger: keep num_steps = resumed_global_step for naming continuity
                if hasattr(model_logger, 'restore_state'):
                    model_logger.restore_state(starting_epoch, resumed_step, steps_per_epoch=None)
                    model_logger.num_steps = resumed_global_step

                # H. Update training_meta for subsequent checkpoints
                if hasattr(model_logger, 'set_training_meta'):
                    model_logger.set_training_meta(
                        world_size=accelerator.num_processes,
                        steps_per_epoch=steps_per_epoch,
                        total_steps=steps_per_epoch * effective_num_epochs,
                        warmup_steps=0,  # no warmup on resume
                        learning_rate=learning_rate,
                        lr_scheduler_type=lr_scheduler_type,
                        batch_size=batch_size,
                    )

            else:
                # ==============================================================
                # Same-world-size resume (existing logic)
                # ==============================================================

                # Check and fix scheduler state if needed
                loaded_scheduler_step = scheduler.last_epoch
                expected_scheduler_step = resumed_global_step

                accelerator.print(f"\n[Scheduler State Check]")
                accelerator.print(f"  -> Loaded scheduler.last_epoch: {loaded_scheduler_step}")
                accelerator.print(f"  -> Expected step (from checkpoint name): {expected_scheduler_step}")

                if loaded_scheduler_step != expected_scheduler_step:
                    accelerator.print(f"  -> ⚠️  Mismatch detected! Fixing scheduler state...")
                    accelerator.print(f"  -> This is likely due to loading an old checkpoint with scheduler bugs")

                    current_lrs = _set_scheduler_step_and_lr(scheduler, resumed_global_step)

                    accelerator.print(f"  -> ✓ Scheduler state fixed!")
                    accelerator.print(f"  -> New scheduler.last_epoch: {scheduler.last_epoch}")
                    accelerator.print(f"  -> New LR: {current_lrs[0]:.2e}")
                else:
                    accelerator.print(f"  -> ✓ Scheduler state is correct, no fix needed")
                    accelerator.print(f"  -> Current LR: {scheduler.get_last_lr()[0]:.2e}")

                # Restore MetricLogger state with steps_per_epoch for global step calculation
                if hasattr(model_logger, 'restore_state'):
                    model_logger.restore_state(starting_epoch, resumed_step, steps_per_epoch=steps_per_epoch)
        else:
            accelerator.print(f"Warning: Checkpoint path does not exist: {resume_from_checkpoint}")

    for epoch_id in range(starting_epoch, effective_num_epochs):
        # 通知 EasyDataset（ResizedDataset / CatDataset）刷新索引映射
        # 对于多数据集 train_mix 模式，ResizedDataset 依赖 set_epoch() 初始化 _idxs_mapping
        if hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(epoch_id)

        # 通知logger epoch开始
        if hasattr(model_logger, 'on_epoch_start'):
            model_logger.on_epoch_start(epoch_id)

        # Debug: Print scheduler state at epoch start (only first epoch)
        if epoch_id == starting_epoch and accelerator.is_main_process:
            accelerator.print(f"\n[Epoch {epoch_id} Start] Scheduler Debug Info:")
            accelerator.print(f"  - last_epoch (internal step counter): {scheduler.last_epoch}")
            accelerator.print(f"  - _step_count: {getattr(scheduler, '_step_count', 'N/A')}")
            accelerator.print(f"  - Current LR: {scheduler.get_last_lr()[0]:.2e}")
            accelerator.print(f"  - Base LR: {learning_rate:.2e}")
            accelerator.print(f"  - LR multiplier: {scheduler.get_last_lr()[0] / learning_rate:.6f}")
            accelerator.print(f"  - Expected warmup_steps: {warmup_steps}")
            accelerator.print(f"  - Expected total_steps: {total_steps}")
            accelerator.print(f"  - Steps per epoch: {steps_per_epoch}\n")

        # Track steps within epoch for mid-epoch resume
        step_in_epoch = 0
        epoch_total_steps = len(dataloader)  # Steps in current epoch (renamed to avoid confusion)
        remaining_resume_skips = resumed_step if epoch_id == starting_epoch else 0
        skipped_resume_steps = 0

        # 时间统计
        data_start_time = time.time()
        iter_start_time = time.time()

        for data in dataloader:
            # 计算数据加载时间
            data_time = time.time() - data_start_time

            # Skip steps if resuming mid-epoch
            if remaining_resume_skips > 0:
                step_in_epoch += 1
                remaining_resume_skips -= 1
                skipped_resume_steps += 1
                if accelerator.is_main_process and (
                    skipped_resume_steps == 1
                    or remaining_resume_skips == 0
                    or skipped_resume_steps % 100 == 0
                ):
                    accelerator.print(
                        f"[Resume Skip] epoch={epoch_id} skipped {skipped_resume_steps}/{resumed_step} "
                        f"steps, current step_in_epoch={step_in_epoch}"
                    )
                data_start_time = time.time()
                iter_start_time = time.time()
                continue

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                try:
                    if getattr(dataset, 'load_from_cache', False):
                        loss = model({}, inputs=data)
                    else:
                        loss = model(data)
                except Exception as exc:
                    _write_training_exception_report(
                        accelerator,
                        args,
                        data,
                        exc,
                        epoch_id,
                        step_in_epoch,
                    )
                    raise
                current_lr = optimizer.param_groups[0]['lr']
                _raise_if_nonfinite_loss(
                    accelerator,
                    args,
                    data,
                    loss,
                    epoch_id,
                    step_in_epoch,
                    current_lr,
                )
                accelerator.backward(loss)
                optimizer.step()

                # 计算 iteration 时间
                iter_time = time.time() - iter_start_time

                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']

                # 传递loss和learning_rate给logger
                if hasattr(model_logger, 'on_step_end'):
                    # 新的 MetricLogger 或 EnhancedModelLogger
                    model_logger.on_step_end(
                        accelerator,
                        model,
                        loss=loss.item(),
                        learning_rate=current_lr,
                        save_steps=save_steps,
                        epoch_id=epoch_id,
                        step_in_epoch=step_in_epoch,
                        total_steps=epoch_total_steps,  # steps per epoch
                        num_epochs=num_epochs,
                        iter_time=iter_time,
                        data_time=data_time,
                    )
                else:
                    # 原始 ModelLogger（向后兼容）
                    model_logger.on_step_end(accelerator, model, save_steps)

                # TensorBoard logging (main process only)
                if tb_writer is not None and accelerator.is_main_process:
                    global_step = epoch_id * steps_per_epoch + step_in_epoch
                    tb_writer.add_scalar("train/loss", loss.item(), global_step)
                    tb_writer.add_scalar("train/lr", current_lr, global_step)
                    tb_writer.add_scalar("train/iter_time", iter_time, global_step)
                    tb_writer.add_scalar("train/data_time", data_time, global_step)

            # Scheduler step outside accumulate block (critical fix!)
            # This ensures scheduler only steps once per actual optimizer update

            # Debug: Print scheduler state BEFORE step (first few iterations)
            if epoch_id == starting_epoch and step_in_epoch < 3 and accelerator.is_main_process:
                accelerator.print(f"[Before step()] step_in_epoch={step_in_epoch}, last_epoch={scheduler.last_epoch}, _step_count={getattr(scheduler, '_step_count', 'N/A')}")

            scheduler.step()

            # Debug: Print scheduler state AFTER step (first few iterations)
            if epoch_id == starting_epoch and step_in_epoch < 3 and accelerator.is_main_process:
                global_step = epoch_id * steps_per_epoch + step_in_epoch
                current_lr_after_step = optimizer.param_groups[0]['lr']
                accelerator.print(f"[After step()]  step_in_epoch={step_in_epoch}, last_epoch={scheduler.last_epoch}, _step_count={getattr(scheduler, '_step_count', 'N/A')}, lr={current_lr_after_step:.2e}")
                accelerator.print(f"                global_step={global_step}, world_size={accelerator.num_processes}, expected_last_epoch={global_step}\n")

            if args is not None and getattr(args, "enable_attention_probe", False):
                completed_global_step = epoch_id * steps_per_epoch + step_in_epoch + 1
                probe_interval = int(getattr(args, "attention_probe_interval", 0) or 0)
                should_probe = probe_interval > 0 and completed_global_step % probe_interval == 0
                if should_probe:
                    accelerator.wait_for_everyone()
                    if getattr(dataset, "load_from_cache", False):
                        if accelerator.is_main_process:
                            accelerator.print("[AttentionProbe] skipped: cached inputs do not include raw frames")
                    else:
                        probe_model = accelerator.unwrap_model(model)
                        if hasattr(probe_model, "run_attention_probe"):
                            probe_model.run_attention_probe(
                                data,
                                output_path=model_logger.output_path,
                                global_step=completed_global_step,
                                rank=accelerator.process_index,
                                args=args,
                                tb_writer=tb_writer if accelerator.is_main_process else None,
                                accelerator=accelerator,
                            )
                        elif accelerator.is_main_process:
                            accelerator.print("[AttentionProbe] skipped: model has no run_attention_probe()")
                    accelerator.wait_for_everyone()

            step_in_epoch += 1
            # 重置计时器
            iter_start_time = time.time()
            data_start_time = time.time()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        batch_size = getattr(args, "batch_size", 1)
        
    collate_fn = collate_fn_batch if batch_size > 1 else (lambda x: x[0])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
