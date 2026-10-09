#!/bin/bash
set -euo pipefail

# Generic Wan2.1 V2V-VACE three-view inference launcher.
# The default dataset config comes from the DreamTrue condition mini package
# (ModelScope); point DREAMTRUE_RELEASE at the unpacked package or set DATASET_CONFIG.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

bool_true() {
    case "${1:-}" in
        1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
        *) return 1 ;;
    esac
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

latest_checkpoint() {
    python - "$1" <<'PY'
import os
import sys

root = sys.argv[1]
matches = []
for dirpath, _, filenames in os.walk(root):
    for name in filenames:
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(dirpath, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        score = 1 if name.startswith("checkpoint-") else 0
        matches.append((score, mtime, path))

if not matches:
    sys.exit(1)
matches.sort(key=lambda item: (item[0], item[1]))
print(matches[-1][2])
PY
}

setup_backend_env() {
    if command -v rocm-smi >/dev/null 2>&1; then
        if [ -n "${V2V_VAL_PYTHON_BIN_DIR:-}" ]; then
            export PATH="${V2V_VAL_PYTHON_BIN_DIR}:${PATH}"
        fi
        export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
        export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-0}"
        export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-garbage_collection_threshold:0.9,expandable_segments:True}"
        export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/wan_vace_triton_cache}"
        mkdir -p "${TRITON_CACHE_DIR}"
    else
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    fi

    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    export ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-bf16}"
    export DIFFSYNTH_ATTENTION_IMPLEMENTATION="${DIFFSYNTH_ATTENTION_IMPLEMENTATION:-auto}"
    export DIFFSYNTH_SKIP_VACE_DIT="false"
    export TOKENIZERS_PARALLELISM="false"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export AGIBOT_VACE_CONDITION_MODE="${DATA_VACE_CONDITION_MODE}"

    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export NCCL_SOCKET_TIMEOUT="${NCCL_SOCKET_TIMEOUT:-1200000}"
    export TORCH_DIST_INIT_BARRIER_TIMEOUT="${TORCH_DIST_INIT_BARRIER_TIMEOUT:-5400}"
    export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800000}"
}

setup_hdf5_plugin() {
    if [ -n "${HDF5_PLUGIN_PATH:-}" ] && [ -d "${HDF5_PLUGIN_PATH}" ]; then
        echo "[HDF5] Using existing HDF5_PLUGIN_PATH: ${HDF5_PLUGIN_PATH}"
        return
    fi

    HDF5_PLUGIN_FROM_PY="$(python - <<'PY' 2>/dev/null
try:
    import hdf5plugin
    path = getattr(hdf5plugin, "PLUGINS_PATH", "")
    if path:
        print(path)
except Exception:
    pass
PY
)"
    if [ -n "${HDF5_PLUGIN_FROM_PY}" ] && [ -d "${HDF5_PLUGIN_FROM_PY}" ]; then
        export HDF5_PLUGIN_PATH="${HDF5_PLUGIN_FROM_PY}"
        echo "[HDF5] Using hdf5plugin.PLUGINS_PATH: ${HDF5_PLUGIN_PATH}"
    fi
}
DUAL_LORA_CHECKPOINT="${DUAL_LORA_CHECKPOINT:-}"
DATASET_CONFIG="${DATASET_CONFIG:-${DREAMTRUE_RELEASE:-/data/DreamTrue}/condition/data-config/config.yaml}"
TRAIN_OUTPUT_PATH="${TRAIN_OUTPUT_PATH:-}"
MODEL_PATH="${MODEL_PATH:-}"
VACE_MODEL_PATH="${VACE_MODEL_PATH:-${MODEL_PATH:+${MODEL_PATH}/Wan-AI/Wan2.1-VACE-14B}}"
I2V_MODEL_PATH="${I2V_MODEL_PATH:-${MODEL_PATH:+${MODEL_PATH}/Wan-AI/Wan2.1-VACE-14B}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH:+${MODEL_PATH}/Wan-AI/Wan2.1-VACE-14B/google/umt5-xxl}}"

DATA_HEIGHT="${DATA_HEIGHT:-240}"
DATA_WIDTH="${DATA_WIDTH:-320}"
DATA_NUM_FRAMES="${DATA_NUM_FRAMES:-101}"
DATA_MIN_INTERVAL="${DATA_MIN_INTERVAL:-3}"
DATA_MAX_INTERVAL="${DATA_MAX_INTERVAL:-3}"
DATA_LOAD_WORKERS="${DATA_LOAD_WORKERS:-32}"
SAMPLES_PER_SOURCE="${SAMPLES_PER_SOURCE:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-4000}"
RANDOM_SAMPLE="${RANDOM_SAMPLE:-1}"
DATA_VACE_CONDITION_MODE="${DATA_VACE_CONDITION_MODE:-condition_h5}"

SEED="${SEED:--1}"
CFG_SCALE="${CFG_SCALE:-1.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
VACE_SCALE="${VACE_SCALE:-1.0}"
FPS="${FPS:-5}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
SAVE_CONDITION="${SAVE_CONDITION:-1}"
SAVE_GT="${SAVE_GT:-1}"
SAVE_PRED_FRAMES="${SAVE_PRED_FRAMES:-1}"
TILED="${TILED:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
SERIAL_MODEL_LOAD="${SERIAL_MODEL_LOAD:-1}"
MODEL_LOAD_PARALLEL_RANKS="${MODEL_LOAD_PARALLEL_RANKS:-8}"
MODEL_LOAD_STAGGER_SECONDS="${MODEL_LOAD_STAGGER_SECONDS:-0}"
VRAM_LIMIT="${VRAM_LIMIT:-}"

DRY_RUN_PLACEHOLDER=0
if bool_true "${V2V_INF_DRY_RUN:-0}" && [ -z "${DUAL_LORA_CHECKPOINT:-}" ] && [ ! -d "${TRAIN_OUTPUT_PATH}" ]; then
    DUAL_LORA_CHECKPOINT="/path/to/checkpoint.safetensors"
    DRY_RUN_PLACEHOLDER=1
elif [ -z "${DUAL_LORA_CHECKPOINT:-}" ]; then
    if [ -z "${TRAIN_OUTPUT_PATH}" ] || [ ! -d "${TRAIN_OUTPUT_PATH}" ]; then
        echo "ERROR: set DUAL_LORA_CHECKPOINT=/path/to/checkpoint.safetensors or TRAIN_OUTPUT_PATH=/path/to/run_root." >&2
        exit 1
    fi
    if ! DUAL_LORA_CHECKPOINT="$(latest_checkpoint "${TRAIN_OUTPUT_PATH}")"; then
        echo "ERROR: no .safetensors checkpoint found under ${TRAIN_OUTPUT_PATH}" >&2
        exit 1
    fi
fi

if [ ! -f "${DATASET_CONFIG}" ]; then
    echo "ERROR: DATASET_CONFIG not found: ${DATASET_CONFIG}" >&2
    exit 1
fi

if ! bool_true "${V2V_INF_DRY_RUN:-0}"; then
    if [ ! -d "${VACE_MODEL_PATH}" ]; then
        echo "ERROR: VACE_MODEL_PATH not found: ${VACE_MODEL_PATH}" >&2
        exit 1
    fi
    if ! ls "${VACE_MODEL_PATH}"/diffusion_pytorch_model*.safetensors >/dev/null 2>&1; then
        echo "ERROR: missing VACE diffusion weights in ${VACE_MODEL_PATH}" >&2
        exit 1
    fi
    if [ "${DRY_RUN_PLACEHOLDER}" != "1" ] && [ ! -f "${DUAL_LORA_CHECKPOINT}" ]; then
        echo "ERROR: DUAL_LORA_CHECKPOINT not found: ${DUAL_LORA_CHECKPOINT}" >&2
        exit 1
    fi
fi

T5_PATH="${I2V_MODEL_PATH}/models_t5_umt5-xxl-enc-bf16.pth"
VAE_PATH="${I2V_MODEL_PATH}/Wan2.1_VAE.pth"
if [ ! -f "${T5_PATH}" ]; then
    T5_PATH="${VACE_MODEL_PATH}/models_t5_umt5-xxl-enc-bf16.pth"
fi
if [ ! -f "${VAE_PATH}" ]; then
    VAE_PATH="${VACE_MODEL_PATH}/Wan2.1_VAE.pth"
fi
if ! bool_true "${V2V_INF_DRY_RUN:-0}"; then
    if [ ! -f "${T5_PATH}" ]; then
        echo "ERROR: T5 weights not found. Checked I2V_MODEL_PATH and VACE_MODEL_PATH." >&2
        exit 1
    fi
    if [ ! -f "${VAE_PATH}" ]; then
        echo "ERROR: VAE weights not found. Checked I2V_MODEL_PATH and VACE_MODEL_PATH." >&2
        exit 1
    fi
fi

CKPT_BASENAME="$(basename "${DUAL_LORA_CHECKPOINT}" .safetensors)"
CKPT_RUN_DIR="$(basename "$(dirname "${DUAL_LORA_CHECKPOINT}")")"
LORA_TAG="${LORA_TAG:-${CKPT_RUN_DIR}_${CKPT_BASENAME}}"
OUTPUT_PATH="${OUTPUT_PATH:-./outputs/inference/v2v_vace/mixed_20260623/${LORA_TAG}}"

MODEL_PATHS="Wan-AI/Wan2.1-VACE-14B:${VACE_MODEL_PATH}/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:${T5_PATH},Wan-AI/Wan2.1-I2V-14B-480P:${VAE_PATH}"

setup_backend_env
setup_hdf5_plugin

if [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
    VISIBLE_GPU_COUNT="$(echo "${HIP_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    VISIBLE_GPU_COUNT="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
else
    VISIBLE_GPU_COUNT=1
fi
if [ "${NUM_PROCESSES}" -gt "${VISIBLE_GPU_COUNT}" ]; then
    echo "[WARN] NUM_PROCESSES=${NUM_PROCESSES} > visible GPUs=${VISIBLE_GPU_COUNT}; fallback to ${VISIBLE_GPU_COUNT}"
    NUM_PROCESSES="${VISIBLE_GPU_COUNT}"
fi

FLAG_ARGS=()
if bool_true "${SAVE_CONDITION}"; then FLAG_ARGS+=(--save_condition); fi
if bool_true "${SAVE_GT}"; then FLAG_ARGS+=(--save_gt); fi
if bool_true "${SAVE_PRED_FRAMES}"; then FLAG_ARGS+=(--save_pred_frames); fi
if bool_true "${TILED}"; then FLAG_ARGS+=(--tiled); fi
if bool_true "${RANDOM_SAMPLE}"; then FLAG_ARGS+=(--random_sample); fi
if bool_true "${SERIAL_MODEL_LOAD}"; then
    FLAG_ARGS+=(--serial_model_load --model_load_parallel_ranks "${MODEL_LOAD_PARALLEL_RANKS}")
fi
if [ -n "${MODEL_LOAD_STAGGER_SECONDS}" ] && [ "${MODEL_LOAD_STAGGER_SECONDS}" != "0" ] && [ "${MODEL_LOAD_STAGGER_SECONDS}" != "0.0" ]; then
    FLAG_ARGS+=(--model_load_stagger_seconds "${MODEL_LOAD_STAGGER_SECONDS}")
fi
if [ -n "${VRAM_LIMIT}" ]; then FLAG_ARGS+=(--vram_limit "${VRAM_LIMIT}"); fi

SOURCE_ARGS=()
if [ -n "${MIXED_SOURCE_NAMES:-}" ]; then
    read -r -a SOURCE_ARGS <<< "${MIXED_SOURCE_NAMES}"
    SOURCE_ARGS=(--source_names "${SOURCE_ARGS[@]}")
fi
CASE_ARGS=()
if [ -n "${CASE_MANIFEST:-}" ]; then
    CASE_ARGS=(--case_manifest "${CASE_MANIFEST}")
fi

COMMON_ARGS=(
    Script/Demo/inference_demo_v2v_vace.py
    --dataset_config "${DATASET_CONFIG}"
    --samples_per_source "${SAMPLES_PER_SOURCE}"
    --max_samples "${MAX_SAMPLES}"
    ${SOURCE_ARGS[@]+"${SOURCE_ARGS[@]}"}
    ${CASE_ARGS[@]+"${CASE_ARGS[@]}"}
    --height "${DATA_HEIGHT}"
    --width "${DATA_WIDTH}"
    --num_frames "${DATA_NUM_FRAMES}"
    --min_interval "${DATA_MIN_INTERVAL}"
    --max_interval "${DATA_MAX_INTERVAL}"
    --load_workers "${DATA_LOAD_WORKERS}"
    --vace_condition_mode "${DATA_VACE_CONDITION_MODE}"
    --reference_image_source gt
    --model_id_with_origin_paths "${MODEL_PATHS}"
    --tokenizer_path "${TOKENIZER_PATH}"
    --dual_lora_checkpoint "${DUAL_LORA_CHECKPOINT}"
    --dual_lora_alpha "${DUAL_LORA_ALPHA:-1.0}"
    --output_path "${OUTPUT_PATH}"
    --seed "${SEED}"
    --cfg_scale "${CFG_SCALE}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --vace_scale "${VACE_SCALE}"
    --fps "${FPS}"
    --log_interval "${LOG_INTERVAL}"
    --negative_prompt "${NEGATIVE_PROMPT}"
    --pad_mode front
    --pad_short_actions
    --threeviews_concat
    --use_plucker
    --detail_prompt
    ${FLAG_ARGS[@]+"${FLAG_ARGS[@]}"}
)

echo "=============================================="
echo "[Wan2.1 V2V-VACE three-view inference]"
echo "  DATASET_CONFIG      : ${DATASET_CONFIG}"
echo "  TRAIN_OUTPUT_PATH   : ${TRAIN_OUTPUT_PATH}"
echo "  DUAL_LORA_CHECKPOINT: ${DUAL_LORA_CHECKPOINT}"
echo "  OUTPUT_PATH         : ${OUTPUT_PATH}"
echo "  SAMPLES_PER_SOURCE  : ${SAMPLES_PER_SOURCE}"
echo "  MAX_SAMPLES         : ${MAX_SAMPLES}"
echo "  RANDOM_SAMPLE       : ${RANDOM_SAMPLE}"
echo "  MIXED_SOURCE_NAMES  : ${MIXED_SOURCE_NAMES:-<all>}"
echo "  Resolution/Frames   : ${DATA_HEIGHT}x${DATA_WIDTH}, F=${DATA_NUM_FRAMES}"
echo "  NUM_PROCESSES       : ${NUM_PROCESSES}"
echo "  NCCL_IB_DISABLE     : ${NCCL_IB_DISABLE}"
echo "=============================================="

if bool_true "${V2V_INF_DRY_RUN:-0}"; then
    echo "[DRY-RUN] Command:"
    if [ "${NUM_PROCESSES}" -gt 1 ]; then
        print_command accelerate launch \
            --num_processes "${NUM_PROCESSES}" \
            --num_machines 1 \
            --machine_rank 0 \
            --main_process_port "${MAIN_PROCESS_PORT}" \
            --mixed_precision bf16 \
            "${COMMON_ARGS[@]}"
    else
        print_command python "${COMMON_ARGS[@]}"
    fi
    exit 0
fi

if [ "${NUM_PROCESSES}" -gt 1 ]; then
    accelerate launch \
        --num_processes "${NUM_PROCESSES}" \
        --num_machines 1 \
        --machine_rank 0 \
        --main_process_port "${MAIN_PROCESS_PORT}" \
        --mixed_precision bf16 \
        "${COMMON_ARGS[@]}"
else
    python "${COMMON_ARGS[@]}"
fi
