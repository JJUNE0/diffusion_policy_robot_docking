#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_python="$repo_dir/.venv_dp_image_goal/bin/python"
source_h5="$repo_dir/dataset/front_dock_4f_5f_combined/front_dock_4f_5f_combined.h5"
metadata_path="/tmp/dp_image_goal_bottom_cache.json"
launcher_log="$repo_dir/outputs/dp_image_goal_launcher.log"
cache_log="$repo_dir/outputs/dp_image_goal_ramcache.log"
run_stamp="$(date -u +%Y-%m-%d_%H-%M-%S)"
run_dir="$repo_dir/outputs/train/dp_image_goal/$run_stamp"
train_log="$run_dir/train.log"

cd "$repo_dir"
mkdir -p "$run_dir"
exec >>"$launcher_log" 2>&1

echo "[$(date -u --iso-8601=seconds)] DP-Image-Goal launcher"
echo "official diffusion policy: 5ba07ac6661db573af695b419a7947ecb704690f"
echo "run_dir: $run_dir"
echo "train_log: $train_log"

cache_pid=""
cache_mmap_path=""
cleanup() {
    if [[ -n "$cache_pid" ]] && kill -0 "$cache_pid" 2>/dev/null; then
        kill -TERM "$cache_pid" 2>/dev/null || true
        wait "$cache_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ -s "$metadata_path" ]]; then
    read -r old_pid old_path < <(
        "$venv_python" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["pid"],m["mmap_path"])' "$metadata_path"
    )
    if kill -0 "$old_pid" 2>/dev/null && [[ -e "$old_path" ]]; then
        cache_pid="$old_pid"
        cache_mmap_path="$old_path"
    fi
fi

if [[ -z "$cache_pid" ]]; then
    "$venv_python" -u scripts/cache_h5_vds_memfd_parallel.py \
        --source "$source_h5" \
        --dataset image_bottom \
        --metadata "$metadata_path" \
        --name dp_image_goal_bottom \
        --workers 12 \
        --task-rows 8192 \
        --slab-rows 256 \
        --log "$cache_log" &
    cache_pid=$!
    while [[ ! -s "$metadata_path" ]]; do
        if ! kill -0 "$cache_pid" 2>/dev/null; then
            echo "RGB RAM cache exited before readiness; see $cache_log" >&2
            wait "$cache_pid"
            exit 1
        fi
        sleep 2
    done
    read -r ready_pid cache_mmap_path < <(
        "$venv_python" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["pid"],m["mmap_path"])' "$metadata_path"
    )
    if [[ "$ready_pid" != "$cache_pid" ]]; then
        echo "RGB cache PID mismatch: $ready_pid != $cache_pid" >&2
        exit 1
    fi
fi

echo "[$(date -u --iso-8601=seconds)] shared RGB RAM cache ready: pid=$cache_pid path=$cache_mmap_path"
export PYTHONPATH="$repo_dir/third_party/diffusion_policy:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export HDF5_USE_FILE_LOCKING=FALSE
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$venv_python" -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    scripts/train_dp_image_goal.py \
    --config configs/robot/dp_image_goal.yaml \
    --output-dir "$run_dir" \
    --image-cache-mmap "$cache_mmap_path" \
    >"$train_log" 2>&1
