#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
metadata_path="/tmp/docking_goal_pool_1cam_cache.json"
launcher_log="$repo_dir/outputs/r_goal_pool_1cam_launcher.log"
cache_log="$repo_dir/outputs/r_goal_pool_1cam_ramcache.log"
train_log="$repo_dir/outputs/r_goal_pool_1cam_all235_color01_scratch_ddp_bf16_ramcache.log"
source_h5="$repo_dir/dataset/front_dock_4f_5f_combined/front_dock_4f_5f_combined_reloc3r_bottom.h5"

cd "$repo_dir"
mkdir -p "$repo_dir/outputs"
exec >>"$launcher_log" 2>&1

cache_pid=""
cache_mmap_path=""

cleanup() {
    if [[ -n "$cache_pid" ]] && kill -0 "$cache_pid" 2>/dev/null; then
        kill -TERM "$cache_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

read_cache_metadata() {
    python -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["pid"], m["mmap_path"])' "$metadata_path"
}

if [[ -s "$metadata_path" ]]; then
    read -r cache_pid cache_mmap_path < <(read_cache_metadata)
    if ! kill -0 "$cache_pid" 2>/dev/null || [[ ! -e "$cache_mmap_path" ]]; then
        cache_pid=""
        cache_mmap_path=""
    fi
fi

if [[ -z "$cache_pid" ]]; then
    python -u scripts/cache_h5_dataset_memfd.py \
        --source "$source_h5" \
        --dataset reloc3r_bottom \
        --metadata "$metadata_path" \
        --name docking_reloc3r_bottom \
        --slab-rows 256 \
        --log "$cache_log" &
    cache_pid=$!

    while [[ ! -s "$metadata_path" ]]; do
        if ! kill -0 "$cache_pid" 2>/dev/null; then
            echo "RAM cache process exited before becoming ready; see $cache_log" >&2
            wait "$cache_pid"
            exit 1
        fi
        sleep 2
    done
    read -r ready_pid cache_mmap_path < <(read_cache_metadata)
    if [[ "$ready_pid" != "$cache_pid" ]]; then
        echo "RAM cache metadata PID mismatch: $ready_pid != $cache_pid" >&2
        exit 1
    fi
fi

echo "shared RAM cache ready: pid=$cache_pid path=$cache_mmap_path"
echo "training log: $train_log"

torchrun --standalone --nproc_per_node=2 scripts/train.py \
    --config-name smr_goal_pool_relfeat1cam_all235 \
    "sensors.reloc3r_pair_bottom.cache_mmap=$cache_mmap_path" \
    >"$train_log" 2>&1
