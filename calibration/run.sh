#!/bin/bash
set -e -o pipefail
clear 

run_script() {
    local script="$1"
    local dir=$(dirname "$script")
    local base=$(basename "$script" .sh)
    local logdir=".logs/$dir"
    echo "================================================$script================================================"
    mkdir -p "$logdir"
    bash "$script" "${@:2}" 2>&1 | tee "$logdir/$base-$(date "+%Y%m%d_%H%M%S").log"
}
clear() {
    true
}
export -f run_script
export -f clear

if [ -f "$1" ]; then
    run_script "$@"
else
    find scripts/ -name "$1" -print0 | sort -zu | xargs -0 -I {} bash -c 'run_script "$@"' _ {} "${@:2}"
fi