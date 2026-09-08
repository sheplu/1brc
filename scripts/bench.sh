#!/usr/bin/env bash
# Warm-cache benchmark. The headline metric for this project is 8 threads.
#
#   scripts/bench.sh v2_mmap v3_hash v4_simd      # compare implementations at 8 threads
#   SWEEP=1 scripts/bench.sh v4_simd              # thread sweep instead
#   DATA=measurements-10m.txt scripts/bench.sh v2_mmap
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

DATA="${DATA:-measurements.txt}"
THREADS="${OBRC_THREADS:-8}"
RUNS="${RUNS:-10}"

if [ ! -f "$DATA" ]; then
    echo "missing $DATA -- run: ./target/release/generate 1000000000 $DATA" >&2
    exit 1
fi

bins=("$@")
[ ${#bins[@]} -eq 0 ] && bins=(v8_pread)

# Pull the file into the unified buffer cache so we measure compute, not the SSD.
echo "priming page cache with $DATA ($(du -h "$DATA" | cut -f1))..."
cat "$DATA" >/dev/null

if [ "${SWEEP:-0}" = "1" ]; then
    for bin in "${bins[@]}"; do
        echo
        echo "=== $bin: thread sweep ==="
        # 6 = fast-tier core count, 18 = all cores. 16 is included because it is the
        # least stable point on this machine, which is worth seeing rather than skipping.
        hyperfine --warmup 2 --runs "$RUNS" \
            --parameter-list t 6,8,12,14,16,18 \
            --setup 'true' \
            --command-name '{t} threads' \
            "OBRC_THREADS={t} ./target/release/$bin $DATA > /dev/null"
    done
else
    cmds=()
    names=()
    for bin in "${bins[@]}"; do
        cmds+=("OBRC_THREADS=$THREADS ./target/release/$bin $DATA > /dev/null")
        names+=(--command-name "$bin")
    done
    args=()
    for i in "${!cmds[@]}"; do
        args+=(--command-name "${bins[$i]}" "${cmds[$i]}")
    done
    echo "=== ${THREADS} threads, warm cache, $RUNS runs ==="
    hyperfine --warmup 2 --runs "$RUNS" "${args[@]}"
fi
