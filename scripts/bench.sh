#!/usr/bin/env bash
# Warm-cache benchmark of ONE binary, for the standard deviation hyperfine reports and
# scripts/batch.sh does not.
#
#   scripts/bench.sh v9_flatscan                  # 8 threads
#   SWEEP=1 scripts/bench.sh v9_flatscan          # thread sweep instead
#   DATA=measurements-10m.txt scripts/bench.sh v9_flatscan
#
# For comparing binaries, use scripts/batch.sh. hyperfine runs each command's runs
# back-to-back, so the last one measured carries all the thermal drift of the ones before
# it -- which on this machine is worth several percent, i.e. the size of most wins here.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

DATA="${DATA:-measurements.txt}"
THREADS="${OBRC_THREADS:-8}"
RUNS="${RUNS:-10}"

if [ ! -f "$DATA" ]; then
    echo "missing $DATA -- run: ./target/release/generate 1000000000 $DATA" >&2
    exit 1
fi

if ! pmset -g batt | grep -q "AC Power"; then
    echo "refusing to measure on battery: $(pmset -g batt | head -1)" >&2
    exit 1
fi

bins=("$@")
[ ${#bins[@]} -eq 0 ] && bins=(v9_flatscan)
if [ ${#bins[@]} -gt 1 ] && [ "${SWEEP:-0}" != "1" ]; then
    echo "note: hyperfine measures these back-to-back, not interleaved." >&2
    echo "      use scripts/batch.sh to compare binaries." >&2
fi

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
