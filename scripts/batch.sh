#!/usr/bin/env bash
# One interleaved measurement batch.
#
# Numbers taken in different batches on a laptop are not comparable — power state, thermal
# headroom and page-cache warmth all drift. So every configuration you intend to compare
# has to be measured *here*, in one run, round-robin rather than back-to-back, so drift
# hits every configuration equally instead of penalising whichever went last.
#
# Reads "label<TAB>command" per line, from a file or stdin:
#
#   scripts/batch.sh configs.tsv
#   printf 'a\tcmd\nb\tcmd\n' | REPS=5 scripts/batch.sh
#
# REPS (default 3) timed passes, preceded by one discarded warmup pass. Reports the median.
# Binaries that print their own "<n> s" (io_floor, hot_floor) are scored on that, which
# excludes process startup and the final merge; everything else is scored on wall clock.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

DATA="${DATA:-measurements.txt}"
REPS="${REPS:-3}"

if ! pmset -g batt | grep -q "AC Power"; then
    echo "refusing to measure on battery: $(pmset -g batt | head -1)" >&2
    exit 1
fi

# Slurped here, not inside python: the heredoc below *is* python's stdin, so a config
# arriving on the pipe has to be read before that redirect takes it away.
BATCH_CONFIG="$(cat -- "${1:-/dev/stdin}")"
export BATCH_CONFIG

echo "priming page cache with $DATA ..." >&2
cat "$DATA" >/dev/null

exec python3 - "$REPS" <<'PY'
import os, re, statistics, subprocess, sys, time

reps = int(sys.argv[1])
lines = os.environ["BATCH_CONFIG"].splitlines()
jobs = [l.split("\t", 1) for l in lines if l.strip() and not l.startswith("#")]
if not jobs:
    sys.exit("no configs: expected 'label<TAB>command' lines")

# The self-reported figure is the last "<float> s" on the line, so a label containing a
# number cannot be mistaken for the timing.
SELF = re.compile(r"([0-9]+\.[0-9]+) s(?!\w)")
times = {label: [] for label, _ in jobs}

for r in range(reps + 1):
    warm = r == 0
    print(f"  {'warmup' if warm else f'pass {r}/{reps}'}", file=sys.stderr, flush=True)
    for label, cmd in jobs:
        t0 = time.perf_counter()
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        if p.returncode != 0:
            print(f"    FAIL {label}: {p.stderr.strip().splitlines()[-1:]}", file=sys.stderr)
            continue
        hits = SELF.findall(p.stdout + p.stderr)
        if not warm:
            times[label].append((float(hits[-1]) if hits else wall, not hits))

print(f"\n{'config':<34}{'median':>9}{'min':>9}{'max':>9}   n")
for label, _ in jobs:
    ts = [t for t, _ in times[label]]
    if not ts:
        print(f"{label:<34}{'FAILED':>9}")
        continue
    tag = "*" if times[label][0][1] else ""
    print(
        f"{label:<34}{statistics.median(ts):>8.3f}{tag:<1}{min(ts):>8.3f} {max(ts):>8.3f}  {len(ts)}"
    )
print("\n* wall clock (no self-reported time); others are the binary's own timed region.")
PY
