#!/usr/bin/env bash
# Differential test: every implementation must agree with the v1 oracle.
#
#   scripts/verify.sh v2_mmap v3_hash ...
#   DATA=measurements.txt scripts/verify.sh v4_simd
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

DATA="${DATA:-measurements-10m.txt}"
bins=("$@")
[ ${#bins[@]} -eq 0 ] && bins=(v2_mmap)

# The output is one long line, and two station names ("Washington, D.C." and
# "Flores,  Petén") contain ", " themselves — so split on the ", " that follows a digit,
# which only ever occurs between records.
split() { perl -pe 's/(?<=\d), /\n/g'; }

ref="$(mktemp)"
trap 'rm -f "$ref"' EXIT
echo "oracle: v1_naive on $DATA"
./target/release/v1_naive "$DATA" | split >"$ref"
echo "        $(wc -l <"$ref" | tr -d ' ') stations"

status=0
for bin in "${bins[@]}"; do
    got="$(mktemp)"
    ./target/release/"$bin" "$DATA" | split >"$got"
    if cmp -s "$ref" "$got"; then
        printf 'OK    %s\n' "$bin"
    else
        n=$(diff "$ref" "$got" | grep -c '^<' || true)
        printf 'DIFF  %s: %s records differ\n' "$bin" "$n"
        diff "$ref" "$got" | head -20
        status=1
    fi
    rm -f "$got"
done
exit $status
