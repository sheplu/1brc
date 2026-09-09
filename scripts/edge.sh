#!/usr/bin/env bash
# Hand-built inputs that the generated dataset never produces, run against the v1 oracle.
#
#   scripts/edge.sh                # every version
#   scripts/edge.sh v9_flatscan    # just one
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

bins=("$@")
[ ${#bins[@]} -eq 0 ] &&
    bins=(v2_mmap v3_hash v4_simd v5_branchless v6_inline v7_pipelined v8_pread v9_flatscan
        v10_rawhash v11_keyed)

dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT

python3 - "$dir" <<'PY'
import os, sys
d = sys.argv[1]
def w(name, text):
    data = text.encode()
    for line in data.split(b"\n"):
        if line:
            assert 1 <= len(line.split(b";")[0]) <= 100, f"{name}: illegal station name"
    open(os.path.join(d, name), "wb").write(data)

w("single.txt", "Abha;18.0\n")

# The spec's separator is '\n', but nothing guarantees one after the final record.
w("no_trailing_newline.txt", "Abha;18.0\nOslo;-5.2\nAbha;-0.4")

# One measurement per station: count == 1, so min == mean == max.
w("one_each.txt", "".join(f"S{i};{i%100}.{i%10}\n" for i in range(50)))

# Every legal magnitude, including the values a branchless parser is most likely to break on.
w("extremes.txt", "".join(f"E;{v}\n" for v in
    ["-99.9", "99.9", "0.0", "-0.0", "-9.9", "9.9", "-0.1", "0.1", "10.0", "-10.0"]))

# A mean that rounds to zero from below must print 0.0, never -0.0.
w("negative_zero.txt", "Z;-0.1\nZ;0.0\n")

# Means landing exactly on a x.x5 tie. Known to diverge from the f64 oracle -- see below.
w("ties.txt", "".join(f"T{i};{i}.1\nT{i};{i}.2\n" for i in range(10)))

# The spec maximum name length, one either side of the 32-byte inline key, and the minimum.
w("long_names.txt", "".join(f"{'x'*(n-3)}{n:03d};1.0\n" for n in (100, 99, 33, 32, 31, 17, 16, 15, 4)) * 3)

# Multi-byte UTF-8, including names whose byte length crosses the 16- and 32-byte lines
# the inline key is built from.
w("utf8.txt", "".join(f"{n};{i}.5\n" for i, n in enumerate(
    ["Ürümqi", "東京", "Ho Chi Minh City", "Флорес", "Ouagadougou", "Bucureşti",
     "Naïve-sur-Mer-en-Provence-Alpes", "Ω", "日本国東京都千代田区千代田"])) * 4)

# Names sharing a 40-byte prefix, so a prefix-only comparison would merge them.
w("shared_prefix.txt", "".join(f"{'p'*40}{s};{i}.0\n" for i, s in enumerate("abcde")) * 3)

PAGE = 16384

def pad_to_page(body, tail=""):
    """Grow `body` with legal records so that `body + tail` is an exact page multiple."""
    need = (-len((body + tail).encode())) % PAGE
    while need < 6:                      # shortest legal record is "Q;1.0\n"
        need += PAGE
    while need > 105:                    # longest is a 100-byte name plus ";1.0\n"
        body += "Q;1.0\n"
        need -= 6
    body += "Q" * (need - len(";1.0\n")) + ";1.0\n"
    out = body + tail
    assert len(out.encode()) % PAGE == 0, len(out.encode())
    return out

# File size an exact multiple of the 16 KB page. mmap maps whole pages, so a read even one
# byte past EOF lands outside the mapping. The generated dataset never hits this
# (13795299516 % 16384 == 4284), which is why it has to be built by hand.
w("page_multiple.txt", pad_to_page("".join(f"P{i%400};{i%100}.{i%10}\n" for i in range(6000))))

# Same, but spanning several 2 MiB work chunks so the chunk-boundary logic runs too.
w("page_multiple_multichunk.txt",
  pad_to_page("".join(f"B{i%400};{i%100}.{i%10}\n" for i in range(700000))))

# A page-multiple file whose very last record carries a maximum-length name, so the tail
# guard is stressed at the one place it matters.
w("page_multiple_long_tail.txt",
  pad_to_page("".join(f"L{i%400};{i%100}.{i%10}\n" for i in range(6000)),
              tail="w" * 100 + ";-99.9\n"))
PY

split() { perl -pe 's/(?<=\d), /\n/g'; }

# v1 accumulates in f64 and divides before rounding, so on a mean that is exactly x.x5 its
# result can land either side of the boundary. The integer path is the correct one there.
# For those inputs the oracle is not authoritative; require instead that every fast version
# agrees with the others.
known_divergent() { [ "$1" = "ties.txt" ]; }

status=0
for f in "$dir"/*.txt; do
    name="$(basename "$f")"
    ref="$dir/ref.out"
    ./target/release/v1_naive "$f" | split >"$ref"

    if known_divergent "$name"; then
        base=""
        line="$(printf '%-30s %8s  vs each other ' "$name" "$(wc -c <"$f" | tr -d ' ')")"
    else
        base="$ref"
        line="$(printf '%-30s %8s  vs oracle     ' "$name" "$(wc -c <"$f" | tr -d ' ')")"
    fi

    for bin in "${bins[@]}"; do
        got="$dir/$bin.out"
        if ! ./target/release/"$bin" "$f" >"$dir/raw" 2>"$dir/err"; then
            line="$line $bin:CRASH"
            status=1
            continue
        fi
        split <"$dir/raw" >"$got"
        [ -z "$base" ] && base="$got"
        if cmp -s "$base" "$got"; then
            line="$line ."
        else
            line="$line $bin:DIFF"
            status=1
        fi
    done
    echo "$line"
done

echo
if [ $status -eq 0 ]; then
    echo "PASS  ${#bins[@]} versions, all cases  (${bins[*]})"
else
    echo "FAIL  see above"
fi
exit $status
