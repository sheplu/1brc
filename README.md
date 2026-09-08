# One Billion Row Challenge, in Rust

Read 13.8 GB of `<station>;<temperature>` lines, compute min/mean/max per station, print them
sorted. [The original challenge](https://github.com/gunnarmorling/1brc) is Java; this is a Rust
port used as an optimization exercise, so every stage of the work is kept as its own binary and
benchmarked against the ones before it.

**83.5 s → 0.68 s.** No external crates: the syscalls are hand-written `extern "C"`, the threading
is `std::thread`, and the only vector code is SWAR in plain `u64`.

## Results

Apple M5 Pro, warm page cache, 1,000,000,000 rows, hyperfine, 10 runs. Every version below
produces output byte-identical to the v1 oracle's on the full dataset.

| | technique | 8 threads | 18 threads |
|---|---|---:|---:|
| `v1_naive` | `BufReader` + `split(';')` + `f64` + `BTreeMap`, single-threaded | 83.5 s | — |
| `v2_mmap` | mmap, work-stealing chunks, hand-rolled Fx hasher | 2.138 s | |
| `v3_hash` | open-addressing table, 32-byte entries, keys as offsets | 2.032 s | |
| `v4_simd` | SWAR `;` scan with the hash fused into it | 1.995 s | |
| `v5_branchless` | branchless temperature parse, word-wise key compare | 1.688 s | |
| `v6_inline` | station name stored *inside* the table entry | 1.479 s | |
| `v7_pipelined` | 3 independent line streams per chunk | 1.440 s | 811 ms |
| **`v8_pread`** | **`pread` into a per-thread buffer instead of mmap** | **1.264 s** | **681 ms** |

8 threads is the ladder's comparison baseline. 18 — every core — is the headline: **681 ms ± 16**,
best run 660 ms.

`v8_pread` thread sweep: 6 → 1.587 s, 8 → 1.264 s, 12 → 887 ms, 14 → 808 ms, 16 → 809 ms,
18 → 681 ms. 16 threads is consistently the *flakiest* point (σ up to 89 ms) while 18 is both the
fastest and among the most stable, which is the opposite of the usual "leave a couple of cores
free" intuition.

## Where the time actually goes

Two ablation harnesses drove almost every decision here. Guessing was wrong more often than not.

`hot_floor <mode>` runs the real hot loop with stages progressively switched on, so each row is
the *marginal* cost of one technique rather than a total (8 threads, one batch):

| stage | time | marginal | |
|---|---:|---:|---|
| `touch` | 444 ms | — | read every byte, XOR-reduce, nothing else |
| `parse` | 1.370 s | +926 ms | SWAR `;` scan + branchless temperature parse |
| `hash` | 1.424 s | +54 ms | fuse the name hash into the scan |
| `nocmp` | 1.433 s | +9 ms | probe and update a table slot |
| `table` | 1.721 s | +288 ms | **compare the key** |

The last two rows are the whole story of v6. Probing the hash table is free; *verifying* the key
cost more than hashing and probing combined. Not because of the bytes compared — because with keys
stored as offsets, checking one means dereferencing into a random spot in a 13.8 GB mapping, a
second scattered memory access on every row. Moving the name onto the entry's own 64-byte cache
line, which the probe has already loaded, deleted that access and 209 ms with it.

The same harness measures instruction-level parallelism headroom by walking N independent line
streams: `parse` 1.370 s → `parse2` 991 ms → `parse4` 898 ms. Rows form a serial dependency chain
— a row's start address is only known once the previous row's temperature has been parsed — so a
single stream leaves most of the out-of-order window idle.

That is also the most instructive failure in the project. Streaming was tried against v5 first and
*lost* (1.76 s vs 1.671 s), and giving each stream its own table was worse still. The offset-keyed
table ended each row with a load-modify-store the streams had to serialize on, which cost more than
the overlap won. The technique only became a win once v6 had made the update L1-resident. **The
1.7× of parsing headroom was real the whole time and unreachable until an unrelated bottleneck
moved.**

`io_floor <mmap|pread>` A/Bs just getting at the bytes, no parsing:

| | 8 threads | 18 threads |
|---|---:|---:|
| mmap | 452 ms | 329 ms |
| pread | 224 ms | 241 ms |

mmap takes a minor fault per 16 KB page — 842k of them — and all workers take them against the same
`vm_map`. `pread` copies 13.8 GB instead and is still half the cost. It stops scaling past 8 threads
while mmap keeps improving, so the floor predicted only ~90 ms of upside at 18; v8 gained 130 ms,
because the faults were also stealing from compute rather than merely adding to it.

Swapping the reader is not free architecturally: `pread` recycles one buffer per chunk, so a key
stored as an offset into it would dangle as soon as the next chunk was read. The table had to be
made to own its keys first.

## What each version actually changed

- **v2** — mmap, and work-stealing over 2 MiB chunks *from the start*. macOS/arm64 exposes no
  thread-affinity API, and this machine has 6 "Super" plus 12 "Performance" cores, so any static
  split leaves a straggler tail. Chunk boundaries are computed so that both sides of a boundary
  evaluate the same expression on the same bytes; workers never communicate to agree on who owns a
  line.
- **v3** — replaces `HashMap` with linear probing over 65,536 slots. The stored hash is dropped to
  keep entries at 32 bytes; `len == 0` is the empty sentinel, which avoids the `hash == 0`
  false-empty bug.
- **v4** — SWAR (`(x - 0x0101..) & !x & 0x8080..` on `word ^ 0x3B3B..`) rather than NEON: 4 scalar
  ops, no cross-domain register move, and station names average ~8 bytes so a 16-byte vector mostly
  does wasted work. The `\n` search is deleted entirely — the dot's position in the parsed value
  gives the next line's offset.
- **v5** — branchless parse: the sign and the one-or-two-integer-digits question are both
  data-dependent and unpredictable. Load 8 bytes, locate the `.` via the ASCII `0x10` bit, and one
  shift lines every layout up for a single magic multiply. The key compare drops `bcmp` for
  word-wise loads with an end-aligned final word, so it never over-reads.
- **v6** — the inline key, above. 32 bytes of the name live in the entry. That covers every name in
  the official list, but the spec allows 100, so longer ones fall back to comparing the remainder
  out of an owned arena — correct, just slower, and never taken on this dataset.
- **v7** — 3 line streams. 2 and 3 are within noise; 4 and above lose, as the streams start
  competing for the table's cache lines faster than they add overlap.
- **v8** — `pread`. Also *removes* code: v6 and v7 need a second, offset-keyed table for the last
  few lines of the file because the inline probe reads a fixed 32 bytes and would run off the end of
  the mapping. A read buffer just has slack past EOF, so the wide path covers the whole file.

## Correctness

The interesting failures here are arithmetic, not performance.

- **`sum` must be `i64`.** 1e9 rows over 413 stations is ~2.42M rows each; `2.42e6 × 999`
  overflows `i32`. This is the real dataset, not an adversarial one.
- **Round half toward positive infinity**, matching Java's `Math.round`. Integer form:
  `(2*sum + count).div_euclid(2*count)` — `div_euclid`, because Rust's `/` truncates toward zero and
  is wrong for negatives. v1 uses `(x + 0.5).floor()` and not `f64::round()`, which rounds half
  *away from* zero and disagrees at `-0.5`.
- **No `-0.0`.** The printed sign comes from the rounded tenths integer, never from `sum < 0`.
- **A hash match is never a key match**, and neither is a 16-byte prefix. The full name is always
  compared. With 10,000 names permitted, anything less would be relying on the dataset.
- **Out-of-bounds reads.** The hot loop reads up to 32 bytes past the current row. Interior chunks
  legitimately over-read into the next one; the hazard is the last chunk when the file size is an
  exact multiple of the 16 KB page, where reading past EOF leaves the mapping. The real dataset does
  *not* exercise this (`13795299516 % 16384 == 4284`), so `scripts/edge.sh` builds files that do.

Testing, in rough order of how much it caught:

- `scripts/edge.sh` — 12 hand-built inputs (single row, no trailing newline, exact page multiples
  including one ending in a 100-byte name, multi-byte UTF-8, names straddling the 16- and 32-byte
  inline-key seams, 40-byte shared prefixes, every rounding extreme) run through all seven versions
  against the oracle.
- `cargo test` — 41 tests. The parser is proved by exhaustion: all 1999 legal temperature strings
  against a reference parse, each with seven different trailing fillers in the 8-byte load. The
  chunk splitter is checked at every chunk size against a full-coverage invariant, and v8's
  local-only boundary rule is asserted equal to the global one for every chunk at every size.
  `InlineTable` is differentially tested against the offset-keyed table.
- `scripts/verify.sh` — v1 against everything else on 10M rows.

One deliberate, documented divergence: v1 accumulates in `f64` and divides before rounding, so a
mean landing exactly on `x.x5` can fall either side of the boundary. The mean of 8.1 and 8.2 is
exactly 8.15; the integer path returns 8.2, and v1 computes 8.149999999999999 and floors to 8.1.
The integer path is the correct one. `edge.sh` has a case built entirely of such ties and requires
the fast versions to agree with *each other* there rather than with the oracle. It never triggers
on the real dataset — with 2.4M samples per station, landing exactly on a tie does not happen.

## Running it

```sh
cargo build --release
./target/release/generate 1000000000 measurements.txt   # ~14 GB, deterministic
./target/release/v8_pread measurements.txt

OBRC_THREADS=18 ./target/release/v8_pread measurements.txt

cargo test
scripts/edge.sh                                          # hand-built edge cases
scripts/verify.sh v8_pread                               # differential vs the oracle
scripts/bench.sh v6_inline v7_pipelined v8_pread         # needs hyperfine
```

The generator is deterministic and parallel: 1,000 chunks of 1M rows, chunk `i` seeded
`splitmix64(seed ^ i)`, so the file is byte-identical regardless of thread scheduling. It clamps to
[-99.9, 99.9], which the Java original does not — across 1e9 gaussian samples at σ=10 a stray
`-100.2` is not unlikely, and it would silently corrupt any parser assuming two integer digits.

`OBRC_THREADS` sets the worker count everywhere (default 8). `OBRC_CHUNK` additionally tunes v8's
read size; it was swept and 64 KB costs 877 ms with double the system time, 1 MiB is the floor at
676 ms, and it is flat past that. Syscall count dominates — keeping the buffer inside L2 turned out
not to matter, which was not the guess.

## Caveats

- Sorting is by raw bytes. That matches Java's `TreeMap` UTF-16 ordering across the whole BMP and
  diverges only above U+10000. Every official station name is BMP, so it is correct here; it is not
  correct in general.
- Numbers are from one laptop and move with power state and thermals. An early batch of results had
  to be thrown out after the machine was unplugged mid-session — a systematic ~13% shift that looks
  exactly like a real regression. Comparisons are only meaningful within a single interleaved batch.
