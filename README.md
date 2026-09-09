# One Billion Row Challenge, in Rust

Read 13.8 GB of `<station>;<temperature>` lines, compute min/mean/max per station, print them
sorted. [The original challenge](https://github.com/gunnarmorling/1brc) is Java; this is a Rust
port used as an optimization exercise, so every stage of the work is kept as its own binary and
benchmarked against the ones before it.

**80.9 s → 0.46 s.** No external crates: the syscalls are hand-written `extern "C"`, the threading
is `std::thread`, and the only vector code is SWAR in plain `u64`.

## Results

Apple M5 Pro, warm page cache, 1,000,000,000 rows. Medians of one interleaved batch — every row
below was measured round-robin in a single run, because numbers from different batches on this
machine are not comparable (see Caveats). Every version produces output byte-identical to the v1
oracle's on the full dataset.

| | technique | 8 threads | 18 threads |
|---|---|---:|---:|
| `v1_naive` | `BufReader` + `split(';')` + `f64` + `BTreeMap`, single-threaded | 80.9 s | — |
| `v2_mmap` | mmap, work-stealing chunks, hand-rolled Fx hasher | 2.181 s | |
| `v3_hash` | open-addressing table, 32-byte entries, keys as offsets | 2.047 s | |
| `v4_simd` | SWAR `;` scan with the hash fused into it | 1.983 s | |
| `v5_branchless` | branchless temperature parse, word-wise key compare | 1.710 s | |
| `v6_inline` | station name stored *inside* the table entry | 1.504 s | |
| `v7_pipelined` | 3 independent line streams per chunk | 1.470 s | 866 ms |
| `v8_pread` | `pread` into a per-thread buffer instead of mmap | 1.271 s | 677 ms |
| `v9_flatscan` | fixed-width branchless `;` scan | 984 ms | 556 ms |
| `v10_rawhash` | hash indexed off high bits, no avalanche | 946 ms | 535 ms |
| **`v11_keyed`** | **the probe key falls out of the scan** | **784 ms** | **463 ms** |

8 threads is the ladder's comparison baseline. 18 — every core — is the headline: **463 ms**.
(`v1_naive` is the one number measured in its own batch, paired against v9, since a 140× ratio does
not need three digits; v9 came out at 591 ms there.)

The last two rows are not the same kind of result. v10 removes five serial operations from the hot
loop and the *engine* gets 3–4% faster for it in every batch it has been measured in; the
whole-binary gap has come out anywhere from 0.3% to 6% depending on the batch, so the 3.9% here is
consistent with the engine number and is not on its own evidence of anything. v11 removes about a
third of the hot loop's *instructions*, and that one shows up everywhere: −13.5% at 18 threads,
−17.1% at 8, and −26% of the compute above the read floor. Both facts are in the tables below.

Thread sweep, v8 against v9 in one batch of their own:

| threads | 6 | 8 | 12 | 14 | 16 | 18 |
|---|---:|---:|---:|---:|---:|---:|
| `v8_pread` | 1.651 s | 1.298 s | 909 ms | 799 ms | 712 ms | 673 ms |
| `v9_flatscan` | 1.362 s | 1.097 s | 803 ms | 708 ms | 641 ms | 608 ms |
| | −17.5% | −15.5% | −11.7% | −11.4% | −10.0% | −9.7% |

The win shrinks as threads are added because ~220 ms of the total is I/O that v9 does not touch,
and that fixed cost is a growing share of a shrinking number. Against compute alone the gain is
roughly 18% at 8 threads and 14% at 18.

Both curves are monotone here. An earlier batch had v8 flat across 14 and 16 threads with σ up to
89 ms at 16, and that did not reproduce — worth recording as a caution about reading structure into
a single sweep. Note too that this batch puts v9 at 608 ms, a v9-only sweep said 579, and the
ladder above says 556. That ±5% is the batch effect this README keeps insisting on, and it is
larger than several of the individual steps in the ladder.

v11 was checked the same way, in a batch of its own: v10 → v11 is −12.1% at 6 threads, −13.2% at 8
and −9.6% at 18 — the same shape as v9 and for the same reason, since the win is on compute and
compute is a shrinking share of the total as threads are added against a fixed ~230 ms of I/O. The
ladder batch puts the same two numbers at −17.1% and −13.5%. Four points of disagreement between
two clean batches, on a win that is real in both, is the size of the effect this README keeps
warning about.

## Where the time actually goes

Two ablation harnesses drove almost every decision here. Guessing was wrong more often than not —
including about which *half* of a technique would be expensive; see the NEON bitmap below.

`hot_floor <mode>` runs the real hot loop with stages progressively switched on, so each row is
the *marginal* cost of one technique rather than a total. `OBRC_READER` picks how the bytes arrive
and the mode picks which table, and **the pair matters**: a mode is only comparable to another
measured the same way.

Under mmap with the offset-keyed table — the configuration that decided v6 (8 threads):

| stage | time | marginal | |
|---|---:|---:|---|
| `touch` | 453 ms | — | read every byte, XOR-reduce, nothing else |
| `parse` | 1.391 s | +938 ms | SWAR `;` scan + branchless temperature parse |
| `hash` | 1.423 s | +32 ms | fuse the name hash into the scan |
| `nocmp` | 1.453 s | +30 ms | probe and update a table slot |
| `table` | 1.677 s | +224 ms | **compare the key** |

The last two rows are the whole story of v6. Probing the hash table is nearly free; *verifying* the
key cost more than hashing and probing combined. Not because of the bytes compared — because with
keys stored as offsets, checking one means dereferencing into a random spot in a 13.8 GB mapping, a
second scattered memory access on every row. Moving the name onto the entry's own 64-byte cache
line, which the probe has already loaded, deleted that access and most of that 224 ms with it.

The same harness measures instruction-level parallelism headroom by walking N independent line
streams: `parse` 1.391 s → `parse2` 994 ms → `parse4` 915 ms. Rows form a serial dependency chain
— a row's start address is only known once the previous row's temperature has been parsed — so a
single stream leaves most of the out-of-order window idle.

That is also the most instructive failure in the project. Streaming was tried against v5 first and
*lost* (1.76 s vs 1.671 s), and giving each stream its own table was worse still. The offset-keyed
table ended each row with a load-modify-store the streams had to serialize on, which cost more than
the overlap won. The technique only became a win once v6 had made the update L1-resident. **The
1.7× of parsing headroom was real the whole time and unreachable until an unrelated bottleneck
moved.**

Under `pread` with the inline-key table — what v8 onward actually ship (18 threads, table held at
2^16 slots throughout so the stage is the only variable):

| stage | time | marginal | |
|---|---:|---:|---|
| `touch` | 237 ms | — | the I/O floor |
| `parse` | 551 ms | +314 ms | scan + parse |
| `hash` | 566 ms | +15 ms | + the fused hash |
| `itable` | 639 ms | +73 ms | + the inline-key upsert; this is v8's engine |
| `fhash` | 492 ms | −74 ms *vs* `hash` | the scan's loop replaced by a fixed window |
| `flat` | 582 ms | −57 ms *vs* `itable` | the same swap, with the table; this is v9's |
| `raw` | 557 ms | −25 ms *vs* `flat` | the hash's final avalanche deleted; v10's |
| `keyed` | 520 ms | −37 ms *vs* `raw` | the probe key taken from the scan; v11's |

That second half was measured last and should have been measured first. For five versions the
harness ran only against mmap and only against the offset-keyed table, so **every marginal cost in
the first table describes v5's engine**, not the shipping one. Fixing the instrument was what
exposed v9, and then v11.

With three interleaved streams, which is what the binaries do, the same four rows are `itable3`
621 ms → `flat3` 533 ms → `raw3` 511 ms → `keyed3` 440 ms. That last step is the largest single
win in the project measured against compute: 71 ms is 26% of the 274 ms sitting above the read
floor.

`io_floor <mode>` A/Bs just getting at the bytes, no parsing:

| | 8 threads | 18 threads |
|---|---:|---:|
| mmap | 450 ms | 329 ms |
| pread | 227 ms | 230 ms |

mmap takes a minor fault per 16 KB page — 842k of them — and all workers take them against the same
`vm_map`. `pread` copies 13.8 GB instead and is still half the cost. It stops scaling past 8 threads
while mmap keeps improving, so the floor predicted only ~90 ms of upside at 18; v8 gained more than
that, because the faults were also stealing from compute rather than merely adding to it.

Four attempts to push that floor lower, none of them a win, at 18 threads:

| | time | |
|---|---:|---|
| `pread` | 220 ms | baseline |
| `pread_fd` | 218 ms | one descriptor per thread instead of a shared `&File` |
| `mmap_seq` | 297 ms | `MADV_SEQUENTIAL` |
| `mmap_shared` | 234–791 ms | `MAP_SHARED`, wildly bimodal run to run |
| `mmap_willneed` | 682 ms | `MADV_WILLNEED` — **2.3× worse** |
| `pread_nored` | 203 ms | *not an attempt* — the XOR reduce removed, so the copy is 92% |

`pread_fd` was the one expected to pay: `pread` is flat from 8 threads to 18 while nothing else on
the machine is saturated, which looks exactly like contention on the shared file object. It is not
— per-thread descriptors change nothing, so the ceiling is in the copy path itself. And the
`madvise` calls, the obvious free win, range from neutral to actively harmful.

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
  gives the next line's offset. (The "16 bytes mostly does wasted work" half of that argument is
  wrong, and v9 is the correction — the wasted work is cheaper than the branch that avoids it.)
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
- **v9** — one branch. v4's `;` scan is a loop over 8-byte words, and station names split almost
  exactly evenly across its exit: 50.4% of the official list is ≤7 bytes (one iteration) and 46.7%
  is 8–15 (two). The generator draws stations i.i.d., so that test is **unpredictable by
  construction** — no branch history beats a coin flip on it, and a billion rows each paid about
  half a misprediction. v9 reads both words up front, selects the delimiter without branching, and
  folds the hash with two multiplies: more instructions, fewer cycles, and bit-identical output to
  the loop it replaces. Names ≥16 bytes (2.9%) take a `#[cold]` re-scan through the old loop, and
  *that* branch is biased ~33:1, so it predicts. Worth 121 ms at 18 threads and 287 ms at 8 —
  roughly a quarter of the compute above the read floor, from a single branch.

  It stayed hidden for five versions because every ablation mode contained it. `parse`, `hash`,
  `table` and `itable` all scanned the same way, so it cancelled out of every difference the
  harness reported and never showed up as a line item.
- **v10** — five operations. The fused hash ends in `h ^= h>>32; h *= K; h ^= h>>32`, an avalanche
  that exists to spread the *low* bits, because the low bits of a multiply are barely mixed — bit 0
  of a product is just the product of the inputs' bit 0s. It is strictly serial and sits between
  the last `mix` and the table load, so nothing in the row proceeds until it retires. Index the
  table off the *high* bits instead, where a bare multiply already spreads well, and the whole
  avalanche is dead code. Engine `flat` → `raw` is −25 ms, `flat3` → `raw3` −22 ms, and 2–4% in
  every batch it has been measured in. End to end it is 21 ms on 556, which is within the batch
  noise on its own and only believable because the engine number is not.

  The table also drops from 2^16 slots to 2^14 — 18 MiB of tables at 18 threads instead of 72. That
  is a footprint change and not a speed one; see below.
- **v11** — stop recomputing what the scan already has. v10's row disassembles to ~148 instructions
  against ~17 measured cycles per row per thread, so at Apple's issue width the loop is
  *instruction-throughput* bound, not latency bound, and roughly half those instructions are not the
  algorithm. The largest single block is the probe key: it loads the name's 32 bytes as two `u128`s,
  indexes a mask table twice and `and`s — four loads, two address computations, two mask loads, four
  masks — and then, because a 32-byte key plus three interleaved row states does not fit in the
  register file, LLVM spills the key to the stack and reloads it four instructions later for the
  compare.

  All of that re-reads bytes the scan is already holding. v9's fixed window loads the name's first
  16 bytes and masks off everything at and past the `;` — it has to, or the trailing garbage would
  reach the hash — and for any name that fits the window that masked pair *is* the probe key, with
  the key's high half zero. So the scan hands the two words back and the compare starts with both
  operands in registers. One select replaces four loads, two mask lookups, four `and`s and a spill
  pair.

  Knowing the name is under 16 bytes then collapses the probe a second time. That length guarantees
  a zero byte inside the first 16, which no stored name of 16 bytes or more has and no empty slot
  lacks — so the key's *low half alone* separates the query from every entry the table can hold.
  The high half, the length check and the long-name comparison all drop out, and the probe reads one
  `ldp` off the entry's cache line instead of two. Names of 16 bytes and over do not fit the window,
  so neither their length nor their key is known there; they take a `#[cold]` copy of v10's row,
  behind the same ~33:1 branch v9 already relied on.

  ~148 instructions down to ~96, and no spill in the hot path. Engine `raw3` → `keyed3` is −71 ms,
  which is 26% of all compute above the read floor; end to end −13.5% at 18 threads and −17.1% at 8.

## Three things that were predicted to work and did not

Recorded because the plans for v10 and v11 said they would, in writing, before the measurement.

**The table would fit the dTLB.** 2^16 slots is 4 MiB per thread holding 413 live entries, scattered
across ~205 distinct 16 KB pages — far more than the L1 dTLB maps. Shrinking to 2^11 (128 KB, 8
pages, permanently resident) was predicted to be worth 40–70 ms. Whole-binary, 18 threads, one
batch:

| slots | 2^11 | 2^12 | 2^13 | 2^14 | 2^15 | 2^16 |
|---|---:|---:|---:|---:|---:|---:|
| | 620 ms | 589 ms | 580 ms | 580 ms | 576 ms | 580 ms |

Flat from 2^13 up, and *worse* below it. Each extra probe is another dependent load, and a low load
factor buys more probes than it saves page walks. The default is 2^14 for the memory, not the time.

**A 6% win that was not there.** An intermediate batch put v9 at 622 ms and v10 at 584 ms — 6.1% —
and a thread sweep showed the gap widening monotonically from 0.1% at 6 threads to 6.1% at 18,
which is a tidy story about shared-cache pressure. Nine reps instead of five turned it into 2.5%,
and a 30-configuration batch turned it into 0.3%. There was no trend; six
noisy 2% effects lined up by chance. The `hot_floor` numbers held steady across all three batches,
because the engine's own timed region excludes ~40 ms of process and merge cost that the wall clock
does not.

**The NEON delimiter bitmap.** This is the technique that separates the fastest C solution from
everything else: compare a whole block against `;` and `\n` at once, collect the results into
bitmasks, and drive the row loop off the bits. Every delimiter is then known before any row is
touched, so rows stop forming a dependency chain. `src/block.rs` does it for 64-byte blocks and
`hot_floor` has ten modes on top of it. One batch, 18 threads:

| | time | |
|---|---:|---|
| `touch` | 235 ms | the read floor |
| `mask_semi` | 241 ms | the `;` bitmap, XORed into the sink and discarded |
| `mask` | 247 ms | both bitmaps |
| `dlrows` | 381 ms | + drain the bits to `(name, semi, nl)` triples |
| `dltable` | 509 ms | + hash, parse and upsert — v10's per-row work, no scanning left |
| `raw3` | 525 ms | v10's engine, three scalar streams |
| `keyed3` | **469 ms** | v11's engine |

The predicted problem was the movemask tax. AVX2 turns 32 bytes into a mask with one
`vpmovmskb`; aarch64 has no such instruction, so each 64-byte mask costs four `cmeq`, four `and`
against a bit-position vector and a four-deep `addp` tree — 13 operations where AVX2 spends 2. That
tax is real and it is *irrelevant*: both bitmaps together cost 12 ms over `touch`, under 5% of a
mode that reads 13.8 GB.

The cost is on the other side. Draining the bits into row positions is +134 ms, eleven times what
building them cost. And the row work that follows cannot use v11's trick: v11's probe key is free
*because the scan already masked those bytes*, and a bitmap design has no scan, so it has to load
and mask the name itself. The bitmap beats v10's engine by 3% and loses to v11's by 8.5%, having
forfeited the thing that made v11 fast.

It is not a small-core artefact either, which was the failure mode the plan called out in advance.
At 6 threads, where the scheduler has the fast cores to hand out, `dltable` is 1.214 s against
`keyed3`'s 996 ms — a 22% loss. The gap is *widest* where per-core throughput is highest, which is
the opposite of the predicted shape. The module stays in the tree as an instrument; nothing ships
it.

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
- **The inline key's length bound is strict, and this is where the argument went wrong.** The table
  skips comparing lengths because a name zero-padded to 32 bytes determines its own length. That is
  true *below* 32 and false *at* 32, where the padding disappears and the key becomes the first 32
  bytes of every longer name that starts the same way. The guard read `len <= INLINE_KEY` from v6
  through v11, so a 32-byte name arriving after a longer namesake merged into it and vanished from
  the output. It never fired here — the longest official name is 26 — but the spec allows 100, and
  the point of the sentence above is not to rely on that. The prefix-chain test missed it because it
  only ever walked short-to-long, and the flaw is one-directional: the *query's* length is what
  decides whether the fallback is taken. Fixed; the test now walks both ways, `scripts/edge.sh` grew
  an input that reaches it end to end, and a unit test forces the hash collision outright.
- **Out-of-bounds reads.** The hot loop reads up to 32 bytes past the current row. Interior chunks
  legitimately over-read into the next one; the hazard is the last chunk when the file size is an
  exact multiple of the 16 KB page, where reading past EOF leaves the mapping. The real dataset does
  *not* exercise this (`13795299516 % 16384 == 4284`), so `scripts/edge.sh` builds files that do.

Testing, in rough order of how much it caught:

- `scripts/edge.sh` — 13 hand-built inputs (single row, no trailing newline, exact page multiples
  including one ending in a 100-byte name, multi-byte UTF-8, names straddling the 16- and 32-byte
  inline-key seams, 40-byte shared prefixes, every rounding extreme) run through all ten fast
  versions against the oracle.
- `cargo test` — 56 tests. The parser is proved by exhaustion: all 1999 legal temperature strings
  against a reference parse, each with seven different trailing fillers in the 8-byte load. The
  chunk splitter is checked at every chunk size against a full-coverage invariant, and v8's
  local-only boundary rule is asserted equal to the global one for every chunk at every size.
  `InlineTable` is differentially tested against the offset-keyed table and again across a growth
  boundary, and v9's fixed-width scan against the byte-at-a-time reference for every name length
  0..=100 and every official station — including a case that varies the bytes *past* the `;`, which
  the window loads but must not hash. v10's raw hash is required to spread its high bits over both
  the official names and 10,000 synthetic ones sharing a 23-byte prefix, at every table size growth
  can reach; that assertion is the entire justification for deleting the avalanche. v11's key is
  checked bit for bit against the one built from memory at every length the scan's window covers
  and on every official station, and the scan is required to *decline* at exactly 16 bytes, since
  past there the key's high half is no longer zero and the collapsed compare would be wrong.
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
./target/release/v11_keyed measurements.txt

OBRC_THREADS=18 ./target/release/v11_keyed measurements.txt

cargo test
scripts/edge.sh                                          # hand-built edge cases
scripts/verify.sh v11_keyed                              # differential vs the oracle
scripts/batch.sh scripts/readme.tsv                      # the ladder and both ablation tables
scripts/batch.sh scripts/neon.tsv                        # the rejected NEON bitmap
```

`scripts/batch.sh` is the only sanctioned way to compare two numbers here: it refuses to start on
battery, primes the page cache, discards a warmup pass, and runs the configurations round-robin so
drift lands on all of them equally rather than on whichever went last. `scripts/readme.tsv` is the
45-configuration batch behind the ladder and both ablation tables; the thread sweeps, the `io_floor`
variants and the NEON table each came from a smaller batch of their own, which is why they do not
agree to the millisecond with the big one.

The generator is deterministic and parallel: 1,000 chunks of 1M rows, chunk `i` seeded
`splitmix64(seed ^ i)`, so the file is byte-identical regardless of thread scheduling. It clamps to
[-99.9, 99.9], which the Java original does not — across 1e9 gaussian samples at σ=10 a stray
`-100.2` is not unlikely, and it would silently corrupt any parser assuming two integer digits.

`OBRC_THREADS` sets the worker count everywhere (default 8). `OBRC_CHUNK` additionally tunes the
read size in v8 onward; it was swept and 64 KB costs 877 ms with double the system time, 1 MiB is
the floor at 676 ms, and it is flat past that. Syscall count dominates — keeping the buffer inside
L2 turned out not to matter, which was not the guess.

## Caveats

- Sorting is by raw bytes. That matches Java's `TreeMap` UTF-16 ordering across the whole BMP and
  diverges only above U+10000. Every official station name is BMP, so it is correct here; it is not
  correct in general.
- Numbers are from one laptop and move with power state and thermals. An early batch of results had
  to be thrown out after the machine was unplugged mid-session — a systematic ~13% shift that looks
  exactly like a real regression. Comparisons are only meaningful within a single interleaved batch,
  and even a *plugged-in* one drifts: v9 measures 556 ms in the 45-configuration batch behind the
  tables above, 590 ms in a 6-configuration batch, and 622 ms in another. Any win smaller than about
  5% needs its own batch to be believed, and a *trend* assembled from several such wins needs more
  than that — see the predictions above that did not survive a rep count.
- For scale: the official Java winner is 1.535 s on 8 Zen2 cores, and the fastest known solution in
  any language is [austindonisan's C](https://github.com/austindonisan/1brc) at 0.577 s on that same
  8-core machine. This does 784 ms on 8 cores of an M5 Pro — different silicon, so not a ranking,
  but it does mean roughly 1.4× per core still separates this from the state of the art, down from
  2× before v11. The obvious explanation was his one big technique — parsing many rows at once off
  SIMD delimiter masks rather than one at a time — and that turns out not to be it: ported to NEON
  and measured, it is 8.5% slower than what ships here (above). Whatever the remaining 1.4× is, this
  project has not found it yet.
