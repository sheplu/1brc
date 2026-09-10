# One Billion Row Challenge, in Rust

Read 13.8 GB of `<station>;<temperature>` lines, compute min/mean/max per station, print them
sorted. [The original challenge](https://github.com/gunnarmorling/1brc) is Java; this is a Rust
port used as an optimization exercise, so every stage of the work is kept as its own binary and
benchmarked against the ones before it.

**76.6 s → 0.38 s**, a 207× ratio measured in one batch. No external crates: the syscalls are
hand-written `extern "C"`, the threading is `std::thread`, and the only vector code is SWAR in
plain `u64`.

## Results

Apple M5 Pro, warm page cache, 1,000,000,000 rows. Medians of one interleaved batch — every row
below was measured round-robin in a single run, because numbers from different batches on this
machine are not comparable (see Caveats). Every version produces output byte-identical to the v1
oracle's on the full dataset.

| | technique | 8 threads | 18 threads |
|---|---|---:|---:|
| `v1_naive` | `BufReader` + `split(';')` + `f64` + `BTreeMap`, single-threaded | 76.6 s | — |
| `v2_mmap` | mmap, work-stealing chunks, hand-rolled Fx hasher | 1.963 s | |
| `v3_hash` | open-addressing table, 32-byte entries, keys as offsets | 1.916 s | |
| `v4_simd` | SWAR `;` scan with the hash fused into it | 1.807 s | |
| `v5_branchless` | branchless temperature parse, word-wise key compare | 1.521 s | |
| `v6_inline` | station name stored *inside* the table entry | 1.338 s | |
| `v7_pipelined` | 3 independent line streams per chunk | 1.348 s | 741 ms |
| `v8_pread` | `pread` into a per-thread buffer instead of mmap | 1.177 s | 607 ms |
| `v9_flatscan` | fixed-width branchless `;` scan | 873 ms | 490 ms |
| `v10_rawhash` | hash indexed off high bits, no avalanche | 845 ms | 472 ms |
| `v11_keyed` | the probe key falls out of the scan | **685 ms** | 398 ms |
| **`v12_pairs`** | **2 interleaved line streams instead of 3** | 711 ms | **381 ms** |
| ~~`v13_hoist`~~ | ~~table header hoisted, bounds checks removed~~ | 724 ms | 393 ms |

18 threads — every core — is the headline: **381 ms**. 8 threads is the ladder's comparison
baseline, and there the fastest binary is v11, not v12; see below. (`v1_naive` is the one number
measured elsewhere, in a three-config batch that also held v12 at 370 ms and v9 at 474 ms. The
207× ratio in the header is that batch's, so it is internally consistent even though it does not
match this table to the millisecond.)

Every figure above comes from the v14 build: a 1 MiB read chunk and `-C target-feature=+cssc`.
Both are described in their own section below; together they are worth about 60 ms of the 441 ms
this table used to report, and they moved almost every other number in this README too, which is
why the smaller sweeps were all re-measured rather than carried forward.

v13 is in the table because it is struck through. It removes 20 instructions per row and is 3%
slower for it, in four separate batches now; it is kept in the tree as a counterexample and nothing
ships it. See below.

The last three rows are not the same kind of result. v10 removes five serial operations from the
hot loop and the *engine* gets 3–4% faster for it in every batch it has been measured in; the
whole-binary gap has come out anywhere from 0.3% to 6% depending on the batch, so the 3.7% here is
consistent with the engine number and is not on its own evidence of anything. v11 removes about a
third of the hot loop's *instructions*, and that one shows up everywhere: −15.7% at 18 threads,
−18.9% at 8, and −17% of the compute above the read.

v12 changes one constant — three interleaved streams down to two — and it is the row that has
aged worst. It is worth −4.3% at 18 threads and **+3.8% at 8**: on this build v11 is the faster
binary on 8 threads by 26 ms. That is not a regression in v12, it is the sweep moving under it.
The optimum is a property of which cores the scheduler hands out, 8 threads land mostly on the
6 wide P-cores and 18 threads mostly on the 12 narrow E-cores, and the two clusters disagree about
how many streams to run. The binary ships 2 because 18 threads is the headline. The stream sweep
below now measures both.

Thread sweep, v8 against v9 in one batch of their own:

| threads | 6 | 8 | 12 | 14 | 16 | 18 |
|---|---:|---:|---:|---:|---:|---:|
| `v8_pread` | 1.484 s | 1.212 s | 868 ms | 756 ms | 683 ms | 621 ms |
| `v9_flatscan` | 1.090 s | 883 ms | 677 ms | 602 ms | 546 ms | 503 ms |
| | −26.6% | −27.1% | −22.0% | −20.4% | −20.1% | −19.0% |

The win shrinks as threads are added because some fixed fraction of the total is I/O that v9 does
not touch, and that cost is a growing share of a shrinking number. *How much* was got wrong for
several versions; see the next section.

This table is the best evidence in the README that the mechanism is the right one, because it was
re-measured after the read got cheaper and it moved in exactly the predicted direction. The same
sweep on the pre-v14 build read −17.5% at 6 threads and −15.5% at 8. Nothing about v9 changed; the
1 MiB chunk cut the read, compute became a larger share of the total, and a pure-compute win became
a larger fraction of it — up to −27%. A story that only ever explains numbers after the fact is
cheap; this one predicted the sign and the rough size of a change before it was measured.

Both curves are monotone here. An earlier batch had v8 flat across 14 and 16 threads with σ up to
89 ms at 16, and that did not reproduce — worth recording as a caution about reading structure into
a single sweep. Note too that this batch puts v9 at 503 ms and the ladder above says 490. That ±3%
is the batch effect this README keeps insisting on, and it is larger than several of the individual
steps in the ladder.

v11 was checked the same way, in a batch of its own: v10 → v11 is −12.1% at 6 threads, −13.2% at 8
and −9.6% at 18 — the same shape as v9 and for the same reason, since the win is on compute and
compute is a shrinking share of the total as threads are added. The ladder batch puts the same two
numbers at −18.9% and −15.7%. Several points of disagreement between two clean batches, on a win
that is real in both, is the size of the effect this README keeps warning about.

## Where the time actually goes

Where v12 spends 381 ms at 18 threads, from its own `OBRC_PHASES` instrumentation (median of three
runs; the workers' figures are per-thread means, and reading and parsing are strictly serialized
within a thread so the two add):

| phase | ms | |
|---|---:|---|
| outside `main` | 24 | exec, dyld, teardown, and the harness's own spawn |
| workers — read | 46 | 13% of worker time |
| workers — **parse** | **310** | **87%** |
| merge + output | 0.6 | 413 stations, 18 tables |
| straggler | 0.7 | spread between the first and last thread to finish |

Seven eighths of the run is the parse loop, and the sections below are mostly about failing to make
it smaller. The read was 78 ms of 441 before the chunk size was fixed and is 46 ms of 381 now,
which is most of why `parse`'s share went up while its absolute cost went down.

The 24 ms outside `main` is the largest thing here nobody has moved. It is not debuginfo —
`strip = true` was built and measured and is worth exactly nothing — so it is exec, dyld and
teardown, and `std::process::exit(0)` already skips as much of the last of those as is safe.

Two ablation harnesses drove almost every decision here. Guessing was wrong more often than not —
including about which *half* of a technique would be expensive; see the NEON bitmap below.

`hot_floor <mode>` runs the real hot loop with stages progressively switched on, so each row is
the *marginal* cost of one technique rather than a total. `OBRC_READER` picks how the bytes arrive
and the mode picks which table, and **the pair matters**: a mode is only comparable to another
measured the same way.

Under mmap with the offset-keyed table — the configuration that decided v6 (8 threads):

| stage | time | marginal | |
|---|---:|---:|---|
| `touch` | 354 ms | — | read every byte, XOR-reduce, nothing else |
| `parse` | 1.219 s | +865 ms | SWAR `;` scan + branchless temperature parse |
| `hash` | 1.240 s | +21 ms | fuse the name hash into the scan |
| `nocmp` | 1.269 s | +29 ms | probe and update a table slot |
| `table` | 1.505 s | +236 ms | **compare the key** |

The last two rows are the whole story of v6. Probing the hash table is nearly free; *verifying* the
key cost more than hashing and probing combined. Not because of the bytes compared — because with
keys stored as offsets, checking one means dereferencing into a random spot in a 13.8 GB mapping, a
second scattered memory access on every row. Moving the name onto the entry's own 64-byte cache
line, which the probe has already loaded, deleted that access and most of that 242 ms with it.

The same harness measures instruction-level parallelism headroom by walking N independent line
streams: `parse` 1.219 s → `parse2` 857 ms → `parse4` 779 ms. Rows form a serial dependency chain
— a row's start address is only known once the previous row's temperature has been parsed — so a
single stream leaves most of the out-of-order window idle.

That is also the most instructive failure in the project. Streaming was tried against v5 first and
*lost* (1.76 s vs 1.671 s, on the build of the day), and giving each stream its own table was worse
still. The offset-keyed
table ended each row with a load-modify-store the streams had to serialize on, which cost more than
the overlap won. The technique only became a win once v6 had made the update L1-resident. **The
1.7× of parsing headroom was real the whole time and unreachable until an unrelated bottleneck
moved.**

Under `pread` with the inline-key table — what v8 onward actually ship (18 threads, table held at
2^16 slots throughout so the stage is the only variable):

| stage | time | marginal | |
|---|---:|---:|---|
| `touch` | 114 ms | — | read every byte and XOR-reduce it; **not the read floor, see below** |
| `parse` | 469 ms | +355 ms | scan + parse |
| `hash` | 479 ms | +10 ms | + the fused hash |
| `itable` | 566 ms | +87 ms | + the inline-key upsert; this is v8's engine |
| `fhash` | 418 ms | −61 ms *vs* `hash` | the scan's loop replaced by a fixed window |
| `flat` | 510 ms | −56 ms *vs* `itable` | the same swap, with the table; this is v9's |
| `raw` | 494 ms | −16 ms *vs* `flat` | the hash's final avalanche deleted; v10's |
| `keyed` | 448 ms | −46 ms *vs* `raw` | the probe key taken from the scan; v11's |

That second half was measured last and should have been measured first. For five versions the
harness ran only against mmap and only against the offset-keyed table, so **every marginal cost in
the first table describes v5's engine**, not the shipping one. Fixing the instrument was what
exposed v9, and then v11.

With interleaved streams, which is what the binaries do, the same four rows are `itable3` 576 ms →
`flat3` 464 ms → `raw3` 446 ms → `keyed3` 371 ms → `keyed2` 359 ms. The `raw3` → `keyed3` step is
the largest single win in the project measured against compute.

### The read floor was an artifact

`io_floor <mode>` A/Bs just getting at the bytes, no parsing:

| | 8 threads | 18 threads |
|---|---:|---:|
| mmap | 350 ms | 237 ms |
| pread | 144 ms | 167 ms |

mmap takes a minor fault per 16 KB page — 842k of them — and all workers take them against the same
`vm_map`. `pread` copies 13.8 GB instead and is still less than half the cost.

`pread` at 18 threads is now *slower* than at 8, which it was not before the chunk size came down:
the old figures were 224 and 225 ms, dead flat. The file is now 13,157 one-mebibyte chunks rather
than 6,579 two-mebibyte ones, so there are twice as many trips into the copy path and half as much
work per trip to hide the contention behind; eighteen threads feel that and eight do not. It does not matter — the real
program is not eighteen threads doing nothing but reading, which is the whole point of the next
subsection — but it does mean the "floor" is not even monotone in thread count.

Attempts to push that floor lower, none of them a win, at 18 threads:

| | time | |
|---|---:|---|
| `pread` | 183 ms | baseline |
| `pread_fd` | 177 ms | one descriptor per thread instead of a shared `&File` |
| `mmap_seq` | 251 ms | `MADV_SEQUENTIAL` |
| `mmap_shared` | 230–779 ms | `MAP_SHARED`, wildly bimodal run to run |
| `mmap_willneed` | 570 ms | `MADV_WILLNEED` — **3.1× worse** |
| `pread_nored` | 175 ms | *not an attempt* — the XOR reduce removed, so the copy is 96% |

`pread_fd` was the one expected to pay: `pread` barely scales from 8 threads to 18 while nothing
else on the machine is saturated, which looks exactly like contention on the shared file object. It
is not — per-thread descriptors are inside the noise, so the ceiling is in the copy path itself.
And the `madvise` calls, the obvious free win, range from neutral to actively harmful.

**All of which pointed at a 230 ms floor that is not there.** Earlier drafts of this README quoted
that number as a fixed I/O cost sitting under every version, and used it to explain why each
compute win shrank as threads were added. It is an artifact of *how* it was measured: `touch` and
`io_floor` do nothing but read, so eighteen threads arrive at the kernel's copy path simultaneously
and queue behind each other. That is not the situation the real program is in.

`OBRC_READER=mmap_warm` prefaults the mapping before the timer starts, so the same mode runs with
the bytes already resident. The difference between a mode and its warm twin is what the read
actually costs *that* mode — and it depends entirely on whether the threads have anything else to
do. Same batch, 18 threads:

| mode | `pread` | warm | the read costs |
|---|---:|---:|---:|
| `touch` | 114 ms | 56 ms | 58 ms |
| `keyed3` | 371 ms | 321 ms | 50 ms |
| `keyed2` | 359 ms | 310 ms | **49 ms** |

v12's own `OBRC_PHASES` says 46 ms independently, from a completely different instrument. So a
*free* reader would be worth about 48 ms of 381, not 230 — the rest overlaps with compute, because
a thread that is parsing is not queueing. Every copy-free design that was going to reclaim
"230 ms" — `mlock`, dedicated toucher threads, a producer-consumer split — was competing against a
phantom, and all of them are now retired on that basis rather than on their own measurements.

The gap between `touch`'s isolated read and the in-place read has also closed on its own: it was
174 ms against 71, and it is now 58 against 49. Shrinking the chunk did nothing to the bytes copied
and everything to how long eighteen idle threads spend queued behind each other, which is one more
way of saying that the isolated number was measuring the harness.

The general lesson is the one the harness keeps teaching in different costumes: **a stage measured
in isolation is not the same stage measured in place.** An ablation that removes everything else
does not price a component, it prices a component under contention it would never otherwise meet.

Swapping the reader is not free architecturally: `pread` recycles one buffer per chunk, so a key
stored as an offset into it would dangle as soon as the next chunk was read. The table had to be
made to own its keys first.

## What each version actually changed

- **v2** — mmap, and work-stealing over fixed-size chunks *from the start*. macOS/arm64 exposes no
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
  roughly a quarter of the compute above the read, from a single branch.

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

  ~148 instructions down to ~96, and no spill in the hot path. Engine `raw3` → `keyed3` is −75 ms,
  which is a fifth of all compute above the read; end to end −15.7% at 18 threads and −18.9% at 8.
- **v12** — one constant. v7 chose 3 interleaved streams and every version since inherited it
  without rechecking, on measurements taken when the loop was a different shape. v11's is smaller
  and holds more live state per stream, so it was worth sweeping again. Warm reader, one batch:

  | streams | 1 | 2 | 3 | 4 | 6 |
  |---|---:|---:|---:|---:|---:|
  | 18 threads | 405 ms | **315 ms** | 330 ms | 339 ms | 354 ms |
  | 8 threads | 861 ms | 624 ms | **599 ms** | 611 ms | 671 ms |

  Monotone worsening past the optimum is the informative part. If the loop were stalling on the
  row-to-row dependency, more streams would keep helping; they do not, so the remaining compute is
  throughput and the extra streams are competing for registers rather than covering latency.

  **The two rows disagree, and that is the result.** 8 threads land mostly on the 6 wide P-cores
  and want 3 streams; 18 threads pull in the 12 narrow E-cores and want 2. The reversal shows up
  under `pread` too (18 threads: 366 ms at N=2 against 380 at N=3; 8 threads: 709 against 682) and
  end to end, where v11 beats v12 by 26 ms on 8 threads and loses to it by 17 on 18. So the shipped
  constant is not "the right number of streams", it is the right number *for the headline
  configuration*, and a build targeting 8 threads should ship v11's 3.

  This is also the second time this exact sweep has moved. Measured on the pre-v14 build it read
  355 ms at N=2 and 374 at N=3 for 18 threads, and 662 against 643 at 8 — same ordering, but the
  8-thread gap has widened from 19 ms to 25 and now survives into the whole-binary numbers, where
  before it was called "a wash". One constant, three sweeps, and the answer depends on the chunk
  size and the ISA flags. Nothing here is a property of the algorithm.
- **v13** — does not ship. The v12 disassembly showed the hot loop reloading four words of the
  table header from the stack on every row, because `&mut InlineTable` stays live across a call
  that might `grow` and replace `entries`, plus about eleven bounds-check instructions from slice
  indexing the compiler cannot prove in range. Both are removable in safe Rust: handing out
  `(&mut [Entry], u32)` lets the borrow checker prove `entries` cannot move, and deriving the mask
  from `slots.len() - 1` makes `x & (len-1) < len` provable — which is exactly how
  `InlineTable::grow` already compiled while `upsert_words` did not. Fixed-size `&[u8; 16]` and
  `&[u8; 8]` windows move the length into the type so the scan and the parser check nothing.

  It all worked. Per row: 116 instructions down to 96, four `ldr [sp, …]` down to zero, no compare
  on the slot index and none on the probe's back-edge. And it is slower — 381 → 393 ms at 18
  threads, 711 → 724 at 8, reproduced in four separate batches across two different builds. See
  below.

## v14 — two things that were not in the source

441 ms → 381 ms without touching the hot loop. Both of these are the same kind of finding: the
program was fine and the *build* was describing a machine slightly different from the one it runs
on. Neither creates a new binary; both apply to every version in the ladder, which is why every
table above was re-measured.

**The chunk was twice the size the sweep said it should be.** Each worker `pread`s one chunk into a
recycled buffer and then parses it. `CHUNK_SIZE` had been 2 MiB since v2. At 18 threads:

| chunk | 128 KiB | 256 KiB | 512 KiB | 768 KiB | 1 MiB | 1.25 MiB | 1.5 MiB | 2 MiB | 4 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coarse sweep | 557 ms | 493 ms | 480 ms | | **430 ms** | | | 441 ms | 443 ms |
| finer sweep | | | | 431 ms | **427 ms** | 432 ms | 432 ms | 436 ms | |

Two batches, each read only against itself — the 2 MiB anchor lands at 441 in one and 436 in the
other, which is the usual spread and the reason the rows are not merged. They agree on the shape
and on the size: 1 MiB is worth 9–11 ms over 2 MiB.

The climb below 512 KiB is bookkeeping — 13.8 GB in 128 KiB pieces is 105k `pread`s. The tilt above
1.5 MiB does not appear at 8 threads (1 MiB 807 ms, 1.25 MiB 801, 2 MiB 801, same batch as the finer
sweep), which points at the shared L2: an E-cluster is 8 MB across 6 cores, so six 1 MiB buffers
survive between the kernel's write and the parse's read and six 2 MiB ones do not.

The first guess was the opposite of this and was wrong in direction, not just in size. The
prediction was that *smaller* would win on per-core L2 capacity, and 128 KiB is the worst point on
the curve by 130 ms. The 8-thread control is what killed the capacity story outright: at 8 threads
512 KiB is 879 ms against 2 MiB's 815, so shrinking the buffer hurts exactly where each core has
the most cache to itself.

The embarrassing part is that this README already contained the sweep. It said, in the *Running it*
section and for several versions, that 1 MiB was the floor and that the curve was flat past it. It
was flat past it — on v8, where the parse was slow enough to hide the difference. "Flat, therefore
the constant does not matter" was true when measured and quietly stopped being true, and nothing
re-checked it because the number had already been swept once.

**`target-cpu=native` is not native.** LLVM resolves `native` to `apple-m4` on this machine.
`sysctl` reports `hw.optional.arm.FEAT_CSSC: 1`, but neither LLVM's `apple-m4` nor its `apple-m5`
model enables the `cssc` feature, so the compiler will not use it — a diff of `rustc --print cfg`
between the two CPU models shows m5 adding only `mte`. FEAT_CSSC provides, among other things, a
GPR count-trailing-zeros; without it LLVM emits `rbit` + `clz` for every `trailing_zeros()`, and
this hot loop calls that **three times per row**, and every one of them is on the chain that
carries `pos` from one row to the next.

Adding `-C target-feature=+cssc` (which merges with `target-cpu` rather than replacing it — all 31
native features survive) turns 44 `rbit`/`clz` pairs into 46 `ctz` across the binary, 9 of them
inside the hot worker. Worth 15 ms at 18 threads and 27 ms on the read-free `hot_floor` at 8, in a
two-build A/B batch.

That sits awkwardly next to v13, which is the project's headline evidence that removing
instructions does not make this loop faster. The two are reconciled under *Fewer instructions per
row* below; the short version is that it depends entirely on whether the instruction was on the
dependency chain or merely in the instruction stream.

It has two costs, and they are why it is worth stating rather than just doing. rustc considers
`cssc` an unstable feature name, so every compile warns. And unlike `target-cpu=native`, which
degrades gracefully, this does not: on an M1 through M3 the binary will `SIGILL`. See Caveats.

## Seven things that were predicted to work and did not

Recorded because the plans said they would, in writing, before the measurement.

**The table would fit the dTLB.** 2^16 slots is 4 MiB per thread holding 413 live entries, scattered
across ~205 distinct 16 KB pages — far more than the L1 dTLB maps. Shrinking to 2^11 (128 KB, 8
pages, permanently resident) was predicted to be worth 40–70 ms. Whole-binary, 18 threads, one
batch:

| slots | 2^11 | 2^12 | 2^13 | 2^14 | 2^15 | 2^16 |
|---|---:|---:|---:|---:|---:|---:|
| | 439 ms | 408 ms | 393 ms | 389 ms | 387 ms | 391 ms |

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
| `touch` | 130 ms | reading alone, which overstates the read — see above |
| `mask_semi` | 133 ms | the `;` bitmap, XORed into the sink and discarded |
| `mask` | 140 ms | both bitmaps |
| `dlrows` | 334 ms | + drain the bits to `(name, semi, nl)` triples |
| `dltable` | 472 ms | + hash, parse and upsert — v10's per-row work, no scanning left |
| `raw3` | 473 ms | v10's engine, three scalar streams |
| `keyed3` | **384 ms** | v11's engine |

The predicted problem was the movemask tax. AVX2 turns 32 bytes into a mask with one
`vpmovmskb`; aarch64 has no such instruction, so each 64-byte mask costs four `cmeq`, four `and`
against a bit-position vector and a four-deep `addp` tree — 13 operations where AVX2 spends 2. That
tax is real and it is *irrelevant*: both bitmaps together cost 10 ms over `touch`, under 8% of a
mode that reads 13.8 GB.

The cost is on the other side. Draining the bits into row positions is +194 ms, nineteen times what
building them cost. And the row work that follows cannot use v11's trick: v11's probe key is free
*because the scan already masked those bytes*, and a bitmap design has no scan, so it has to load
and mask the name itself. The bitmap now merely *ties* v10's engine and loses to v11's by 23%,
having forfeited the thing that made v11 fast.

It is not a small-core artefact either, which was the failure mode the plan called out in advance.
At 6 threads, where the scheduler has the fast cores to hand out, `dltable` is 1.076 s against
`keyed3`'s 789 ms — a 36% loss. The gap is *widest* where per-core throughput is highest, which is
the opposite of the predicted shape. The module stays in the tree as an instrument; nothing ships
it.

Re-measuring this on the v14 build was the one re-run expected to change a conclusion, and it
changed it in the wrong direction. Draining a bitmap is a loop whose body is almost entirely
`trailing_zeros`, so `+cssc`'s single-instruction `CTZ` should favour the bitmap more than anything
else in the tree. It went from 8.5% behind `keyed3` to 23% behind, and from 3% ahead of `raw3` to
level with it. Both designs got faster; the scalar one got faster faster.

**Reclaiming the read.** A whole plan was written against the 230 ms read floor: `mlock` the
mapping, or dedicate threads to touching pages ahead of the parsers, or split producer from
consumer outright. Every design in it was arithmetic on a number that turned out to be an artifact
of idle threads queueing on the kernel copy. Measured in place the read costs 46–50 ms, three
different instruments agree, and the plan was deleted unimplemented. That is the cheapest failure
in the project and the only reason it was cheap is that the premise got measured before the code
got written. The section above has the numbers.

**Fewer instructions per row.** v13, above. Four results now say the parse loop is not bound by
anything the source can address: v12's stream sweep says it is not latency past two streams (a
third is worse at 18 threads and a fourth at 8), the NEON bitmap says it is not byte scanning (both
delimiter masks cost 10 ms over `touch` and the design still lost), pairing the accumulate says it
is not store ports (half the memory operations, 1.5%), and v13 says it is not instruction count
either — 20 fewer per row, 3% slower.

`+cssc` is the apparent exception and is worth holding next to this one, because it removes
instructions and *does* pay. The difference is which instructions, and the distinction is sharper
than "on the dependency chain" — there are two chains, and only one of them matters.

A row's work forks. One branch runs from the name's bytes through the hash to the table slot, the
probe, and the accumulate. The other computes where the *next* row starts:

```
pos → load 16B → semi_mask → tz → len → semi → load 8B at semi+1 → tz → dot_bit → next pos
```

Only the second is loop-carried. The table branch hangs off `len` and never feeds back, so an
out-of-order window that is already running two streams has slack to hide all of it. That
recurrence contains two dependent loads and two `trailing_zeros`, and `+cssc` shortens both of the
latter — which is the whole of its 15 ms.

This is also the answer to the obvious next move, which was on the candidate list and is now
withdrawn. `mix` ends in a 3–4 cycle multiply sitting directly between the masked name word and the
table load, the hash only picks a slot, and 413 stations in 16384 make a weaker index nearly free —
so replacing the multiply with two shifted `eor`s looks like a cycle for nothing. It is a cycle off
the branch that is not loop-carried. Two streams already cover the recurrence — that is what the
sweep's interior optimum means — so a chain with strictly more slack than the recurrence cannot be
what the loop is waiting on. Not attempted, and not because it would be hard.

v13's mechanism is worth stating because it generalises. The eleven bounds checks it deleted
branched to *panic* blocks: never taken, macro-fused, and with nothing live at the target, so the
register allocator could ignore them entirely. What replaced them — `get(pos..)?`, `first_chunk()?`,
and a probe that returns `false` instead of inserting — are real loop exits, and every
loop-carried value has to be materialized at each one. v13's preamble spills registers v12 never
needed to. **An instruction count is not a cost model**: it does not price live ranges, and on a
core this wide a never-taken fused compare-and-branch is nearly free while a constraint on the
allocator is not.

Ablations, warm reader at 18 threads, from a batch of their own where the spreads were tight
(`keyed2` 333–351, `hot2` 349–357):

| | time | |
|---|---:|---|
| `keyed2` | 310 ms | v12's engine |
| `win2` | 311 ms | + fixed-size windows |
| `hdr2` | 317 ms | + the header hoist |
| `hot2` | 323 ms | both; the costs add |

Same ordering at 8 threads and under `pread`, and the same ordering on the v14 build as on the one
this was first measured on — though the windows half has shrunk to 1 ms and is now only really
supported by `hot2` being worse than `hdr2` by about the same amount. The header hoist is the half
that reliably costs. `src/bin/v13_hoist.rs` stays in the tree with the measurement in its module
doc, and `scripts/edge.sh` runs it against the oracle like any other version — it is correct, just
slower.

**Pairing the accumulate.** Every repeat row does `sum += value; count += 1` — two loads, two adds,
two stores, on fields sitting four bytes apart in the entry. `sum` is `i64` and `count` is `u32`,
so nothing pairs; widening `count` to `u64` makes them two adjacent 8-byte fields at an 8-aligned
offset, which is the `ldp`/`stp` shape. `Entry` stays at 64 bytes by shrinking the stored hash to
its top 32 bits, which is all a slot index ever reads.

The codegen worked, differently from the prediction: LLVM did not emit `ldp`/`stp`, it vectorised,
turning four loads and two stores into one `ldr q` and one `str q`. Half the memory operations for
the same instruction count. Measured against a snapshot of the identical source with `count: u32`,
11 reps, two configs: median **402 ms → 396 ms**, minimum 396 → 392. Six milliseconds against a
gate of eight, in exchange for permanently truncating the stored hash. Reverted; the `Entry` doc
comment carries the measurement so the next person does not retry it.

It took three attempts to get that number, and the first two are the more useful part. The opening
batch produced a 3.366 s outlier in a field of 0.4 s. The follow-up entered *the same binary* twice
under two labels and got 636 and 619 ms for one, 614 and 646 for the other — a 32 ms noise floor
around a 6 ms effect. Only the third, run on an idle machine, could resolve the sign. A gate of
8 ms is not conservatism; it is roughly what this laptop can actually distinguish on a good day.

Worth noting what this rules out. Halving the store traffic of the one operation every single row
performs is worth about 1.5%, which puts a ceiling on how much of this loop is store-port bound,
and it is a low ceiling.

**More threads than cores, and stripping the binary.** Two free knobs, swept together, both zero.
Thread counts of 20, 24, 28 and 36 against the default 18 came in at 439, 443, 442 and 446 ms
against 441 — flat, no trend, the best of them inside a millisecond of the default.
Oversubscription can only pay if workers stall, and the read is 46 ms of 381 and stalls nobody.
`strip = true` in place of `debug = 1` measured 441 against 441; it cannot change codegen, and the
hope was that it would shorten exec and dyld, which is 24 ms of the run. It does not measurably.
Both knobs are left where they were.

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
  inline-key seams, 40-byte shared prefixes, every rounding extreme) run through all twelve fast
  versions against the oracle.
- `cargo test` — 58 tests, in both debug and release. The parser is proved by exhaustion: all 1999 legal temperature strings
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
./target/release/v12_pairs measurements.txt

OBRC_THREADS=18 ./target/release/v12_pairs measurements.txt

cargo test
scripts/edge.sh                                          # hand-built edge cases
scripts/verify.sh v12_pairs                              # differential vs the oracle
scripts/batch.sh scripts/readme.tsv                      # the ladder and both ablation tables
scripts/batch.sh scripts/v14-sweeps.tsv                  # threads, table size, read floor
scripts/batch.sh scripts/v14-streams.tsv                 # the stream sweep, both clusters
scripts/batch.sh scripts/v14-stage1.tsv                  # the coarse chunk sweep
scripts/batch.sh scripts/v14-chunk.tsv                   # the finer one, around the optimum
scripts/batch.sh scripts/neon.tsv                        # the rejected NEON bitmap
```

`scripts/batch.sh` is the only sanctioned way to compare two numbers here: it refuses to start on
battery, primes the page cache, discards a warmup pass, and runs the configurations round-robin so
drift lands on all of them equally rather than on whichever went last. `scripts/readme.tsv` is the
49-configuration batch behind the ladder, both ablation tables, the v13 ablations and the read-cost
correction. Everything else came from a smaller batch of its own, which is why those numbers do not
agree to the millisecond with the big one: `v14-sweeps` holds the thread sweep, the table-size sweep
and the `io_floor` variants, `v14-streams` the stream sweep, `v14-stage1` and `v14-chunk` the two
chunk sweeps, `v14-naive` the `v1_naive` pairing, and `neon` the rejected bitmap. All of them were
re-run for v14 — a build change invalidates an old number more firmly than batch drift does.

The generator is deterministic and parallel: 1,000 chunks of 1M rows, chunk `i` seeded
`splitmix64(seed ^ i)`, so the file is byte-identical regardless of thread scheduling. It clamps to
[-99.9, 99.9], which the Java original does not — across 1e9 gaussian samples at σ=10 a stray
`-100.2` is not unlikely, and it would silently corrupt any parser assuming two integer digits.

`OBRC_THREADS` sets the worker count everywhere (default 8; 18 is the headline). `OBRC_TABLE_BITS`
sets slots per table as a power of two (default 14). `OBRC_CHUNK` tunes the read size in v8 onward;
the default is 1 MiB and the sweep behind that is in the v14 section above. `OBRC_PHASES` prints
the phase breakdown, and `OBRC_READER` picks how `hot_floor` gets its bytes (`pread`, `mmap`,
`mmap_warm`).

## Caveats

- **The release build requires FEAT_CSSC, which means an Apple M4 or newer.** `.cargo/config.toml`
  passes `-C target-feature=+cssc`, and unlike `target-cpu=native` that flag does not degrade: on an
  M1 through M3 the binary compiles and then `SIGILL`s. Delete the second flag in that file to build
  there, at a cost of about 4%. rustc also considers the feature name unstable, so every build
  prints a warning; that is expected and is the reason the flag carries a comment.
- Sorting is by raw bytes. That matches Java's `TreeMap` UTF-16 ordering across the whole BMP and
  diverges only above U+10000. Every official station name is BMP, so it is correct here; it is not
  correct in general.
- Numbers are from one laptop and move with power state and thermals. An early batch of results had
  to be thrown out after the machine was unplugged mid-session — a systematic ~13% shift that looks
  exactly like a real regression. Comparisons are only meaningful within a single interleaved batch,
  and even a *plugged-in* one drifts: on the v14 build v9 measures 490 ms in the 49-configuration
  batch behind the tables above, 503 ms in the thread sweep, and 474 ms in the three-config
  `v1_naive` pairing — a 6% spread on an unchanged binary in one afternoon. Any win smaller than
  about 5% needs its own batch to be believed, and a *trend* assembled from several such wins needs
  more than that — see the predictions above that did not survive a rep count.

  A batch is also only valid against a quiet machine, which is not the same as a plugged-in one.
  Three batches in this round had to be discarded after a duplicate-arm control — the same binary
  entered twice under two labels — came back 17 to 32 ms apart. The culprit was ordinary desktop
  load, and the fix was to wait, not to tune anything. Entering one arm twice is cheap and is the
  only way to measure the noise floor rather than assume it.
- For scale: the official Java winner is 1.535 s on 8 Zen2 cores, and the fastest known solution in
  any language is [austindonisan's C](https://github.com/austindonisan/1brc) at 0.577 s on that same
  8-core machine. This does 685 ms on 8 cores of an M5 Pro — different silicon, so not a ranking,
  but it does mean roughly 1.19× per core still separates this from the state of the art, down from
  1.4× before v14 and 2× before v11. (685 ms is v11; v12's two streams are tuned for 18 threads and
  cost 26 ms at 8.) The obvious explanation was his one big technique — parsing many rows at once
  off SIMD delimiter masks rather than one at a time — and that turns out not to be it: ported to
  NEON and measured, it is 23% slower than what ships here (above). Whatever the remaining 1.19×
  is, this project has not found it yet.
