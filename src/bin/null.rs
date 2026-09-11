//! Does nothing, on purpose.
//!
//! `v12_pairs` reports 381 ms of wall clock and about 357 of it from the top of `main`. The
//! remaining ~24 ms is `exec`, dynamic linking, Rust runtime setup and whatever teardown
//! survives `std::process::exit(0)` — 6% of the budget that no in-process timer can see and
//! that nobody has ever separated. This binary is the floor of that: same profile, same `lto`
//! and `debug = 1`, same dynamic dependencies, no work.
//!
//! Read it with [`spawn`](../spawn/index.html), which adds only the threads.
//!
//! **The 24 ms was mostly the harness.** In `scripts/v15-bounds.tsv` this binary came back at
//! 18-21 ms, which looked like a confirmation and was not one. `batch.sh` times
//! `subprocess.run(..., shell=True)`, so every number it has ever printed contains a Python
//! `fork`/`exec` and a `/bin/sh`. Priced against a system binary that cannot possibly be slow:
//!
//! ```text
//!                              via sh -c      direct exec
//!   /usr/bin/true                 17.1 ms         5.5 ms
//!   null                          17.5 ms         6.1 ms
//! ```
//!
//! `/usr/bin/true` is indistinguishable from `null`, so essentially none of the 18-21 ms is ours.
//! `exec` + dyld + Rust runtime setup for this binary, over the cheapest process the system can
//! make, is **0.6 ms**. The rest is the cost of asking for a process at all, and ~11 ms of it is
//! the shell that `batch.sh` needs in order to accept `VAR=x cmd >/dev/null` as a config line.
//!
//! The same split on the real binary, 18 threads, page cache primed:
//!
//! ```text
//!   v12_pairs via sh -c                     370.1 ms
//!   v12_pairs direct exec                   360.4 ms
//!   v12_pairs from the top of main          356.5 ms   (OBRC_PHASES)
//! ```
//!
//! So the budget outside `main` is **~4 ms, not ~24**, and 0.6 ms of that is attributable to this
//! binary rather than to Unix. There is nothing there to reclaim, and the plan's Stage 6 is closed
//! before it was opened. [`spawn`](../spawn/index.html) closes the other half: threads are free.
//!
//! Two things follow for everything else in the repo. The honest figure for `v12_pairs` run from a
//! prompt is **~360 ms**, not the 381 the README quotes; the difference is a shell and a Python
//! interpreter. And `batch.sh` numbers carry a ~10-15 ms additive offset that varies a little with
//! how long the payload runs, so they remain good for the comparisons they exist to make and
//! should not be quoted as absolute costs. `batch.sh` is deliberately left alone — changing it now
//! would break comparability with every measurement already recorded here.

fn main() {}
