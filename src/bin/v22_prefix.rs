//! v22: the key mask is built without a scalar, to test whether crossings are the term.
//!
//! **The finding this acts on.** v21 removed 16.5 integer operations a row — over twice v19's 7.5 —
//! and returned 8 ms where v19 returned 23. The account written down for that was the register-file
//! crossing: v19's extracts produced values the row wanted in general-purpose registers anyway,
//! while v21 has to send `len` *back* into the vector unit to build a lane mask and pull the masked
//! words out again, in series on the chain feeding the hash.
//!
//! That is an account, not a measurement. This version is the experiment that decides it.
//!
//! **The change.** The prefix mask does not actually need `len`. A prefix-OR across the `cmeq`
//! result — `vextq_u8(zero, p, 16 - k)` is a left shift by `k` lanes, and four of them at
//! k = 1, 2, 4, 8 propagate every set lane forward into all higher lanes — marks every byte at or
//! after the first `;`. One `bic` then keeps exactly the bytes before it. No scalar is involved at
//! any point, so `klo` and `khi` stop depending on `len` and the `dup` disappears.
//!
//! The row now has two chains that do not wait on each other, where it had one long one:
//!
//! ```text
//!   v21   ldr q → cmeq → shrn → fmov → ctz → dup → cmhi → and.16b → fmov → hash
//!   v22   ldr q → cmeq → ext/orr ×4 → bic → fmov → hash
//!                     └→ shrn → fmov → ctz → semi
//! ```
//!
//! `len` is still computed off the same `shrn` — the row needs it for `semi` and for the `short`
//! test — but nothing on the key's path waits for it any more.
//!
//! **What deliberately does not move.** Everything else. Same hash, same `short = len < 8`, same
//! two `csel`, same parser, same `upsert_words`, same table, same two streams, same chunking. This
//! differs from v21 in one function call.
//!
//! **Building this corrected the instrument.** `scripts/loopcount.py` classified by mnemonic, and
//! two of its buckets were wrong in the direction that flattered the argument they were supporting:
//! `orr.16b` fell through to `const` and `ext.16b` to `other`, so eight of v22's vector operations
//! would have been counted as integer work — and `dup.16b v1, w4`, a general-purpose register being
//! written into the vector file, was filed as ordinary vector work even though the entire v21
//! write-up turns on counting exactly that. It now decides by which register files an instruction
//! names. The corrected counts, with the versions this changes marked:
//!
//! ```text
//!                  insns/row   integer/row   vector/row   xfer/row   inbound   measured vs v12
//!   v12                107.0          79.5          0.0        0.0         0                 —
//!   v19                105.0        † 72.0          2.0      † 2.0         0            −23 ms
//!   v20                 95.5        † 69.5          2.0      † 2.0         0            −16 ms
//!   v21                 91.5        † 55.5        † 5.0      † 5.0         1             −8 ms
//!   v22                 97.5          55.5         13.0        4.0         0                 ?
//! ```
//!
//! **v19 did not add zero crossings; it added two.** They are `fmov x8, d0` and
//! `mov.d x9, v0[1]`, the two extracts of the compare result, and both are *outbound* — they
//! produce the delimiter masks in the general-purpose registers where `trailing_zeros` needs them
//! anyway. The qualitative claim in v19's write-up survives; the number in it does not, and the
//! README's crossing table is corrected alongside this.
//!
//! v21's five are those two, plus the `dup` that sends `len` back in, plus a second outbound
//! extract, plus one value LLVM parks in `d0` and reloads — register pressure relieved through the
//! vector file rather than the stack. v22 has four: the same set less the `dup`. So the column
//! that actually separates them is the last one. **v21 is the only version with an inbound
//! crossing, and v22 exists to delete it.**
//!
//! **What makes this a clean experiment.** The integer column does not move at all — 55.5 in both.
//! That has never happened before here; every previous version changed the integer count and the
//! vector count together, so no measurement could tell which one it had paid for. This one varies
//! the vector and crossing mix with the integer side pinned exactly.
//!
//! **The three models disagree, which is the point.**
//!
//! ```text
//!   instruction count (retired by v20)   +6.0 a row              slower
//!   integer operations (retired by v21)  no change               nothing
//!   crossings (the live one)             −1, and the inbound one faster
//! ```
//!
//! And a fourth reading says slower for a different reason: the prefix-OR is *itself* a serial
//! chain. Four dependent `ext`/`orr` pairs at roughly two cycles each is about sixteen cycles to
//! the mask, where v21's `dup → cmhi → and` is about eight once `len` exists. v22 wins only if
//! detaching the key from `len` is worth more than lengthening the key's own path.
//!
//! Nothing here supports a number, so the band written down in advance is **−10 to +10 ms**, and
//! the informative result is the sign and whether it clears the duplicate-arm spread. A v22 that
//! is not slower despite eight more vector operations a row says the vector unit still has the
//! slack v19 assumed; a v22 that is slower by about the instruction count says that era is over
//! and vector issue is now a real cost.
//!
//! **Measured** (`scripts/v22-prefix.tsv`, REPS=7):
//!
//! ```text
//!                      t18            t8
//!   v12           401 ms         722 ms
//!   v19           375
//!   v21           365  364       672
//!   v22           359  358       667
//! ```
//!
//! **−6 ms at 18 threads and −5 at 8, on duplicate spreads of 1 ms in both pairs.** Six times the
//! noise floor, and the fastest version the project has: 358 ms median, 353 ms best. (This batch
//! reads v12 at 401 where the two before it read 390 and 384, so the machine is some 4% slow
//! today and only the within-batch ordering is quoted. It is monotone: v22 < v21 < v19 < v12.)
//!
//! **The crossing model is the only one that got the sign right.** Instruction count said slower
//! and was wrong for the fourth time. Integer operations said nothing at all — the column is
//! identical — and a change worth 6 ms happened anyway, which is the cleanest possible refutation
//! of it, because no other variable moved to argue about. Only the crossing account predicted a
//! win, and the win is where it said it would be.
//!
//! So an inbound register-file crossing on the row's critical path is worth about 6 ms here, and
//! v21 was paying it. That is a real term, and it is the first one this project has *predicted*
//! before measuring rather than fitted afterwards.
//!
//! **It still does not make the integer column a rate.** Crediting v21 with the 6 ms it was losing
//! to the `dup`:
//!
//! ```text
//!                        integer removed   crossings added   measured   per integer op
//!   v19 vs v12                       7.5             2 out     −23 ms          −3.1 ms
//!   v21 vs v19                      16.5    2 out, 1 inbound   −10.5            −0.6
//!   v21 vs v19, less the `dup`      16.5             2 out     −16.5            −1.0
//! ```
//!
//! v19 is still three times the rate per integer operation removed, with the crossings netted out.
//! The gap it opened is smaller than it was and it has not closed.
//!
//! **And the vector unit still has slack.** Eight more vector operations a row, thirteen in total
//! against v19's two, and the row got faster. Every version since v19 has rested on the premise
//! that the unit beside the integer ALUs is idle; this is the first evidence for it that is not
//! v19's own arm.
//!
//! **What is left.** Two things this cannot see. The row's chain still terminates in a dependent
//! load from a table whose 413 live entries occupy 413 distinct cache lines — 53 KB against the
//! 64 KB L1d on twelve of this machine's eighteen cores — and `OBRC_TABLE_BITS` has never varied
//! that, because changing the slot count does not change the live-set footprint. And the three
//! remaining outbound crossings are all real work, but two of them feed `upsert_words`, which
//! compares the key in general-purpose registers. A `cmeq`/`bic` comparison against a vector-laid
//! table would delete both. See "A crossing has a price" in the README.
//!
//! Usage: v22_prefix [path]   (OBRC_THREADS, default 8; OBRC_TABLE_BITS, default 14)

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use obrc::chunk::{local_bounds, num_chunks, split_streams, CHUNK_SIZE};
use obrc::inline_table::{InlineTable, KEY_SLACK};
use obrc::output::{write_results, Stats};
use obrc::parse::parse_temp_branchless;
use obrc::swar::{find_semi_and_hash_flat_raw, scan_flat_keyed_pfx, FLAT_SLACK};
use obrc::sys::set_thread_qos_user_interactive;
use obrc::MAX_LINE_LEN;

/// See v10: 2^13..2^16 all measure the same, and this is the smallest of them.
const DEFAULT_TABLE_BITS: u32 = 14;

/// v12's pair loop, unchanged.
const STREAMS: usize = 2;

/// Read past the chunk so the line straddling its end can be finished locally.
const OVERLAP: usize = MAX_LINE_LEN;

/// Slack past the read that the wide path may touch: the flat scan reads a 16-byte window
/// from the name start, the parser loads 8 bytes, and the cold path's inline probe reads 32.
const TAIL_GUARD: usize = MAX_LINE_LEN + FLAT_SLACK;
const _: () = assert!(TAIL_GUARD >= KEY_SLACK);

/// v21's row with the prefix-OR scan substituted. One call differs.
#[inline(always)]
fn step(buf: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    let Some((semi, hash, klo, khi)) = scan_flat_keyed_pfx(buf, pos) else {
        return step_long(buf, table, pos);
    };
    let (value, next) = parse_temp_branchless(buf, semi + 1);
    table.upsert_words(buf, pos, semi - pos, hash, value, klo, khi);
    next
}

/// A name of 16 bytes or more: v10's row, out of line so it shares no registers with the hot
/// path. 2.9% of the official station list and none of this dataset's common names.
#[cold]
#[inline(never)]
fn step_long(buf: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    let (semi, hash) = find_semi_and_hash_flat_raw(buf, pos);
    let (value, next) = parse_temp_branchless(buf, semi + 1);
    table.upsert(buf, pos, semi - pos, hash, value);
    next
}

/// What one worker spent its life on. Three `Instant::now()` per chunk is ~730 chunks a
/// thread and under 30 µs all told, so this is measured unconditionally and only *reported*
/// under `OBRC_PHASES`. Reading and parsing are strictly serialized within a thread, so the two
/// do add up — the interesting quantity is the ratio, and how far the slowest thread runs past
/// the fastest.
#[derive(Default, Clone, Copy)]
struct Spent {
    read: Duration,
    parse: Duration,
    total: Duration,
}

fn worker(
    file: &File,
    len: usize,
    chunk: usize,
    bits: u32,
    cursor: &AtomicUsize,
    total: usize,
) -> (InlineTable, Spent) {
    let t_start = Instant::now();
    set_thread_qos_user_interactive();
    let mut table = InlineTable::new(bits);
    let mut buf = vec![0u8; chunk + OVERLAP + TAIL_GUARD];
    let mut spent = Spent::default();

    loop {
        let i = cursor.fetch_add(1, Ordering::Relaxed);
        if i >= total {
            break;
        }
        let base = i * chunk;
        if base >= len {
            continue;
        }
        let avail = (chunk + OVERLAP).min(len - base);
        let t0 = Instant::now();
        file.read_exact_at(&mut buf[..avail], base as u64).expect("pread");
        let t1 = Instant::now();
        spent.read += t1 - t0;
        let view = &buf[..avail];

        let Some((start, end)) = local_bounds(view, base, chunk) else {
            continue;
        };

        let mut s = split_streams::<STREAMS>(view, start, end);

        loop {
            let mut all = true;
            for k in 0..STREAMS {
                all &= s[k].0 < s[k].1;
            }
            if !all {
                break;
            }
            for k in 0..STREAMS {
                s[k].0 = step(&buf, &mut table, s[k].0);
            }
        }

        for k in 0..STREAMS {
            while s[k].0 < s[k].1 {
                s[k].0 = step(&buf, &mut table, s[k].0);
            }
        }
        spent.parse += t1.elapsed();
    }
    spent.total = t_start.elapsed();
    (table, spent)
}

fn main() {
    let t_main = Instant::now();
    let path = env::args().nth(1).unwrap_or_else(|| "measurements.txt".to_string());
    let threads =
        env::var("OBRC_THREADS").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(8);

    let chunk = env::var("OBRC_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > MAX_LINE_LEN)
        .unwrap_or(CHUNK_SIZE);

    let bits = env::var("OBRC_TABLE_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&b| (1..=24).contains(&b))
        .unwrap_or(DEFAULT_TABLE_BITS);

    let file = File::open(&path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    let len = file.metadata().expect("stat").len() as usize;
    let total = num_chunks(len, chunk);
    let cursor = AtomicUsize::new(0);

    let t_setup = t_main.elapsed();

    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| scope.spawn(|| worker(&file, len, chunk, bits, &cursor, total)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });
    let t_workers = t_main.elapsed();

    let mut merged: BTreeMap<&[u8], Stats> = BTreeMap::new();
    for (t, _) in &results {
        for (name, stats) in t.iter() {
            merged.entry(name).and_modify(|s| s.merge(&stats)).or_insert(stats);
        }
    }

    let entries: Vec<(Vec<u8>, Stats)> = merged.into_iter().map(|(k, v)| (k.to_vec(), v)).collect();
    let t_merge = t_main.elapsed();

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    write_results(&mut out, &entries).unwrap();
    out.flush().unwrap();
    let t_out = t_main.elapsed();

    if env::var_os("OBRC_PHASES").is_some() {
        report_phases(&results, t_setup, t_workers, t_merge, t_out);
    }
    // Deliberate: 18 tables, each with a key arena, would otherwise be walked and freed here
    // for no purpose. That teardown is part of the wall clock a benchmark sees.
    std::process::exit(0);
}

/// Prints where the run went. Everything is measured from the top of `main`, so the gap between
/// `t_out` and the wall clock a benchmark reports is `exec` plus dynamic linking plus teardown —
/// real cost that no in-process timer can see.
fn report_phases(
    results: &[(InlineTable, Spent)],
    t_setup: Duration,
    t_workers: Duration,
    t_merge: Duration,
    t_out: Duration,
) {
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let mut spent: Vec<Spent> = results.iter().map(|(_, s)| *s).collect();

    eprintln!("  setup (open+stat)   {:7.1} ms", ms(t_setup));
    eprintln!("  workers             {:7.1} ms", ms(t_workers - t_setup));
    eprintln!("  merge               {:7.1} ms", ms(t_merge - t_workers));
    eprintln!("  output              {:7.1} ms", ms(t_out - t_merge));
    eprintln!("  --- from main entry {:7.1} ms", ms(t_out));

    // Per-thread, so these are core-milliseconds and sum to more than the wall clock.
    let sum = |f: fn(&Spent) -> Duration| spent.iter().map(|s| ms(f(s))).sum::<f64>();
    let n = spent.len() as f64;
    eprintln!(
        "  per thread, mean of {}: read {:.1} ms, parse {:.1} ms  ({:.0}% read)",
        spent.len(),
        sum(|s| s.read) / n,
        sum(|s| s.parse) / n,
        100.0 * sum(|s| s.read) / (sum(|s| s.read) + sum(|s| s.parse)),
    );

    // A thread that finishes early has idled while another still held a chunk. The spread is
    // the load imbalance, and it is wall clock nobody can use.
    spent.sort_by_key(|s| s.total);
    let (lo, hi) = (ms(spent[0].total), ms(spent[spent.len() - 1].total));
    eprintln!("  thread lifetime     {lo:7.1} .. {hi:.1} ms  (straggler costs {:.1} ms)", hi - lo);
}
