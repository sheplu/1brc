//! v16: take the next row's start from the newline, not from the value.
//!
//! **What v15 established.** Removing eleven instructions from the pair loop made it 9-14 ms
//! *slower*, and the disassembly said why: the instructions LLVM chose to add back — a
//! `mov`/`movk` rebuilding `0x10101000` — landed inside the loop-carried chain instead of beside
//! it. Two cycles of dependency cost more than eleven instructions of throughput saved. Read
//! together with `+cssc` (15 ms for a shorter `ctz`) and v13 (slower while removing twenty
//! instructions a row), the rule the project now has is that only the recurrence pays.
//!
//! **The recurrence.** One row's `pos` reaches the next one's like this:
//!
//! ```text
//!   pos -> load 16B -> semi mask -> ctz -> len -> semi+1 -> load 8B -> dot mask -> ctz -> pos'
//! ```
//!
//! Two `trailing_zeros` in series with a load between them, and that middle load is the worst
//! link in it: its *address* is what the first `ctz` produces, so it cannot even be issued until
//! the delimiter has been found. `+cssc` has already taken what there was to take from the two
//! `ctz`. Nothing had been tried on the load.
//!
//! **The change.** The next row starts one byte after this row's `\n`, and that fact needs no
//! knowledge of where the `;` is. [`line_len_flat`] scans for the `\n` over a 24-byte window
//! taken from `pos` — the sixteen bytes the delimiter scan already loads, plus one more word.
//! All three addresses are `pos` plus a constant, so all three loads issue together, and the
//! recurrence collapses to:
//!
//! ```text
//!   pos -> load 24B -> newline mask -> ctz -> pos'
//! ```
//!
//! One `ctz` instead of two, no dependent load, and nothing between the loads and the answer but
//! the mask. The value is still loaded from `semi + 1` and still parsed by the dot search, but
//! that whole chain now hangs off the loop rather than being part of it — its result reaches the
//! table and stops. `parse_temp_branchless` returns the next position too; it goes unused here
//! and LLVM drops the arithmetic that produced it.
//!
//! 24 bytes is not new slack. A name that fits the 16-byte window puts the `;` at offset 15 at
//! worst, and the parser then loads eight bytes from offset 16 — v12 already reads exactly this
//! far. `TAIL_GUARD` is unchanged and the edge cases it exists for are unchanged with it.
//!
//! **The bet, stated before measuring.** This buys latency with instructions: one more load, four
//! more ALU ops for the mask, and the selects, against roughly six cycles off the chain. Every
//! previous attempt in this project to trade the other way has lost, and v15's two cycles were
//! worth 9-14 ms, so the arithmetic says this is worth 30-40. What it cannot say is what the
//! register allocator does with a third word held live across the hash — which is precisely how
//! v15 lost, and the reason this is a separate binary rather than an edit to v12.
//!
//! **Measured** (`scripts/v16-newline.tsv`, REPS=5, duplicate arms 1-5 ms apart — the tightest
//! floor this project has had):
//!
//! ```text
//!                      t18            t8
//!   v12_pairs      376  376 ms   680  679 ms
//!   v16_newline    413  415 ms   716  720 ms
//!   v17_cursors    418  419 ms   737  740 ms
//! ```
//!
//! **Wrong, by 38 ms in the wrong direction.** The bet was 30-40 ms faster; the result is 10%
//! slower, at both thread counts, well outside a 5 ms floor. The recurrence *was* shortened — the
//! disassembly confirms all three loads issuing off `pos` and one `ctz` where there were two — and
//! it bought nothing.
//!
//! The cost side is what landed. The hot pair loop goes from 107 instructions a row to 143: three
//! `nl_mask`es are twelve ALU ops, the combine is six more, the 24-byte window brings its own
//! bounds check, and the 8- and 16-byte checks it should have subsumed stayed. That is three times
//! the adder budgeted above, and 38 ms across 1e9 rows and 18 threads is about 1 ms per
//! instruction-per-row — which is the rate the project should have been using all along.
//!
//! [`v17_cursors`](../v17_cursors/index.html) is the control that makes this conclusive rather than
//! merely negative, and it fails harder. See its doc comment: it removes a store-to-load round trip
//! from the loop-carried `pos` — four to six cycles, on the chain, exactly the thing this version
//! exists to shorten — and is 5 ms slower at t18 and 20 ms slower at t8. A change that removes
//! recurrence latency and loses is not consistent with a latency-bound loop. Taken with this
//! version, the pair loop is throughput-bound, and the model the last three versions were designed
//! against is inverted.
//!
//! One suspect for *which* throughput: this adds one 8-byte load a row to a body that had three, on
//! a machine where twelve of eighteen cores are the narrow tier. +33% on the load/store ports
//! predicts a slowdown of this size far better than +34% on instructions does. Untested here, but
//! it is the only remaining reading that also explains why `+cssc` — one instruction, no load —
//! is the single instruction-count change that ever paid.
//!
//! Everything else is v12: two interleaved streams, `pread` into a recycled per-thread buffer,
//! the fixed 16-byte `;` window, no final avalanche, the keyed probe, a table that grows on
//! demand.
//!
//! Usage: v16_newline [path]   (OBRC_THREADS, default 8; OBRC_TABLE_BITS, default 14)

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
use obrc::swar::{find_semi_and_hash_flat_raw, line_len_flat, scan_flat_keyed, FLAT_SLACK};
use obrc::sys::set_thread_qos_user_interactive;
use obrc::MAX_LINE_LEN;

/// See v10: 2^13..2^16 all measure the same, and this is the smallest of them.
const DEFAULT_TABLE_BITS: u32 = 14;

/// v12's pair loop, unchanged. Re-swept in `scripts/v16-newline.tsv` once the chain moved.
const STREAMS: usize = 2;

/// Read past the chunk so the line straddling its end can be finished locally.
const OVERLAP: usize = MAX_LINE_LEN;

/// Slack past the read that the wide path may touch: the flat scan reads a 16-byte window
/// from the name start, the parser loads 8 bytes, and the cold path's inline probe reads 32.
const TAIL_GUARD: usize = MAX_LINE_LEN + FLAT_SLACK;
const _: () = assert!(TAIL_GUARD >= KEY_SLACK);

/// Consumes the line at `pos` and returns the start of the next one.
///
/// The newline scan comes first because it is the only thing the next iteration waits on. What
/// follows it — the delimiter, the hash, the value, the table — is a side effect of this row that
/// no later row reads, so the two halves overlap instead of queueing.
#[inline(always)]
fn step(buf: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    let next = pos + line_len_flat(buf, pos) + 1;
    let Some((semi, hash, klo, khi)) = scan_flat_keyed(buf, pos) else {
        return step_long(buf, table, pos);
    };
    let (value, _) = parse_temp_branchless(buf, semi + 1);
    table.upsert_words(buf, pos, semi - pos, hash, value, klo, khi);
    next
}

/// A name of 16 bytes or more: v10's row, out of line so it shares no registers with the hot
/// path. 2.9% of the official station list and none of this dataset's common names.
///
/// This one keeps the dot search, because a row this long may end past the newline scan's
/// window and the scan says so by declining rather than by guessing.
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
/// under `OBRC_PHASES`.
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

/// Prints where the run went. Everything is measured from the top of `main`.
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

    let sum = |f: fn(&Spent) -> Duration| spent.iter().map(|s| ms(f(s))).sum::<f64>();
    let n = spent.len() as f64;
    eprintln!(
        "  per thread, mean of {}: read {:.1} ms, parse {:.1} ms  ({:.0}% read)",
        spent.len(),
        sum(|s| s.read) / n,
        sum(|s| s.parse) / n,
        100.0 * sum(|s| s.read) / (sum(|s| s.read) + sum(|s| s.parse)),
    );

    spent.sort_by_key(|s| s.total);
    let (lo, hi) = (ms(spent[0].total), ms(spent[spent.len() - 1].total));
    eprintln!("  thread lifetime     {lo:7.1} .. {hi:.1} ms  (straggler costs {:.1} ms)", hi - lo);
}
